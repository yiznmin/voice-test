"""
即時 Whisper 語言偵測

系統流程:
1) sounddevice 讀取麥克風串流
2) 以 WebRTC VAD 判斷語音片段
3) 將語音片段送入 Whisper 做語言偵測 + 轉寫
"""

import time
import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import threading
import sys
import os

import numpy as np

from config import StreamConfig, LANGUAGE_NAMES, SUPPORTED_LANGUAGES

# 強制 UTF-8 輸出，避免 Windows 終端機顯示亂碼
os.environ.setdefault("PYTHONUTF8", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# 依賴套件檢查
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except Exception:
    SOUNDDEVICE_AVAILABLE = False

try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
except Exception:
    WEBRTCVAD_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False


class LanguageStateMachine:
    """語言穩定狀態機：避免語言標籤快速跳動"""
    def __init__(self, config: StreamConfig):
        self.config = config
        self.state = "UNDECIDED"
        self.locked = None
        self.has_locked_once = False
        self.history = deque(maxlen=5)

    def update(self, lang: str, prob: float) -> str:
        # 低信心或不支援語言先視為 UNKNOWN
        if lang not in SUPPORTED_LANGUAGES or prob < self.config.confidence_threshold:
            self.history.append(("UNKNOWN", prob))
        else:
            self.history.append((lang, prob))

        # 尚未鎖定語言時，連續多次一致才鎖定
        if self.state == "UNDECIDED":
            valid = [
                l for l, p in self.history
                if l != "UNKNOWN" and p >= self.config.confidence_threshold
            ]

            if len(valid) >= self.config.stable_count:
                if len(set(valid[-self.config.stable_count:])) == 1:
                    self.locked = valid[-1]
                    self.state = "LOCKED"
                    self.has_locked_once = True

        # 已鎖定語言時，若新語言高信心連續出現才切換
        elif self.state == "LOCKED":
            switch = [
                l for l, p in self.history
                if p >= self.config.switch_threshold and l != self.locked
            ]
            if len(switch) >= self.config.switch_count:
                if len(set(switch[-self.config.switch_count:])) == 1:
                    self.locked = switch[-1]

        return self.locked if self.locked else "UNKNOWN"


class StreamRecognizer:
    """即時辨識主體"""
    def __init__(self, model_name: str = "base", device: str = None,
                 stream_config: StreamConfig = None, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.config = stream_config or StreamConfig()

        self.model_name = model_name
        self.device = device
        self.model = None

        # frame 與 buffer 參數
        self._frame_size = int(self.config.sample_rate * self.config.frame_ms / 1000)
        self._window_samples = int(self.config.sample_rate * self.config.stream_window)

        # ring buffer 儲存最近一段音訊
        self._ring_buffer = deque(maxlen=self._window_samples)
        # 接收 callback 來的 PCM frame
        self._pcm_frames = deque()
        self._last_infer_time = 0.0

        self._stop = False
        self._vad = None
        self._state = LanguageStateMachine(self.config)
        self._lang_executor = ThreadPoolExecutor(max_workers=1) if self.config.parallel_language_detection else None
        self._lang_future = None
        self._last_lang_result = None
        self._model_lock = threading.Lock()

        self._init_dependencies()

    def _init_dependencies(self):
        # 檢查依賴
        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError("未安裝 sounddevice，請執行: pip install sounddevice")
        if not WEBRTCVAD_AVAILABLE:
            raise RuntimeError("未安裝 webrtcvad，請執行: pip install webrtcvad")
        if not WHISPER_AVAILABLE:
            raise RuntimeError("未安裝 openai-whisper，請執行: pip install openai-whisper")

        # 初始化 VAD 與 Whisper
        self._vad = webrtcvad.Vad(self.config.vad_aggressiveness)
        self.logger.info(f"載入 Whisper 模型: {self.model_name}...")
        self.model = whisper.load_model(self.model_name, device=self.device)
        self.logger.info("Whisper 模型載入完成")

    def _audio_callback(self, indata, frames, time_info, status):
        # 串流回呼：只做最小處理，避免阻塞
        if status:
            self.logger.debug(status)
        pcm = (indata[:, 0] * 32768).astype(np.int16)
        for i in range(0, len(pcm), self._frame_size):
            frame = pcm[i:i + self._frame_size]
            if len(frame) < self._frame_size:
                continue
            self._pcm_frames.append(frame)

    def _collect_window(self) -> np.ndarray:
        # 將新進 PCM frame 放入 ring buffer
        while self._pcm_frames:
            frame = self._pcm_frames.popleft()
            self._ring_buffer.extend(frame)

        # ring buffer 不足就等待
        if len(self._ring_buffer) < self._window_samples:
            return None

        # VAD 過濾出語音區段
        audio_np = np.array(self._ring_buffer, dtype=np.int16)
        vad_frames = []
        for i in range(0, len(audio_np), self._frame_size):
            frame = audio_np[i:i + self._frame_size]
            if len(frame) < self._frame_size:
                continue
            if self._vad.is_speech(frame.tobytes(), self.config.sample_rate):
                vad_frames.extend(frame)

        # 語音太短就略過
        if len(vad_frames) < int(self.config.sample_rate * self.config.min_speech_duration):
            return None

        # 轉成 Whisper 需要的 float32
        audio_float = np.array(vad_frames, dtype=np.float32) / 32768.0
        return audio_float

    def _detect_language_topk(self, audio_float: np.ndarray, allow_skip: bool = False) -> dict:
        if allow_skip:
            acquired = self._model_lock.acquire(blocking=False)
            if not acquired:
                return None
        else:
            self._model_lock.acquire()

        start = time.perf_counter()
        try:
            audio_padded = whisper.pad_or_trim(audio_float)
            mel = whisper.log_mel_spectrogram(audio_padded).to(self.model.device)
            _, probs = self.model.detect_language(mel)
            topk = sorted(probs.items(), key=lambda x: x[1], reverse=True)[: self.config.topk_languages]
            detected_lang, confidence = topk[0]
            detect_ms = (time.perf_counter() - start) * 1000.0
            return {
                "detected_lang": detected_lang,
                "confidence": float(confidence),
                "topk": topk,
                "detect_ms": detect_ms,
            }
        finally:
            self._model_lock.release()

    def _get_latest_language(self, audio_float: np.ndarray) -> dict:
        if self._lang_future and self._lang_future.done():
            try:
                latest = self._lang_future.result()
                if latest:
                    self._last_lang_result = latest
                    self._last_lang_result["stable_lang"] = self._state.update(
                        self._last_lang_result["detected_lang"],
                        self._last_lang_result["confidence"],
                    )
            except Exception as exc:
                self.logger.exception("語言偵測背景執行失敗: %s", exc)
            finally:
                self._lang_future = None

        if self._last_lang_result is None:
            self._last_lang_result = self._detect_language_topk(audio_float, allow_skip=False)
            if self._last_lang_result:
                self._last_lang_result["stable_lang"] = self._state.update(
                    self._last_lang_result["detected_lang"],
                    self._last_lang_result["confidence"],
                )

        if self._lang_executor and self._lang_future is None:
            audio_copy = np.array(audio_float, copy=True)
            self._lang_future = self._lang_executor.submit(self._detect_language_topk, audio_copy, True)

        return self._last_lang_result

    def _transcribe(self, audio_float: np.ndarray, language: str) -> dict:
        start = time.perf_counter()
        use_fp16 = self.model.device.type == "cuda"
        with self._model_lock:
            transcribe_result = self.model.transcribe(
                audio_float,
                language=language,
                fp16=use_fp16
            )
        transcribe_ms = (time.perf_counter() - start) * 1000.0
        return {
            "result": transcribe_result,
            "transcribe_ms": transcribe_ms,
        }

    def _infer(self, audio_float: np.ndarray, lang_result: dict) -> dict:
        detected_lang = lang_result["detected_lang"]
        confidence = lang_result["confidence"]
        stable_lang = lang_result.get("stable_lang", "UNKNOWN")
        lang_for_text = stable_lang if stable_lang != "UNKNOWN" else detected_lang
        display_lang = lang_for_text

        transcribe_out = self._transcribe(audio_float, lang_for_text)
        transcribe_result = transcribe_out["result"]

        return {
            "success": True,
            "language": display_lang,
            "language_name": LANGUAGE_NAMES.get(display_lang, display_lang),
            "raw_language": detected_lang,
            "confidence": confidence,
            "top5": lang_result.get("topk", []),
            "detect_ms": lang_result.get("detect_ms", 0.0),
            "transcribe_ms": transcribe_out["transcribe_ms"],
            "text": transcribe_result.get("text", "").strip(),
            "audio_duration": len(audio_float) / self.config.sample_rate,
        }

    def run(self, on_result=None):
        # 主迴圈：收音 -> VAD -> Whisper
        if not self.model:
            raise RuntimeError("Whisper 模型尚未載入")

        self.logger.info("開始即時串流，按 Ctrl+C 停止")
        self._stop = False

        with sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=1,
            blocksize=int(self.config.sample_rate * self.config.block_sec),
            dtype="float32",
            callback=self._audio_callback,
        ):
            try:
                while not self._stop:
                    if not self._pcm_frames:
                        time.sleep(0.01)
                        continue

                    now = time.time()
                    if now - self._last_infer_time < self.config.infer_interval_sec:
                        time.sleep(0.01)
                        continue

                    collect_start = time.perf_counter()
                    audio_float = self._collect_window()
                    if audio_float is None:
                        continue
                    collect_ms = (time.perf_counter() - collect_start) * 1000.0

                    self._last_infer_time = now
                    lang_result = self._get_latest_language(audio_float)
                    infer_start = time.perf_counter()
                    result = self._infer(audio_float, lang_result)
                    infer_ms = (time.perf_counter() - infer_start) * 1000.0
                    result["timing"] = {
                        "collect_ms": collect_ms,
                        "detect_ms": result.get("detect_ms", 0.0),
                        "transcribe_ms": result.get("transcribe_ms", 0.0),
                        "infer_ms": infer_ms,
                        "total_ms": collect_ms + infer_ms,
                    }

                    if on_result:
                        on_result(result)
                    else:
                        self._print_result(result)

            except KeyboardInterrupt:
                self.logger.info("使用者停止串流")

    def stop(self):
        self._stop = True

    def _print_result(self, result: dict):
        # 預設輸出格式
        lang = result.get("language", "unknown")
        lang_name = result.get("language_name", lang)
        conf = result.get("confidence", 0.0)
        text = result.get("text", "")
        duration = result.get("audio_duration", 0.0)
        top5 = result.get("top5", [])
        timing = result.get("timing", {})

        print("\n" + "=" * 60)
        print(f"語言 : {lang_name} ({lang})  信心={conf:.2%}  長度={duration:.2f}s")
        if top5:
            top5_text = ", ".join(
                f"{LANGUAGE_NAMES.get(code, code)}({code})={prob:.2%}"
                for code, prob in top5
            )
            print(f"Top5 : {top5_text}")
        if timing:
            print(
                "時間 : "
                f"collect={timing.get('collect_ms', 0.0):.1f}ms, "
                f"detect={timing.get('detect_ms', 0.0):.1f}ms, "
                f"transcribe={timing.get('transcribe_ms', 0.0):.1f}ms, "
                f"total={timing.get('total_ms', 0.0):.1f}ms"
            )
        if text:
            print(f"文字 : {text}")
        else:
            print("文字 : <無>")

    def close(self):
        self.stop()
        if self._lang_executor:
            self._lang_executor.shutdown(wait=False, cancel_futures=True)
        self.model = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    with StreamRecognizer() as recognizer:
        recognizer.run()
