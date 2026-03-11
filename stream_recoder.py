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

    def _infer(self, audio_float: np.ndarray) -> dict:
        # Whisper 語言偵測
        audio_padded = whisper.pad_or_trim(audio_float)
        mel = whisper.log_mel_spectrogram(audio_padded).to(self.model.device)
        _, probs = self.model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)
        confidence = float(probs[detected_lang])

        # 使用狀態機穩定語言
        stable_lang = self._state.update(detected_lang, confidence)
        lang_for_text = stable_lang if stable_lang != "UNKNOWN" else detected_lang

        # 顯示語言規則:
        # 1) 初始未鎖定時先顯示英文
        # 2) 一旦曾鎖定過，未穩定時顯示 UNKNOWN
        if self._state.locked is None and not self._state.has_locked_once:
            display_lang = "en"
        elif self._state.locked is None:
            display_lang = "UNKNOWN"
        else:
            display_lang = self._state.locked

        # Whisper 轉寫文字
        use_fp16 = self.model.device.type == "cuda"
        transcribe_result = self.model.transcribe(
            audio_float,
            language=lang_for_text,
            fp16=use_fp16
        )

        return {
            "success": True,
            "language": display_lang,
            "language_name": LANGUAGE_NAMES.get(display_lang, display_lang),
            "raw_language": detected_lang,
            "confidence": confidence,
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

                    audio_float = self._collect_window()
                    if audio_float is None:
                        continue

                    self._last_infer_time = now
                    result = self._infer(audio_float)

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

        print("\n" + "=" * 60)
        print(f"語言 : {lang_name} ({lang})  信心={conf:.2%}  長度={duration:.2f}s")
        if text:
            print(f"文字 : {text}")
        else:
            print("文字 : <無>")

    def close(self):
        self.stop()
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
