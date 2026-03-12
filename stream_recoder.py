"""
即時語言偵測

系統流程:
1) sounddevice 讀取麥克風串流
2) 以 WebRTC VAD 判斷語音片段
3) Denoiser 降噪
4) SpeechBrain 語言分類
5) Whisper 轉寫（初始語系只用一次 Whisper 偵測）
"""

import time
import logging
from collections import deque
from typing import Deque, List, Optional, Dict
import threading
import sys
import os

import numpy as np

from config import StreamConfig, ModelConfig, LANGUAGE_NAMES, SUPPORTED_LANGUAGES

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
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    from speechbrain.inference import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
except Exception:
    SPEECHBRAIN_AVAILABLE = False

try:
    from denoiser import pretrained
    DENOISER_AVAILABLE = True
except Exception:
    DENOISER_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False

# SpeechBrain (voxlingua) 語言碼轉換為 ISO-639-1
LANGUAGE_CODE_MAP: Dict[str, str] = {
    "cmn": "zh",
    "yue": "zh",
    "zho": "zh",
    "eng": "en",
    "jpn": "ja",
    "kor": "ko",
}


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

    def is_locked(self) -> bool:
        return self.locked is not None

    def display_lang(self, fallback_lang: str) -> str:
        return self.locked if self.locked else fallback_lang


class StreamRecognizer:
    """即時辨識主體"""
    def __init__(
        self,
        model_name: str = "base",
        stream_config: StreamConfig = None,
        model_config: ModelConfig = None,
        logger: logging.Logger = None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.config = stream_config or StreamConfig()
        self.model_config = model_config or ModelConfig()

        self.model_name = model_name
        self.model = None
        self._classifier = None
        self._denoiser = None

        # frame 與 buffer 參數
        self._frame_size = int(self.config.sample_rate * self.config.frame_ms / 1000)
        self._window_samples = int(self.config.sample_rate * self.config.stream_window)

        # ring buffer 儲存最近一段音訊
        self._ring_buffer: Deque[int] = deque(maxlen=self._window_samples)
        # 接收 callback 來的 PCM frame
        self._pcm_frames: Deque[np.ndarray] = deque()
        self._last_infer_time = 0.0

        self._stop = False
        self._vad = None
        self._state = LanguageStateMachine(self.config)
        self._model_lock = threading.Lock()
        self._initial_lang: Optional[str] = None

        self._classifier_device = self.model_config.classifier_device
        self._whisper_device = self.model_config.whisper_device

        self._init_dependencies()

    def _init_dependencies(self):
        # 檢查依賴
        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError("未安裝 sounddevice，請執行: pip install sounddevice")
        if not WEBRTCVAD_AVAILABLE:
            raise RuntimeError("未安裝 webrtcvad，請執行: pip install webrtcvad")
        if not TORCH_AVAILABLE:
            raise RuntimeError("未安裝 torch，請執行: pip install torch")
        if not SPEECHBRAIN_AVAILABLE:
            raise RuntimeError("未安裝 speechbrain，請執行: pip install speechbrain")
        if not WHISPER_AVAILABLE:
            raise RuntimeError("未安裝 openai-whisper，請執行: pip install openai-whisper")
        if self.model_config.use_denoiser and not DENOISER_AVAILABLE:
            raise RuntimeError("未安裝 denoiser，請執行: pip install denoiser")

        if self._classifier_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("classifier_device 設為 cuda 但 torch.cuda.is_available() 為 False")
        if self._whisper_device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("whisper_device=cpu (cuda 不可用，已自動改為 cpu)")
            self._whisper_device = "cpu"

        # 初始化 VAD
        self._vad = webrtcvad.Vad(self.config.vad_aggressiveness)

        # 初始化 SpeechBrain 分類器與降噪
        torch.set_grad_enabled(False)
        self._classifier = EncoderClassifier.from_hparams(
            source=self.model_config.classifier_source,
            run_opts={"device": self._classifier_device},
        )
        if self.model_config.use_denoiser:
            self._denoiser = pretrained.dns64().to(self._classifier_device).eval()

        # 初始化 Whisper
        self.logger.info(f"載入 Whisper 模型: {self.model_name}...")
        self.model = whisper.load_model(self.model_name, device=self._whisper_device)
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
            if self._vad.is_speech(frame.tobytes(), self.config.sample_rate):
                self._pcm_frames.append(frame)

    def _collect_window(self) -> Optional[np.ndarray]:
        # 將新進 PCM frame 放入 ring buffer
        while self._pcm_frames:
            frame = self._pcm_frames.popleft()
            self._ring_buffer.extend(frame)

        # ring buffer 不足就等待
        if len(self._ring_buffer) < self._window_samples:
            return None

        # VAD 過濾出語音區段
        audio_np = np.array(self._ring_buffer, dtype=np.float32) / 32768.0
        vad_frames: List[float] = []
        for i in range(0, len(audio_np), self._frame_size):
            frame = audio_np[i:i + self._frame_size]
            if len(frame) < self._frame_size:
                continue
            if self._vad.is_speech((frame * 32768).astype(np.int16).tobytes(), self.config.sample_rate):
                vad_frames.extend(frame)

        if len(vad_frames) < int(self.config.sample_rate * self.config.min_speech_duration):
            return None

        return np.array(vad_frames, dtype=np.float32)

    def _detect_initial_language(self, audio_float: np.ndarray) -> str:
        start = time.perf_counter()
        with self._model_lock:
            audio_padded = whisper.pad_or_trim(audio_float)
            mel = whisper.log_mel_spectrogram(audio_padded).to(self.model.device)
            _, probs = self.model.detect_language(mel)
        topk = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        detected_lang = topk[0][0] if topk else "en"
        self.logger.debug("初始語系偵測耗時: %.1fms", (time.perf_counter() - start) * 1000.0)
        return detected_lang if detected_lang in SUPPORTED_LANGUAGES else "en"

    def _classify_language(self, audio_float: np.ndarray) -> Dict[str, object]:
        start = time.perf_counter()

        tensor = torch.from_numpy(audio_float).unsqueeze(0).to(self._classifier_device)
        with torch.no_grad():
            if self._denoiser:
                tensor = self._denoiser(tensor).squeeze(1)
            else:
                tensor = tensor.squeeze(1)

        out = self._classifier.classify_batch(tensor)
        logits = out[0]
        probs = F.softmax(logits, dim=1)

        top5_prob, top5_idx = torch.topk(probs, 5, dim=1)
        top5_langs = self._classifier.hparams.label_encoder.decode_torch(top5_idx[0])
        top_lang = top5_langs[0]
        top_lang_code = top_lang.split(":")[0]
        top_prob_val = top5_prob[0][0].item()

        mapped_lang = LANGUAGE_CODE_MAP.get(top_lang_code, "unknown")
        self._state.update(mapped_lang, top_prob_val)

        return {
            "mapped_lang": mapped_lang,
            "raw_language": top_lang_code,
            "confidence": top_prob_val,
            "top5_langs": top5_langs,
            "top5_probs": top5_prob[0].tolist(),
            "detect_ms": (time.perf_counter() - start) * 1000.0,
        }

    def _transcribe(self, audio_float: np.ndarray, language: str) -> Dict[str, object]:
        start = time.perf_counter()
        use_fp16 = self.model.device.type == "cuda"
        with self._model_lock:
            transcribe_result = self.model.transcribe(
                audio_float,
                language=language,
                fp16=use_fp16,
            )
        transcribe_ms = (time.perf_counter() - start) * 1000.0
        return {
            "result": transcribe_result,
            "transcribe_ms": transcribe_ms,
        }

    def _infer(self, audio_float: np.ndarray) -> Dict[str, object]:
        classify_result = self._classify_language(audio_float)
        mapped_lang = classify_result["mapped_lang"]
        confidence = classify_result["confidence"]

        if self._state.is_locked():
            lang_for_text = self._state.display_lang(mapped_lang)
        else:
            if self._initial_lang is None:
                self._initial_lang = self._detect_initial_language(audio_float)
            lang_for_text = self._initial_lang

        display_lang = lang_for_text
        transcribe_out = self._transcribe(audio_float, lang_for_text)
        transcribe_result = transcribe_out["result"]

        return {
            "success": True,
            "language": display_lang,
            "language_name": LANGUAGE_NAMES.get(display_lang, display_lang),
            "raw_language": classify_result["raw_language"],
            "confidence": confidence,
            "top5_langs": classify_result.get("top5_langs", []),
            "top5_probs": classify_result.get("top5_probs", []),
            "detect_ms": classify_result.get("detect_ms", 0.0),
            "transcribe_ms": transcribe_out["transcribe_ms"],
            "text": transcribe_result.get("text", "").strip(),
            "audio_duration": len(audio_float) / self.config.sample_rate,
        }

    def run(self, on_result=None):
        # 主迴圈：收音 -> VAD -> SpeechBrain -> Whisper
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
                    infer_start = time.perf_counter()
                    result = self._infer(audio_float)
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

    def _print_result(self, result: Dict[str, object]):
        # 預設輸出格式
        lang = result.get("language", "unknown")
        lang_name = result.get("language_name", lang)
        conf = result.get("confidence", 0.0)
        text = result.get("text", "")
        duration = result.get("audio_duration", 0.0)
        top5_langs = result.get("top5_langs", [])
        top5_probs = result.get("top5_probs", [])
        timing = result.get("timing", {})

        print("\n" + "=" * 60)
        print(f"語言 : {lang_name} ({lang})  信心={conf:.2%}  長度={duration:.2f}s")
        if top5_langs:
            top5_text = ", ".join(
                f"{code}={prob:.2%}" for code, prob in zip(top5_langs, top5_probs)
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
        self.model = None
        self._classifier = None
        self._denoiser = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    with StreamRecognizer() as recognizer:
        recognizer.run()
