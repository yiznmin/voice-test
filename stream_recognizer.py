"""
即時語言辨識核心

系統流程:
1) sounddevice 收音 -> PCM frame
2) WebRTC VAD 過濾語音片段
3) Denoiser 降噪
4) SpeechBrain 語言分類
5) 狀態機穩定輸出語言
"""

import time
import logging
from collections import deque
from typing import Deque, List, Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F

from speechbrain.inference import EncoderClassifier
from denoiser import pretrained
import sounddevice as sd
import webrtcvad

from .config import StreamConfig, StateConfig, ModelConfig, LANGUAGE_NAMES
from .state_machine import LanguageStateMachine


class StreamRecognizer:
    """即時串流辨識"""
    def __init__(
        self,
        stream_cfg: StreamConfig,
        state_cfg: StateConfig,
        model_cfg: ModelConfig,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.stream_cfg = stream_cfg
        self.state_cfg = state_cfg
        self.model_cfg = model_cfg

        self.frame_size = int(self.stream_cfg.sample_rate * self.stream_cfg.frame_ms / 1000)
        self.window_samples = int(self.stream_cfg.sample_rate * self.stream_cfg.stream_window)

        self.ring_buffer: Deque[int] = deque(maxlen=self.window_samples)
        self.pcm_frames: Deque[np.ndarray] = deque()

        self.state_machine = LanguageStateMachine(self.state_cfg)

        self.device = self.model_cfg.device
        torch.set_grad_enabled(False)

        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/lang-id-voxlingua107-ecapa",
            run_opts={"device": self.device}
        )
        self.denoiser = pretrained.dns64().to(self.device).eval()
        self.vad = webrtcvad.Vad(self.stream_cfg.vad_aggressiveness)

        self._stop = False
        self._inference_times: List[float] = []

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            self.logger.debug(status)
        pcm = (indata[:, 0] * 32768).astype(np.int16)
        for i in range(0, len(pcm), self.frame_size):
            frame = pcm[i:i + self.frame_size]
            if len(frame) < self.frame_size:
                continue
            if self.vad.is_speech(frame.tobytes(), self.stream_cfg.sample_rate):
                self.pcm_frames.append(frame)

    def _collect_window(self) -> Optional[np.ndarray]:
        while self.pcm_frames:
            frame = self.pcm_frames.popleft()
            self.ring_buffer.extend(frame)

        if len(self.ring_buffer) < self.window_samples:
            return None

        audio_np = np.array(self.ring_buffer, dtype=np.float32) / 32768.0
        vad_frames: List[float] = []
        for i in range(0, len(audio_np), self.frame_size):
            frame = audio_np[i:i + self.frame_size]
            if len(frame) < self.frame_size:
                continue
            if self.vad.is_speech((frame * 32768).astype(np.int16).tobytes(), self.stream_cfg.sample_rate):
                vad_frames.extend(frame)

        if not vad_frames:
            return None

        return np.array(vad_frames, dtype=np.float32)

    def _infer(self, audio_float: np.ndarray) -> Dict[str, object]:
        # 降噪
        tensor = torch.from_numpy(audio_float).unsqueeze(0)
        with torch.no_grad():
            tensor = self.denoiser(tensor).squeeze(1)

        # 語言分類
        out = self.classifier.classify_batch(tensor)
        logits = out[0]
        probs = F.softmax(logits, dim=1)

        top5_prob, top5_idx = torch.topk(probs, 5, dim=1)
        top5_langs = self.classifier.hparams.label_encoder.decode_torch(top5_idx[0])
        top_lang = top5_langs[0]
        top_lang_code = top_lang.split(":")[0]
        top_prob_val = top5_prob[0][0].item()

        # 狀態機穩定輸出
        _ = self.state_machine.update(top_lang_code, top_prob_val)
        display_lang = self.state_machine.display_lang(top_lang_code)

        return {
            "success": True,
            "language": display_lang,
            "language_name": LANGUAGE_NAMES.get(display_lang, display_lang),
            "raw_language": top_lang_code,
            "confidence": top_prob_val,
            "top5_langs": top5_langs,
            "top5_probs": top5_prob[0].tolist(),
            "audio_duration": len(audio_float) / self.stream_cfg.sample_rate,
        }

    def run(self):
        self.logger.info("開始即時串流，按 Ctrl+C 停止")

        with sd.InputStream(
            samplerate=self.stream_cfg.sample_rate,
            channels=1,
            blocksize=int(self.stream_cfg.sample_rate * self.stream_cfg.block_sec),
            dtype="float32",
            callback=self._audio_callback,
        ):
            try:
                while not self._stop:
                    if not self.pcm_frames:
                        time.sleep(0.01)
                        continue

                    audio_float = self._collect_window()
                    if audio_float is None:
                        time.sleep(0.01)
                        continue

                    result = self._infer(audio_float)
                    self._print_result(result)

            except KeyboardInterrupt:
                self.logger.info("使用者停止串流")

    def stop(self):
        self._stop = True

    def _print_result(self, result: Dict[str, object]):
        lang = result.get("language", "unknown")
        lang_name = result.get("language_name", lang)
        conf = result.get("confidence", 0.0)
        duration = result.get("audio_duration", 0.0)
        top5_langs = result.get("top5_langs", [])
        top5_probs = result.get("top5_probs", [])

        print("\n" + "=" * 60)
        print(f"語言: {lang_name} ({lang})  信心: {conf:.2%}  長度: {duration:.2f}s")
        print("Top-5 可信度:")
        for i in range(min(5, len(top5_langs))):
            bar_len = int(top5_probs[i] * 20)
            bar = "#" * bar_len
            print(f"{top5_langs[i]:<12} {bar:<20} {top5_probs[i]*100:6.2f}%")
