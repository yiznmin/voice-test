"""
即時 Whisper 語言偵測設定
"""

from dataclasses import dataclass
from typing import Dict

# 支援語言清單 (語言代碼 -> 顯示名稱)
SUPPORTED_LANGUAGES: Dict[str, str] = {
    "zh": "中文 (Chinese)",
    "en": "英文 (English)",
    "ja": "日文 (Japanese)",
    "ko": "韓文 (Korean)",
}

# 語言代碼對應簡短名稱
LANGUAGE_NAMES: Dict[str, str] = {
    "zh": "中文",
    "en": "英文",
    "ja": "日文",
    "ko": "韓文",
    "unknown": "未知",
}


@dataclass
class StreamConfig:
    """即時串流設定 (Whisper + VAD)"""
    # 音訊取樣設定
    sample_rate: int = 16000   # 16kHz，符合 Whisper 預期
    frame_ms: int = 30         # VAD 允許 10/20/30 ms

    # 串流視窗設定
    stream_window: float = 1.0       # 分析視窗長度 (秒)
    block_sec: float = 0.5           # 音訊區塊大小 (秒)
    infer_interval_sec: float = 0.5  # 推論最小間隔 (秒)

    # 語音片段判斷
    min_speech_duration: float = 0.3 # 最短語音片段 (秒)
    vad_aggressiveness: int = 2      # VAD 強度 (0-3)

    # 語言穩定判斷 (避免快速抖動)
    confidence_threshold: float = 0.60
    stable_count: int = 3
    switch_count: int = 4
    switch_threshold: float = 0.93

    # 語言偵測輸出與並行設定
    topk_languages: int = 5
    parallel_language_detection: bool = True


@dataclass
@dataclass
class ModelConfig:
    """模型與裝置設定"""
    classifier_device: str = "cpu"  # SpeechBrain / Denoiser 用的裝置
    whisper_device: str = "cpu"     # Whisper 用的裝置
    use_denoiser: bool = True
    classifier_source: str = "speechbrain/lang-id-voxlingua107-ecapa"
