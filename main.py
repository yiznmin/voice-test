#!/usr/bin/env python3
"""
即時 Whisper 語言偵測主程式

流程說明:
1) 解析參數 (模型、裝置、串流設定)
2) 初始化串流設定
3) 啟動 StreamRecognizer 即時辨識

使用方式:
    python main.py --stream
"""

import logging
import argparse
import sys
import os

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

from stream_recoder import StreamRecognizer
from config import StreamConfig, ModelConfig

# 強制 UTF-8 輸出，避免 Windows 終端機顯示亂碼
os.environ.setdefault("PYTHONUTF8", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def setup_logging(verbose: bool = False):
    """設定 logging（是否顯示詳細訊息）"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )


def main():
    # === 參數解析 ===
    parser = argparse.ArgumentParser(
        description="即時 Whisper 語言偵測",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  python main.py --stream
  python main.py --stream -m small
  python main.py --stream --device cuda
  python main.py --stream --stream-window 1.0 --stream-interval 0.5
        """
    )

    # Whisper 模型設定
    parser.add_argument("-m", "--model", default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper 模型 (預設: base)")
    parser.add_argument("--device", default=None,
                        choices=["cpu", "cuda"],
                        help="運算設備 (cpu/cuda)")
    parser.add_argument("--classifier-device", default="cpu",
                        choices=["cpu", "cuda"],
                        help="語言分類/降噪設備 (預設: cpu)")

    # 即時串流參數
    parser.add_argument("--stream", action="store_true",
                        help="啟用即時串流辨識")
    parser.add_argument("--stream-window", type=float, default=None,
                        help="串流視窗大小 (秒)；越大越穩定但延遲較高")
    parser.add_argument("--stream-interval", type=float, default=None,
                        help="推論間隔 (秒)；越小越即時但計算量較高")
    parser.add_argument("--stream-vad", type=int, default=None,
                        help="WebRTC VAD 強度 (0-3)；越大越保守")

    # 日誌輸出
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="輸出詳細 log")

    args = parser.parse_args()

    setup_logging(args.verbose)

    # 沒有指定 --stream 就顯示說明
    if not args.stream:
        parser.print_help()
        return

    # === 建立串流設定 ===
    stream_config = StreamConfig()
    if args.stream_window is not None:
        stream_config.stream_window = args.stream_window
    if args.stream_interval is not None:
        stream_config.infer_interval_sec = args.stream_interval
    if args.stream_vad is not None:
        stream_config.vad_aggressiveness = args.stream_vad

    # === 預設裝置：Whisper 走 GPU(若可用)，否則 CPU ===
    device = args.device
    if device is None:
        if TORCH_AVAILABLE and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    model_config = ModelConfig(
        classifier_device=args.classifier_device,
        whisper_device=device,
    )

    # === 啟動即時辨識 ===
    with StreamRecognizer(
        model_name=args.model,
        stream_config=stream_config,
        model_config=model_config,
    ) as recognizer:
        recognizer.run()


if __name__ == "__main__":
    main()
