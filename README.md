# 即時語言偵測 (Whisper)

此專案提供即時語言偵測與轉寫。

## 系統流程
1. 麥克風即時收音 (sounddevice)
2. VAD 過濾語音片段 (webrtcvad)
3. Whisper 語言偵測 + 轉寫

## 安裝
```bash
pip install -r requirements.txt
```

## 執行
```bash
python main.py --stream
```

## 參數說明
- `-m, --model` : Whisper 模型 (tiny/base/small/medium/large)
- `--device` : 運算設備 (cpu/cuda)
- `--stream-window` : 串流視窗大小 (秒)
- `--stream-interval` : 推論間隔 (秒)
- `--stream-vad` : WebRTC VAD 強度 (0-3)
