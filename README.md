# 即時語言偵測 (SpeechBrain + Whisper)

此專案提供即時語言偵測與轉寫。

## 系統流程
1. 麥克風即時收音 (sounddevice)
2. VAD 過濾語音片段 (webrtcvad)
3. Denoiser 降噪 (denoiser)
4. SpeechBrain 語言分類
5. Whisper 轉寫（初始語系只用一次 Whisper 偵測）

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
- `--device` : Whisper 運算設備 (cpu/cuda)
- `--classifier-device` : 語言分類/降噪設備 (cpu/cuda)
- `--stream-window` : 串流視窗大小 (秒)
- `--stream-interval` : 推論間隔 (秒)
- `--stream-vad` : WebRTC VAD 強度 (0-3)

## 語言邏輯
- 語系主要依賴 SpeechBrain + 狀態機穩定
- 初始語系只用 Whisper 偵測一次
- 初始語系僅限中/英/日/韓，其餘直接當作英文
- 語言標籤與轉寫參數都使用這套結果
