# python3 test.py
# test_ecapa_speed.py
import time
import torch
import torchaudio
import webrtcvad
import numpy as np
from collections import Counter
import torch.nn.functional as F
from speechbrain.inference import EncoderClassifier
from denoiser import pretrained
import subprocess
import soundfile as sf
import tempfile
import os
import sounddevice as sd
import queue
from collections import deque
# ==========================
# 參數設定
# ==========================

print("🎙️ Real-time Language Detection (CPU Version)")
print("Press Ctrl+C to stop\n")
# ==========================
# 語言控制設定
# ==========================

ALLOWED_LANGS = ["zh", "en", "ja", "ko"]   # 點餐機允許語言
CONF_THRESHOLD = 0.85                     # 鎖定最低信心
UNKNOWN_THRESHOLD = 0.60                  # 低於這個就視為 UNKNOWN


STATE = "UNDECIDED"   # UNDECIDED / LOCKED
LOCKED_LANGUAGE = None
history = deque(maxlen=5)   # 最近5次結果
history.clear()
STABLE_COUNT = 3            # 連續幾次才鎖定
SWITCH_COUNT = 4            # 鎖定後幾次強反證才切換
SWITCH_THRESHOLD = 0.90     # 切換要更高信心

SAMPLE_RATE = 16000
VAD_MODE = 2      # 0~3，越大越嚴格
FRAME_MS = 30     # WebRTC VAD 建議 10/20/30
fs = SAMPLE_RATE  # 給 prepare_speech 用

BUFFER_SEC = 1.5
BLOCK_SEC = 0.5

audio_queue = queue.Queue()
buffer_data = []
# sd.default.device = (2, None)
# ==========================
# 載入模型
# ==========================
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-voxlingua107-ecapa",
    savedir="pretrained_models/ecapa_vox",
    run_opts={"device": "cpu" }
)
denoiser_model = pretrained.dns64().eval()  # 去噪
torch.set_grad_enabled(False)


# ==========================
# 處理音訊 (VAD + 去噪可選)
# ==========================
def prepare_speech(tensor, use_vad=True, use_denoise=True, denoise_model="dns64"):
    pcm = (tensor[0].numpy() * 32768).astype(np.int16)

    # VAD
    if use_vad:
        vad = webrtcvad.Vad()
        vad.set_mode(VAD_MODE)
        frame_size = int(FRAME_MS * fs / 1000)
        segments = []
        for start in range(0, len(pcm), frame_size):
            end = start + frame_size
            frame = pcm[start:end]
            if len(frame) < frame_size:
                frame = np.pad(frame, (0, frame_size - len(frame)), mode='constant')
            if vad.is_speech(frame.tobytes(), fs):
                segments.append(frame)
        if not segments:
            segments = [pcm]
        pcm = np.concatenate(segments)

    tensor = torch.from_numpy(pcm.astype(np.float32)/32768.0).unsqueeze(0)

    # 去噪
    if use_denoise:
        if denoise_model == "dns64":
            tensor = denoiser_model(tensor).squeeze(1)
        else:
            raise ValueError("Unknown denoise model")
    run_opts={"device": "cpu"}
    
    return tensor


def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def print_confidence_bar(lang, prob):
    bar_len = int(prob * 20)  # 20格寬
    bar = "█" * bar_len
    print(f"{lang:<4} {bar:<20} {prob*100:6.2f}%")

# ---------- 顯示語言穩定度 ----------
def print_history(history):
    print("\n📝 Recent Language History (last {}):".format(len(history)))
    for lang, prob in history:
        bar_len = int(prob * 20)
        bar = "█" * bar_len
        print(f"{lang:<7} {bar:<20} {prob*100:6.2f}%")
    print("-"*40)

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    blocksize=int(SAMPLE_RATE * BLOCK_SEC),
    callback=audio_callback
):
    try:
        inference_times = []
        while True:
            data = audio_queue.get()
            buffer_data.append(data)

            total_len = sum(len(b) for b in buffer_data) / SAMPLE_RATE

            if total_len >= BUFFER_SEC:
                audio_np = np.concatenate(buffer_data, axis=0).T
                tensor = torch.from_numpy(audio_np)

                # VAD + 去噪
                tensor = prepare_speech(
                    tensor,
                    use_vad=True,
                    use_denoise=True
                )

                # 推論
                # ===== 效能計時開始 =====
                start_time = time.time()

                predictions = classifier.classify_batch(tensor)

                end_time = time.time()
                inference_time = end_time - start_time
                inference_times.append(inference_time)

                # 計算音訊長度
                audio_duration = tensor.shape[1] / SAMPLE_RATE
                rtf = inference_time / audio_duration
                # ===== 效能計時結束 =====＼

                logits = predictions[0]
                probs = F.softmax(logits, dim=1)

                top5_prob, top5_idx = torch.topk(probs, 5, dim=1)
                top5_langs = classifier.hparams.label_encoder.decode_torch(top5_idx[0])
                # 取第一名
                top_lang = top5_langs[0]
                top_prob_value = top5_prob[0][0].item()

                # 只取語言代碼 (例如 "zh: Chinese" → "zh")
                top_lang_code = top_lang.split(":")[0]

                # ==========================
                # Soft Lock 語言穩定機制
                # ==========================

                # 先過濾不允許語言或太低信心
                if top_lang_code not in ALLOWED_LANGS or top_prob_value < UNKNOWN_THRESHOLD:
                    history.append(("UNKNOWN", top_prob_value))
                else:
                    history.append((top_lang_code, top_prob_value))

                detected_language = "UNKNOWN"

                # -------- 尚未鎖定 --------
                if STATE == "UNDECIDED":
                    valid_history = [
                        lang for lang, prob in history
                        if lang != "UNKNOWN" and prob >= CONF_THRESHOLD
                    ]

                    if len(valid_history) >= STABLE_COUNT:
                        # 檢查是否連續一致
                        if len(set(valid_history[-STABLE_COUNT:])) == 1:
                            LOCKED_LANGUAGE = valid_history[-1]
                            STATE = "LOCKED"

                    detected_language = LOCKED_LANGUAGE if LOCKED_LANGUAGE else "UNKNOWN"


                # -------- 已鎖定 --------
                elif STATE == "LOCKED":

                    detected_language = LOCKED_LANGUAGE

                    # 檢查是否有強反證
                    switch_candidates = [
                        lang for lang, prob in history
                        if prob >= SWITCH_THRESHOLD and lang != LOCKED_LANGUAGE
                    ]

                    if len(switch_candidates) >= SWITCH_COUNT:
                        if len(set(switch_candidates[-SWITCH_COUNT:])) == 1:
                            LOCKED_LANGUAGE = switch_candidates[-1]
                            detected_language = LOCKED_LANGUAGE
                print("\n" + "="*40)
                print(f"⏱ Inference Time: {inference_time:.3f} sec")
                print(f"🎧 Audio Length: {audio_duration:.3f} sec")
                print(f"⚡ RTF: {rtf:.3f}")

                avg_time = sum(inference_times) / len(inference_times)
                print(f"📊 Avg Inference Time: {avg_time:.3f} sec")
                print("🎯 Detected Language:")
                print(f"{detected_language} ({top_prob_value*100:.2f}%)\n")
                print("Top-5 Confidence:")
                for i in range(5):
                    lang = top5_langs[i]
                    prob = top5_prob[0][i].item()
                    print_confidence_bar(lang, prob)
                # ---------- 印出最近 N 次歷史 ----------
                print_history(history)
                buffer_data = []

    except KeyboardInterrupt:
        print("\nStopped.")