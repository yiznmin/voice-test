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
# ==========================
# 參數設定
# ==========================

print("🎙️ Real-time Language Detection (CPU Version)")
print("Press Ctrl+C to stop\n")
SAMPLE_RATE = 16000
VAD_MODE = 2      # 0~3，越大越嚴格
FRAME_MS = 30     # WebRTC VAD 建議 10/20/30
fs = SAMPLE_RATE  # 給 prepare_speech 用

BUFFER_SEC = 3
BLOCK_SEC = 0.5

audio_queue = queue.Queue()
buffer_data = []

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

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    blocksize=int(SAMPLE_RATE * BLOCK_SEC),
    callback=audio_callback
):
    try:
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
                predictions = classifier.classify_batch(tensor)
                logits = predictions[0]
                probs = F.softmax(logits, dim=1)

                top5_prob, top5_idx = torch.topk(probs, 5, dim=1)
                top5_langs = classifier.hparams.label_encoder.decode_torch(top5_idx[0])

                print("\n" + "="*40)
                print("🎯 Detected Language:")
                print(f"{top5_langs[0]} ({top5_prob[0][0].item()*100:.2f}%)\n")

                print("Top-5 Confidence:")
                for i in range(5):
                    lang = top5_langs[i]
                    prob = top5_prob[0][i].item()
                    print_confidence_bar(lang, prob)

                buffer_data = []

    except KeyboardInterrupt:
        print("\nStopped.")