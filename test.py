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
# ==========================
# 參數設定
# ==========================
TEST_AUDIO = "/workspace/KoClip2.mp3"
SAMPLE_RATE = 16000
VAD_MODE = 2
FRAME_MS = 30
RUNS = 5
SEG_SEC = 5  # 分段秒數

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
# 讀音訊
# ==========================
signal, fs = torchaudio.load(TEST_AUDIO)
if fs != SAMPLE_RATE:
    signal = torchaudio.functional.resample(signal, fs, SAMPLE_RATE)
fs = SAMPLE_RATE
signal = signal.mean(dim=0, keepdim=True)  # stereo -> mono

# ==========================
# 音訊長度輔助函數
# ==========================
def get_audio_length(tensor, fs):
    return tensor.shape[1] / fs

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

    if torch.cuda.is_available():
        tensor = tensor.to("cuda").half()
    
    return tensor

# ==========================
# 單段推理函數
# ==========================
def infer_and_print(tensor, description=""):
    classifier.to("cpu")
    times = []
    for i in range(RUNS):
        start = time.time()
        predictions = classifier.classify_batch(tensor)
        elapsed = time.time() - start
        times.append(elapsed)
    
    avg_time = sum(times)/RUNS
    audio_len = tensor.shape[1]/fs
    print(f"\n--- {description} ---")
    print(f"Average latency: {avg_time:.4f} sec, RTF: {avg_time/audio_len:.4f}")
    
    # Top-5
    logits = predictions[0]
    probs = F.softmax(logits, dim=1)
    top5_prob, top5_idx = torch.topk(probs, 5, dim=1)
    top5_langs = classifier.hparams.label_encoder.decode_torch(top5_idx[0])
    
    print("Top-5 Predictions:")
    for i in range(5):
        print(f"{top5_langs[i]} : {top5_prob[0][i].item()*100:.2f}%")

# ==========================
# 分段 + 投票
# ==========================
def segment_and_vote(signal, fs, use_vad=True, use_denoise=True):
    total_samples = signal.shape[1]
    seg_samples = int(SEG_SEC * fs)
    segments = []

    for start in range(0, total_samples, seg_samples):
        end = min(start + seg_samples, total_samples)
        seg_tensor = signal[:, start:end]
        segments.append(seg_tensor)

    votes = []
    for i, seg in enumerate(segments):
        tensor = prepare_speech(seg, use_vad=use_vad, use_denoise=use_denoise)
        predictions = classifier.classify_batch(tensor)
        top_idx = predictions[0].argmax(dim=1)
        lang = classifier.hparams.label_encoder.decode_torch(top_idx)[0]
        votes.append(lang)
        print(f"Segment {i+1}/{len(segments)} top prediction: {lang}")

    vote_counts = Counter(votes)
    most_common_lang, count = vote_counts.most_common(1)[0]
    print(f"\nFinal language prediction by majority vote: {most_common_lang} ({count}/{len(segments)} segments)")
    return most_common_lang, vote_counts

# ==========================
# 1️⃣ 原始音訊
# ==========================
raw_len_sec = get_audio_length(signal, fs)
print(f"Original audio length: {raw_len_sec:.2f} sec")
speech_tensor_raw = prepare_speech(signal, use_vad=False, use_denoise=False)
infer_and_print(speech_tensor_raw, "Raw audio (no VAD, no denoise)")

# ==========================
# 2️⃣ VAD + 去噪
# ==========================
speech_tensor_clean = prepare_speech(signal, use_vad=True, use_denoise=True, denoise_model="dns64")
clean_len_sec = get_audio_length(speech_tensor_clean, fs)
print(f"After VAD + denoise audio length: {clean_len_sec:.2f} sec")
infer_and_print(speech_tensor_clean, "VAD + denoise audio")

# ==========================
# 3️⃣ 分段 + 投票
# ==========================
print("\n--- Segment + Majority Vote ---")
segment_and_vote(signal, fs, use_vad=True, use_denoise=True)