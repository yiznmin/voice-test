import time
import torch
import torchaudio
import webrtcvad
import numpy as np
from speechbrain.inference import EncoderClassifier
from denoiser import pretrained

# ==========================
# 參數設定
# ==========================
TEST_AUDIO = "/workspace/JaClip1.mp3"  # 你的音檔
SAMPLE_RATE = 16000
VAD_MODE = 2          # 0~3，越大越嚴格
FRAME_MS = 30         # 每個 frame 的長度 (ms)
RUNS = 5              # 重複推理次數計算平均時間

# ==========================
# 1️⃣ 載入模型
# ==========================
start_load = time.time()
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-voxlingua107-ecapa",
    savedir="pretrained_models/ecapa_vox",
    run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)
denoiser_model = pretrained.dns64().eval()  # CPU-friendly 去噪
load_time = time.time() - start_load
print(f"Model load time: {load_time:.3f} sec")

torch.set_grad_enabled(False)

# ==========================
# 2️⃣ 讀音訊
# ==========================
signal, fs = torchaudio.load(TEST_AUDIO)
if fs != SAMPLE_RATE:
    signal = torchaudio.functional.resample(signal, fs, SAMPLE_RATE)
    fs = SAMPLE_RATE

signal = signal.mean(dim=0, keepdim=True)  # stereo -> mono
print(f"Audio length: {signal.shape[1]/fs:.2f} sec, sample rate: {fs}")

# ==========================
# 3️⃣ VAD 過濾非語音
# ==========================
vad = webrtcvad.Vad()
vad.set_mode(VAD_MODE)

pcm = (signal[0].numpy() * 32768).astype(np.int16)
frame_size = int(FRAME_MS * fs / 1000)
speech_segments = []

for start in range(0, len(pcm), frame_size):
    end = start + frame_size
    frame = pcm[start:end]
    if len(frame) < frame_size:
        frame = np.pad(frame, (0, frame_size - len(frame)), mode='constant')
    if vad.is_speech(frame.tobytes(), fs):
        speech_segments.append(frame)

if not speech_segments:
    print("Warning: No speech detected by VAD, using full audio.")
    speech_segments = [pcm]

speech_pcm = np.concatenate(speech_segments)
speech_tensor = torch.from_numpy(speech_pcm.astype(np.float32) / 32768.0).unsqueeze(0)

# ==========================
# 4️⃣ 去噪
# ==========================
with torch.no_grad():
    speech_tensor = denoiser_model(speech_tensor)
    # DeepFilterNet 回傳 [1,1,N] -> squeeze 成 [1,N]
    speech_tensor = speech_tensor.squeeze(1)
# ==========================
# 5️⃣ 移到 GPU + fp16 (如果有)
# ==========================
if torch.cuda.is_available():
    classifier = classifier.to("cuda")
    speech_tensor = speech_tensor.to("cuda").half()

# ==========================
# 6️⃣ 推理並計算速度
# ==========================
times = []
for i in range(RUNS):
    start = time.time()
    predictions = classifier.classify_batch(speech_tensor)
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"Run {i+1}: latency = {elapsed:.4f} sec")

avg_time = sum(times) / RUNS
audio_len = speech_tensor.shape[1] / fs
print(f"\nAverage latency over {RUNS} runs: {avg_time:.4f} sec")
print(f"Realtime factor (RTF = latency / audio_length): {avg_time / audio_len:.4f}")

# ==========================
# 7️⃣ 顯示預測結果
# ==========================
print("\nPredictions:")
print(predictions)