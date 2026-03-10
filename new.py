import time
import torch
import numpy as np
import sounddevice as sd
import webrtcvad
from collections import deque
import torch.nn.functional as F
from speechbrain.inference import EncoderClassifier
from denoiser import pretrained

# ===============================
# 商用參數
# ===============================
SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)

STREAM_WINDOW = 0.8
BLOCK_SEC = 0.5

CONF_THRESHOLD = 0.90
UNKNOWN_THRESHOLD = 0.60
STABLE_COUNT = 3
SWITCH_COUNT = 4
SWITCH_THRESHOLD = 0.93

ALLOWED_LANGS = ["zh", "en", "ja", "ko"]

# ===============================
# 商用狀態機
# ===============================
class LanguageStateMachine:
    def __init__(self):
        self.state = "UNDECIDED"
        self.locked = None
        self.history = deque(maxlen=5)

    def update(self, lang, prob):

        if lang not in ALLOWED_LANGS or prob < UNKNOWN_THRESHOLD:
            self.history.append(("UNKNOWN", prob))
        else:
            self.history.append((lang, prob))

        if self.state == "UNDECIDED":
            valid = [
                l for l, p in self.history
                if l != "UNKNOWN" and p >= CONF_THRESHOLD
            ]

            if len(valid) >= STABLE_COUNT:
                if len(set(valid[-STABLE_COUNT:])) == 1:
                    self.locked = valid[-1]
                    self.state = "LOCKED"

        elif self.state == "LOCKED":
            switch = [
                l for l, p in self.history
                if p >= SWITCH_THRESHOLD and l != self.locked
            ]
            if len(switch) >= SWITCH_COUNT:
                if len(set(switch[-SWITCH_COUNT:])) == 1:
                    self.locked = switch[-1]

        return self.locked if self.locked else "UNKNOWN"

# ===============================
# 載入模型（CPU 安全）
# ===============================
device = "cpu"
torch.set_grad_enabled(False)

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-voxlingua107-ecapa",
    run_opts={"device": device}
)

denoiser = pretrained.dns64().to(device).eval()

vad = webrtcvad.Vad(2)

# ===============================
# Ring Buffer
# ===============================
ring_buffer = deque(maxlen=int(SAMPLE_RATE * STREAM_WINDOW))
speech_frames = deque()

state_machine = LanguageStateMachine()

# ===============================
# Print 工具
# ===============================
def print_confidence_bar(lang, prob):
    bar_len = int(prob * 20)
    bar = "█" * bar_len
    print(f"{lang:<7} {bar:<20} {prob*100:6.2f}%")

def print_history(history):
    print("\n📝 Recent Language History (last {}):".format(len(history.history)))
    for lang, prob in history.history:
        print_confidence_bar(lang, prob)
    print("-"*40)

# ===============================
# Audio Callback（只做輕量工作）
# ===============================
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    pcm = (indata[:, 0] * 32768).astype(np.int16)
    for i in range(0, len(pcm), FRAME_SIZE):
        frame = pcm[i:i + FRAME_SIZE]
        if len(frame) < FRAME_SIZE:
            continue
        if vad.is_speech(frame.tobytes(), SAMPLE_RATE):
            speech_frames.append(frame)

# ===============================
# Streaming 主迴圈
# ===============================
with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    blocksize=int(SAMPLE_RATE * BLOCK_SEC),
    dtype="float32",
    callback=audio_callback
):
    print("🚀 Jetson 商用 Streaming LangID 啟動")
    print("Ctrl+C 結束\n")

    inference_times = []

    try:
        while True:

            if len(speech_frames) == 0:
                time.sleep(0.01)
                continue

            # 收集語音 frame
            frame_collect_start = time.time()
            frames_for_inference = []
            while len(speech_frames) > 0:
                frame = speech_frames.popleft()
                ring_buffer.extend(frame)
            frame_collect_end = time.time()

            if len(ring_buffer) < SAMPLE_RATE * STREAM_WINDOW:
                continue

            # ===== Step 1: VAD 過濾 frame =====
            vad_start = time.time()
            audio_np = np.array(ring_buffer, dtype=np.float32) / 32768.0
            vad_frames = []
            for i in range(0, len(audio_np), FRAME_SIZE):
                frame = audio_np[i:i+FRAME_SIZE]
                if len(frame) < FRAME_SIZE:
                    continue
                if vad.is_speech((frame*32768).astype(np.int16).tobytes(), SAMPLE_RATE):
                    vad_frames.extend(frame)
            vad_end = time.time()
            vad_time = vad_end - vad_start

            # ===== Step 2: Denoiser =====
            denoise_start = time.time()
            tensor = torch.from_numpy(np.array(vad_frames, dtype=np.float32)).unsqueeze(0)
            with torch.no_grad():
                tensor = denoiser(tensor).squeeze(1)
            denoise_end = time.time()
            denoise_time = denoise_end - denoise_start

            # ===== Step 3: Classifier 推論 =====
            inference_start = time.time()
            out = classifier.classify_batch(tensor)
            logits = out[0]
            probs = F.softmax(logits, dim=1)
            inference_end = time.time()
            classifier_time = inference_end - inference_start
            inference_times.append(classifier_time)

            # ===== Top-5 計算 =====
            top5_prob, top5_idx = torch.topk(probs, 5, dim=1)
            top5_langs = classifier.hparams.label_encoder.decode_torch(top5_idx[0])
            top_lang = top5_langs[0]
            top_lang_code = top_lang.split(":")[0]
            top_prob_val = top5_prob[0][0].item()

            # ===== 狀態機更新 =====
            detected = state_machine.update(top_lang_code, top_prob_val)
            # 啟動階段：尚未鎖定語言
            if state_machine.locked is None:
                display_lang = "en" if detected == "UNKNOWN" else detected
            else:
                display_lang = detected
            # ===== frame rate 計算 =====
            frames_processed = len(vad_frames) / FRAME_SIZE
            frame_rate_per_sec = frames_processed / STREAM_WINDOW

            # ===== RTF / Avg =====
            rtf = classifier_time / STREAM_WINDOW
            avg_time = sum(inference_times) / len(inference_times)

            # ===== Debug 輸出 =====
            print("\n" + "="*60)
            print(f"⏱ Frame Collect : {frame_collect_end - frame_collect_start:.3f}s")
            print(f"⏱ VAD Time      : {vad_time:.3f}s")
            print(f"⏱ Denoise Time  : {denoise_time:.3f}s")
            print(f"⏱ Inference Time: {classifier_time:.3f}s")
            print(f"🎧 Audio Length : {STREAM_WINDOW:.3f}s")
            print(f"⚡ RTF           : {rtf:.3f}")
            print(f"📊 Avg Inference : {avg_time:.3f}s")
            print(f"📊 Frames/sec    : {frame_rate_per_sec:.1f}")
            print("🎯 Detected Language:")
            print(f"{display_lang} ({top_prob_val*100:.2f}%)")
            print("Top-5 Confidence:")
            for i in range(5):
                print_confidence_bar(top5_langs[i], top5_prob[0][i].item())
            print_history(state_machine)

    except KeyboardInterrupt:
        print("\n🛑 停止")