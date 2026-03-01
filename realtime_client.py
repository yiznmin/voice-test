import queue
import time

import httpx
import numpy as np
import sounddevice as sd


API_URL = "http://127.0.0.1:8010/infer"
SAMPLE_RATE = 16000
BLOCK_SEC = 0.5
WINDOW_SEC = 3.0


audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()
buffer_data = []


def audio_callback(indata, frames, callback_time, status):
    del frames, callback_time
    if status:
        print(f"[audio] {status}")
    audio_queue.put(indata.copy())


def main():
    print("Realtime language test started. Press Ctrl+C to stop.")
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=int(SAMPLE_RATE * BLOCK_SEC),
        callback=audio_callback,
    ):
        try:
            while True:
                data = audio_queue.get()
                buffer_data.append(data)

                total_sec = sum(len(block) for block in buffer_data) / SAMPLE_RATE
                if total_sec < WINDOW_SEC:
                    continue

                audio_np = np.concatenate(buffer_data, axis=0).squeeze(1)
                buffer_data.clear()

                payload = {"audio": audio_np.tolist()}
                start = time.time()

                try:
                    resp = httpx.post(API_URL, json=payload, timeout=30.0)
                    latency = (time.time() - start) * 1000
                    if resp.status_code == 200:
                        result = resp.json()
                        lang = result.get("language", "unknown")
                        score = result.get("score")
                        print(f"[{latency:7.1f} ms] language={lang} score={score}")
                    else:
                        print(f"[api error] {resp.status_code}: {resp.text}")
                except Exception as exc:
                    print(f"[request error] {exc}")
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
