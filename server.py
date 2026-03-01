from typing import List

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from speechbrain.inference import EncoderClassifier


app = FastAPI(title="Realtime Language ID API")

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-voxlingua107-ecapa",
    savedir="pretrained_models/ecapa_vox",
    run_opts={"device": "cpu"},
)


class InferRequest(BaseModel):
    audio: List[float] = Field(..., min_length=160)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/infer")
def infer(payload: InferRequest) -> dict:
    try:
        audio = torch.tensor(payload.audio, dtype=torch.float32).unsqueeze(0)
        _, score, _, text_lab = classifier.classify_batch(audio)
        return {"language": text_lab[0], "score": float(score[0].item())}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Inference failed: {exc}") from exc


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
