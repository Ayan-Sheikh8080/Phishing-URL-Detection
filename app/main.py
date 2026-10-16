import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.features import FEATURE_NAMES, extract_features

MODEL_PATH = Path("models/model.pkl")
LOG_PATH = Path("logs/predictions.jsonl")

app = FastAPI(title="Phishing URL Detection API", version="1.0.0")
model = joblib.load(MODEL_PATH)

# SHAP explainer only works for tree models; skip gracefully otherwise
try:
    import shap
    explainer = shap.TreeExplainer(model)
except Exception:
    explainer = None

stats = {"requests": 0, "phishing": 0, "legitimate": 0, "total_latency_ms": 0.0}


class PredictRequest(BaseModel):
    url: str = Field(..., min_length=3, max_length=2048)


def top_factors(X: pd.DataFrame, k: int = 3):
    if explainer is None:
        return []
    try:
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        vals = sv[0]
        idx = np.argsort(-np.abs(vals))[:k]
        return [
            {
                "feature": FEATURE_NAMES[i],
                "value": float(X.iloc[0, i]),
                "effect": "raises phishing risk" if vals[i] > 0 else "lowers phishing risk",
                "impact": round(float(vals[i]), 4),
            }
            for i in idx
        ]
    except Exception:
        return []


def log_prediction(url, features, label, prob):
    LOG_PATH.parent.mkdir(exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "url": url,
        "prediction": label,
        "probability": prob,
        **features,
    }
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    start = time.perf_counter()
    feats = extract_features(req.url)
    X = pd.DataFrame([feats])[FEATURE_NAMES]

    prob = float(model.predict_proba(X)[0, 1])
    label = int(prob >= 0.5)
    name = "phishing" if label else "legitimate"

    latency = (time.perf_counter() - start) * 1000
    stats["requests"] += 1
    stats[name] += 1
    stats["total_latency_ms"] += latency
    log_prediction(req.url, feats, label, prob)

    return {
        "url": req.url,
        "prediction": name,
        "is_phishing": bool(label),
        "probability": round(prob, 4),
        "top_factors": top_factors(X),
    }


@app.get("/metrics")
def metrics():
    n = max(stats["requests"], 1)
    return {
        "total_requests": stats["requests"],
        "phishing_predictions": stats["phishing"],
        "legitimate_predictions": stats["legitimate"],
        "phishing_rate": round(stats["phishing"] / n, 4),
        "avg_latency_ms": round(stats["total_latency_ms"] / n, 2),
    }