import os
from pathlib import Path
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib
matplotlib.use("Agg")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# ── startup — load everything once ───────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS = PROJECT_ROOT / "artifacts"

print("loading artifacts...")

weekly_data = joblib.load(ARTIFACTS / "weekly_data.pkl")
selection   = joblib.load(ARTIFACTS / "best_model_selection.pkl")

model_results = {
    "ARIMA":   joblib.load(ARTIFACTS / "arima_results.pkl"),
    "Prophet": joblib.load(ARTIFACTS / "prophet_results.pkl"),
    "XGBoost": joblib.load(ARTIFACTS / "xgb_results.pkl"),
    "LSTM":    joblib.load(ARTIFACTS / "lstm_results.pkl"),
}

STATES = sorted(weekly_data.keys())
print(f"ready — {len(STATES)} states, 4 models")

# ── app ───────────────────────────────────────────────────
app = FastAPI(
    title="Sales Forecasting API",
    description=(
        "8-week beverage sales forecasts for 43 US states. "
        "Auto-selects best model per state from ARIMA, Prophet, XGBoost, LSTM."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/dashboard", include_in_schema=False)
def dashboard():
    return FileResponse(PROJECT_ROOT / "dashboard.html")


# ── helpers ───────────────────────────────────────────────
def _match_state(raw: str):
    needle = raw.lower().replace("-", " ")
    return next((s for s in STATES if s.lower() == needle), None)

def _match_model(raw: str):
    return next((m for m in model_results if m.lower() == raw.lower()), None)

def _serialize(state: str, model_name: str, res: dict) -> dict:
    return {
        "state":    state,
        "model":    model_name,
        "mape":     round(res["mape"], 2),
        "rmse":     round(res["rmse"], 2),
        "mae":      round(res["mae"], 2),
        "forecast": [
            {"week": i + 1, "date": str(d.date()), "predicted_sales": round(float(v), 2)}
            for i, (d, v) in enumerate(res["forecast"].items())
        ],
        "validation": [
            {"date": str(d.date()), "actual_sales": round(float(v), 2)}
            for d, v in res["actual"].items()
        ],
    }


# ── endpoints ─────────────────────────────────────────────
@app.get("/", tags=["health"])
def root():
    return {
        "status":  "running",
        "states":  len(STATES),
        "models":  list(model_results.keys()),
        "docs":    "/docs",
        "ui":      "/dashboard",
    }


@app.get("/states", tags=["info"])
def get_states():
    return {"total": len(STATES), "states": STATES}


@app.get("/models", tags=["info"])
def get_models():
    summary = {}
    for name, res in model_results.items():
        mapes = [v["mape"] for v in res.values() if v]
        rmses = [v["rmse"] for v in res.values() if v]
        summary[name] = {
            "states_trained": len(mapes),
            "avg_mape":  round(float(np.mean(mapes)), 2),
            "avg_rmse":  round(float(np.mean(rmses)), 2),
            "best_mape": round(float(np.min(mapes)), 2),
            "worst_mape": round(float(np.max(mapes)), 2),
        }
    ranked = sorted(summary.items(), key=lambda x: x[1]["avg_mape"])
    return {
        "rankings": [
            {"rank": i + 1, "model": name, **stats}
            for i, (name, stats) in enumerate(ranked)
        ]
    }


@app.get("/predict/{state}", tags=["forecast"])
def predict(state: str):
    matched = _match_state(state)
    if not matched:
        raise HTTPException(404, f"state '{state}' not found — see /states")

    info  = selection[matched]
    name  = info["best_model"]
    res   = model_results[name][matched]
    return {"best_model": name, **_serialize(matched, name, res)}


@app.get("/predict/{state}/{model}", tags=["forecast"])
def predict_with_model(state: str, model: str):
    matched = _match_state(state)
    if not matched:
        raise HTTPException(404, f"state '{state}' not found")

    m_name = _match_model(model)
    if not m_name:
        raise HTTPException(400, f"model '{model}' not found — choose from: {list(model_results)}")

    res = model_results[m_name].get(matched)
    if res is None:
        raise HTTPException(404, f"no {m_name} result for {matched}")

    return _serialize(matched, m_name, res)


@app.get("/best-model/{state}", tags=["info"])
def best_model(state: str):
    matched = _match_state(state)
    if not matched:
        raise HTTPException(404, f"state '{state}' not found")

    info  = selection[matched]
    mapes = {
        name: round(res["mape"], 2)
        for name, res in model_results.items()
        if res.get(matched)
    }
    return {
        "state":      matched,
        "best_model": info["best_model"],
        "best_mape":  round(info["best_mape"], 2),
        "best_rmse":  round(info["rmse"], 2),
        "all_models": mapes,
        "reason":     f"{info['best_model']} had the lowest MAPE ({info['best_mape']:.1f}%) for {matched}",
    }


@app.get("/compare/{state}", tags=["forecast"])
def compare(state: str):
    matched = _match_state(state)
    if not matched:
        raise HTTPException(404, f"state '{state}' not found")

    out = {}
    for name, res in model_results.items():
        r = res.get(matched)
        if not r:
            continue
        out[name] = {
            "mape": round(r["mape"], 2),
            "rmse": round(r["rmse"], 2),
            "forecast": [
                {"week": i + 1, "date": str(d.date()), "predicted_sales": round(float(v), 2)}
                for i, (d, v) in enumerate(r["forecast"].items())
            ],
        }

    return {
        "state":      matched,
        "best_model": selection[matched]["best_model"],
        "models":     out,
    }