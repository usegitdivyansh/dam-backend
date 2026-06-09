from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LIGHTGBM_ADDITIONAL_FRAMEWORK_LOADING_STRATEGY"] = "dlopen"

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
import subprocess
import sys
import io
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime

app = FastAPI(
    title="DAM Price Forecast API",
    description="Electricity price forecasting for IEX DAM market",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
PREDICTIONS_FILE = BASE_DIR / "data" / "fresh_predictions.csv"
METADATA_FILE = BASE_DIR / "model_metadata.json"
FEATURE_LIST_FILE = BASE_DIR / "feature_list.json"
INFERENCE_SCRIPT = BASE_DIR / "src" / "inference.py"

models = {}
feature_list = []

def load_models():
    global models, feature_list
    try:
        models_dir = BASE_DIR / "models"
        with open(METADATA_FILE) as f:
            metadata = json.load(f)
        model_files = metadata["model_files"]
        models["classifier"] = joblib.load(models_dir / model_files["regime_classifier"])
        models["low"] = joblib.load(models_dir / model_files["regime_low"])
        models["mid"] = joblib.load(models_dir / model_files["regime_mid"])
        models["high"] = joblib.load(models_dir / model_files["regime_high"])
        models["p50"] = joblib.load(models_dir / model_files["p50"]) if model_files.get("p50") else None
        with open(FEATURE_LIST_FILE) as f:
            feature_list = json.load(f)["feature_list"]
        print(f"✅ DAM Models loaded! Features: {len(feature_list)}")
    except Exception as e:
        print(f"⚠️ Model loading failed: {e}")

predictions_df = None

def load_predictions():
    global predictions_df
    if PREDICTIONS_FILE.exists():
        predictions_df = pd.read_csv(PREDICTIONS_FILE, parse_dates=["timestamp"])
        print(f"✅ Loaded {len(predictions_df)} predictions")
        return
    print("⚠️ No predictions file found!")

@app.on_event("startup")
async def startup_event():
    print("🚀 DAM API starting up...")
    load_predictions()
    load_models()
    print("✅ Startup complete!")

def format_forecast(row):
    result = {
        "timestamp": row["timestamp"].isoformat(),
        "predicted_price": round(float(row["prediction"]), 2),
        "regime_probability": round(float(row["regime_probability"]), 4)
    }
    if "p10" in row and pd.notna(row["p10"]):
        result["p10"] = round(float(row["p10"]), 2)
    if "p90" in row and pd.notna(row["p90"]):
        result["p90"] = round(float(row["p90"]), 2)
    return result

@app.get("/")
def health_check():
    return {
        "status": "running",
        "message": "DAM Price Forecast API is live",
        "market": "DAM",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "predictions_loaded": predictions_df is not None,
        "total_predictions": len(predictions_df) if predictions_df is not None else 0
    }

@app.get("/api/metadata")
def get_metadata():
    if not METADATA_FILE.exists():
        raise HTTPException(status_code=404, detail="Metadata file not found")
    with open(METADATA_FILE) as f:
        return json.load(f)

@app.get("/api/forecasts/latest")
def get_latest_forecasts():
    if predictions_df is None:
        raise HTTPException(status_code=503, detail="Predictions not loaded")
    df = predictions_df.tail(96)
    return {
        "market": "DAM",
        "total_records": len(df),
        "forecasts": [format_forecast(row) for _, row in df.iterrows()]
    }

@app.get("/api/forecasts/summary")
def get_summary():
    if predictions_df is None:
        raise HTTPException(status_code=503, detail="Predictions not loaded")
    df = predictions_df.tail(96)
    return {
        "market": "DAM",
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "average_price": round(float(df["prediction"].mean()), 2),
            "min_price": round(float(df["prediction"].min()), 2),
            "max_price": round(float(df["prediction"].max()), 2),
            "total_blocks": len(df)
        }
    }

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """
    Send raw CSV → get DAM price predictions.
    Required columns: datetime, dam_price, gdam_price, rtm_price,
    gdam_buy_mw, gdam_sell_mw, gdam_solar, gdam_temp, gdam_cloud,
    gdam_wind, gdam_humidity, gdam_rain. Minimum 200 rows.
    """
    try:
        contents = await file.read()
        input_df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        temp_input = BASE_DIR / "data" / "temp_input.csv"
        temp_output = BASE_DIR / "data" / "temp_output.csv"
        input_df.to_csv(temp_input, index=False)
        result = subprocess.run(
            [sys.executable, str(INFERENCE_SCRIPT),
             "--input", str(temp_input),
             "--output", str(temp_output),
             "--deploy-dir", str(BASE_DIR)],
            capture_output=True, text=True, cwd=str(BASE_DIR)
        )
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Inference failed: {result.stderr}")
        output_df = pd.read_csv(temp_output, parse_dates=["timestamp"])
        return {
            "status": "success",
            "market": "DAM",
            "total_predictions": len(output_df),
            "predictions": [format_forecast(row) for _, row in output_df.iterrows()]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/features")
async def predict_features(file: UploadFile = File(...)):
    """
    Send 96 rows of pre-computed features → get DAM predictions directly.
    """
    try:
        if not models:
            raise HTTPException(status_code=503, detail="Models not loaded")
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        missing = [f for f in feature_list if f not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing}")
        X = df[feature_list]
        classifier = models["classifier"]
        probs = classifier.predict_proba(X)
        classes = getattr(classifier, "classes_", np.array([0]))
        pred_low = models["low"].predict(X)
        pred_mid = models["mid"].predict(X)
        pred_high = models["high"].predict(X)
        def class_prob(target):
            if probs.shape[1] == 1:
                return np.ones(len(X)) if classes[0] == target else np.zeros(len(X))
            if target not in classes:
                return np.zeros(len(X))
            idx = int(np.where(classes == target)[0][0])
            return probs[:, idx]
        normal_prob = class_prob(0)
        med_prob = class_prob(1)
        ext_prob = class_prob(2)
        spike_prob = med_prob + ext_prob
        preds = normal_prob * pred_low + med_prob * pred_mid + ext_prob * pred_high
        p50_preds = models["p50"].predict(X) if models.get("p50") else None
        results = []
        for i in range(len(preds)):
            row = {
                "block": i + 1,
                "predicted_price": round(float(preds[i]), 2),
                "regime_probability": round(float(spike_prob[i]), 4),
            }
            if p50_preds is not None:
                row["p50"] = round(float(p50_preds[i]), 2)
            if "datetime" in df.columns:
                row["timestamp"] = str(df["datetime"].iloc[i])
            results.append(row)
        return {
            "status": "success",
            "market": "DAM",
            "total_predictions": len(results),
            "predictions": results
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))