"""
FastAPI app for the shared-input architecture.

Docker does NOT download market data.
It reads shared_data/latest_input.json and runs the existing scaler + LSTM.
"""

import json
import os
import sys
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from predict import FEATURES, LOOKBACK, predict_oil_price


DEFAULT_INPUT_PATH = BASE_DIR / "shared_data" / "latest_input.json"
LATEST_INPUT_PATH = Path(
    os.getenv("LATEST_INPUT_PATH", str(DEFAULT_INPUT_PATH))
)

app = FastAPI(
    title="Oil Price Prediction API",
    description="Latest market input is prepared outside Docker.",
    version="3.0.0",
)


class PredictionRequest(BaseModel):
    data: List[List[float]] = Field(
        ...,
        description="Exactly 60 rows x 26 features",
    )


def load_latest_payload():
    if not LATEST_INPUT_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Input file not found: {LATEST_INPUT_PATH}. "
                "Run python prepare_latest_input.py outside Docker first."
            ),
        )

    try:
        payload = json.loads(
            LATEST_INPUT_PATH.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read latest_input.json: {exc}",
        ) from exc

    data = payload.get("data")

    if not isinstance(data, list) or len(data) != LOOKBACK:
        raise HTTPException(
            status_code=500,
            detail="latest_input.json must contain exactly 60 rows.",
        )

    for i, row in enumerate(data):
        if not isinstance(row, list) or len(row) != len(FEATURES):
            raise HTTPException(
                status_code=500,
                detail=f"Row {i} must contain exactly 26 features.",
            )

    return payload


@app.get("/")
def home():
    return {
        "status": "ok",
        "version": "3.0.0",
        "automatic_endpoint": "/predict/latest",
        "input_path": str(LATEST_INPUT_PATH),
    }


@app.get("/features")
def get_features():
    return {
        "lookback": LOOKBACK,
        "number_of_features": len(FEATURES),
        "features": FEATURES,
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    if len(request.data) != LOOKBACK:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {LOOKBACK} rows.",
        )

    for i, row in enumerate(request.data):
        if len(row) != len(FEATURES):
            raise HTTPException(
                status_code=400,
                detail=f"Row {i} must contain {len(FEATURES)} features.",
            )

    try:
        return {
            "predicted_oil_price": predict_oil_price(request.data)
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@app.get("/predict/latest")
def predict_latest():
    payload = load_latest_payload()

    try:
        predicted = predict_oil_price(payload["data"])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc

    return {
        "predicted_oil_price": predicted,
        "input_shape": payload.get("input_shape", [60, 26]),
        "first_input_date": payload.get("first_input_date"),
        "latest_input_date": payload.get("latest_input_date"),
        "generated_at_utc": payload.get("generated_at_utc"),
        "sources": payload.get("sources"),
    }


@app.get("/input/status")
def input_status():
    payload = load_latest_payload()

    return {
        "available": True,
        "generated_at_utc": payload.get("generated_at_utc"),
        "first_input_date": payload.get("first_input_date"),
        "latest_input_date": payload.get("latest_input_date"),
        "input_shape": payload.get("input_shape"),
    }
