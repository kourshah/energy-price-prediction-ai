"""
External live prediction runner.

IMPORTANT:
- Run this OUTSIDE the inference Docker container.
- It downloads current market data.
- It recreates the 26 engineered features.
- It selects the latest complete 60 x 26 input.
- It sends ONLY that prepared matrix to the LOCAL Docker FastAPI /predict endpoint.
- The local Docker container performs scaling + LSTM inference only.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from live_data.pipeline import prepare_latest_model_input


# Change this only if your Render service URL changes.
PREDICTION_API_URL = "http://localhost:8000/predict"

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "shared_data"
OUTPUT_PATH = OUTPUT_DIR / "latest_input.json"


def prepare_live_input(history_calendar_days: int = 500):
    """Create the latest complete 60 x 26 model input outside Docker."""
    latest_60, metadata = prepare_latest_model_input(
        history_calendar_days=history_calendar_days
    )

    if latest_60.shape != (60, 26):
        raise ValueError(
            f"Expected prepared input shape (60, 26), got {latest_60.shape}."
        )

    return latest_60, metadata


def save_input_copy(latest_60, metadata):
    """
    Save a local audit/debug copy.

    This file is NOT required by the deployed Docker service.
    """
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_shape": [60, 26],
        "first_input_date": metadata["first_input_date"],
        "latest_input_date": metadata["latest_input_date"],
        "sources": metadata["sources"],
        "data": latest_60.values.tolist(),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    return payload


def send_to_prediction_api(latest_60):
    """
    Send the already prepared 60 x 26 matrix to FastAPI.

    The deployed API's POST /predict endpoint then:
      1. validates the matrix,
      2. applies the saved scaler,
      3. runs the LSTM,
      4. returns the oil-price prediction.
    """
    request_body = {
        "data": latest_60.values.tolist()
    }

    response = requests.post(
        PREDICTION_API_URL,
        json=request_body,
        timeout=180,
    )
    response.raise_for_status()

    return response.json()


def main():
    print("=" * 70)
    print("LIVE OIL PRICE PREDICTION - EXTERNAL DATA PREPARATION")
    print("=" * 70)

    # STEP 1: outside Docker
    latest_60, metadata = prepare_live_input()

    # Optional local record for inspection/debugging.
    save_input_copy(latest_60, metadata)

    print()
    print("Prepared outside Docker:")
    print("Shape:", latest_60.shape)
    print("First input date:", metadata["first_input_date"])
    print("Latest input date:", metadata["latest_input_date"])

    # STEP 2: send prepared matrix to Docker/Render
    print()
    print("Sending prepared 60 x 26 input to:")
    print(PREDICTION_API_URL)

    result = send_to_prediction_api(latest_60)

    # Display exactly what the local FastAPI endpoint returned.
    print()
    print("API response:")
    print(json.dumps(result, indent=2))

    predicted_price = result.get("predicted_oil_price")
    if predicted_price is None:
        predicted_price = result.get("prediction")

    if predicted_price is None:
        raise KeyError(
            "The API returned HTTP 200, but the response does not contain "
            "'predicted_oil_price' or 'prediction'. See the API response above."
        )

    print()
    print("=" * 70)
    print("PREDICTION COMPLETE")
    print("=" * 70)
    print(f"Predicted oil price: ${float(predicted_price):.2f}")
    print("Latest market date:", metadata["latest_input_date"])
    print("=" * 70)

    return {
        "prediction": predicted_price,
        "metadata": metadata,
        "api_response": result,
    }


if __name__ == "__main__":
    main()
