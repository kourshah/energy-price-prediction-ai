"""
Test the Dockerized FastAPI prediction endpoint.
"""

import pandas as pd
import requests

from predict import FEATURES, LOOKBACK

API_URL = "http://127.0.0.1:8000/predict"
DATA_PATH = "data/final_dataset_26_features.csv"

df = pd.read_csv(DATA_PATH)

if len(df) < LOOKBACK + 1:
    raise ValueError(
        f"Dataset needs at least {LOOKBACK + 1} rows, "
        f"but contains only {len(df)}."
    )

test_input = df[FEATURES].iloc[-(LOOKBACK + 1):-1].copy()
actual_price = float(df["Oil_Price"].iloc[-1])

payload = {
    "data": test_input.values.tolist()
}

import json
with open("postman_body.json", "w") as f:
    json.dump(payload, f, indent=2)

try:
    response = requests.post(
        API_URL,
        json=payload,
        timeout=30
    )

    print("=" * 60)
    print("FASTAPI / DOCKER PREDICTION TEST")
    print("=" * 60)
    print("Request URL: ", API_URL)
    print("Input shape: ", test_input.shape)
    print("Status code:", response.status_code)

    response.raise_for_status()

    result = response.json()
    predicted_price = float(result["predicted_oil_price"])

    print(f"Predicted price: {predicted_price:.4f}")
    print(f"Actual price:    {actual_price:.4f}")
    print(f"Absolute error:  {abs(predicted_price - actual_price):.4f}")
    print("=" * 60)

except requests.exceptions.ConnectionError as exc:
    raise SystemExit(
        "Could not connect to FastAPI. "
        "Make sure the Docker container is running on port 8000."
    ) from exc

except requests.exceptions.RequestException as exc:
    body = response.text if "response" in locals() else "No response"
    raise SystemExit(
        f"API request failed: {exc}\nResponse body: {body}"
    ) from exc
