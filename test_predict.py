"""
Single real test prediction using the last available historical sample.

The script:
1. Loads data/final_dataset_26_features.csv
2. Takes the 60 rows immediately BEFORE the final row
3. Uses predict_oil_price(...) from predict.py
4. Compares the prediction with the actual Oil_Price in the final row
"""

from pathlib import Path

import pandas as pd

from predict import FEATURES, LOOKBACK, predict_oil_price


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "final_dataset_26_features.csv"


if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place final_dataset_26_features.csv inside the data/ folder."
    )


df = pd.read_csv(DATA_PATH)

missing = [feature for feature in FEATURES if feature not in df.columns]
if missing:
    raise ValueError(
        "Dataset is missing required features: " + ", ".join(missing)
    )

if len(df) < LOOKBACK + 1:
    raise ValueError(
        f"Dataset needs at least {LOOKBACK + 1} rows, but has only {len(df)}."
    )

test_input = df[FEATURES].iloc[-(LOOKBACK + 1):-1].copy()

actual_price = float(df["Oil_Price"].iloc[-1])

predicted_price = predict_oil_price(test_input)

print("=" * 60)
print("REAL SINGLE-SAMPLE TEST")
print("=" * 60)
print(f"Input shape:     {test_input.shape}")
print(f"Predicted price: {predicted_price:.4f}")
print(f"Actual price:    {actual_price:.4f}")
print(f"Absolute error:  {abs(predicted_price - actual_price):.4f}")
print("=" * 60)
