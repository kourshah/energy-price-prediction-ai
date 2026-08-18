"""
Prediction utilities for the Oil Price LSTM model.

This module:
1. Loads the trained Keras LSTM model.
2. Loads the fitted MinMaxScaler.
3. Validates 60 rows x 26 features.
4. Scales the input using the same scaler used during training.
5. Runs the LSTM prediction.
6. Converts the scaled prediction back to the original oil-price scale.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


# ----------------------------------------------------------
# PROJECT PATHS
# ----------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "oil_lstm_26_features.keras"
SCALER_PATH = BASE_DIR / "models" / "oil_scaler_26_features.pkl"


# ----------------------------------------------------------
# MODEL INPUT CONFIGURATION
# ----------------------------------------------------------

LOOKBACK = 60

FEATURES = [
    "Oil_Price",
    "Natural_Gas",
    "USD_Index",
    "VIX",
    "Gold",
    "Oil_Lag_1",
    "Oil_Lag_5",
    "Oil_Lag_10",
    "Oil_Lag_21",
    "RollMean_7",
    "RollMean_21",
    "RollMean_60",
    "RollStd_7",
    "Oil_LogRet",
    "Gas_LogRet",
    "Gold_LogRet",
    "VIX_LogRet",
    "Momentum_7",
    "Momentum_21",
    "Volatility_21",
    "Volatility_60",
    "Month",
    "Quarter",
    "DayOfWeek",
    "Gold_Oil_Ratio",
    "VIX_Spike",
]

N_FEATURES = len(FEATURES)


# ----------------------------------------------------------
# LOAD TRAINED ARTIFACTS
# ----------------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}\n"
        "Place oil_lstm_26_features.keras inside the models/ folder."
    )

if not SCALER_PATH.exists():
    raise FileNotFoundError(
        f"Scaler file not found: {SCALER_PATH}\n"
        "Place oil_scaler_26_features.pkl inside the models/ folder."
    )

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# ----------------------------------------------------------
# INPUT VALIDATION
# ----------------------------------------------------------

def _prepare_input(input_data) -> pd.DataFrame:
    """
    Convert input_data to a DataFrame and verify that it contains
    exactly 60 time steps and the 26 features used during training.

    Accepted input:
      - pandas.DataFrame with the required feature columns
      - list/tuple/NumPy array with shape (60, 26)
    """

    if isinstance(input_data, pd.DataFrame):
        missing = [name for name in FEATURES if name not in input_data.columns]

        if missing:
            raise ValueError(
                "Missing required feature columns: " + ", ".join(missing)
            )

        frame = input_data[FEATURES].copy()

    else:
        array = np.asarray(input_data, dtype=float)

        if array.shape != (LOOKBACK, N_FEATURES):
            raise ValueError(
                f"Expected input shape ({LOOKBACK}, {N_FEATURES}), "
                f"but received {array.shape}."
            )

        frame = pd.DataFrame(array, columns=FEATURES)

    if frame.shape != (LOOKBACK, N_FEATURES):
        raise ValueError(
            f"Expected exactly {LOOKBACK} rows and {N_FEATURES} features, "
            f"but received {frame.shape}."
        )

    if frame.isnull().any().any():
        raise ValueError("Input data contains missing (NaN) values.")

    return frame


# ----------------------------------------------------------
# PREDICTION
# ----------------------------------------------------------

def predict_oil_price(input_data) -> float:
    """
    Predict the next oil price.

    Parameters
    ----------
    input_data:
        The most recent 60 observations of all 26 model features,
        in the same feature order/meaning used during training.

    Returns
    -------
    float
        Predicted oil price in the original (unscaled) price units.
    """

    frame = _prepare_input(input_data)

    # Use the exact scaler fitted during training.
    scaled = scaler.transform(frame)

    # LSTM expects: (batch, time_steps, features)
    model_input = scaled.reshape(1, LOOKBACK, N_FEATURES)

    # Prediction is the scaled Oil_Price value (feature index 0).
    scaled_prediction = model.predict(model_input, verbose=0).reshape(-1)[0]

    # Reproduce the inverse-scaling method used in the training script:
    # put the prediction into feature 0 and use zeros for the other features.
    inverse_row = np.zeros((1, N_FEATURES))
    inverse_row[0, 0] = scaled_prediction

    predicted_price = scaler.inverse_transform(inverse_row)[0, 0]

    return float(predicted_price)


# ----------------------------------------------------------
# EXPLANATION HELPER (for explain.py)
# ----------------------------------------------------------

# A small, interpretable subset of FEATURES to surface in the LLM explanation.
# Picked because they're intuitive to a non-technical reader — momentum,
# volatility, and the gold/oil ratio are easy to explain in one phrase.
EXPLAIN_FEATURE_SUBSET = [
    "Momentum_7",
    "Momentum_21",
    "Volatility_21",
    "VIX_Spike",
    "Gold_Oil_Ratio",
]


def get_explanation_inputs(input_data) -> dict:
    """
    Pulls the current (most recent) raw Oil_Price plus a handful of
    interpretable feature values from the latest row of the input matrix.
    Use this to build the arguments for explain.explain_prediction().

    Returns
    -------
    dict with keys: "current_price", "top_features"
    """
    frame = _prepare_input(input_data)
    latest_row = frame.iloc[-1]

    return {
        "current_price": float(latest_row["Oil_Price"]),
        "top_features": {name: float(latest_row[name]) for name in EXPLAIN_FEATURE_SUBSET},
    }


# ----------------------------------------------------------
# OPTIONAL QUICK CHECK
# ----------------------------------------------------------

if __name__ == "__main__":
    print("Prediction module loaded successfully.")
    print(f"Model:  {MODEL_PATH}")
    print(f"Scaler: {SCALER_PATH}")
    print(f"Expected input shape: ({LOOKBACK}, {N_FEATURES})")
