# ==========================================================
# OIL PRICE FORECASTING USING LSTM
# 26 FEATURES DATASET
# Input Shape = (60,26)
# ==========================================================

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv("final_dataset_26_features.csv")

print(df.head())
print(df.shape)

# ==========================================================
# DATE CONVERSION
# ==========================================================

df["Date"] = pd.to_datetime(df["Date"])

# ==========================================================
# FEATURES (26)
# ==========================================================

features = [

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
    "VIX_Spike"
]

# ==========================================================
# DATA INFORMATION
# ==========================================================

print("\nDataset Information")
print(df.info())

print("\nSummary Statistics")
print(df.describe())

# ==========================================================
# CORRELATION MATRIX
# ==========================================================

plt.figure(figsize=(18,12))

corr = df[features].corr()

sns.heatmap(
    corr,
    cmap="coolwarm"
)

plt.title("Feature Correlation Matrix")

plt.tight_layout()

plt.show()

# ==========================================================
# SCALING
# ==========================================================

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(
    df[features]
)

# ==========================================================
# CREATE SEQUENCES
# ==========================================================

LOOKBACK = 60

X = []
y = []

for i in range(
    LOOKBACK,
    len(scaled_data)
):

    X.append(
        scaled_data[
            i-LOOKBACK:i
        ]
    )

    y.append(
        scaled_data[i,0]
    )

X = np.array(X)
y = np.array(y)

print("\nX Shape:", X.shape)
print("y Shape:", y.shape)

# Expected:
# (samples,60,26)

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

train_size = int(
    len(X) * 0.8
)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

print("\nTraining Samples:",
      len(X_train))

print("Testing Samples:",
      len(X_test))

# ==========================================================
# LSTM MODEL
# ==========================================================

model = Sequential()

model.add(
    LSTM(
        128,
        return_sequences=True,
        input_shape=(60,26)
    )
)

model.add(
    Dropout(0.2)
)

model.add(
    LSTM(
        64,
        return_sequences=False
    )
)

model.add(
    Dropout(0.2)
)

model.add(
    Dense(
        32,
        activation='relu'
    )
)

model.add(
    Dense(1)
)

# ==========================================================
# COMPILE MODEL
# ==========================================================

model.compile(
    optimizer='adam',
    loss='mse'
)

model.summary()

# ==========================================================
# CALLBACKS
# ==========================================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    verbose=1
)

# ==========================================================
# TRAIN MODEL
# ==========================================================

history = model.fit(

    X_train,
    y_train,

    epochs=150,

    batch_size=32,

    validation_split=0.2,

    callbacks=[
        early_stop,
        reduce_lr
    ],

    verbose=1
)

# ==========================================================
# PREDICTION
# ==========================================================

predictions = model.predict(
    X_test
)

# ==========================================================
# INVERSE SCALING
# ==========================================================

pred_dummy = np.zeros(
    (
        len(predictions),
        len(features)
    )
)

pred_dummy[:,0] = (
    predictions.flatten()
)

predictions_inverse = (
    scaler.inverse_transform(
        pred_dummy
    )[:,0]
)

test_dummy = np.zeros(
    (
        len(y_test),
        len(features)
    )
)

test_dummy[:,0] = y_test

y_test_inverse = (
    scaler.inverse_transform(
        test_dummy
    )[:,0]
)

# ==========================================================
# METRICS
# ==========================================================

mae = mean_absolute_error(
    y_test_inverse,
    predictions_inverse
)

rmse = np.sqrt(
    mean_squared_error(
        y_test_inverse,
        predictions_inverse
    )
)

mape = np.mean(
    np.abs(
        (
            y_test_inverse
            -
            predictions_inverse
        )
        /
        y_test_inverse
    )
) * 100

r2 = r2_score(
    y_test_inverse,
    predictions_inverse
)

# ==========================================================
# RESULTS
# ==========================================================

print("\n")
print("="*60)
print("MODEL PERFORMANCE")
print("="*60)

print("MAE  =", round(mae,4))
print("RMSE =", round(rmse,4))
print("MAPE =", round(mape,4),"%")
print("R²   =", round(r2,4))

# ==========================================================
# ACTUAL VS PREDICTED
# ==========================================================

plt.figure(figsize=(14,6))

plt.plot(
    y_test_inverse,
    label="Actual Oil Price"
)

plt.plot(
    predictions_inverse,
    label="Predicted Oil Price"
)

plt.title(
    "Actual vs Predicted Oil Price"
)

plt.xlabel("Time")

plt.ylabel("Oil Price")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

# ==========================================================
# LAST 200 SAMPLES
# ==========================================================

plt.figure(figsize=(14,6))

plt.plot(
    y_test_inverse[-200:],
    label="Actual"
)

plt.plot(
    predictions_inverse[-200:],
    label="Predicted"
)

plt.title(
    "Last 200 Samples"
)

plt.xlabel("Time")

plt.ylabel("Oil Price")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

# ==========================================================
# TRAINING LOSS
# ==========================================================

plt.figure(figsize=(10,5))

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title(
    'Training vs Validation Loss'
)

plt.xlabel(
    'Epoch'
)

plt.ylabel(
    'MSE'
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

# ==========================================================
# SAVE MODEL AND SCALER
# ==========================================================

# Save the trained LSTM model in the modern Keras format.
# FastAPI will load this file later for prediction.
model.save(
    "oil_lstm_26_features.keras"
)

# Save the fitted MinMaxScaler used during training.
# The API must use the exact same scaler for incoming data.
joblib.dump(
    scaler,
    "oil_scaler_26_features.pkl"
)

print(
    "\nModel and scaler saved successfully."
)
