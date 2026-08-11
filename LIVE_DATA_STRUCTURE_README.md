# Reorganized Live-Data Architecture

## Structure

```text
OilPricePrediction/
│
├── api/
│   └── app.py
│
├── live_data/
│   ├── __init__.py
│   ├── fetch_data.py
│   ├── feature_engineering.py
│   └── pipeline.py
│
├── models/
│   ├── oil_lstm_26_features.keras
│   └── oil_scaler_26_features.pkl
│
├── data/
│   └── final_dataset_26_features.csv
│
├── predict.py
├── test_api.py
├── test_live_api.py
├── train.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

## Responsibilities

- `live_data/fetch_data.py`: downloads Oil, Natural Gas, USD Index, VIX and Gold.
- `live_data/feature_engineering.py`: converts the five raw variables to the same 26 model features.
- `live_data/pipeline.py`: creates the latest complete `(60, 26)` model input.
- `predict.py`: keeps using the saved scaler and LSTM.
- `api/app.py`: adds `GET /predict/latest`.

## Data flow

```text
User
  ↓
GET /predict/latest
  ↓
FastAPI
  ↓
live_data/pipeline.py
  ↓
live_data/fetch_data.py
  ↓
FRED + Stooq
  ↓
5 raw variables
  ↓
live_data/feature_engineering.py
  ↓
26 features
  ↓
latest 60 complete rows
  ↓
predict.py
  ↓
saved scaler
  ↓
saved LSTM
  ↓
predicted oil price
```

## Dependency

Ensure `requests` is in `requirements.txt`.

## Test locally

```powershell
python test_live_api.py
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

Then execute:

```text
GET /predict/latest
```
