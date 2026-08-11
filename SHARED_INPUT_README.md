# Shared Input Architecture

## Idea

Local Python downloads and prepares the current 60 x 26 input.
Docker only serves the model and API.

```text
Local Python
    -> FRED + XAUS
    -> 5 raw variables
    -> same 26 features
    -> latest 60 rows
    -> shared_data/latest_input.json

Docker
    -> FastAPI
    -> reads latest_input.json
    -> scaler
    -> LSTM
    -> prediction
```

## Files

Add:
- `prepare_latest_input.py`
- `test_shared_api.py`
- `shared_data/`

Replace:
- `api/app.py`

Keep unchanged:
- `predict.py`
- `live_data/feature_engineering.py`
- `live_data/pipeline.py`
- model and scaler files

## Step 1: prepare fresh data outside Docker

```powershell
python prepare_latest_input.py
```

This creates:

```text
shared_data/latest_input.json
```

## Step 2: build Docker

```powershell
docker build -t oil-price-api:3.0 .
```

## Step 3: run Docker with the shared folder mounted

From the project root:

```powershell
docker run --name oil-price-live-30 -p 8000:8000 -v "${PWD}/shared_data:/app/shared_data" oil-price-api:3.0
```

## Step 4: test

Open:

```text
http://127.0.0.1:8000/docs
```

Run:

```text
GET /predict/latest
```

or:

```powershell
python test_shared_api.py
```

## Later / next week

Just run again:

```powershell
python prepare_latest_input.py
```

The JSON file is overwritten with the newest 60 x 26 input.
Because the folder is mounted into Docker, the running container sees the new file immediately.
No Docker rebuild is needed when only the market data changes.
