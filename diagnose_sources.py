import time
from datetime import date, timedelta

from live_data.fetch_data import (
    fetch_fred_series,
    fetch_yfinance_series,
)

start_date = date.today() - timedelta(days=500)

# USD_Index and VIX -- same-day publishers, stay on FRED
fred_tests = [
    ("USD", "DTWEXBGS"),
    ("VIX", "VIXCLS"),
]

# Oil_Price, Natural_Gas, Gold -- moved to yfinance (FRED/EIA batches
# these ~weekly even though they're labeled "Daily")
yfinance_tests = [
    ("Oil", "CL=F"),
    ("Gas", "NG=F"),
    ("Gold", "GC=F"),
]

print("START", flush=True)

for name, series_id in fred_tests:
    t0 = time.perf_counter()

    fetch_fred_series(
        series_id=series_id,
        output_column=name,
        start_date=start_date,
    )

    elapsed = time.perf_counter() - t0
    print(f"{name} = {elapsed:.2f} sec", flush=True)

for name, ticker in yfinance_tests:
    t0 = time.perf_counter()

    fetch_yfinance_series(
        ticker=ticker,
        output_column=name,
        start_date=start_date,
    )

    elapsed = time.perf_counter() - t0
    print(f"{name} = {elapsed:.2f} sec", flush=True)

print("DONE", flush=True)
