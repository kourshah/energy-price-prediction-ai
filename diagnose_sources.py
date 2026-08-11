import time
from datetime import date, timedelta

from live_data.fetch_data import (
    fetch_fred_series,
    fetch_gold_xauusd,
)

start_date = date.today() - timedelta(days=500)

tests = [
    ("Oil", "DCOILWTICO"),
    ("Gas", "DHHNGSP"),
    ("USD", "DTWEXBGS"),
    ("VIX", "VIXCLS"),
]

print("START", flush=True)

for name, series_id in tests:
    t0 = time.perf_counter()

    fetch_fred_series(
        series_id=series_id,
        output_column=name,
        start_date=start_date,
    )

    elapsed = time.perf_counter() - t0
    print(f"{name} = {elapsed:.2f} sec", flush=True)

t0 = time.perf_counter()

fetch_gold_xauusd(
    start_date=start_date
)

elapsed = time.perf_counter() - t0
print(f"Gold = {elapsed:.2f} sec", flush=True)

print("DONE", flush=True)
