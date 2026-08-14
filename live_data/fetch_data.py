"""
live_data/fetch_data.py

PRODUCTION FIX -- switched from FRED's unofficial chart-export CSV
endpoint to FRED's official, documented REST API.

Why the previous version was unreliable:
`https://fred.stlouisfed.org/graph/fredgraph.csv` is the CSV-export
endpoint behind FRED's website *graphs* -- built to feed browser chart
widgets, not documented or rate-limit-guaranteed as a public API. It is
a known target for bot-mitigation against traffic from cloud/datacenter
IP ranges (exactly what Streamlit Community Cloud's outbound IPs look
like). That mitigation typically holds the connection open and never
responds, rather than cleanly rejecting -- which produces exactly a
"Read timed out" rather than a clean error, and explains why the same
code sometimes worked and sometimes hung.

The fix: `https://api.stlouisfed.org/fred/series/observations` is
FRED's actual documented, supported API for automated/programmatic
access, with clear rate limits (120 req/min) and clean JSON responses
instead of silent hangs.

Setup required
--------------
1. Get a free FRED API key: https://fredaccount.stlouisfed.org/apikeys
2. Set it as an environment variable / Streamlit secret named FRED_API_KEY
   - Local / Docker: export FRED_API_KEY=your_key_here
   - Streamlit Community Cloud: add FRED_API_KEY under
     App settings -> Secrets, as:  FRED_API_KEY = "your_key_here"

Sources
-------
FRED (official API):
- DCOILWTICO -> Oil_Price
- DHHNGSP    -> Natural_Gas
- DTWEXBGS   -> USD_Index
- VIXCLS     -> VIX

Stooq (CSV export, unchanged from the earlier fix):
- XAU/USD daily history -> Gold
"""

from __future__ import annotations

import os
import socket
from datetime import date, timedelta
from io import StringIO

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Global socket timeout -- covers DNS resolution too, which requests'
# own `timeout=` parameter does not reliably bound. Keep this LOW; if a
# host is going to hang, fail fast rather than block the Streamlit UI.
socket.setdefaulttimeout(20)

FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
STOOQ_GOLD_URL = "https://stooq.com/q/d/l/"

FRED_SERIES = {
    "Oil_Price": "DCOILWTICO",
    "Natural_Gas": "DHHNGSP",
    "USD_Index": "DTWEXBGS",
    "VIX": "VIXCLS",
}


class LiveDataError(RuntimeError):
    """Raised when live market data cannot be downloaded or prepared."""


def _get_fred_api_key() -> str:
    """
    Reads the FRED API key from environment variable first (works for
    local runs and Docker), falling back to Streamlit secrets if running
    inside a Streamlit app.
    """
    key = os.environ.get("FRED_API_KEY")
    if key:
        return key

    try:
        import streamlit as st
        key = st.secrets.get("FRED_API_KEY")
        if key:
            return key
    except Exception:
        pass

    raise LiveDataError(
        "FRED_API_KEY is not set. Get a free key at "
        "https://fredaccount.stlouisfed.org/apikeys and set it as an "
        "environment variable (local/Docker) or a Streamlit secret "
        "(Streamlit Community Cloud)."
    )


def _session_with_retries() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=2,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def fetch_fred_series(
    series_id: str,
    output_column: str,
    start_date: date,
) -> pd.DataFrame:
    """
    Download one FRED series via the official observations API,
    already filtered server-side to `start_date` onward (this API
    supports date filtering reliably, unlike the old CSV endpoint).
    """
    print(f"Downloading FRED {series_id} -> {output_column} (official API) ...")

    api_key = _get_fred_api_key()
    session = _session_with_retries()

    response = session.get(
        FRED_API_URL,
        params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start_date.isoformat(),
        },
        # (connect_timeout, read_timeout) -- short and explicit. If FRED
        # doesn't respond quickly, fail fast with a clear error instead
        # of freezing the Streamlit UI for up to 60s per attempt.
        timeout=(5, 15),
    )
    response.raise_for_status()

    payload = response.json()
    observations = payload.get("observations")

    if not observations:
        raise LiveDataError(
            f"FRED API returned no observations for {series_id}. "
            f"Response: {payload}"
        )

    df = pd.DataFrame(observations)[["date", "value"]]
    df = df.rename(columns={"date": "Date", "value": output_column})

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    # FRED uses "." for missing values (e.g. holidays) -- coerce handles this
    df[output_column] = pd.to_numeric(df[output_column], errors="coerce")

    df = (
        df.dropna(subset=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if df.empty:
        raise LiveDataError(f"No usable rows for {series_id} after parsing.")

    print(f"  OK: {series_id} -> {len(df)} rows, latest={df['Date'].max().date()}")
    return df


def fetch_gold_xauusd(start_date: date) -> pd.DataFrame:
       """Download daily gold price history using yfinance (COMEX Gold futures, GC=F)."""
       import yfinance as yf

       print("Downloading Gold (GC=F, COMEX futures) via yfinance ...")

       ticker = yf.Ticker("GC=F")
       hist = ticker.history(start=start_date.isoformat())

       if hist.empty:
           raise LiveDataError("yfinance returned no data for GC=F (Gold).")

       df = hist.reset_index()[["Date", "Close"]].rename(columns={"Close": "Gold"})
       df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
       df["Gold"] = pd.to_numeric(df["Gold"], errors="coerce")

       df = df.dropna().sort_values("Date").reset_index(drop=True)

       if df.empty:
           raise LiveDataError("Gold data was empty after cleaning.")

       print(f"  OK: Gold (GC=F) -> {len(df)} rows, latest={df['Date'].max().date()}")
       return df


def fetch_raw_market_data(history_calendar_days: int = 500) -> pd.DataFrame:
    """
    Download enough raw market history to build the latest complete
    60 x 26 model input. Unchanged in structure/output from before --
    only the FRED transport underneath changed.
    """
    if history_calendar_days < 250:
        raise ValueError("history_calendar_days must be at least 250.")

    start_date = date.today() - timedelta(days=history_calendar_days)

    print("=" * 70)
    print("DOWNLOADING LIVE MARKET DATA")
    print("=" * 70)
    print("Local start date:", start_date.isoformat())
    print()

    frames = []
    for output_column, series_id in FRED_SERIES.items():
        frames.append(fetch_fred_series(series_id, output_column, start_date))

    frames.append(fetch_gold_xauusd(start_date))

    print()
    print("Merging market variables ...")

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="Date", how="outer")

    merged = (
        merged.sort_values("Date")
        .drop_duplicates(subset=["Date"])
        .reset_index(drop=True)
    )
    merged = merged[merged["Date"].dt.dayofweek < 5].copy()

    feature_columns = ["Natural_Gas", "USD_Index", "VIX", "Gold"]
    merged[feature_columns] = merged[feature_columns].ffill(limit=3)
    merged = merged.dropna(subset=["Oil_Price"]).copy()
    merged[feature_columns] = merged[feature_columns].interpolate(
        method="linear", limit_direction="both"
    )

    merged = (
        merged.dropna(subset=["Oil_Price", "Natural_Gas", "USD_Index", "VIX", "Gold"])
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if len(merged) < 125:
        raise LiveDataError(
            f"Not enough aligned observations were downloaded. "
            f"Received only {len(merged)} usable rows."
        )

    print(f"Final raw rows: {len(merged)}")
    print("Latest raw date:", merged["Date"].max())
    print("=" * 70)

    return merged[["Date", "Oil_Price", "Natural_Gas", "USD_Index", "VIX", "Gold"]]


if __name__ == "__main__":
    df = fetch_raw_market_data()
    print()
    print("LATEST RAW ROWS:")
    print(df.tail())
    print()
    print("Shape:", df.shape)
