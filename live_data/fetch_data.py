"""
live_data/fetch_data.py

FINAL SIMPLIFIED LIVE-DATA FIX

Main change:
- FRED is downloaded WITHOUT server-side date filtering (`cosd`).
- The full CSV series is downloaded first.
- The requested date range is then filtered LOCALLY with pandas.

Why:
The plain FRED CSV URLs worked successfully inside Docker, while the
FRED requests that included `cosd=...` repeatedly timed out.

Sources
-------
FRED:
- DCOILWTICO -> Oil_Price
- DHHNGSP    -> Natural_Gas
- DTWEXBGS   -> USD_Index
- VIXCLS     -> VIX

XAUS:
- XAU/USD daily history -> Gold

No API key is required.
"""

from __future__ import annotations

from datetime import date, timedelta
from io import StringIO

import pandas as pd
import requests


FRED_SERIES = {
    "Oil_Price": "DCOILWTICO",
    "Natural_Gas": "DHHNGSP",
    "USD_Index": "DTWEXBGS",
    "VIX": "VIXCLS",
}

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
XAUS_HISTORY_URL = "https://xaus.com/api/v1/history"


class LiveDataError(RuntimeError):
    """Raised when live market data cannot be downloaded or prepared."""


def _get(
    url: str,
    params: dict | None = None,
    timeout: int = 60,
) -> requests.Response:
    """
    Simple HTTP GET.

    No custom Session.
    No retry adapter.
    No special backoff logic.
    """

    try:
        response = requests.get(
            url,
            params=params,
            headers={
                "User-Agent": "OilPricePredictionProject/2.3"
            },
            timeout=timeout,
        )

        response.raise_for_status()

        return response

    except requests.exceptions.RequestException as exc:
        raise LiveDataError(
            f"HTTP request failed for {url}: {exc}"
        ) from exc


def fetch_fred_series(
    series_id: str,
    output_column: str,
    start_date: date,
) -> pd.DataFrame:
    """
    Download one COMPLETE FRED CSV series,
    then filter the dates locally.

    Important:
    We intentionally do NOT send `cosd` to FRED.
    """

    print(
        f"Downloading FRED {series_id} "
        f"-> {output_column} ..."
    )

    response = _get(
        FRED_CSV_URL,
        params={
            "id": series_id,
        },
        timeout=60,
    )

    df = pd.read_csv(
        StringIO(response.text)
    )

    if df.empty or df.shape[1] < 2:
        raise LiveDataError(
            f"Unexpected FRED response for {series_id}."
        )

    date_col = df.columns[0]
    value_col = df.columns[1]

    df = df.rename(
        columns={
            date_col: "Date",
            value_col: output_column,
        }
    )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    df[output_column] = pd.to_numeric(
        df[output_column],
        errors="coerce",
    )

    # -----------------------------------------------------
    # LOCAL DATE FILTER
    # -----------------------------------------------------
    start_timestamp = pd.Timestamp(start_date)

    df = df[
        df["Date"] >= start_timestamp
    ].copy()

    df = (
        df[["Date", output_column]]
        .dropna(subset=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if df.empty:
        raise LiveDataError(
            f"No rows remained for {series_id} "
            f"after local date filtering."
        )

    print(
        f"  OK: {series_id} -> "
        f"{len(df)} rows"
    )

    return df


def fetch_gold_xauusd(
    start_date: date,
) -> pd.DataFrame:
    """
    Download daily XAU/USD history from XAUS.

    Mapping:
    - d -> Date
    - c -> Gold
    """

    print(
        "Downloading XAUS XAU/USD -> Gold ..."
    )

    response = _get(
        XAUS_HISTORY_URL,
        timeout=60,
    )

    try:
        payload = response.json()
    except ValueError as exc:
        raise LiveDataError(
            "XAUS did not return valid JSON."
        ) from exc

    points = payload.get("points")

    if not points:
        raise LiveDataError(
            "XAUS returned no historical XAU/USD points."
        )

    df = pd.DataFrame(points)

    if (
        "d" not in df.columns
        or "c" not in df.columns
    ):
        raise LiveDataError(
            "Unexpected XAUS response format. "
            f"Columns received: {df.columns.tolist()}"
        )

    df = df.rename(
        columns={
            "d": "Date",
            "c": "Gold",
        }
    )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    df["Gold"] = pd.to_numeric(
        df["Gold"],
        errors="coerce",
    )

    df = (
        df[["Date", "Gold"]]
        .dropna()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Local date filtering for Gold too.
    df = df[
        df["Date"] >= pd.Timestamp(start_date)
    ].copy()

    if df.empty:
        raise LiveDataError(
            "XAUS returned history, but no rows "
            "remained after local date filtering."
        )

    print(
        f"  OK: XAUS XAU/USD -> "
        f"{len(df)} rows"
    )

    return df.reset_index(drop=True)


def fetch_raw_market_data(
    history_calendar_days: int = 500,
) -> pd.DataFrame:
    """
    Download enough raw market history to create
    the latest complete 60 x 26 model input.

    More than 60 calendar days are needed because
    the original feature engineering contains
    60-period rolling calculations.
    """

    if history_calendar_days < 250:
        raise ValueError(
            "history_calendar_days must be at least 250."
        )

    start_date = (
        date.today()
        - timedelta(
            days=history_calendar_days
        )
    )

    print("=" * 70)
    print("DOWNLOADING LIVE MARKET DATA")
    print("=" * 70)
    print("Local start date:", start_date.isoformat())
    print()

    frames = []

    # -----------------------------------------------------
    # 1. Four FRED variables
    # -----------------------------------------------------
    for (
        output_column,
        series_id,
    ) in FRED_SERIES.items():

        frame = fetch_fred_series(
            series_id=series_id,
            output_column=output_column,
            start_date=start_date,
        )

        frames.append(frame)

    # -----------------------------------------------------
    # 2. Gold XAU/USD
    # -----------------------------------------------------
    frames.append(
        fetch_gold_xauusd(
            start_date=start_date
        )
    )

    # -----------------------------------------------------
    # 3. Merge the five raw variables
    # -----------------------------------------------------
    print()
    print("Merging market variables ...")

    merged = frames[0]

    for frame in frames[1:]:
        merged = merged.merge(
            frame,
            on="Date",
            how="outer",
        )

    merged = (
        merged
        .sort_values("Date")
        .drop_duplicates(subset=["Date"])
        .reset_index(drop=True)
    )

    # Weekdays only.
    merged = merged[
        merged["Date"].dt.dayofweek < 5
    ].copy()

    feature_columns = [
        "Natural_Gas",
        "USD_Index",
        "VIX",
        "Gold",
    ]

    # Short publication/holiday gaps.
    merged[feature_columns] = (
        merged[feature_columns]
        .ffill(limit=3)
    )

    # Oil is the target; it must exist.
    merged = merged.dropna(
        subset=["Oil_Price"]
    ).copy()

    # Fill any remaining feature gaps.
    merged[feature_columns] = (
        merged[feature_columns]
        .interpolate(
            method="linear",
            limit_direction="both",
        )
    )

    # Final safety cleanup.
    merged = (
        merged
        .dropna(
            subset=[
                "Oil_Price",
                "Natural_Gas",
                "USD_Index",
                "VIX",
                "Gold",
            ]
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if len(merged) < 125:
        raise LiveDataError(
            "Not enough aligned observations were downloaded. "
            f"Received only {len(merged)} usable rows."
        )

    print(
        f"Final raw rows: {len(merged)}"
    )

    print(
        "Latest raw date:",
        merged["Date"].max()
    )

    print("=" * 70)

    return merged[
        [
            "Date",
            "Oil_Price",
            "Natural_Gas",
            "USD_Index",
            "VIX",
            "Gold",
        ]
    ]


if __name__ == "__main__":
    df = fetch_raw_market_data()

    print()
    print("LATEST RAW ROWS:")
    print(df.tail())
    print()
    print("Shape:", df.shape)
