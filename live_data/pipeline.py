"""
Main live-data orchestration.
"""

from __future__ import annotations

from live_data.fetch_data import fetch_raw_market_data
from live_data.feature_engineering import create_26_features, get_latest_60_rows


def prepare_latest_model_input(history_calendar_days: int = 500):
    raw_df = fetch_raw_market_data(
        history_calendar_days=history_calendar_days
    )

    engineered_df = create_26_features(raw_df)
    latest_60 = get_latest_60_rows(engineered_df)

    metadata = {
        "raw_rows": int(len(raw_df)),
        "engineered_rows": int(len(engineered_df)),
        "first_input_date": engineered_df.iloc[-60]["Date"].date().isoformat(),
        "latest_input_date": engineered_df.iloc[-1]["Date"].date().isoformat(),
        "input_shape": [60, 26],
        "sources": {
            "Oil_Price": "yfinance CL=F (WTI Crude futures)",
            "Natural_Gas": "yfinance NG=F (Henry Hub Natural Gas futures)",
            "USD_Index": "FRED DTWEXBGS",
            "VIX": "FRED VIXCLS",
            "Gold": "yfinance GC=F (COMEX Gold futures)",
        },
    }

    return latest_60, metadata


if __name__ == "__main__":
    latest_60, metadata = prepare_latest_model_input()

    print("=" * 70)
    print("LATEST LIVE MODEL INPUT")
    print("=" * 70)
    print("Shape:", latest_60.shape)
    print("First input date:", metadata["first_input_date"])
    print("Latest input date:", metadata["latest_input_date"])
    print("Sources:", metadata["sources"])
    print("=" * 70)
