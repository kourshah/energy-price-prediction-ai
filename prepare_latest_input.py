"""
Run this OUTSIDE Docker.

It downloads the latest market data through the existing live_data pipeline,
creates the latest 60 x 26 input, and saves it to shared_data/latest_input.json.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from live_data.pipeline import prepare_latest_model_input


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "shared_data"
OUTPUT_PATH = OUTPUT_DIR / "latest_input.json"


def main():
    latest_60, metadata = prepare_latest_model_input(
        history_calendar_days=500
    )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_shape": [60, 26],
        "first_input_date": metadata["first_input_date"],
        "latest_input_date": metadata["latest_input_date"],
        "sources": metadata["sources"],
        "data": latest_60.values.tolist(),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print("=" * 70)
    print("LATEST INPUT CREATED")
    print("=" * 70)
    print("Saved:", OUTPUT_PATH)
    print("Shape:", latest_60.shape)
    print("First input date:", metadata["first_input_date"])
    print("Latest input date:", metadata["latest_input_date"])
    print("=" * 70)


if __name__ == "__main__":
    main()
