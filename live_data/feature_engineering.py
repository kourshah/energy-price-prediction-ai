"""
Reusable version of the original create_new_dataset_26.py logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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

LOOKBACK = 60


def create_26_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Lag Features
    df["Oil_Lag_1"] = df["Oil_Price"].shift(1)
    df["Oil_Lag_5"] = df["Oil_Price"].shift(5)
    df["Oil_Lag_10"] = df["Oil_Price"].shift(10)
    df["Oil_Lag_21"] = df["Oil_Price"].shift(21)

    # Rolling Statistics
    df["RollMean_7"] = df["Oil_Price"].rolling(7).mean()
    df["RollMean_21"] = df["Oil_Price"].rolling(21).mean()
    df["RollMean_60"] = df["Oil_Price"].rolling(60).mean()
    df["RollStd_7"] = df["Oil_Price"].rolling(7).std()

    # Log Returns
    df["Oil_LogRet"] = np.log(df["Oil_Price"] / df["Oil_Price"].shift(1))
    df["Gas_LogRet"] = np.log(df["Natural_Gas"] / df["Natural_Gas"].shift(1))
    df["Gold_LogRet"] = np.log(df["Gold"] / df["Gold"].shift(1))
    df["VIX_LogRet"] = np.log(df["VIX"] / df["VIX"].shift(1))

    # Momentum & Volatility
    df["Momentum_7"] = df["Oil_Price"] / df["Oil_Price"].shift(7) - 1
    df["Momentum_21"] = df["Oil_Price"] / df["Oil_Price"].shift(21) - 1
    df["Volatility_21"] = df["Oil_LogRet"].rolling(21).std()
    df["Volatility_60"] = df["Oil_LogRet"].rolling(60).std()

    # Calendar Features
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    # Cross Asset Features
    df["Gold_Oil_Ratio"] = df["Gold"] / df["Oil_Price"]

    df["VIX_Spike"] = (
        df["VIX"]
        >
        (
            df["VIX"].rolling(21).mean()
            + 2 * df["VIX"].rolling(21).std()
        )
    ).astype(int)

    df = (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )

    return df


def get_latest_60_rows(engineered_df: pd.DataFrame) -> pd.DataFrame:
    if len(engineered_df) < LOOKBACK:
        raise ValueError(
            f"Need at least {LOOKBACK} complete engineered rows, "
            f"but only have {len(engineered_df)}."
        )

    latest = engineered_df[FEATURES].tail(LOOKBACK).copy()

    if latest.shape != (LOOKBACK, len(FEATURES)):
        raise ValueError(f"Expected shape (60, 26), got {latest.shape}.")

    return latest
