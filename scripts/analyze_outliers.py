#!/usr/bin/env python3
"""
Batch outlier analysis — identifies worst predictions across the full dataset.
Uses vectorized (batch) prediction per tier to avoid sklearn warning spam.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
import warnings
warnings.filterwarnings("ignore")


sys.path.insert(0, str(Path(__file__).parent))
from model_utils import LogTargetPipeline, LocationAwarePipeline  # noqa: F401

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data/processed/nationwide_single_family_training_data.jsonl"
MODELS = ROOT / "models"

NUMERIC_COLS = [
    "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
    "GrLivArea", "FullBath", "HalfBath", "BedroomAbvGr", "TotRmsAbvGrd",
    "Fireplaces", "GarageCars", "GarageArea", "NeighborhoodScore",
]
STR_COLS = ["PropertyType", "HouseStyle", "Neighborhood"]
FEATURE_COLS = NUMERIC_COLS + STR_COLS


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    records = [json.loads(l) for l in DATA.open()]
    rows = []
    prices = []
    for r in records:
        row = {}
        for col in NUMERIC_COLS:
            v = r.get(col)
            row[col] = float(v) if v is not None else 0.0
        for col in STR_COLS:
            v = r.get(col)
            row[col] = str(v) if v is not None else ""
        rows.append(row)
        prices.append(float(r.get("SalePrice", 0)))
    return pd.DataFrame(rows), pd.Series(prices)


def main():
    print("Loading data...")
    X, y = load_data()
    n = len(X)
    print(f"  {n:,} records loaded")

    print("Loading models...")
    budget_m = joblib.load(MODELS / "nationwide_budget_model.joblib")
    low_m = joblib.load(MODELS / "nationwide_low_price_model.joblib")
    mid_m = joblib.load(MODELS / "nationwide_mid_price_model_v2.joblib")
    luxury_m = joblib.load(MODELS / "nationwide_luxury_model.joblib")
    print("  Models loaded")

    # --- Batch prediction by tier mask ---
    budget_mask = y < 200_000
    low_mask = (y >= 200_000) & (y < 500_000)
    mid_mask = (y >= 500_000) & (y < 1_500_000)
    luxury_mask = y >= 1_500_000

    preds = np.zeros(n)
    tiers = np.empty(n, dtype=object)

    print("Predicting (batch)...")
    if budget_mask.any():
        preds[budget_mask] = budget_m.predict(X[budget_mask])
        tiers[budget_mask] = "budget"
    if low_mask.any():
        preds[low_mask] = low_m.predict(X[low_mask])
        tiers[low_mask] = "low"
    if mid_mask.any():
        preds[mid_mask] = mid_m.predict(X[mid_mask])
        tiers[mid_mask] = "mid"
    if luxury_mask.any():
        raw_lux = luxury_m.predict(X[luxury_mask])
        mid_lux = mid_m.predict(X[luxury_mask])
        preds[luxury_mask] = 0.6 * raw_lux + 0.4 * mid_lux
        tiers[luxury_mask] = "luxury"
    print("  Done")

    # --- Compute error ---
    y_arr = y.values
    err_abs = np.abs(preds - y_arr)
    err_pct = err_abs / y_arr * 100

    df = X.copy()
    df["SalePrice"] = y_arr
    df["Predicted"] = preds
    df["ErrorAbs"] = err_abs
    df["ErrorPct"] = err_pct
    df["Tier"] = tiers

    # --- Overall accuracy summary ---
    print("\n" + "="*70)
    print("OVERALL ACCURACY — FULL DATASET")
    print("="*70)
    finite = np.isfinite(err_pct)
    print(f"  Mean Error %     : {err_pct[finite].mean():.1f}%")
    print(f"  Median Error %   : {np.median(err_pct[finite]):.1f}%")
    print(f"  MAE              : ${err_abs[finite].mean():,.0f}")
    print(f"  >30% errors      : {(err_pct[finite] > 30).sum():,} "
          f"({(err_pct[finite] > 30).mean()*100:.1f}%)")

    for tier in ["budget", "low", "mid", "luxury"]:
        mask = tiers == tier
        if not mask.any():
            continue
        ep = err_pct[mask & finite]
        print(f"\n  [{tier.upper()}] n={mask.sum():,}  "
              f"mean={ep.mean():.1f}%  "
              f"median={np.median(ep):.1f}%  "
              f"p90={np.percentile(ep,90):.1f}%  "
              f">30%={(ep>30).sum()}")

    # --- Top 30 worst predictions ---
    worst = df[np.isfinite(df["ErrorPct"])].nlargest(30, "ErrorPct")

    print("\n" + "="*70)
    print("TOP 30 WORST PREDICTIONS")
    print("="*70)
    display_cols = ["Neighborhood", "PropertyType", "GrLivArea", "OverallQual",
                    "YearBuilt", "SalePrice", "Predicted", "ErrorPct", "Tier"]
    pd.set_option("display.float_format", "{:,.0f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 140)
    print(worst[display_cols].to_string(index=False))

    # --- ZIP-level error analysis ---
    df["zip5"] = df["Neighborhood"].astype(str).str[:5]
    zip_stats = (
        df[np.isfinite(df["ErrorPct"])]
        .groupby("zip5")
        .agg(
            count=("ErrorPct", "size"),
            mean_err=("ErrorPct", "mean"),
            median_err=("ErrorPct", "median"),
            p90_err=("ErrorPct", lambda x: np.percentile(x, 90)),
            median_price=("SalePrice", "median"),
        )
        .query("count >= 3")
        .sort_values("mean_err", ascending=False)
        .head(20)
    )
    print("\n" + "="*70)
    print("TOP 20 WORST-PERFORMING ZIPs (≥3 samples)")
    print("="*70)
    print(zip_stats.to_string())

    # --- Smoothing opportunity: ZIPs with few training samples ---
    zip_counts = df.groupby("zip5").size().rename("n_samples")
    zip_err = (
        df[np.isfinite(df["ErrorPct"])]
        .groupby("zip5")["ErrorPct"]
        .mean()
        .rename("mean_err")
    )
    small_zip = (
        pd.concat([zip_counts, zip_err], axis=1)
        .query("n_samples <= 5")
        .sort_values("mean_err", ascending=False)
        .head(15)
    )
    print("\n" + "="*70)
    print("SMALL-SAMPLE ZIPs (≤5 records) — highest error")
    print("="*70)
    print(small_zip.to_string())


if __name__ == "__main__":
    main()
