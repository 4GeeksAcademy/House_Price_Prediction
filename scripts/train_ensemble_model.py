#!/usr/bin/env python3
"""
Train the stacking ensemble model that combines all four price-tier champion
architectures into a single level-2 model.

The existing champion models are NOT modified:
  models/nationwide_budget_model.joblib
  models/nationwide_low_price_model.joblib
  models/nationwide_mid_price_model_v2.joblib
  models/nationwide_luxury_model.joblib

Output: models/nationwide_ensemble_model.joblib
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
from model_utils import LogTargetPipeline, LocationAwarePipeline, EnsembleModel  # noqa: F401

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data/processed/nationwide_single_family_training_data.jsonl"
MODELS = ROOT / "models"

NUMERIC_COLS = [
    "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
    "GrLivArea", "FullBath", "HalfBath", "BedroomAbvGr", "TotRmsAbvGrd",
    "Fireplaces", "GarageCars", "GarageArea", "NeighborhoodScore",
]
STR_COLS = ["PropertyType", "HouseStyle", "Neighborhood"]


def load_data() -> tuple:
    records = [json.loads(line) for line in DATA.open()]
    rows, prices = [], []
    for r in records:
        sp = r.get("SalePrice")
        if sp is None or float(sp) <= 0:
            continue  # skip records without a valid sale price
        row = {}
        for col in NUMERIC_COLS:
            v = r.get(col)
            row[col] = float(v) if v is not None else 0.0
        for col in STR_COLS:
            v = r.get(col)
            row[col] = str(v) if v is not None else ""
        rows.append(row)
        prices.append(float(sp))
    return pd.DataFrame(rows), pd.Series(prices)


def four_tier_predict(X: pd.DataFrame, y_actual: np.ndarray,
                      budget_m, low_m, mid_m, lux_m) -> np.ndarray:
    """Batch 4-tier champion routing for fair comparison."""
    n = len(X)
    preds = np.zeros(n)

    b_mask = y_actual < 200_000
    l_mask = (y_actual >= 200_000) & (y_actual < 500_000)
    m_mask = (y_actual >= 500_000) & (y_actual < 1_500_000)
    x_mask = y_actual >= 1_500_000

    if b_mask.any():
        preds[b_mask] = budget_m.predict(X[b_mask])
    if l_mask.any():
        preds[l_mask] = low_m.predict(X[l_mask])
    if m_mask.any():
        preds[m_mask] = mid_m.predict(X[m_mask])
    if x_mask.any():
        raw = lux_m.predict(X[x_mask])
        mid = mid_m.predict(X[x_mask])
        preds[x_mask] = 0.6 * raw + 0.4 * mid

    return preds


def report(label: str, preds: np.ndarray, actuals: np.ndarray) -> float:
    err_abs = np.abs(preds - actuals)
    err_pct = err_abs / actuals * 100
    fin = np.isfinite(err_pct)
    mean_pct = err_pct[fin].mean()
    print(f"  {label}")
    print(f"    Mean Error %   : {mean_pct:.2f}%")
    print(f"    Median Error % : {np.median(err_pct[fin]):.2f}%")
    print(f"    MAE            : ${err_abs[fin].mean():,.0f}")
    print(f"    >30%% errors   : {(err_pct[fin] > 30).sum()} "
          f"({(err_pct[fin] > 30).mean()*100:.1f}%)")
    return mean_pct


def main():
    print("\n" + "=" * 70)
    print("  ENSEMBLE MODEL TRAINING")
    print("  4 diverse LightGBM sub-models + LightGBM meta-learner")
    print("=" * 70)

    print("\n[1] Loading data...")
    X, y = load_data()
    print(f"  {len(X):,} records  |  "
          f"${y.min():,.0f} – ${y.max():,.0f}  |  median ${y.median():,.0f}")

    print("\n[2] Training EnsembleModel (all data, 5-fold OOF)...")
    ensemble = EnsembleModel()
    ensemble.fit(X, y, n_folds=5)

    print("\n[3] Evaluating on full dataset (in-sample, consistent with champion baseline)...")
    ens_preds = ensemble.predict(X)
    y_arr = y.values

    print("\n" + "=" * 70)
    print("  PERFORMANCE COMPARISON — FULL 17,753 RECORDS")
    print("=" * 70)

    ens_mean = report("ENSEMBLE MODEL", ens_preds, y_arr)

    # Load champions for comparison
    print()
    budget_m = joblib.load(MODELS / "nationwide_budget_model.joblib")
    low_m = joblib.load(MODELS / "nationwide_low_price_model.joblib")
    mid_m = joblib.load(MODELS / "nationwide_mid_price_model_v2.joblib")
    lux_m = joblib.load(MODELS / "nationwide_luxury_model.joblib")

    champ_preds = four_tier_predict(X, y_arr, budget_m, low_m, mid_m, lux_m)
    champ_mean = report("4-TIER CHAMPION ROUTING", champ_preds, y_arr)

    delta = champ_mean - ens_mean
    print(f"\n  Delta (champion - ensemble): {delta:+.2f}%  "
          f"({'ENSEMBLE WINS' if delta > 0 else 'champions still best'})")

    # Per-tier breakdown for ensemble
    print("\n" + "=" * 70)
    print("  ENSEMBLE TIER BREAKDOWN (by actual price range)")
    print("=" * 70)
    for label, mask in [
        ("Budget  (<$200K)   ", y_arr < 200_000),
        ("Low     ($200-500K)", (y_arr >= 200_000) & (y_arr < 500_000)),
        ("Mid     ($500K-1.5M)", (y_arr >= 500_000) & (y_arr < 1_500_000)),
        ("Luxury  (>$1.5M)   ", y_arr >= 1_500_000),
    ]:
        if mask.any():
            ep = np.abs(ens_preds[mask] - y_arr[mask]) / y_arr[mask] * 100
            fin = np.isfinite(ep)
            print(f"  {label}  n={mask.sum():,}  "
                  f"mean={ep[fin].mean():.1f}%  "
                  f"median={np.median(ep[fin]):.1f}%  "
                  f">30%={(ep[fin]>30).sum()}")

    # Save
    out_path = MODELS / "nationwide_ensemble_model.joblib"
    joblib.dump(ensemble, out_path)
    print(f"\n[4] Saved  ->  {out_path}")
    print("    Champion models remain intact at their original paths.\n")


if __name__ == "__main__":
    main()
