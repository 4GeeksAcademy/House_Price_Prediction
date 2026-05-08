#!/usr/bin/env python3
"""
Test the ensemble model against the 4-tier champion routing system.
Uses the same 1000-property random sample (seed=44) as the champion baseline.
"""

from argparse import ArgumentParser
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
import warnings
warnings.filterwarnings("ignore")


sys.path.insert(0, str(Path(__file__).parent))
from model_utils import LogTargetPipeline, LocationAwarePipeline, EnsembleModel, SmartRouter  # noqa: F401

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data/processed/nationwide_single_family_training_data.jsonl"
MODELS = ROOT / "models"

NUMERIC_COLS = [
    "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
    "GrLivArea", "FullBath", "HalfBath", "BedroomAbvGr", "TotRmsAbvGrd",
    "Fireplaces", "GarageCars", "GarageArea", "NeighborhoodScore",
]
STR_COLS = ["PropertyType", "HouseStyle", "Neighborhood"]


def load_sample(sample_size: int, seed: int) -> tuple:
    records = [json.loads(l) for l in DATA.open()]
    np.random.seed(seed)
    idx = np.random.choice(len(records), sample_size, replace=False)
    records = [records[i] for i in idx]

    rows, prices = [], []
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
    return pd.DataFrame(rows), np.array(prices)


def four_tier(X: pd.DataFrame, y: np.ndarray,
              b_m, l_m, m_m, x_m) -> np.ndarray:
    """Batch 4-tier champion routing — ORACLE version (uses actual price)."""
    n = len(X)
    preds = np.zeros(n)
    for mask, fn in [
        (y < 200_000, lambda i: b_m.predict(X.iloc[i:i+1])[0]),
        ((y >= 200_000) & (y < 500_000),
         lambda i: l_m.predict(X.iloc[i:i+1])[0]),
        ((y >= 500_000) & (y < 1_500_000),
         lambda i: m_m.predict(X.iloc[i:i+1])[0]),
        (y >= 1_500_000, lambda i: 0.6 *
         x_m.predict(X.iloc[i:i+1])[0] + 0.4*m_m.predict(X.iloc[i:i+1])[0]),
    ]:
        for i in np.where(mask)[0]:
            preds[i] = fn(i)
    return preds


def print_stats(label: str, preds: np.ndarray, actuals: np.ndarray,
                show_extremes: bool = False) -> float:
    err_abs = np.abs(preds - actuals)
    err_pct = err_abs / actuals * 100
    fin = np.isfinite(err_pct)
    mean_pct = err_pct[fin].mean()

    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  {label}")
    print(bar)
    print(f"  Samples         : {fin.sum():,}")
    print(f"  Mean Error %    : {mean_pct:.2f}%")
    print(f"  Median Error %  : {np.median(err_pct[fin]):.2f}%")
    print(f"  MAE             : ${err_abs[fin].mean():,.0f}")
    print(f"  RMSE            : ${np.sqrt((err_abs[fin]**2).mean()):,.0f}")
    print(f"  >30%% errors    : {(err_pct[fin] > 30).sum()} "
          f"({(err_pct[fin] > 30).mean()*100:.1f}%)")

    if show_extremes:
        best_i = np.argsort(err_pct[fin])[:5]
        worst_i = np.argsort(err_pct[fin])[-5:][::-1]
        fin_idx = np.where(fin)[0]
        print("\n  Best 5:")
        for i in fin_idx[best_i]:
            print(f"    Actual ${actuals[i]:>12,.0f}  →  "
                  f"Predicted ${preds[i]:>12,.0f}  ({err_pct[i]:.1f}%)")
        print("\n  Worst 5:")
        for i in fin_idx[worst_i]:
            print(f"    Actual ${actuals[i]:>12,.0f}  →  "
                  f"Predicted ${preds[i]:>12,.0f}  ({err_pct[i]:.1f}%)")

    return mean_pct


def print_tier_breakdown(label: str, preds: np.ndarray, actuals: np.ndarray):
    print(f"\n  {label} — by price tier:")
    for tier_label, lo, hi in [
        ("Budget  (<$200K)    ", 0,         200_000),
        ("Low     ($200-500K) ", 200_000,   500_000),
        ("Mid     ($500K-1.5M)", 500_000, 1_500_000),
        ("Luxury  (>$1.5M)    ", 1_500_000, 1e15),
    ]:
        mask = (actuals >= lo) & (actuals < hi)
        if not mask.any():
            continue
        ep = np.abs(preds[mask] - actuals[mask]) / actuals[mask] * 100
        fin = np.isfinite(ep)
        print(f"    {tier_label}  n={mask.sum():3d}  "
              f"mean={ep[fin].mean():.1f}%  "
              f"median={np.median(ep[fin]):.1f}%  "
              f">30%={(ep[fin]>30).sum()}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed",        type=int, default=44)
    parser.add_argument(
        "--ensemble-model",
        default=str(MODELS / "nationwide_ensemble_model.joblib"),
    )
    parser.add_argument(
        "--smart-router",
        default=str(MODELS / "nationwide_smart_router.joblib"),
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  ENSEMBLE vs SMART ROUTER vs CHAMPION COMPARISON TEST")
    print(f"  Sample: {args.sample_size:,} properties  |  seed={args.seed}")
    print("=" * 70)

    print("\nLoading test data...")
    X, y = load_sample(args.sample_size, args.seed)
    print(f"  {len(X):,} records  |  ${y.min():,.0f} – ${y.max():,.0f}")

    print("\nLoading models...")
    ensemble = joblib.load(args.ensemble_model)
    budget_m = joblib.load(MODELS / "nationwide_budget_model.joblib")
    low_m = joblib.load(MODELS / "nationwide_low_price_model.joblib")
    mid_m = joblib.load(MODELS / "nationwide_mid_price_model_v2.joblib")
    lux_m = joblib.load(MODELS / "nationwide_luxury_model.joblib")

    smart_router_path = Path(args.smart_router)
    has_sr = smart_router_path.exists()
    if has_sr:
        sr_model = joblib.load(smart_router_path)
        print("  All models loaded (including SmartRouter).")
    else:
        sr_model = None
        print("  SmartRouter not found — run train_smart_router.py first.")
        print("  All other models loaded.")

    # ── Predictions ──────────────────────────────────────────────────────
    print("\nRunning stacking ensemble predictions...")
    ens_preds = ensemble.predict(X)

    print("Running oracle champion routing (actual price used for tier)...")
    champ_preds = four_tier(X, y, budget_m, low_m, mid_m, lux_m)

    if has_sr:
        print("Running SmartRouter predictions (feature-based tier routing)...")
        sr_preds = sr_model.predict(X)

    # ── Results ──────────────────────────────────────────────────────────
    ens_mean = print_stats("STACKING ENSEMBLE",
                           ens_preds,   y, show_extremes=True)
    champ_mean = print_stats("CHAMPIONS — oracle routing\n  "
                             "(⚠ uses actual sale price for tier — not fair in production)",
                             champ_preds, y)
    if has_sr:
        sr_mean = print_stats("SMART ROUTER — feature routing\n  "
                              "(combines all 4 champions via learned tier classifier)",
                              sr_preds, y, show_extremes=True)

    print_tier_breakdown("STACKING ENSEMBLE", ens_preds,   y)
    print_tier_breakdown("CHAMPIONS (oracle)", champ_preds, y)
    if has_sr:
        print_tier_breakdown("SMART ROUTER", sr_preds, y)

    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)
    print(f"  Stacking ensemble (retrained sub-models) : {ens_mean:.2f}%")
    print(f"  Champions  — oracle routing  (UNFAIR)    : {champ_mean:.2f}%")
    if has_sr:
        print(f"  SmartRouter — feature routing (FAIR)     : {sr_mean:.2f}%")
        best = min(ens_mean, sr_mean)
        print(f"\n  Best production model: "
              f"{'SmartRouter' if sr_mean < ens_mean else 'Stacking Ensemble'}  "
              f"({best:.2f}% vs oracle {champ_mean:.2f}%)")
    print()


if __name__ == "__main__":
    main()
