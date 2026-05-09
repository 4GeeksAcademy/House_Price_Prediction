#!/usr/bin/env python3
"""
Train SmartRouter: a production-ready ensemble that combines all 4 champion
models via a learned tier classifier — no oracle price required at inference.

Architecture
------------
  1. Load all 17 K+ training records.
  2. Fit SmartRouter:
       a. Compute Bayesian-smoothed ZIP medians from training prices.
       b. Load the 4 existing champion joblib files (never retrained).
       c. Train a LightGBM multiclass classifier to predict price tier
          (budget / low / mid / luxury) from property features.
  3. Evaluate with 5-fold cross-val on the tier classifier (unbiased routing
     accuracy); then evaluate final model on full dataset (biased reference).
  4. Save -> models/nationwide_smart_router.joblib

Comparison
----------
  Champions (oracle routing)   : ~8.8%  (uses actual sale price for routing)
  SmartRouter (feature routing): ~9–10% (no price known at inference time)
  Stacking ensemble            : ~14%   (uses CV sub-models, not champions)
"""

from sklearn.model_selection import StratifiedKFold
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import json
import math
import sys
import warnings
warnings.filterwarnings("ignore")


sys.path.insert(0, str(Path(__file__).parent))
from model_utils import LogTargetPipeline, LocationAwarePipeline, SmartRouter  # noqa: F401

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data/processed/nationwide_single_family_training_data.jsonl"
MODELS = ROOT / "models"

NUMERIC_COLS = [
    "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
    "GrLivArea", "FullBath", "HalfBath", "BedroomAbvGr", "TotRmsAbvGrd",
    "Fireplaces", "GarageCars", "GarageArea", "NeighborhoodScore",
]
STR_COLS = ["PropertyType", "HouseStyle", "Neighborhood"]


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_all():
    records = [json.loads(l) for l in DATA.open(encoding="utf-8")]
    rows, prices = [], []
    skipped = 0
    for r in records:
        sp = r.get("SalePrice")
        if sp is None or math.isnan(float(sp)) or float(sp) <= 0:
            skipped += 1
            continue
        row = {col: float(r.get(col) or 0) for col in NUMERIC_COLS}
        row.update({col: str(r.get(col) or "") for col in STR_COLS})
        rows.append(row)
        prices.append(float(sp))
    print(f"  {len(rows):,} valid records  ({skipped} dropped — null SalePrice)")
    return pd.DataFrame(rows), np.array(prices)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────

def mean_pct_error(preds, actuals):
    pct = np.abs(preds - actuals) / actuals * 100
    fin = np.isfinite(pct)
    return pct[fin].mean()


def print_tier_breakdown(preds, actuals):
    tiers = [
        ("Budget  (<$200K)    ", 0,         200_000),
        ("Low     ($200-500K) ", 200_000,   500_000),
        ("Mid     ($500K-1.5M)", 500_000, 1_500_000),
        ("Luxury  (>$1.5M)    ", 1_500_000, 1e15),
    ]
    for label, lo, hi in tiers:
        mask = (actuals >= lo) & (actuals < hi)
        if not mask.any():
            continue
        ep = np.abs(preds[mask] - actuals[mask]) / actuals[mask] * 100
        fin = np.isfinite(ep)
        print(f"    {label}  n={mask.sum():4d}  "
              f"mean={ep[fin].mean():.1f}%  median={np.median(ep[fin]):.1f}%  "
              f">30%={(ep[fin]>30).sum()}")


def four_tier_oracle(X, y, budget_m, low_m, mid_m, luxury_m):
    """Baseline: oracle routing using actual prices (unfair — prices known)."""
    n = len(X)
    preds = np.zeros(n)
    for mask, fn in [
        (y < 200_000,
         lambda i: budget_m.predict(X.iloc[i:i+1])[0]),
        ((y >= 200_000) & (y < 500_000),
         lambda i: low_m.predict(X.iloc[i:i+1])[0]),
        ((y >= 500_000) & (y < 1_500_000),
         lambda i: mid_m.predict(X.iloc[i:i+1])[0]),
        (y >= 1_500_000,
         lambda i: 0.6*luxury_m.predict(X.iloc[i:i+1])[0]
         + 0.4*mid_m.predict(X.iloc[i:i+1])[0]),
    ]:
        for i in np.where(mask)[0]:
            preds[i] = fn(i)
    return preds


# ──────────────────────────────────────────────────────────────────────────────
# CV evaluation for tier classifier routing accuracy
# ──────────────────────────────────────────────────────────────────────────────

def cv_routing_accuracy(X, y, n_folds=5):
    """
    Estimate out-of-fold tier-classifier routing accuracy to check that the
    classifier generalises — not a biased train-set number.
    """
    from lightgbm import LGBMClassifier

    router_tmp = SmartRouter()
    router_tmp.global_median_ = float(np.median(y))
    router_tmp.zip_medians_ = router_tmp._compute_zip_medians(X, y)

    X_enc = router_tmp._set_cat_dtypes(router_tmp._enrich(X.copy()))
    cat_cols = [c for c in ["PropertyType", "HouseStyle", "Neighborhood", "zip3"]
                if c in X_enc.columns]
    tier_labels = router_tmp._label_tiers(y)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(y), dtype=int)

    for fold, (tr, va) in enumerate(skf.split(X_enc, tier_labels), 1):
        clf = LGBMClassifier(
            n_estimators=400, max_depth=8, learning_rate=0.05,
            num_leaves=48, min_child_samples=5, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, class_weight="balanced",
            verbose=-1, n_jobs=1, random_state=42,
        )
        X_tr = X_enc.iloc[tr].copy()
        X_va = X_enc.iloc[va].copy()
        for col in cat_cols:
            X_tr[col] = X_tr[col].astype("category")
            X_va[col] = X_va[col].astype("category")
        clf.fit(X_tr, tier_labels[tr], categorical_feature=cat_cols)
        oof_pred[va] = clf.predict(X_va)
        print(f"    Fold {fold}/{n_folds}  acc={( oof_pred[va]==tier_labels[va]).mean():.1%}",
              flush=True)

    oof_acc = (oof_pred == tier_labels).mean()
    print(f"  OOF tier-routing accuracy: {oof_acc:.1%}")

    # Per-tier confusion
    for i, name in enumerate(["budget", "low", "mid", "luxury"]):
        mask = tier_labels == i
        acc_i = (oof_pred[mask] == tier_labels[mask]).mean()
        print(f"    [{name}]  n={mask.sum():4d}  acc={acc_i:.1%}")

    return oof_acc


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    bar = "=" * 70

    print(f"\n{bar}")
    print("  SMART ROUTER TRAINING")
    print(bar)

    print("\nLoading training data...")
    X, y = load_all()
    print(f"  Price range: ${y.min():,.0f} – ${y.max():,.0f}  |  "
          f"median ${np.median(y):,.0f}")

    # ── Cross-val routing accuracy ─────────────────────────────────────
    print(f"\n{bar}")
    print("  5-FOLD TIER-CLASSIFIER CV (unbiased routing accuracy)")
    print(bar)
    oof_acc = cv_routing_accuracy(X, y, n_folds=5)

    # ── Train final SmartRouter ────────────────────────────────────────
    print(f"\n{bar}")
    print("  TRAINING FINAL SMART ROUTER")
    print(bar)
    router = SmartRouter()
    router.fit(X, y, models_dir=MODELS)

    # ── Full-dataset evaluation (biased — same data as training) ──────
    print(f"\n{bar}")
    print("  FULL-DATASET EVALUATION  (champions trained on this data)")
    print(bar)

    print("\nSmartRouter predictions (batch)...")
    sr_preds = router.predict(X)
    sr_err = mean_pct_error(sr_preds, y)

    print("Oracle champion routing predictions (sample: 2000 for speed)...")
    np.random.seed(42)
    samp = np.random.choice(len(X), min(2000, len(X)), replace=False)
    X_s, y_s = X.iloc[samp].reset_index(drop=True), y[samp]
    oc_preds = four_tier_oracle(
        X_s, y_s,
        router.champions_["budget"],
        router.champions_["low"],
        router.champions_["mid"],
        router.champions_["luxury"],
    )
    oc_err = mean_pct_error(oc_preds, y_s)

    print(f"\n  SmartRouter  (feature routing)  : {sr_err:.2f}%")
    print(
        f"  Oracle champions (oracle routing): {oc_err:.2f}%  [unfair — price known]")
    print(f"  Gap                              : {sr_err - oc_err:+.2f} pp")

    print("\n  SmartRouter tier breakdown:")
    print_tier_breakdown(sr_preds, y)

    # ── Save ─────────────────────────────────────────────────────────
    out = MODELS / "nationwide_smart_router.joblib"
    joblib.dump(router, out, compress=3)
    print(f"\n{bar}")
    print(f"  Saved -> {out}")
    print(bar)


if __name__ == "__main__":
    main()
