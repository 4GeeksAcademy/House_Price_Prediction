#!/usr/bin/env python
"""
test_nationwide_model.py

Live testing script comparing King County vs Nationwide models.
Tests both models on the same addresses and compares predictions.

Usage:
  python scripts/test_nationwide_model.py \\
    --model-kc models/house_price_model.joblib \\
    --model-nationwide models/nationwide_single_family_model.joblib \\
    --test-data data/processed/nationwide_single_family_training_data.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def load_test_data(path: Path, sample_size: int | None = None) -> pd.DataFrame:
    """Load test data from JSONL."""
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))

    df = pd.DataFrame(records)

    if sample_size and len(df) > sample_size:
        # Sample stratified by price range
        df = df.sample(n=sample_size, random_state=42)

    return df


def prepare_features_for_kc_model(row: dict[str, Any]) -> dict[str, Any]:
    """Prepare features for King County model (expects Redfin schema)."""
    return {
        "BEDS": row.get("BedroomAbvGr"),
        "BATHS": row.get("FullBath", 0) + row.get("HalfBath", 0) * 0.5,
        "SQUARE FEET": row.get("GrLivArea"),
        "LOT SIZE": row.get("LotArea"),
        "YEAR BUILT": row.get("YearBuilt"),
        "DAYS ON MARKET": 30,  # Default assumption
        "$/SQUARE FEET": row.get("GrLivArea", 1000) and row.get("SalePrice", 0) / row.get("GrLivArea", 1000) or 0,
        "HOA/MONTH": 0,  # Default assumption
        "LATITUDE": 47.5,  # King County default
        "LONGITUDE": -122.0,  # King County default
        "SALE TYPE": "Resale",
        "PROPERTY TYPE": "Single Family Residential",
        "CITY": row.get("Neighborhood", ""),
        "STATE OR PROVINCE": "WA",
        "ZIP OR POSTAL CODE": row.get("Neighborhood", ""),
        "LOCATION": "Urban",
        "STATUS": "Sold",
    }


def prepare_features_for_nationwide_model(row: dict[str, Any]) -> dict[str, Any]:
    """Prepare features for Nationwide model (16-feature schema)."""
    return {
        "LotArea": row.get("LotArea"),
        "OverallQual": row.get("OverallQual"),
        "OverallCond": row.get("OverallCond"),
        "YearBuilt": row.get("YearBuilt"),
        "YearRemodAdd": row.get("YearRemodAdd"),
        "GrLivArea": row.get("GrLivArea"),
        "FullBath": row.get("FullBath"),
        "HalfBath": row.get("HalfBath"),
        "BedroomAbvGr": row.get("BedroomAbvGr"),
        "TotRmsAbvGrd": row.get("TotRmsAbvGrd"),
        "Fireplaces": row.get("Fireplaces"),
        "GarageCars": row.get("GarageCars"),
        "GarageArea": row.get("GarageArea"),
        "NeighborhoodScore": row.get("NeighborhoodScore", 50),
        "PropertyType": row.get("PropertyType", "single_family"),
        "HouseStyle": row.get("HouseStyle", "2Story"),
        "Neighborhood": str(row.get("Neighborhood", "00000")),
    }


def predict_models(
    models: dict[str, Any],
    test_data: pd.DataFrame,
) -> pd.DataFrame:
    """Make predictions with both models."""
    results = []

    for idx, (_, row) in enumerate(test_data.iterrows()):
        actual_price = row.get("SalePrice")

        try:
            features_kc = prepare_features_for_kc_model(row.to_dict())
            df_kc = pd.DataFrame([features_kc])
            pred_kc = models["kc"].predict(df_kc)[0]
        except Exception as e:
            pred_kc = np.nan

        try:
            features_nw = prepare_features_for_nationwide_model(row.to_dict())
            df_nw = pd.DataFrame([features_nw])
            pred_nw = models["nationwide"].predict(df_nw)[0]
        except Exception as e:
            pred_nw = np.nan

        results.append({
            "actual_price": actual_price,
            "kc_prediction": pred_kc,
            "nw_prediction": pred_nw,
            "kc_error": abs(actual_price - pred_kc) if not np.isnan(pred_kc) else np.nan,
            "nw_error": abs(actual_price - pred_nw) if not np.isnan(pred_nw) else np.nan,
            "kc_error_pct": abs(actual_price - pred_kc) / actual_price * 100 if not np.isnan(pred_kc) and actual_price != 0 else np.nan,
            "nw_error_pct": abs(actual_price - pred_nw) / actual_price * 100 if not np.isnan(pred_nw) and actual_price != 0 else np.nan,
        })

    return pd.DataFrame(results)


def evaluate_predictions(results: pd.DataFrame) -> dict[str, float]:
    """Calculate evaluation metrics."""
    metrics = {}

    for model_name in ["kc", "nw"]:
        error_col = f"{model_name}_error"
        error_pct_col = f"{model_name}_error_pct"

        valid_errors = results[error_col].dropna()
        valid_pcts = results[error_pct_col].dropna()

        if len(valid_errors) > 0:
            metrics[f"{model_name}_mae"] = valid_errors.mean()
            metrics[f"{model_name}_rmse"] = np.sqrt((valid_errors ** 2).mean())
            metrics[f"{model_name}_median_error"] = valid_errors.median()
            metrics[f"{model_name}_mean_error_pct"] = valid_pcts.mean()
            metrics[f"{model_name}_count"] = len(valid_errors)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live test comparing King County vs Nationwide models."
    )
    parser.add_argument(
        "--model-kc",
        default="models/house_price_model.joblib",
        help="Path to King County model.",
    )
    parser.add_argument(
        "--model-nationwide",
        default="models/nationwide_single_family_model.joblib",
        help="Path to Nationwide model.",
    )
    parser.add_argument(
        "--test-data",
        default="data/processed/nationwide_single_family_training_data.jsonl",
        help="Path to test data.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of samples to test (default: 100).",
    )
    args = parser.parse_args()

    # Load models
    print("=" * 80)
    print("  LIVE MODEL COMPARISON TEST")
    print("=" * 80)
    print()
    print("[1/3] Loading models...")

    kc_model_path = Path(args.model_kc)
    nw_model_path = Path(args.model_nationwide)

    if not kc_model_path.exists():
        print(f"ERROR: King County model not found: {kc_model_path}")
        return
    if not nw_model_path.exists():
        print(f"ERROR: Nationwide model not found: {nw_model_path}")
        return

    models = {
        "kc": joblib.load(kc_model_path),
        "nationwide": joblib.load(nw_model_path),
    }
    print(
        f"      ✓ King County model loaded ({kc_model_path.stat().st_size / 1e6:.1f} MB)")
    print(
        f"      ✓ Nationwide model loaded ({nw_model_path.stat().st_size / 1e6:.1f} MB)")
    print()

    # Load test data
    print("[2/3] Loading test data...")
    test_data = load_test_data(
        Path(args.test_data), sample_size=args.sample_size)
    print(f"      Loaded {len(test_data):,} test samples")
    print(
        f"      Price range: ${test_data['SalePrice'].min():,.0f} - ${test_data['SalePrice'].max():,.0f}")
    print(f"      Median price: ${test_data['SalePrice'].median():,.0f}")
    print()

    # Make predictions
    print("[3/3] Making predictions...")
    results = predict_models(models, test_data)
    metrics = evaluate_predictions(results)
    print(f"      ✓ Completed {len(results):,} predictions")
    print()

    # Results
    print("=" * 80)
    print("  TEST RESULTS")
    print("=" * 80)
    print()
    print(f"  {'Metric':<25} {'King County':<20} {'Nationwide':<20} {'Winner':<15}")
    print(f"  {'-'*80}")

    # MAE comparison
    kc_mae = metrics.get("kc_mae", np.nan)
    nw_mae = metrics.get("nw_mae", np.nan)
    mae_winner = "Nationwide ✓" if nw_mae < kc_mae else "King County ✓" if kc_mae < nw_mae else "Tie"
    print(f"  {'Mean Absolute Error':<25} ${kc_mae:>17,.0f} ${nw_mae:>17,.0f} {mae_winner:<15}")

    # RMSE comparison
    kc_rmse = metrics.get("kc_rmse", np.nan)
    nw_rmse = metrics.get("nw_rmse", np.nan)
    rmse_winner = "Nationwide ✓" if nw_rmse < kc_rmse else "King County ✓" if kc_rmse < nw_rmse else "Tie"
    print(f"  {'Root Mean Squared Error':<25} ${kc_rmse:>17,.0f} ${nw_rmse:>17,.0f} {rmse_winner:<15}")

    # Median error comparison
    kc_med = metrics.get("kc_median_error", np.nan)
    nw_med = metrics.get("nw_median_error", np.nan)
    med_winner = "Nationwide ✓" if nw_med < kc_med else "King County ✓" if kc_med < nw_med else "Tie"
    print(f"  {'Median Absolute Error':<25} ${kc_med:>17,.0f} ${nw_med:>17,.0f} {med_winner:<15}")

    # Error percentage
    kc_pct = metrics.get("kc_mean_error_pct", np.nan)
    nw_pct = metrics.get("nw_mean_error_pct", np.nan)
    pct_winner = "Nationwide ✓" if nw_pct < kc_pct else "King County ✓" if kc_pct < nw_pct else "Tie"
    print(f"  {'Mean Error % of Price':<25} {kc_pct:>18.2f}% {nw_pct:>18.2f}% {pct_winner:<15}")

    print(f"  {'-'*80}")
    print()

    # Show sample predictions
    print("  Sample Predictions (first 10):")
    print(f"  {'-'*80}")
    print(f"  {'Actual':<15} {'KC Pred':<15} {'NW Pred':<15} {'KC Error %':<15} {'NW Error %':<15}")
    print(f"  {'-'*80}")

    for i in range(min(10, len(results))):
        row = results.iloc[i]
        actual = row["actual_price"]
        kc_pred = row["kc_prediction"]
        nw_pred = row["nw_prediction"]
        kc_err_pct = row["kc_error_pct"]
        nw_err_pct = row["nw_error_pct"]

        print(
            f"  ${actual:>13,.0f} ${kc_pred:>13,.0f} ${nw_pred:>13,.0f} {kc_err_pct:>13.1f}% {nw_err_pct:>13.1f}%")

    print()
    print("=" * 80)

    # Recommendations
    print()
    print("  RECOMMENDATIONS:")
    print("  " + "-" * 76)
    if nw_mae < kc_mae * 0.95:
        print("  ✓ NATIONWIDE model is performing significantly better (5%+ improvement)")
        print("    → Consider deploying nationwide model to production")
    elif kc_mae < nw_mae * 0.95:
        print("  ✓ KING COUNTY model is still better (5%+ edge)")
        print("    → Consider keeping KC model or using ensemble approach")
    else:
        print("  → Models perform similarly; hybrid deployment recommended")
        print("    → Use KC for local predictions, NW for national coverage")
    print()


if __name__ == "__main__":
    main()
