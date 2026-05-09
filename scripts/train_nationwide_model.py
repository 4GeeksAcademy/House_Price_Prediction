#!/usr/bin/env python
"""
train_nationwide_model.py

Specialized training script for nationwide single-family homes dataset.
Loads 16-feature canonical JSONL format and trains LightGBM model.

Usage:
  python scripts/train_nationwide_model.py \\
    --data data/processed/nationwide_single_family_training_data.jsonl \\
    --output models/nationwide_single_family_model.joblib
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
    "GrLivArea", "FullBath", "HalfBath", "BedroomAbvGr", "TotRmsAbvGrd",
    "Fireplaces", "GarageCars", "GarageArea", "NeighborhoodScore",
    "PropertyType", "HouseStyle", "Neighborhood"
]

NUMERIC_FEATURES = [
    "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
    "GrLivArea", "FullBath", "HalfBath", "BedroomAbvGr", "TotRmsAbvGrd",
    "Fireplaces", "GarageCars", "GarageArea", "NeighborhoodScore"
]

CATEGORICAL_FEATURES = ["PropertyType", "HouseStyle", "Neighborhood"]

TARGET_COLUMN = "SalePrice"


def load_jsonl_data(path: Path) -> pd.DataFrame:
    """Load JSONL training data."""
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    return df


def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Prepare and split data."""
    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()

    # Drop rows with missing target
    df = df.dropna(subset=[TARGET_COLUMN])

    # Fill missing numeric features with median
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Fill missing categorical features with mode
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(
                df[col].mode()[0] if not df[col].mode().empty else "Unknown")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def build_model() -> Pipeline:
    """Build sklearn pipeline with preprocessing and LightGBM."""

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("log_transform", FunctionTransformer(np.log1p, validate=True))
                ]),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline([
                    ("onehot", OneHotEncoder(
                        handle_unknown="ignore", sparse_output=False))
                ]),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LGBMRegressor(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ))
    ])

    return model


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Evaluate model on test set."""
    y_pred = model.predict(X_test)

    # Handle log-scale predictions (if used)
    if np.any(y_pred < 0):
        y_pred = np.maximum(y_pred, 0)

    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred))
    r2 = 1 - (np.sum((y_test - y_pred) ** 2) /
              np.sum((y_test - y_test.mean()) ** 2))

    return {
        "test_r2": r2,
        "test_rmse": rmse,
        "test_mae": mae,
        "test_mse": mse,
        "test_median_error": float(np.median(np.abs(y_test - y_pred))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train LightGBM model on nationwide single-family homes data."
    )
    parser.add_argument(
        "--data",
        default="data/processed/nationwide_single_family_training_data.jsonl",
        help="Path to nationwide training JSONL data.",
    )
    parser.add_argument(
        "--output",
        default="models/nationwide_single_family_model.joblib",
        help="Output path for trained model.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  NATIONWIDE MODEL TRAINING")
    print("=" * 70)
    print(f"  Data: {data_path}")
    print(f"  Output: {output_path}")
    print()

    # Load data
    print("[1/4] Loading data...")
    df = load_jsonl_data(data_path)
    print(f"      Loaded {len(df):,} records")
    print(f"      Columns: {list(df.columns)[:5]}...")
    print()

    # Prepare data
    print("[2/4] Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(f"      Train: {len(X_train):,} samples")
    print(f"      Test:  {len(X_test):,} samples")
    print(f"      Price range: ${y_train.min():,.0f} - ${y_train.max():,.0f}")
    print()

    # Build and train model
    print("[3/4] Training model...")
    model = build_model()
    model.fit(X_train, y_train)
    print("      Training complete")
    print()

    # Evaluate
    print("[4/4] Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    print(f"      R² Score:        {metrics['test_r2']:.4f}")
    print(f"      RMSE:            ${metrics['test_rmse']:,.2f}")
    print(f"      MAE:             ${metrics['test_mae']:,.2f}")
    print(f"      Median Error:    ${metrics['test_median_error']:,.2f}")
    print()

    # Save model
    print("[5/5] Saving model...")
    joblib.dump(model, output_path)
    print(f"      Saved to {output_path}")

    # Summary
    print()
    print("=" * 70)
    print("  TRAINING SUMMARY")
    print("=" * 70)
    print(f"  Model type:        LightGBM (sklearn Pipeline)")
    print(
        f"  Features:          {len(FEATURE_COLUMNS)} (16-feature canonical schema)")
    print(f"  Training samples:  {len(X_train):,}")
    print(f"  Test samples:      {len(X_test):,}")
    print(f"  R² (test):         {metrics['test_r2']:.4f}")
    print(f"  MAE (test):        ${metrics['test_mae']:,.2f}")
    print(f"  RMSE (test):       ${metrics['test_rmse']:,.2f}")
    print(f"  Timestamp:         {datetime.now(UTC).isoformat()}")
    print("=" * 70)

    # Comparison with King County baseline
    print()
    print("  COMPARISON TO KING COUNTY BASELINE")
    print("  ──────────────────────────────────────────────────")
    print(f"  Metric           │ Nationwide │ King County │ Difference")
    print(f"  ──────────────────┼────────────┼─────────────┼──────────────")
    print(
        f"  R² Score         │   {metrics['test_r2']:>7.4f}   │  0.9266     │  {metrics['test_r2']-0.9266:>+7.4f}")
    print(
        f"  MAE              │ ${metrics['test_mae']:>10,.0f} │ $16,573     │  {metrics['test_mae']-16573:>+10,.0f}")
    print(
        f"  RMSE             │ ${metrics['test_rmse']:>10,.0f} │ $20,904     │  {metrics['test_rmse']-20904:>+10,.0f}")
    print()


if __name__ == "__main__":
    main()
