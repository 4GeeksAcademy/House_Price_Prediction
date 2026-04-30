"""
Train model using prepared splits from the training pipeline.
Loads train.jsonl, val.jsonl, and test.jsonl from data/processed/training_pipeline/splits/
"""
import warnings
from pathlib import Path
import pickle

import numpy as np
import joblib
from lightgbm import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
import pandas as pd

from house_price_prediction.config import load_settings
from house_price_prediction.features import build_preprocessor


def load_splits(splits_dir: Path = Path("data/processed/training_pipeline/splits")):
    """Load train, validation, and test splits from JSONL files."""
    train_path = splits_dir / "train.jsonl"
    val_path = splits_dir / "val.jsonl"
    test_path = splits_dir / "test.jsonl"

    if not all(p.exists() for p in [train_path, val_path, test_path]):
        raise FileNotFoundError(
            f"Training splits not found in {splits_dir}. "
            "Run scripts/build_training_pipeline.py first."
        )

    # Load JSONL files
    train_df = pd.read_json(train_path, lines=True)
    val_df = pd.read_json(val_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)

    return train_df, val_df, test_df


def prepare_data(train_df, val_df, test_df, target_column: str = "SalePrice"):
    """Prepare features and targets from loaded splits."""
    # Extract target and features
    x_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    x_val = val_df.drop(columns=[target_column])
    y_val = val_df[target_column]

    x_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    return x_train, y_train, x_val, y_val, x_test, y_test


GROUND_TRUTH_CSV = Path("data/raw/ground_truth_prices.csv")
GROUND_TRUTH_REPEATS = 8  # oversample real labels to increase their influence


def load_ground_truth(train_df: pd.DataFrame) -> pd.DataFrame | None:
    """Load verified Zillow ground truth rows and align columns with training splits."""
    if not GROUND_TRUTH_CSV.exists():
        return None
    gt = pd.read_csv(GROUND_TRUTH_CSV, dtype=str)  # read all as str first
    # Align columns: keep only columns present in training splits
    common_cols = [c for c in train_df.columns if c in gt.columns]
    if "SalePrice" not in common_cols:
        return None
    gt = gt[common_cols].copy()
    # Cast columns to match the dtype of the corresponding training split column
    for col in gt.columns:
        if col not in train_df.columns:
            continue
        target_dtype = train_df[col].dtype
        try:
            if target_dtype in ["float64", "float32"]:
                gt[col] = pd.to_numeric(
                    gt[col], errors="coerce").astype("float64")
            elif target_dtype in ["int64", "int32"]:
                gt[col] = pd.to_numeric(
                    gt[col], errors="coerce").astype("float64")
            else:
                # Keep as string to match object/str dtype from pipeline
                gt[col] = gt[col].astype(str)
        except Exception:
            pass
    return gt


def train_and_save_model_from_splits(settings) -> dict[str, float]:
    """Train model using prepared pipeline splits augmented with ground truth."""
    print("[train] Loading pipeline splits...")
    train_df, val_df, test_df = load_splits()

    # Augment training set with verified real-market labels
    gt = load_ground_truth(train_df)
    if gt is not None and not gt.empty:
        gt_repeated = pd.concat([gt] * GROUND_TRUTH_REPEATS, ignore_index=True)
        train_df = pd.concat([train_df, gt_repeated], ignore_index=True).sample(
            frac=1, random_state=42
        ).reset_index(drop=True)
        print(
            f"[train] Ground truth: +{len(gt)} rows x{GROUND_TRUTH_REPEATS} repeats → train now {len(train_df)} rows")
    else:
        print("[train] No ground truth CSV found — training on pipeline splits only")

    print(
        f"[train] Loaded: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"[train] Features: {list(train_df.columns)}")

    # Prepare data
    x_train, y_train, x_val, y_val, x_test, y_test = prepare_data(
        train_df, val_df, test_df, target_column="SalePrice"
    )

    print(
        f"[train] Building preprocessor on {len(x_train)} training samples...")
    preprocessor = build_preprocessor(x_train)

    print("[train] Creating LGBMRegressor model...")
    regressor = LGBMRegressor(
        objective="huber",       # less bias toward under-prediction than 'fair'
        alpha=0.9,               # huber quantile — controls outlier tolerance
        n_estimators=1200,
        learning_rate=0.018,
        num_leaves=95,
        min_child_samples=6,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.01,
        reg_lambda=6.0,
        n_jobs=-1,
        random_state=settings.random_state,
        verbose=-1,
    )

    transformed_regressor = TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", transformed_regressor),
        ]
    )

    print("[train] Training model on training set...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_train, y_train)

        print("[train] Evaluating on validation set...")
        val_predictions = model.predict(x_val)

        print("[train] Evaluating on test set...")
        test_predictions = model.predict(x_test)

    # Compute metrics on training set (to detect overfitting)
    print("[train] Computing metrics on training set...")
    train_predictions = model.predict(x_train)
    train_non_zero_mask = y_train != 0
    train_mape = float(
        mean_absolute_percentage_error(
            y_train[train_non_zero_mask], train_predictions[train_non_zero_mask]
        ) * 100
    ) if train_non_zero_mask.any() else float("nan")
    train_metrics = {
        "train_mae": float(mean_absolute_error(y_train, train_predictions)),
        "train_mape": train_mape,
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, train_predictions))),
        "train_r2": float(r2_score(y_train, train_predictions)),
    }

    # Compute metrics on test set
    print("[train] Computing metrics on test set...")
    non_zero_mask = y_test != 0
    if non_zero_mask.any():
        mape = float(
            mean_absolute_percentage_error(
                y_test[non_zero_mask], test_predictions[non_zero_mask]
            )
            * 100
        )
    else:
        mape = float("nan")

    metrics = {
        "mae": float(mean_absolute_error(y_test, test_predictions)),
        "mape": mape,
        "rmse": float(np.sqrt(mean_squared_error(y_test, test_predictions))),
        "r2": float(r2_score(y_test, test_predictions)),
    }

    # Also compute validation metrics for monitoring
    val_non_zero_mask = y_val != 0
    if val_non_zero_mask.any():
        val_mape = float(
            mean_absolute_percentage_error(
                y_val[val_non_zero_mask], val_predictions[val_non_zero_mask]
            )
            * 100
        )
    else:
        val_mape = float("nan")

    val_metrics = {
        "val_mae": float(mean_absolute_error(y_val, val_predictions)),
        "val_mape": val_mape,
        "val_rmse": float(np.sqrt(mean_squared_error(y_val, val_predictions))),
        "val_r2": float(r2_score(y_val, val_predictions)),
    }

    metrics.update(val_metrics)
    metrics.update(train_metrics)

    # Save model
    print(f"[train] Saving model to {settings.model_path}...")
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.model_path, 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    return metrics


if __name__ == "__main__":
    settings = load_settings()
    metrics = train_and_save_model_from_splits(settings)

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Model saved to: {settings.model_path}")

    def _fmt(name: str, value: float) -> str:
        if "mape" in name:
            return f"  {name:.<40} {value:>10.2f}%"
        if "mae" in name:
            return f"  {name:.<40} ${value:>10,.2f}"
        return f"  {name:.<40} {value:>10.4f}"

    groups = [
        ("Train", ["train_r2", "train_mae", "train_mape", "train_rmse"]),
        ("Val",   ["val_r2",   "val_mae",   "val_mape",   "val_rmse"]),
        ("Test",  ["r2",       "mae",       "mape",       "rmse"]),
    ]
    for label, keys in groups:
        print(f"\n  --- {label} ---")
        for k in keys:
            if k in metrics:
                print(_fmt(k, metrics[k]))

    # Overfitting diagnosis
    train_mape = metrics.get("train_mape", 0)
    test_mape  = metrics.get("mape", 0)
    gap = test_mape - train_mape
    print("\n  --- Overfitting Diagnosis ---")
    print(f"  train_mape → test_mape gap:         {gap:+.2f}%")
    if gap < 1.0:
        print("  Status: LOW RISK — train/test gap < 1 pp")
    elif gap < 3.0:
        print("  Status: MODERATE — consider more regularization")
    else:
        print("  Status: HIGH — model is overfitting; reduce complexity or add data")
    print("="*60)
