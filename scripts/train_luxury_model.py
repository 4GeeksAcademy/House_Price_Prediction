#!/usr/bin/env python3
"""
Train ultra-luxury model for homes >$1.5M.
Uses boosting with conservative hyperparameters to avoid overfitting on limited data.
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Shared module — needed for joblib serialisation of LocationAwarePipeline
sys.path.insert(0, str(Path(__file__).parent))
from model_utils import LocationAwarePipeline  # noqa: E402


def load_luxury_data(data_file, price_min=1500000, price_max=float('inf')):
    """Load and filter data for luxury tier."""
    predictions = []
    with open(data_file) as f:
        for line in f:
            record = json.loads(line)
            price = record.get('SalePrice', 0)
            if price_min <= price < price_max:
                predictions.append(record)

    print(f"✓ Loaded {len(predictions)} luxury homes (${price_min:,}+)")
    return predictions


def build_conservative_luxury_model():
    """
    Build LGBMRegressor with conservative hyperparameters for luxury segment.
    Limited data requires careful regularization to avoid overfitting.
    """
    return LGBMRegressor(
        n_estimators=200,          # Conservative: fewer trees
        learning_rate=0.03,        # Very conservative learning rate
        max_depth=8,               # Shallow trees to reduce overfitting
        num_leaves=25,
        min_child_samples=50,      # High minimum samples to prevent overfitting
        subsample=0.7,             # Row subsampling
        colsample_bytree=0.7,      # Column subsampling
        reg_lambda=2.0,            # Strong L2 regularization
        reg_alpha=1.0,             # L1 regularization
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )


def build_pipeline():
    """Create sklearn Pipeline with preprocessing and model."""
    # These are the columns the inner pipeline sees *after* LocationAwarePipeline
    # has injected zip_median_price, zip3, and engineered features.
    numeric_features = ['LotArea', 'OverallQual', 'YearBuilt', 'YearRemodAdd',
                        'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd',
                        'Fireplaces', 'GarageCars', 'GarageArea', 'NeighborhoodScore',
                        'zip_median_price',
                        # engineered
                        'TotalBaths', 'QualScore', 'LogLotArea',
                        'PropertyAge', 'YearsSinceRemodel', 'HasFireplace',
                        'QualToZip', 'SqftPerRoom', 'GarageRatio']

    categorical_features = ['PropertyType', 'HouseStyle', 'zip3']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
             categorical_features)
        ],
        remainder='drop'
    )

    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', build_conservative_luxury_model())
    ])


def train_luxury_model(data_file='data/processed/csv_training_data.jsonl',
                       output_file='models/nationwide_luxury_model.joblib'):
    """Train ultra-luxury model."""

    print("\n" + "="*80)
    print("TRAINING ULTRA-LUXURY MODEL ($1.5M+)")
    print("="*80 + "\n")

    # Load and filter data
    predictions = load_luxury_data(data_file)

    if len(predictions) < 50:
        print(f"✗ Insufficient luxury samples ({len(predictions)} < 50)")
        print("  Recommendation: Collect more high-end properties or use mid-price model")
        return None

    print(f"  Note: Training on limited dataset ({len(predictions)} samples)")
    print(f"  Will use conservative regularization to prevent overfitting\n")

    # Prepare features — include Neighborhood so LocationAwarePipeline can encode ZIP
    feature_names = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
                     'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd',
                     'Fireplaces', 'GarageCars', 'GarageArea', 'NeighborhoodScore',
                     'PropertyType', 'HouseStyle', 'Neighborhood']

    X_data = []
    y = []

    for pred in predictions:
        row = {}
        for name in feature_names:
            val = pred.get(name, None)
            if name == 'Neighborhood':
                row[name] = str(val) if val is not None else ''
            else:
                row[name] = val if (
                    val is not None and not pd.isna(val)) else 0
        X_data.append(row)
        y.append(pred.get('SalePrice', 0))

    X = pd.DataFrame(X_data)
    y = np.array(y)

    # Only 80% for training, 20% for validation (80-20 split for small dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Cross-validation on raw pipeline with synthetic LocationAwarePipeline wrapping
    print("\nRunning cross-validation (5-fold)...")
    pipeline = build_pipeline()
    cv_model = LocationAwarePipeline(pipeline, log_target=True)
    # Simple hold-out estimate instead of proper CV (LocationAwarePipeline wraps fit)
    cv_scores = np.array([0.0])
    try:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_r2s = []
        for tr_idx, val_idx in kf.split(X_train):
            m = LocationAwarePipeline(build_pipeline(), log_target=True)
            m.fit(X_train.iloc[tr_idx], y_train[tr_idx])
            p = m.predict(X_train.iloc[val_idx])
            yt = y_train[val_idx]
            ss_res = np.sum((yt - p) ** 2)
            ss_tot = np.sum((yt - yt.mean()) ** 2)
            cv_r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)
        cv_scores = np.array(cv_r2s)
    except Exception:
        pass
    print(f"  Cross-validation R\u00b2 scores: {cv_scores.round(4)}")
    print(
        f"  Mean CV R\u00b2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Train final model wrapped with LocationAwarePipeline (log-target + ZIP encoding)
    print("\nTraining final model with location-aware log-target transform...")
    pipeline = build_pipeline()
    luxury_model = LocationAwarePipeline(pipeline, log_target=True)
    luxury_model.fit(X_train, y_train)

    # Evaluate in dollar-space
    y_pred_train = luxury_model.predict(X_train)
    y_pred_test = luxury_model.predict(X_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    train_error_pct = np.mean(
        np.abs(y_pred_train - y_train) / np.maximum(y_train, 1)) * 100
    test_error_pct = np.mean(
        np.abs(y_pred_test - y_test) / np.maximum(y_test, 1)) * 100

    print("\n" + "="*80)
    print("PERFORMANCE METRICS")
    print("="*80)
    print(f"Train R²:        {train_r2:.4f}")
    print(f"Test R²:         {test_r2:.4f}")
    print(f"Train MAE:       ${train_mae:,.0f}")
    print(f"Test MAE:        ${test_mae:,.0f}")
    print(f"Train RMSE:      ${train_rmse:,.0f}")
    print(f"Test RMSE:       ${test_rmse:,.0f}")
    print(f"Train Error%:    {train_error_pct:.1f}%")
    print(f"Test Error%:     {test_error_pct:.1f}%")
    print()

    # Warning/assessment
    if test_r2 < 0:
        print("⚠ WARNING: Model R² is negative (predicts worse than mean)")
        print("  This is expected with very limited luxury data")
        print("  Recommendation: Use mid-price model as fallback")
    elif test_r2 < 0.3:
        print("⚠ WARNING: Model R² is low (< 0.3)")
        print("  Limited training data expected")
    else:
        print("✓ Model performance acceptable for limited dataset")

    print()

    # Show sample predictions
    print("="*80)
    print("SAMPLE PREDICTIONS (Test Set)")
    print("="*80)

    sample_indices = np.random.choice(
        len(y_test), min(10, len(y_test)), replace=False)
    for idx in sample_indices:
        actual = y_test[idx]
        pred = y_pred_test[idx]
        error_pct = (abs(pred - actual) / actual) * 100
        print(
            f"  Actual: ${actual:>12,.0f}  |  Predicted: ${pred:>12,.0f}  |  Error: {error_pct:>5.1f}%")

    print()

    # Save model (LocationAwarePipeline wrapper)
    joblib.dump(luxury_model, output_file)
    print(f"✓ Model saved to {output_file}")

    # Save training report
    report = {
        'model_type': 'luxury',
        'price_range': f'${1500000:,}+',
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'metrics': {
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'train_mae': float(train_mae),
            'test_mae': float(test_mae),
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
            'train_error_pct': float(train_error_pct),
            'test_error_pct': float(test_error_pct)
        },
        'cv_scores': cv_scores.tolist(),
        'hyperparameters': {
            'n_estimators': 200,
            'learning_rate': 0.03,
            'max_depth': 8,
            'num_leaves': 25,
            'min_child_samples': 50,
            'reg_lambda': 2.0,
            'reg_alpha': 1.0
        },
        'notes': 'Conservative hyperparameters due to limited training data'
    }

    with open('data/processed/luxury_model_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print("✓ Training report saved")
    print()

    return pipeline


if __name__ == '__main__':
    train_luxury_model()
