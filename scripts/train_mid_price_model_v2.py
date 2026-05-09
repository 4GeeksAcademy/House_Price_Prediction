#!/usr/bin/env python3
"""
Retrain mid-price model ($500K-$2M) with optimized hyperparameters.
Focuses on the segment with highest error rates and most improvement potential.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')


def prepare_mid_price_data(data_file, price_min=500000, price_max=2000000):
    """Load and filter data for mid-price tier."""
    predictions = []
    with open(data_file) as f:
        for line in f:
            record = json.loads(line)
            price = record.get('SalePrice', 0)
            if price_min <= price < price_max:
                predictions.append(record)

    print(
        f"✓ Loaded {len(predictions)} mid-price homes (${price_min:,} - ${price_max:,})")
    return predictions


def build_optimized_model():
    """Build LGBMRegressor with improved hyperparameters for mid-price segment."""
    return LGBMRegressor(
        n_estimators=400,          # Increased from 300
        learning_rate=0.05,        # Reduced from 0.1 for better generalization
        max_depth=13,              # Slightly deeper trees for complexity
        num_leaves=50,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,            # L2 regularization
        reg_alpha=0.5,             # L1 regularization
        random_state=42,
        n_jobs=-1,
        verbose=0
    )


def build_pipeline():
    """Create sklearn Pipeline with preprocessing and model."""
    feature_names = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
                     'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd',
                     'Fireplaces', 'GarageCars', 'GarageArea', 'NeighborhoodScore']

    categorical_features = ['PropertyType', 'HouseStyle']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', feature_names),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
             categorical_features)
        ],
        remainder='drop'
    )

    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', build_optimized_model())
    ])


def train_mid_price_model(data_file='data/processed/nationwide_single_family_training_data.jsonl',
                          output_file='models/nationwide_mid_price_model_v2.joblib'):
    """Train improved mid-price model."""

    print("\n" + "="*80)
    print("TRAINING IMPROVED MID-PRICE MODEL ($500K-$2M)")
    print("="*80 + "\n")

    # Load and filter data
    predictions = prepare_mid_price_data(data_file)

    if len(predictions) < 100:
        print("✗ Not enough mid-price samples for training")
        return

    # Prepare features
    feature_names = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
                     'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd',
                     'Fireplaces', 'GarageCars', 'GarageArea', 'NeighborhoodScore',
                     'PropertyType', 'HouseStyle']

    X_data = []
    y = []

    for pred in predictions:
        row = {}
        for name in feature_names:
            val = pred.get(name, 0)
            # Handle NaN/None
            row[name] = val if (val is not None and not pd.isna(val)) else 0
        X_data.append(row)
        y.append(pred.get('SalePrice', 0))

    X = pd.DataFrame(X_data)
    y = np.array(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print()

    # Build and train pipeline
    pipeline = build_pipeline()
    print("Training pipeline...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Error percentages
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

    # Check for overfitting
    overfitting = train_mae - test_mae
    if overfitting > 50000:
        print(
            f"⚠ WARNING: Possible overfitting detected (${overfitting:,.0f} gap)")
    else:
        print(
            f"✓ Model generalization: acceptable gap of ${abs(overfitting):,.0f}")

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

    # Save model
    joblib.dump(pipeline, output_file)
    print(f"✓ Model saved to {output_file}")

    # Save training report
    report = {
        'model_type': 'mid_price_v2',
        'price_range': f'${500000:,} - ${2000000:,}',
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
        'hyperparameters': {
            'n_estimators': 400,
            'learning_rate': 0.05,
            'max_depth': 13,
            'num_leaves': 50,
            'reg_lambda': 1.0,
            'reg_alpha': 0.5
        }
    }

    with open('data/processed/mid_price_model_v2_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print("✓ Training report saved")
    print()

    return pipeline


if __name__ == '__main__':
    train_mid_price_model()
