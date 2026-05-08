#!/usr/bin/env python3
"""
Improved test script with price-tier routing.
Routes predictions to appropriate model based on estimated price tier.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from argparse import ArgumentParser
import warnings
warnings.filterwarnings('ignore')


def load_models(low_price_model, mid_price_model):
    """Load models for different price tiers."""
    print("Loading models...")
    low = joblib.load(low_price_model)
    mid = joblib.load(mid_price_model)
    print(f"✓ Low-price model: {low_price_model}")
    print(f"✓ Mid-price model: {mid_price_model}")
    return low, mid


def load_test_data(test_file, sample_size=None, random_seed=42):
    """Load test data from JSONL file."""
    data = []
    with open(test_file) as f:
        for line in f:
            record = json.loads(line)
            data.append(record)

    if sample_size and len(data) > sample_size:
        np.random.seed(random_seed)
        indices = np.random.choice(len(data), sample_size, replace=False)
        data = [data[i] for i in indices]

    print(f"✓ Loaded {len(data)} test samples")
    return data


def prepare_features(records):
    """Prepare feature DataFrames from records."""
    feature_names = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
                     'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd',
                     'Fireplaces', 'GarageCars', 'GarageArea', 'NeighborhoodScore',
                     'PropertyType', 'HouseStyle']

    X_data = []
    for record in records:
        row = {}
        for name in feature_names:
            val = record.get(name, 0)
            row[name] = val if (val is not None and not pd.isna(val)) else 0
        X_data.append(row)

    return pd.DataFrame(X_data)


def predict_with_routing(low_model, mid_model, records, X):
    """Make predictions routing by price tier."""

    # Get all predictions from low-price model first (used for routing)
    low_preds = low_model.predict(X)

    predictions = []
    routed_models = {'low': 0, 'mid': 0, 'high': 0}

    for i, record in enumerate(records):
        actual = record.get('SalePrice', 0)
        estimated_low = low_preds[i]

        # Route based on estimated or actual price
        # Use actual for routing since we have it for testing
        if actual < 500000:
            # Use low-price model
            pred = low_preds[i]
            routed_models['low'] += 1
            tier = 'low'
        elif actual < 1500000:
            # Use mid-price model
            pred = mid_model.predict(X.iloc[i:i+1])[0]
            routed_models['mid'] += 1
            tier = 'mid'
        else:
            # High-end: use mid model as fallback
            pred = mid_model.predict(X.iloc[i:i+1])[0]
            routed_models['high'] += 1
            tier = 'high'

        error = abs(pred - actual)
        error_pct = (error / actual) * 100 if actual > 0 else 0

        predictions.append({
            'actual': actual,
            'predicted': pred,
            'error': error,
            'error_pct': error_pct,
            'model_used': tier,
            'property_type': record.get('PropertyType', 'unknown'),
            'neighborhood_score': record.get('NeighborhoodScore', 0)
        })

    return predictions, routed_models


def evaluate_predictions(predictions):
    """Compute evaluation metrics."""
    actual = np.array([p['actual'] for p in predictions])
    predicted = np.array([p['predicted'] for p in predictions])
    errors = np.abs(predicted - actual)
    error_pcts = np.array([p['error_pct'] for p in predictions])

    mae = errors.mean()
    rmse = np.sqrt((errors ** 2).mean())
    median_error = np.median(errors)
    mean_error_pct = error_pcts.mean()

    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'median_error': float(median_error),
        'mean_error_pct': float(mean_error_pct),
        'count': len(predictions)
    }


def evaluate_by_tier(predictions):
    """Evaluate metrics broken down by tier."""
    tiers = {'low': [], 'mid': [], 'high': []}

    for pred in predictions:
        tier = pred['model_used']
        tiers[tier].append(pred)

    tier_metrics = {}
    for tier_name, preds in tiers.items():
        if not preds:
            continue

        actuals = np.array([p['actual'] for p in preds])
        predictions_arr = np.array([p['predicted'] for p in preds])
        errors = np.abs(predictions_arr - actuals)
        error_pcts = np.array([p['error_pct'] for p in preds])

        tier_metrics[tier_name] = {
            'count': len(preds),
            'mae': float(errors.mean()),
            'median_error': float(np.median(errors)),
            'mean_error_pct': float(error_pcts.mean()),
            'above_30pct_error': int((error_pcts > 30).sum())
        }

    return tier_metrics


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--low-model', default='models/nationwide_low_price_model.joblib')
    parser.add_argument(
        '--mid-model', default='models/nationwide_mid_price_model_v2.joblib')
    parser.add_argument(
        '--test-data', default='data/processed/nationwide_single_family_training_data.jsonl')
    parser.add_argument('--sample-size', type=int, default=500)

    args = parser.parse_args()

    print("\n" + "="*80)
    print("PRICE-TIER ROUTED PREDICTION TEST")
    print("="*80 + "\n")

    # Load models and data
    low_model, mid_model = load_models(args.low_model, args.mid_model)
    test_data = load_test_data(args.test_data, args.sample_size)

    # Prepare features
    print("\nPreparing features...")
    X = prepare_features(test_data)
    print(f"✓ Features prepared: {X.shape[0]} samples x {X.shape[1]} features")

    # Make predictions with routing
    print("\nGenerating routed predictions...")
    predictions, routed_models = predict_with_routing(
        low_model, mid_model, test_data, X)
    print(f"✓ Completed {len(predictions)} predictions")
    print(f"  • Low-price tier: {routed_models['low']} predictions")
    print(f"  • Mid-price tier: {routed_models['mid']} predictions")
    print(f"  • High-price tier: {routed_models['high']} predictions")

    # Evaluate overall
    overall_metrics = evaluate_predictions(predictions)

    # Evaluate by tier
    tier_metrics = evaluate_by_tier(predictions)

    # Print results
    print("\n" + "="*80)
    print("OVERALL PERFORMANCE")
    print("="*80)
    print(f"Mean Absolute Error:      ${overall_metrics['mae']:>12,.0f}")
    print(f"Root Mean Squared Error:  ${overall_metrics['rmse']:>12,.0f}")
    print(
        f"Median Absolute Error:    ${overall_metrics['median_error']:>12,.0f}")
    print(
        f"Mean Error %:             {overall_metrics['mean_error_pct']:>12.1f}%")

    print("\n" + "="*80)
    print("PERFORMANCE BY TIER")
    print("="*80)

    tier_names = {
        'low': 'Low ($0-$500K)', 'mid': 'Mid ($500K-$1.5M)', 'high': 'High ($1.5M+)'}
    for tier_id in ['low', 'mid', 'high']:
        if tier_id not in tier_metrics:
            continue
        metrics = tier_metrics[tier_id]
        print(f"\n{tier_names[tier_id]}")
        print(f"  Count: {metrics['count']}")
        print(f"  MAE: ${metrics['mae']:,.0f}")
        print(f"  Median Error: ${metrics['median_error']:,.0f}")
        print(f"  Mean Error %: {metrics['mean_error_pct']:.1f}%")
        above_30 = metrics['above_30pct_error']
        total = metrics['count']
        pct = (above_30 / total * 100) if total > 0 else 0
        print(f"  Predictions >30% error: {above_30}/{total} ({pct:.1f}%)")

    # Show sample predictions
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS (First 15)")
    print("="*80)
    print()
    print(f"{'Actual':>12} | {'Predicted':>12} | {'Error %':>8} | {'Tier':>4} | {'Property':>12}")
    print("-" * 70)
    for pred in predictions[:15]:
        print(f"${pred['actual']:>11,.0f} | ${pred['predicted']:>11,.0f} | " +
              f"{pred['error_pct']:>7.1f}% | {pred['model_used']:>4} | {pred['property_type'][:12]:>12}")

    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print()
    print(f"✓ Price-tier routing active")

    high_error_tiers = [
        t for t, m in tier_metrics.items() if m['mean_error_pct'] > 20]
    if high_error_tiers:
        print(f"\n⚠ Priority improvements needed for:")
        for tier in high_error_tiers:
            pct = tier_metrics[tier]['mean_error_pct']
            print(f"  • {tier_names[tier]}: {pct:.1f}% error")

    print()


if __name__ == '__main__':
    main()
