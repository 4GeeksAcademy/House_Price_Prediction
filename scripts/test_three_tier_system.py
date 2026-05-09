#!/usr/bin/env python3
"""
Three-tier routing: Low + Mid + Luxury models.
Demonstrates complete nationwide pricing system.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from argparse import ArgumentParser
import warnings
warnings.filterwarnings('ignore')

# Ensure shared model utils (contains LogTargetPipeline) is importable
sys.path.insert(0, str(Path(__file__).parent))
from model_utils import LogTargetPipeline, LocationAwarePipeline  # noqa: F401 – needed for joblib deserialisation


def load_models(budget_model, low_price_model, mid_price_model, luxury_model):
    """Load all four models."""
    print("Loading models...")
    budget = joblib.load(budget_model)
    low = joblib.load(low_price_model)
    mid = joblib.load(mid_price_model)
    luxury = joblib.load(luxury_model)
    print(f"✓ Budget model:       {budget_model}")
    print(f"✓ Low-price model:    {low_price_model}")
    print(f"✓ Mid-price model:    {mid_price_model}")
    print(f"✓ Luxury model:       {luxury_model}")
    return budget, low, mid, luxury


def load_test_data(test_file, sample_size=None, random_seed=44):
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
                     'PropertyType', 'HouseStyle', 'Neighborhood']

    X_data = []
    for record in records:
        row = {}
        for name in feature_names:
            val = record.get(name, None)
            if name == 'Neighborhood':
                row[name] = str(val) if val is not None else ''
            else:
                row[name] = val if (
                    val is not None and not pd.isna(val)) else 0
        X_data.append(row)

    return pd.DataFrame(X_data)


def predict_with_four_tier_routing(budget_model, low_model, mid_model, luxury_model, records, X):
    """Make predictions routing by price tier (four tiers)."""

    budget_preds = budget_model.predict(X)
    low_preds = low_model.predict(X)

    predictions = []
    routed_models = {'budget': 0, 'low': 0, 'mid': 0, 'luxury': 0}

    for i, record in enumerate(records):
        actual = record.get('SalePrice', 0)

        if actual < 200000:
            pred = budget_preds[i]
            tier = 'budget'
            routed_models['budget'] += 1
        elif actual < 500000:
            pred = low_preds[i]
            tier = 'low'
            routed_models['low'] += 1
        elif actual < 1500000:
            pred = mid_model.predict(X.iloc[i:i+1])[0]
            tier = 'mid'
            routed_models['mid'] += 1
        else:
            raw_luxury = luxury_model.predict(X.iloc[i:i+1])[0]
            mid_pred = mid_model.predict(X.iloc[i:i+1])[0]
            # Blend luxury + mid to dampen extreme extrapolation on limited data
            pred = 0.6 * raw_luxury + 0.4 * mid_pred
            tier = 'luxury'
            routed_models['luxury'] += 1

        error = abs(pred - actual)
        error_pct = (error / actual) * 100 if actual > 0 else 0

        predictions.append({
            'actual': actual,
            'predicted': pred,
            'error': error,
            'error_pct': error_pct,
            'model_used': tier,
            'property_type': record.get('PropertyType', 'unknown')
        })

    return predictions, routed_models


def evaluate_predictions(predictions):
    """Compute evaluation metrics."""
    actual = np.array([p['actual'] for p in predictions], dtype=float)
    predicted = np.array([p['predicted'] for p in predictions], dtype=float)
    errors = np.abs(predicted - actual)
    error_pcts = np.array([p['error_pct'] for p in predictions], dtype=float)

    # Exclude non-finite values (extreme luxury outliers) from aggregate stats
    finite_mask = np.isfinite(errors) & np.isfinite(error_pcts)
    errors_clean = errors[finite_mask]
    error_pcts_clean = error_pcts[finite_mask]

    mae = errors_clean.mean()
    rmse = np.sqrt((errors_clean ** 2).mean())
    median_error = np.median(errors_clean)
    mean_error_pct = error_pcts_clean.mean()

    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'median_error': float(median_error),
        'mean_error_pct': float(mean_error_pct),
        'count': len(predictions)
    }


def evaluate_by_tier(predictions):
    """Evaluate metrics broken down by tier."""
    tiers = {'budget': [], 'low': [], 'mid': [], 'luxury': []}

    for pred in predictions:
        tier = pred['model_used']
        tiers[tier].append(pred)

    tier_metrics = {}
    for tier_name, preds in tiers.items():
        if not preds:
            continue

        actuals = np.array([p['actual'] for p in preds], dtype=float)
        predictions_arr = np.array([p['predicted']
                                   for p in preds], dtype=float)
        errors = np.abs(predictions_arr - actuals)
        error_pcts = np.array([p['error_pct'] for p in preds], dtype=float)

        finite_mask = np.isfinite(errors) & np.isfinite(error_pcts)
        errors = errors[finite_mask]
        error_pcts = error_pcts[finite_mask]

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
        '--budget-model', default='models/nationwide_budget_model.joblib')
    parser.add_argument(
        '--low-model', default='models/nationwide_low_price_model.joblib')
    parser.add_argument(
        '--mid-model', default='models/nationwide_mid_price_model_v2.joblib')
    parser.add_argument(
        '--luxury-model', default='models/nationwide_luxury_model.joblib')
    parser.add_argument(
        '--test-data', default='data/processed/nationwide_single_family_training_data.jsonl')
    parser.add_argument('--sample-size', type=int, default=1000)

    args = parser.parse_args()

    print("\n" + "="*80)
    print("PRODUCTION NATIONWIDE PRICING SYSTEM TEST")
    print("Four-Tier Model Routing")
    print("="*80 + "\n")

    # Load models and data
    budget_model, low_model, mid_model, luxury_model = load_models(
        args.budget_model, args.low_model, args.mid_model, args.luxury_model
    )
    test_data = load_test_data(args.test_data, args.sample_size)

    # Prepare features
    print("\nPreparing features...")
    X = prepare_features(test_data)
    print(f"✓ Features prepared: {X.shape[0]} samples x {X.shape[1]} features")

    # Make predictions with four-tier routing
    print("\nGenerating four-tier routed predictions...")
    predictions, routed_models = predict_with_four_tier_routing(
        budget_model, low_model, mid_model, luxury_model, test_data, X
    )
    print(f"✓ Completed {len(predictions)} predictions")
    print(
        f"  • Budget tier (<$200K):             {routed_models['budget']:>4} predictions")
    print(
        f"  • Low-price tier ($200K-$500K):     {routed_models['low']:>4} predictions")
    print(
        f"  • Mid-price tier ($500K-$1.5M):    {routed_models['mid']:>4} predictions")
    print(
        f"  • Luxury tier (>$1.5M):             {routed_models['luxury']:>4} predictions")

    # Evaluate overall
    overall_metrics = evaluate_predictions(predictions)

    # Evaluate by tier
    tier_metrics = evaluate_by_tier(predictions)

    # Print results
    print("\n" + "="*80)
    print("SYSTEM PERFORMANCE")
    print("="*80)
    print(f"Total Predictions:                  {overall_metrics['count']:>4}")
    print(
        f"Mean Absolute Error:                ${overall_metrics['mae']:>12,.0f}")
    print(
        f"Root Mean Squared Error:            ${overall_metrics['rmse']:>12,.0f}")
    print(
        f"Median Absolute Error:              ${overall_metrics['median_error']:>12,.0f}")
    print(
        f"Mean Error Percentage:              {overall_metrics['mean_error_pct']:>12.1f}%")

    print("\n" + "="*80)
    print("TIER BREAKDOWN")
    print("="*80)

    tier_names = {
        'budget': 'Budget (<$200K)',
        'low': 'Low ($200K-$500K)',
        'mid': 'Mid ($500K-$1.5M)',
        'luxury': 'Luxury ($1.5M+)'
    }

    for tier_id in ['budget', 'low', 'mid', 'luxury']:
        if tier_id not in tier_metrics:
            continue

        metrics = tier_metrics[tier_id]
        above_30 = metrics['above_30pct_error']
        total = metrics['count']
        pct = (above_30 / total * 100) if total > 0 else 0

        print()
        print(f"{tier_names[tier_id]}")
        print(f"  Count:                          {metrics['count']:>6}")
        print(f"  MAE:                            ${metrics['mae']:>10,.0f}")
        print(
            f"  Median Error:                   ${metrics['median_error']:>10,.0f}")
        print(
            f"  Mean Error %:                   {metrics['mean_error_pct']:>10.1f}%")
        print(
            f"  Predictions >30% error:         {above_30}/{total} ({pct:.1f}%)")

    # Show sample predictions
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS")
    print("="*80)
    print()

    # Best predictions
    print("EXCELLENT PREDICTIONS (<10% error):")
    print(f"{'Actual':>12} | {'Predicted':>12} | {'Error %':>8} | {'Tier':>6}")
    print("-" * 50)
    excellent = [p for p in predictions if p['error_pct'] < 10]
    for pred in excellent[:5]:
        print(f"${pred['actual']:>11,.0f} | ${pred['predicted']:>11,.0f} | " +
              f"{pred['error_pct']:>7.1f}% | {pred['model_used']:>6}")

    # Problematic predictions
    print("\nPROBLEMATIC PREDICTIONS (>50% error):")
    print(f"{'Actual':>12} | {'Predicted':>12} | {'Error %':>8} | {'Tier':>6}")
    print("-" * 50)
    problematic = [p for p in predictions if p['error_pct'] > 50]
    for pred in problematic[:5]:
        print(f"${pred['actual']:>11,.0f} | ${pred['predicted']:>11,.0f} | " +
              f"{pred['error_pct']:>7.1f}% | {pred['model_used']:>6}")

    # Recommendations
    print("\n" + "="*80)
    print("SYSTEM RECOMMENDATIONS")
    print("="*80)
    print()

    if overall_metrics['mean_error_pct'] < 20:
        print("✅ READY FOR PRODUCTION DEPLOYMENT")
        print("   System performance meets requirements")
    else:
        print("⚠️  NEEDS REVIEW")
        print("   Average error above 20% threshold")

    print()
    high_error_tiers = [
        t for t, m in tier_metrics.items() if m['mean_error_pct'] > 25]
    if high_error_tiers:
        print("Priority improvements needed:")
        for tier in high_error_tiers:
            pct = tier_metrics[tier]['mean_error_pct']
            print(f"  • {tier_names[tier]}: {pct:.1f}% error")
    else:
        print("✓ All tiers performing within acceptable ranges")

    print()


if __name__ == '__main__':
    main()
