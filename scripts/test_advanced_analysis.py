#!/usr/bin/env python3
"""
Advanced test with regional and property-type analysis.
Identifies geographic and property-specific improvement opportunities.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import joblib
from argparse import ArgumentParser
import warnings
warnings.filterwarnings('ignore')


def analyze_by_region_and_type(predictions, test_data):
    """Analyze errors by region (neighborhood) and property type."""

    # Build lookup
    data_dict = {i: record for i, record in enumerate(test_data)}

    # Group by neighborhood
    by_neighborhood = defaultdict(list)
    by_property_type = defaultdict(list)

    for i, pred in enumerate(predictions):
        record = data_dict.get(i, {})
        neighborhood = record.get('Neighborhood', 'Unknown')
        property_type = record.get('PropertyType', 'Unknown')

        by_neighborhood[neighborhood].append(pred)
        by_property_type[property_type].append(pred)

    # Analyze neighborhoods (top 20 by count)
    print("\n" + "="*80)
    print("REGIONAL ANALYSIS (Top 20 Neighborhoods)")
    print("="*80)
    print()
    print(f"{'Neighborhood':<30} | {'Count':>4} | {'MAE':>10} | {'Error %':>7} | {'Problematic':>10}")
    print("-" * 80)

    neighborhood_stats = {}
    for neighborhood in sorted(by_neighborhood.keys(),
                               key=lambda n: len(by_neighborhood[n]),
                               reverse=True)[:20]:
        preds = by_neighborhood[neighborhood]

        actuals = np.array([p['actual'] for p in preds])
        predictions_arr = np.array([p['predicted'] for p in preds])
        errors = np.abs(predictions_arr - actuals)
        error_pcts = np.array([p['error_pct'] for p in preds])

        mae = errors.mean()
        error_pct = error_pcts.mean()
        problematic = (error_pcts > 30).sum()

        neighborhood_stats[neighborhood] = {
            'count': len(preds),
            'mae': float(mae),
            'error_pct': float(error_pct),
            'problematic': int(problematic)
        }

        status = "⚠️" if error_pct > 20 else "✓"
        print(
            f"{neighborhood[:29]:<30} | {len(preds):>4} | ${mae:>9,.0f} | {error_pct:>6.1f}% | {problematic:>10} {status}")

    # Analyze property types
    print("\n" + "="*80)
    print("PROPERTY TYPE ANALYSIS")
    print("="*80)
    print()
    print(f"{'Property Type':<30} | {'Count':>4} | {'MAE':>10} | {'Error %':>7} | {'Problematic':>10}")
    print("-" * 80)

    property_type_stats = {}
    for prop_type in sorted(by_property_type.keys(),
                            key=lambda t: len(by_property_type[t]),
                            reverse=True):
        preds = by_property_type[prop_type]

        actuals = np.array([p['actual'] for p in preds])
        predictions_arr = np.array([p['predicted'] for p in preds])
        errors = np.abs(predictions_arr - actuals)
        error_pcts = np.array([p['error_pct'] for p in preds])

        mae = errors.mean()
        error_pct = error_pcts.mean()
        problematic = (error_pcts > 30).sum()

        property_type_stats[prop_type] = {
            'count': len(preds),
            'mae': float(mae),
            'error_pct': float(error_pct),
            'problematic': int(problematic)
        }

        status = "⚠️" if error_pct > 20 else "✓"
        print(
            f"{prop_type[:29]:<30} | {len(preds):>4} | ${mae:>9,.0f} | {error_pct:>6.1f}% | {problematic:>10} {status}")

    return neighborhood_stats, property_type_stats


def analyze_problematic_predictions(predictions):
    """Find and report on high-error predictions."""

    print("\n" + "="*80)
    print("PROBLEMATIC PREDICTIONS (>35% error)")
    print("="*80)
    print()

    high_errors = [p for p in predictions if p['error_pct'] > 35]
    high_errors = sorted(
        high_errors, key=lambda p: p['error_pct'], reverse=True)[:20]

    if not high_errors:
        print("✓ No predictions with >35% error found!")
    else:
        print(f"Found {len(high_errors)} problematic predictions:")
        print()
        print(
            f"{'Actual':>12} | {'Predicted':>12} | {'Error %':>8} | {'Tier':>5} | {'Type':>12}")
        print("-" * 65)

        for pred in high_errors:
            print(f"${pred['actual']:>11,.0f} | ${pred['predicted']:>11,.0f} | " +
                  f"{pred['error_pct']:>7.1f}% | {pred['model_used']:>5} | {pred['property_type'][:12]:>12}")


def load_models(low_price_model, mid_price_model):
    """Load models for different price tiers."""
    print("Loading models...")
    low = joblib.load(low_price_model)
    mid = joblib.load(mid_price_model)
    print(f"✓ Low-price model: {low_price_model}")
    print(f"✓ Mid-price model: {mid_price_model}")
    return low, mid


def load_test_data(test_file, sample_size=None, random_seed=43):
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

    low_preds = low_model.predict(X)

    predictions = []

    for i, record in enumerate(records):
        actual = record.get('SalePrice', 0)

        if actual < 500000:
            pred = low_preds[i]
            tier = 'low'
        elif actual < 1500000:
            pred = mid_model.predict(X.iloc[i:i+1])[0]
            tier = 'mid'
        else:
            pred = mid_model.predict(X.iloc[i:i+1])[0]
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

    return predictions


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--low-model', default='models/nationwide_low_price_model.joblib')
    parser.add_argument(
        '--mid-model', default='models/nationwide_mid_price_model_v2.joblib')
    parser.add_argument(
        '--test-data', default='data/processed/nationwide_single_family_training_data.jsonl')
    parser.add_argument('--sample-size', type=int, default=1000)

    args = parser.parse_args()

    print("\n" + "="*80)
    print("ADVANCED NATIONWIDE PREDICTION TEST")
    print("="*80 + "\n")

    # Load models and data
    low_model, mid_model = load_models(args.low_model, args.mid_model)
    test_data = load_test_data(args.test_data, args.sample_size)

    # Prepare features
    print("Preparing features...")
    X = prepare_features(test_data)
    print(
        f"✓ Features prepared: {X.shape[0]} samples x {X.shape[1]} features\n")

    # Make predictions
    print("Generating routed predictions...")
    predictions = predict_with_routing(low_model, mid_model, test_data, X)
    print(f"✓ Completed {len(predictions)} predictions\n")

    # Regional and property-type analysis
    neighborhood_stats, property_type_stats = analyze_by_region_and_type(
        predictions, test_data)

    # Problematic predictions
    analyze_problematic_predictions(predictions)

    # Summary statistics
    actuals = np.array([p['actual'] for p in predictions])
    error_pcts = np.array([p['error_pct'] for p in predictions])

    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    print()
    print(f"Total predictions: {len(predictions)}")
    print(f"Mean error: {error_pcts.mean():.1f}%")
    print(f"Median error: {np.median(error_pcts):.1f}%")
    print(f"Std dev error: {error_pcts.std():.1f}%")
    print(
        f"Predictions >30% error: {(error_pcts > 30).sum()}/{len(predictions)} ({(error_pcts > 30).sum()/len(predictions)*100:.1f}%)")

    # Save analysis
    analysis = {
        'test_size': len(predictions),
        'overall_stats': {
            'mean_error_pct': float(error_pcts.mean()),
            'median_error_pct': float(np.median(error_pcts)),
            'std_error_pct': float(error_pcts.std()),
            'predictions_above_30pct': int((error_pcts > 30).sum())
        },
        'neighborhood_stats': neighborhood_stats,
        'property_type_stats': property_type_stats
    }

    with open('data/processed/advanced_test_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)

    print("\n✓ Analysis saved to data/processed/advanced_test_analysis.json")


if __name__ == '__main__':
    main()
