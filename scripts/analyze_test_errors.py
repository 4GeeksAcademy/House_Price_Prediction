#!/usr/bin/env python3
"""
Analyze test errors by segment to identify improvement opportunities.
Groups predictions by price tier and calculates segment-specific metrics.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict


def analyze_errors_by_segment():
    """Analyze prediction errors grouped by price tier."""

    # Load test data
    test_file = Path(
        "data/processed/nationwide_single_family_training_data.jsonl")

    predictions = []
    with open(test_file) as f:
        for line in f:
            record = json.loads(line)
            predictions.append(record)

    # Sample similar to test
    np.random.seed(42)
    if len(predictions) > 500:
        sample_indices = np.random.choice(len(predictions), 500, replace=False)
        predictions = [predictions[i] for i in sample_indices]

    # Load model and make predictions
    import joblib
    from lightgbm import LGBMRegressor
    import warnings
    warnings.filterwarnings('ignore')

    model = joblib.load("models/nationwide_low_price_model.joblib")

    # Prepare data
    feature_names = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
                     'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd',
                     'Fireplaces', 'GarageCars', 'GarageArea', 'NeighborhoodScore',
                     'PropertyType', 'HouseStyle']

    X_data = []
    y = []
    data_for_analysis = []

    for pred in predictions:
        row = {}
        for name in feature_names:
            row[name] = pred.get(name, 0)
        X_data.append(row)
        actual = pred.get('SalePrice', 0)
        y.append(actual)
        data_for_analysis.append({
            'actual': actual,
            'neighborhood': pred.get('Neighborhood', 'Unknown'),
            'property_type': pred.get('PropertyType', 'Unknown')
        })

    # Convert to DataFrame
    X = pd.DataFrame(X_data)
    y = np.array(y)

    # Make predictions
    preds = model.predict(X)

    # Calculate errors
    errors = np.abs(preds - y)
    error_pcts = (errors / np.maximum(y, 1)) * 100

    # Group by price tiers
    segments = {
        'Low ($0-$300K)': (0, 300000),
        'Mid-Low ($300K-$500K)': (300000, 500000),
        'Mid-High ($500K-$750K)': (500000, 750000),
        'High ($750K-$1.5M)': (750000, 1500000),
        'Luxury ($1.5M+)': (1500000, float('inf'))
    }

    print("\n" + "="*80)
    print("ERROR ANALYSIS BY PRICE SEGMENT")
    print("="*80)
    print()

    segment_stats = {}
    for segment_name, (min_price, max_price) in segments.items():
        mask = (y >= min_price) & (y < max_price)
        count = mask.sum()

        if count == 0:
            print(f"{segment_name:25} | Count: 0 (no samples)")
            continue

        seg_errors = errors[mask]
        seg_errors_pct = error_pcts[mask]
        seg_actuals = y[mask]
        seg_preds = preds[mask]

        mae = seg_errors.mean()
        median_error = np.median(seg_errors)
        rmse = np.sqrt((seg_errors ** 2).mean())
        mean_error_pct = seg_errors_pct.mean()

        segment_stats[segment_name] = {
            'count': int(count),
            'mae': float(mae),
            'median_error': float(median_error),
            'rmse': float(rmse),
            'mean_error_pct': float(mean_error_pct),
            'min_actual': float(seg_actuals.min()),
            'max_actual': float(seg_actuals.max()),
        }

        print(f"{segment_name:25} | n={count:3d} | MAE=${mae:10,.0f} | " +
              f"Median=${median_error:10,.0f} | Error%={mean_error_pct:5.1f}%")

        # Show problematic predictions
        worst_mask = seg_errors_pct > 30
        if worst_mask.sum() > 0:
            worst_count = worst_mask.sum()
            print(
                f"  ⚠ {worst_count} predictions with >30% error (needs improvement)")

    print("\n" + "="*80)
    print("IMPROVEMENT STRATEGY")
    print("="*80)
    print()

    # Identify highest-opportunity segment
    high_error_segments = [seg for seg, stats in segment_stats.items()
                           if stats['mean_error_pct'] > 20]

    if high_error_segments:
        print(f"→ Priority segments for retraining (>20% error):")
        for seg in high_error_segments:
            pct = segment_stats[seg]['mean_error_pct']
            print(f"  • {seg}: {pct:.1f}% error")
        print()

    print("RECOMMENDED ACTIONS:")
    print(f"  1. Retrain mid-price model ($500K-$1.5M) with:")
    print(f"     • Additional hyperparameter tuning (learning_rate, max_depth)")
    print(f"     • Feature engineering for luxury segment")
    print(f"     • Consider creating ultra-luxury tier ($1.5M+)")
    print()
    print(f"  2. Create regional adjustments for high-error neighborhoods")
    print()
    print(f"  3. Improve feature estimation for luxury homes:")
    print(f"     • Pool/spa presence indicator")
    print(f"     • Lot size premium factor")
    print(f"     • Custom garage/driveway estimation")
    print()

    # Save segment stats
    with open('data/processed/segment_analysis.json', 'w') as f:
        json.dump(segment_stats, f, indent=2)

    print("✓ Segment analysis saved to data/processed/segment_analysis.json")


if __name__ == '__main__':
    analyze_errors_by_segment()
