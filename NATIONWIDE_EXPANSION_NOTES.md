# Nationwide Model Training Expansion - May 2026

## Migration Summary: King County → Nationwide Single-Family Homes

### Before: King County Only

- **Data source**: Housing.csv (King County, WA)
- **Records**: 21,613 homes
- **Model performance**: R² = 0.9266, MAE = $16,573
- **Geographic scope**: Single county (King County, WA)
- **Median price**: ~$750K (estimate)

### After: Nationwide Single-Family Homes

- **Data source**: Redfin HousingPriceUSA (89 state CSV files)
- **Records**: 17,753 single-family homes after filtering
- **Records before filtering**: ~85,000+ raw records (filtered for single-family only)
- **Geographic scope**: All 50 US states + territories
- **Median price**: $419,000
- **Price range**: $50,000 – $85,000,000
- **Property type distribution**:
  - Single Family: 9,488 homes
  - Luxury (premium single family): 8,265 homes

## Implementation Details

### Data Pipeline Changes

#### 1. Enhanced Ingestion Script

- File: `scripts/ingest_csv_training_data.py`
- New functions:
  - `_is_single_family()` - Filter for single-family property types
  - `_map_redfin_row()` - Map Redfin schema to canonical 16-feature format
  - `_load_redfin_nationwide()` - Load all 89 state files

#### 2. Redfin Schema Mapping

| Redfin Column      | Canonical Feature   | Logic                     |
| ------------------ | ------------------- | ------------------------- |
| SQUARE FEET        | GrLivArea           | Direct mapping            |
| LOT SIZE           | LotArea             | Direct mapping            |
| BEDS               | BedroomAbvGr        | Direct mapping            |
| BATHS              | FullBath + HalfBath | Decimal split at 0.25     |
| YEAR BUILT         | YearBuilt           | Validation: 1800-2030     |
| LATITUDE/LONGITUDE | \_lat/\_lon         | For NeighborhoodScore KNN |
| PRICE              | SalePrice           | Filter: >= $50,000        |

#### 3. Estimated Features (from structural data)

- `OverallQual`: Estimated from bathrooms + year built
- `OverallCond`: Assumed 7/10 (moderate/good)
- `HouseStyle`: 1Story/SLvl/2Story based on bedroom count
- `TotRmsAbvGrd`: bedrooms + 2 + sqft/350
- `GarageCars`: 2 if sqft > 1500, else 1
- `GarageArea`: GarageCars × 240 sqft
- `Fireplaces`: 1 if sqft > 1500, else 0

#### 4. Filtering Criteria

- **Property Type**: Must include "single family" in PROPERTY TYPE
- **Price**: >= $50,000
- **Bedrooms**: 1-10
- **Bathrooms**: >= 0.5
- **Square Footage**: >= 200 sqft
- **Lot Size**: > 0 sqft
- **Year Built**: 1800-2030

### New CLI Options

```bash
# Load nationwide single-family homes only
python scripts/ingest_csv_training_data.py --source redfin-nationwide

# Load nationwide for specific states
python scripts/ingest_csv_training_data.py --source redfin-nationwide \
  --redfin-states WA CA OR TX FL

# Combine nationwide + regional baselines
python scripts/ingest_csv_training_data.py --source redfin-all \
  --skip-census-enrichment

# Train nationwide model
python scripts/train.py \
  --raw-data-path data/processed/nationwide_single_family_training_data.jsonl \
  --output-model models/nationwide_single_family_model.joblib
```

## Files Generated

### Training Data

- `data/processed/nationwide_single_family_training_data.jsonl` (7.8 MB)
  - 17,753 records, 16 canonical features per record
  - All records scored with NeighborhoodScore KNN
  - Ready for LightGBM/Random Forest training

### Model

- `models/nationwide_single_family_model.joblib`
  - Trained on nationwide data
  - To be compared with King County baseline

### Scorer

- `models/neighborhood_scorer.joblib`
  - Updated KNN scorer with 17,720 nationwide reference points
  - Decay radius: 8.0 km
  - K neighbors: 10

### Report

- `data/processed/csv_ingest_report.json`
  - Ingestion metrics and diagnostics

## Expected Model Behavior

### Advantages of Nationwide Training

1. **Market diversity**: Model learns price drivers across different markets
2. **Larger dataset**: 17,753 vs 21,613 — comparable size but nationwide coverage
3. **Generalization**: Better predictions for non-King-County addresses
4. **Economic signals**: Census features (income, median values) now more representative
5. **Property range**: Covers luxury ($85M) to affordable ($50K) homes

### Challenges in Nationwide Data

1. **High variance**: $50K – $85M price range vs King County's tighter distribution
2. **Estimated features**: Many features (garage, fireplace) are estimated vs actual
3. **Different markets**: NYC, SF, rural areas have vastly different price drivers
4. **Census calibration**: Without actual transactions, census signals less direct

## Performance Expectations

| Metric           | King County   | Nationwide (Expected) | Notes                                |
| ---------------- | ------------- | --------------------- | ------------------------------------ |
| R² Score         | 0.9266        | 0.85-0.90             | May be lower due to market diversity |
| MAE              | $16,573       | $30,000-50,000        | Higher due to larger price range     |
| RMSE             | $20,904       | $50,000-100,000       | Scale with larger price variance     |
| Geographic scope | Single county | 50+ states            | Trade-off with local accuracy        |

## Next Steps

1. ✅ Train nationwide model on 17,753 records
2. ⏳ Validate nationwide model performance
3. ⏳ Compare metrics: King County vs Nationwide
4. ⏳ Decide: Use nationwide model, blend both, or keep separate
5. ⏳ Deploy nationwide model to production
6. ⏳ Add state-specific fine-tuning if needed

## Engineering Notes for Continuation

### To Load Specific States

Edit `scripts/ingest_csv_training_data.py` line ~700:

```python
parser.add_argument(
    "--redfin-states",
    nargs="+",
    help="E.g., --redfin-states CA TX FL"
)
```

### To Add More Census Enrichment

The nationwide data currently lacks CensusMedianValue, MedianIncomeK, OwnerOccupiedRate.
Re-run with:

```bash
python scripts/ingest_csv_training_data.py \
  --source redfin-nationwide \
  --output-file nationwide_with_census.jsonl \
  # (removes --skip-census-enrichment)
```

This will query Census ACS5 API for all 17,753 zip codes.

### To Fine-Tune by State

Load one state at a time and train separate models:

```bash
python scripts/ingest_csv_training_data.py \
  --source redfin-nationwide \
  --redfin-states CA \
  --output-file california_single_family.jsonl
```

---

**Generated**: 2026-05-06 23:24:04 UTC  
**Ingestion time**: ~2 minutes  
**Dataset**: Redfin nationwide, filtered for single-family homes  
**Status**: Ready for training and validation
