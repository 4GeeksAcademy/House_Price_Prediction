"""
ingest_csv_training_data.py

Maps raw CSV datasets into the canonical 20-feature JSONL format used by
scripts/train.py and scripts/build_training_pipeline.py.

Sources
-------
king-county   data/raw/Housing.csv   (21,613 rows, King County WA)
              Has: real sale prices, lat/lon, grade, condition, sqft, bedrooms,
                   bathrooms, floors, year built.
              Estimated: GarageCars, GarageArea, Fireplaces, TotRmsAbvGrd
              Census enrichment: CensusMedianValue, MedianIncomeK, OwnerOccupiedRate
                   fetched from Census ACS5 API by zipcode (ZCTA),
                   cached to data/processed/zcta_census_stats.json.

ames          data/raw/housing.csv   (500 rows, Ames Iowa)
              Already in the 20-feature schema format, no mapping needed.

NeighborhoodScore
-----------------
Fitted exclusively on King County (lat, lon, actual sale price).
Using actual transaction prices as the KNN signal is non-circular:
these are real market observations, not model predictions.
The fitted NeighborhoodScoreService is saved to --model-dir for use at
inference time.  Ames rows receive None (no lat/lon in that CSV).

Generalization beyond King County
----------------------------------
The model is price-calibrated on King County data.  It generalizes to other
US addresses because:
  - Structural features (squft → GrLivArea, grade → OverallQual, bedrooms,
    year built) are universal predictors whose price relationships generalize.
  - NeighborhoodScore degrades gracefully for faraway addresses: the Gaussian
    decay produces near-zero weights at distance >> decay_km, so the score
    approaches 50 (neutral).  The live Census features (CensusMedianValue,
    MedianIncomeK, OwnerOccupiedRate) then carry the market-calibration signal.
  - PropertyType is rule-based and works for any address.
To improve national price coverage: add more regional CSVs (Zillow Research,
ATTOM, HUD) with lat/lon + sale price and re-run this script.

King County column mapping
--------------------------
KC column        Schema column     Notes
sqft_lot         LotArea
grade (1-13)     OverallQual       linear remap to 1-10
condition (1-5)  OverallCond       linear remap to 1-9
yr_built         YearBuilt
yr_renovated     YearRemodAdd      0 → yr_built (never renovated)
sqft_living      GrLivArea
bathrooms        FullBath + HalfBath  decimal split
bedrooms         BedroomAbvGr
floors           HouseStyle        1.0→1Story, 1.5→SLvl, ≥2→2Story
zipcode          Neighborhood
grade            GarageCars        estimated: <5→0, 5-7→1, 8-10→2, 11+→3
grade            Fireplaces        estimated: <9→0, 9-10→1, 11+→2
bedrooms/sqft    TotRmsAbvGrd      estimated: bedrooms+2+round(sqft_above/350)
lat/long         NeighborhoodScore KNN LOO on actual prices
—                CensusMedianValue None — live Census API provides at inference
—                MedianIncomeK     None
—                OwnerOccupiedRate None
—                PropertyType      rule-based classifier

Outputs  (--output-dir, default data/processed/)
  csv_training_data.jsonl   Canonicalized rows for scripts/train.py
  csv_ingest_report.json    Row counts, quality flags, scorer diagnostics

Scorer output  (--model-dir, default models/)
  neighborhood_scorer.joblib   Fitted KNN scorer for inference time

Usage
-----
  python scripts/ingest_csv_training_data.py
  python scripts/ingest_csv_training_data.py --source king-county
  python scripts/ingest_csv_training_data.py --source ames
  python scripts/ingest_csv_training_data.py --source both \\
      --kc-csv data/raw/Housing.csv --ames-csv data/raw/housing.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# src/ package imports
from house_price_prediction.application.services.neighborhood_score_service import (
    NeighborhoodScoreService,
)
from house_price_prediction.infrastructure.providers.property_type_classifier import (
    classify_property_type,
)

# ---------------------------------------------------------------------------
# Census ZCTA enrichment constants and helpers
# ---------------------------------------------------------------------------

_CENSUS_ACS5_YEAR = "2022"
_CENSUS_SUPPRESSED = -666_666_666   # Census sentinel for suppressed/missing data
_CENSUS_BATCH_SIZE = 50             # max ZCTAs per Census API request


def _parse_census_int(value: Any) -> int | None:
    """Parse a Census API integer; return None for suppressed (-666666666) or invalid."""
    try:
        v = int(value)
        return None if v == _CENSUS_SUPPRESSED or v < 0 else v
    except (TypeError, ValueError):
        return None


def _fetch_zcta_census_stats(
    zipcodes: list[str],
    cache_path: Path | None = None,
    timeout: float = 45.0,
) -> dict[str, dict[str, float | None]]:
    """Fetch or load ACS5 stats for the given ZIP codes from the Census ZCTA endpoint.

    Returns {zipcode: {CensusMedianValue, MedianIncomeK, OwnerOccupiedRate}}.
    Results are cached to cache_path (JSON) to avoid repeated API calls.
    Returns an empty dict on any network failure so callers continue gracefully.

    ACS5 variables fetched:
      B25077_001E  Median home value (USD)
      B19013_001E  Median household income (USD)
      B25003_002E  Owner-occupied housing units
      B25003_001E  Total occupied housing units
    """
    if cache_path and cache_path.exists():
        print(f"  [census] Loading ZCTA stats from cache: {cache_path}")
        with cache_path.open("r", encoding="utf-8") as fh:
            cached: dict[str, dict[str, float | None]] = json.load(fh)
        return {z: cached[z] for z in zipcodes if z in cached}

    variables = "B25077_001E,B19013_001E,B25003_002E,B25003_001E"
    stats: dict[str, dict[str, float | None]] = {}
    unique_zips = sorted(set(zipcodes))
    batches = [
        unique_zips[i: i + _CENSUS_BATCH_SIZE]
        for i in range(0, len(unique_zips), _CENSUS_BATCH_SIZE)
    ]
    print(
        f"  [census] Querying ACS5 for {len(unique_zips)} unique ZCTAs "
        f"in {len(batches)} batch(es) ..."
    )

    for batch_n, batch in enumerate(batches, 1):
        zcta_param = ",".join(batch)
        url = (
            f"https://api.census.gov/data/{_CENSUS_ACS5_YEAR}/acs/acs5"
            f"?get={variables}&for=zip+code+tabulation+area:{zcta_param}"
        )
        try:
            req = urllib.request.Request(
                url, headers={
                    "User-Agent": "house-price-prediction-ingest/1.0"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data: list[list[str]] = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            print(
                f"  [census] WARNING: batch {batch_n}/{len(batches)} failed: {exc}")
            continue

        headers = data[0]
        zcta_col = headers.index("zip code tabulation area")
        for row in data[1:]:
            zipcode = str(row[zcta_col]).zfill(5)
            median_value = _parse_census_int(row[headers.index("B25077_001E")])
            median_income = _parse_census_int(
                row[headers.index("B19013_001E")])
            owner_units = _parse_census_int(row[headers.index("B25003_002E")])
            total_units = _parse_census_int(row[headers.index("B25003_001E")])

            owner_rate: float | None = None
            if owner_units is not None and total_units is not None and total_units > 0:
                owner_rate = round(owner_units / total_units, 4)

            stats[zipcode] = {
                "CensusMedianValue": float(median_value) if median_value is not None else None,
                "MedianIncomeK": round(median_income / 1000, 2) if median_income is not None else None,
                "OwnerOccupiedRate": owner_rate,
            }

    if stats and cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2)
        print(f"  [census] Cached {len(stats):,} ZCTA entries → {cache_path}")

    return stats


def _apply_census_enrichment(
    rows: list[dict[str, Any]],
    zcta_stats: dict[str, dict[str, float | None]],
) -> int:
    """Fill census features in-place for KC rows whose zipcode has ZCTA stats.

    Returns the count of rows that received at least one non-null census value.
    """
    enriched = 0
    for row in rows:
        if row.get("_source") != "king-county":
            continue
        zipcode = str(row.get("Neighborhood", "")).strip().zfill(5)
        census = zcta_stats.get(zipcode)
        if not census:
            continue
        changed = False
        for field in ("CensusMedianValue", "MedianIncomeK", "OwnerOccupiedRate"):
            if row.get(field) is None and census.get(field) is not None:
                row[field] = census[field]
                changed = True
        if changed:
            enriched += 1
    return enriched


# ---------------------------------------------------------------------------
# King County column mapping helpers
# ---------------------------------------------------------------------------

def _kc_grade_to_overall_qual(grade: int) -> int:
    """Map KC building grade 1-13 → OverallQual 1-10 (linear remap)."""
    return max(1, min(10, round(grade * 10 / 13)))


def _kc_condition_to_overall_cond(condition: int) -> int:
    """Map KC condition 1-5 → OverallCond 1-9 (linear remap: 1,3,5,7,9)."""
    return 1 + (int(condition) - 1) * 2


def _kc_bathrooms_split(bathrooms: float) -> tuple[int, int]:
    """Split KC decimal bathrooms → (FullBath, HalfBath).
    e.g. 2.5 → (2, 1),  2.0 → (2, 0),  1.75 → (1, 1)
    """
    full = int(bathrooms)
    half = 1 if (bathrooms - full) >= 0.25 else 0
    return full, half


def _kc_floors_to_housestyle(floors: float) -> str:
    if floors <= 1.0:
        return "1Story"
    if floors <= 1.5:
        return "SLvl"
    return "2Story"


def _kc_estimate_total_rooms(bedrooms: int, sqft_above: int) -> int:
    """Estimate TotRmsAbvGrd from bedrooms + above-grade sqft.
    Rule of thumb: ~350 sqft per room above grade.
    """
    estimated = int(bedrooms) + 2 + round(int(sqft_above) / 350)
    return max(4, min(14, estimated))


def _kc_estimate_garage_cars(grade: int) -> int:
    if grade < 5:
        return 0
    if grade <= 7:
        return 1
    if grade <= 10:
        return 2
    return 3


def _kc_estimate_fireplaces(grade: int) -> int:
    if grade < 9:
        return 0
    if grade < 11:
        return 1
    return 2


# ---------------------------------------------------------------------------
# Per-row mapping
# ---------------------------------------------------------------------------

def _map_kc_row(row: pd.Series, min_price: float, max_bedrooms: int) -> dict[str, Any] | None:
    """Map one King County CSV row to the training schema dict.
    Returns None to drop the row (outlier / bad data).
    Internal keys _lat and _lon are used for KNN scoring and stripped before output.
    """
    try:
        price = float(row["price"])
        bedrooms = int(row["bedrooms"])
        sqft_living = int(row["sqft_living"])
        sqft_lot = int(row["sqft_lot"])
        sqft_above = int(row.get("sqft_above", sqft_living))
        sqft_basement = int(row.get("sqft_basement", 0) or 0)
        grade = int(row["grade"])
        condition = int(row["condition"])
        bathrooms = float(row["bathrooms"])
        floors = float(row["floors"])
        yr_built = int(row["yr_built"])
        yr_renovated = int(row.get("yr_renovated", 0))
        waterfront = int(row.get("waterfront", 0) or 0)
        view = int(row.get("view", 0) or 0)
        zipcode = str(int(row["zipcode"]))
        lat = float(row["lat"])
        lon = float(row["long"])
    except (KeyError, TypeError, ValueError):
        return None

    # Outlier guards: prices below threshold, unrealistic bedroom counts, tiny homes
    if price < min_price:
        return None
    if bedrooms > max_bedrooms or bedrooms < 1:
        return None
    if sqft_living < 200:
        return None

    full_bath, half_bath = _kc_bathrooms_split(bathrooms)
    overall_qual = _kc_grade_to_overall_qual(grade)
    overall_cond = _kc_condition_to_overall_cond(condition)
    house_style = _kc_floors_to_housestyle(floors)
    tot_rms = _kc_estimate_total_rooms(bedrooms, sqft_above)
    garage_cars = _kc_estimate_garage_cars(grade)
    garage_area = garage_cars * 240
    fireplaces = _kc_estimate_fireplaces(grade)
    year_remod = yr_renovated if yr_renovated > 0 else yr_built

    # PropertyType classifier uses the already-mapped structural features.
    # CensusMedianValue / OwnerOccupiedRate are not in the KC CSV so we pass
    # None — the classifier degrades gracefully (defaults: owner_rate=0.7).
    property_type = classify_property_type({
        "OverallQual": overall_qual,
        "GrLivArea": sqft_living,
        "TotRmsAbvGrd": tot_rms,
        "BedroomAbvGr": bedrooms,
        "HouseStyle": house_style,
        "CensusMedianValue": None,
        "OwnerOccupiedRate": None,
    })

    return {
        "LotArea": sqft_lot,
        "OverallQual": overall_qual,
        "OverallCond": overall_cond,
        "YearBuilt": yr_built,
        "YearRemodAdd": year_remod,
        "GrLivArea": sqft_living,
        "BasementSF": sqft_basement,
        "FullBath": full_bath,
        "HalfBath": half_bath,
        "BedroomAbvGr": bedrooms,
        "TotRmsAbvGrd": tot_rms,
        "Fireplaces": fireplaces,
        "GarageCars": garage_cars,
        "GarageArea": garage_area,
        "Waterfront": waterfront,
        "ViewScore": view,
        "PropertyType": property_type,
        "HouseStyle": house_style,
        "NeighborhoodScore": None,   # filled in after KNN fitting
        "CensusMedianValue": None,   # absent from KC CSV; Census API at inference
        "MedianIncomeK": None,       # absent from KC CSV
        "OwnerOccupiedRate": None,   # absent from KC CSV
        "Neighborhood": zipcode,
        "SalePrice": price,
        "_lat": lat,
        "_lon": lon,
        "_source": "king-county",
    }


def _map_ames_row(row: pd.Series, min_price: float) -> dict[str, Any] | None:
    """Map one Ames CSV row (already in schema format) to the training dict."""
    try:
        price = float(row["SalePrice"])
        if price < min_price:
            return None

        property_type = classify_property_type({
            "OverallQual": row["OverallQual"],
            "GrLivArea": row["GrLivArea"],
            "TotRmsAbvGrd": row["TotRmsAbvGrd"],
            "BedroomAbvGr": row["BedroomAbvGr"],
            "HouseStyle": str(row["HouseStyle"]),
            "CensusMedianValue": None,
            "OwnerOccupiedRate": None,
        })

        return {
            "LotArea": int(row["LotArea"]),
            "OverallQual": int(row["OverallQual"]),
            "OverallCond": int(row["OverallCond"]),
            "YearBuilt": int(row["YearBuilt"]),
            "YearRemodAdd": int(row["YearRemodAdd"]),
            "GrLivArea": int(row["GrLivArea"]),
            "FullBath": int(row["FullBath"]),
            "HalfBath": int(row["HalfBath"]),
            "BedroomAbvGr": int(row["BedroomAbvGr"]),
            "TotRmsAbvGrd": int(row["TotRmsAbvGrd"]),
            "Fireplaces": int(row["Fireplaces"]),
            "GarageCars": int(row["GarageCars"]),
            "GarageArea": int(row["GarageArea"]),
            "PropertyType": property_type,
            "HouseStyle": str(row["HouseStyle"]),
            "NeighborhoodScore": None,       # filled in after national scorer pass
            "CensusMedianValue": None,
            "MedianIncomeK": None,
            "OwnerOccupiedRate": None,
            "Neighborhood": str(row["Neighborhood"]),
            "SalePrice": price,
            # Ames, Iowa centroid — centre of Story County.  All Ames training
            # rows share one city, so a single centroid gives a consistent and
            # realistic NeighborhoodScore from the national ZCTA scorer.
            "_lat": 42.0308,
            "_lon": -93.6319,
            "_source": "ames",
        }
    except (KeyError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# NeighborhoodScore: KNN LOO on KC actual prices (or pre-built national scorer)
# ---------------------------------------------------------------------------

def _fit_and_assign_neighborhood_scores(
    rows: list[dict[str, Any]],
    k: int,
    decay_km: float,
    scorer_path: Path,
    national_scorer_path: Path | None = None,
) -> NeighborhoodScoreService | None:
    """Assign NeighborhoodScore to every row that has lat/lon.

    Two modes:

    1. ``national_scorer_path`` is provided and the file exists:
       Load the pre-built national ZCTA scorer (built by
       ``seed_national_neighborhood_scorer.py``) and use ``.score()``
       directly for each training row.  The national scorer's reference
       points are ZCTA *centroids*, not individual training rows, so
       leave-one-out is not required.  The scorer file is NOT re-saved
       (it has already been built and placed at ``national_scorer_path``).

    2. No ``national_scorer_path`` (legacy / offline mode):
       Fit a fresh scorer from King County lat/lon + CensusMedianValue
       (falling back to SalePrice) and compute LOO scores to avoid label
       leakage.  Saves the KC-only scorer to ``scorer_path``.
    """
    # Rows that have geocoordinates — includes KC and any others
    valid_indices = [
        i for i, r in enumerate(rows)
        if (
            r.get("_lat") is not None
            and r.get("_lon") is not None
            and np.isfinite(r["_lat"])
            and np.isfinite(r["_lon"])
        )
    ]

    if not valid_indices:
        print(
            "  [scorer] No geocoded rows — NeighborhoodScore left as None for all rows.")
        return None

    # ------------------------------------------------------------------
    # Mode 1: pre-built national scorer
    # ------------------------------------------------------------------
    if national_scorer_path is not None and national_scorer_path.exists():
        print(
            f"  [scorer] Loading pre-built national scorer from {national_scorer_path} ...")
        svc = NeighborhoodScoreService.load(national_scorer_path)
        scored = 0
        for global_i in valid_indices:
            lat = rows[global_i]["_lat"]
            lon = rows[global_i]["_lon"]
            s = svc.score(lat, lon)
            rows[global_i]["NeighborhoodScore"] = round(s, 2)
            scored += 1
        print(
            f"  [scorer] Scored {scored:,} rows using national ZCTA reference "
            f"({svc.diagnostics().get('reference_point_count', '?'):,} ZCTA points, "
            f"decay_km={svc.decay_km:.0f})."
        )
        # Copy the national scorer to scorer_path so it is discoverable at inference
        if national_scorer_path.resolve() != scorer_path.resolve():
            import shutil
            scorer_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(national_scorer_path, scorer_path)
            print(f"  [scorer] Copied national scorer to {scorer_path}")
        return svc

    # ------------------------------------------------------------------
    # Mode 2: KC-only LOO scorer (legacy / offline mode)
    # ------------------------------------------------------------------
    kc_indices = [
        i for i in valid_indices
        if rows[i].get("SalePrice") is not None and rows[i]["SalePrice"] > 0
    ]
    if not kc_indices:
        print(
            "  [scorer] No geocoded rows with SalePrice — NeighborhoodScore left as None.")
        return None

    lats = [rows[i]["_lat"] for i in kc_indices]
    lons = [rows[i]["_lon"] for i in kc_indices]

    # Prefer CensusMedianValue as the KNN signal: it is a stable, non-circular
    # external Census signal that matches what NeighborhoodScoreService.fit()
    # expects (parameter is named census_median_values).  Fall back to SalePrice
    # for rows where ZCTA enrichment produced no census value.
    knn_signal = [
        rows[i]["CensusMedianValue"]
        if rows[i].get("CensusMedianValue") is not None
        else rows[i]["SalePrice"]
        for i in kc_indices
    ]
    census_signal_count = sum(
        1 for i in kc_indices if rows[i].get("CensusMedianValue") is not None
    )
    print(
        f"  [scorer] Fitting KNN scorer on {len(lats):,} King County rows ..."
        f" (census signal: {census_signal_count:,}, "
        f"sale-price fallback: {len(lats) - census_signal_count:,})"
    )
    svc = NeighborhoodScoreService(k=k, decay_km=decay_km)
    svc.fit(lats, lons, knn_signal)

    print("  [scorer] Computing leave-one-out scores (no label leakage) ...")
    loo_scores = svc.score_loo_batch()

    # loo_scores has same length as kc_indices because fit() received the
    # same rows in the same order (all are finite, no internal drops)
    for local_i, global_i in enumerate(kc_indices):
        rows[global_i]["NeighborhoodScore"] = round(loo_scores[local_i], 2)

    scorer_path.parent.mkdir(parents=True, exist_ok=True)
    svc.save(scorer_path)
    print(f"  [scorer] Saved to {scorer_path}")
    return svc


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _load_king_county(path: Path, min_price: float, max_bedrooms: int) -> list[dict[str, Any]]:
    print(f"  [kc] Reading {path} ...")
    df = pd.read_csv(path)
    print(f"  [kc] {len(df):,} raw rows")

    rows: list[dict[str, Any]] = []
    dropped = 0
    for _, row in df.iterrows():
        mapped = _map_kc_row(row, min_price=min_price,
                             max_bedrooms=max_bedrooms)
        if mapped is None:
            dropped += 1
        else:
            rows.append(mapped)

    print(f"  [kc] {len(rows):,} rows mapped  ({dropped} dropped as outliers)")
    return rows


def _load_ames(path: Path, min_price: float) -> list[dict[str, Any]]:
    print(f"  [ames] Reading {path} ...")
    df = pd.read_csv(path)
    print(f"  [ames] {len(df):,} raw rows")

    rows: list[dict[str, Any]] = []
    dropped = 0
    for _, row in df.iterrows():
        mapped = _map_ames_row(row, min_price=min_price)
        if mapped is None:
            dropped += 1
        else:
            rows.append(mapped)

    print(f"  [ames] {len(rows):,} rows mapped  ({dropped} dropped)")
    return rows


# ---------------------------------------------------------------------------
# Redfin nationwide single-family mapper
# ---------------------------------------------------------------------------

def _is_single_family(property_type: str) -> bool:
    """Filter Redfin PROPERTY TYPE to accept only single-family homes."""
    if property_type is None:
        return False
    pt = str(property_type).strip().lower()
    single_family_keywords = ["single family", "single-family", "house"]
    return any(keyword in pt for keyword in single_family_keywords)


def _map_redfin_row(
    row: pd.Series,
    state: str,
    min_price: float,
    max_bedrooms: int,
) -> dict[str, Any] | None:
    """Map one Redfin CSV row (nationwide HousingPriceUSA) to the training schema dict.

    Redfin schema:
      PROPERTY TYPE, PRICE, BEDS, BATHS, SQUARE FEET, LOT SIZE, YEAR BUILT,
      LATITUDE, LONGITUDE, STATE OR PROVINCE, ZIP OR POSTAL CODE, etc.
    """
    try:
        # Filter for single-family homes
        if not _is_single_family(row.get("PROPERTY TYPE")):
            return None

        price = float(row["PRICE"])
        if price < min_price:
            return None

        bedrooms = int(row.get("BEDS", 0))
        if bedrooms < 1 or bedrooms > max_bedrooms:
            return None

        bathrooms = float(row.get("BATHS", 0))
        if bathrooms < 0.5:
            return None

        sqft_living = int(row.get("SQUARE FEET", 0))
        if sqft_living < 200:
            return None

        sqft_lot = int(row.get("LOT SIZE", 0))
        if sqft_lot <= 0:
            return None

        year_built = int(row.get("YEAR BUILT", 1950))
        if year_built < 1800 or year_built > 2030:
            return None

        lat = float(row.get("LATITUDE"))
        lon = float(row.get("LONGITUDE"))

        zipcode = str(row.get("ZIP OR POSTAL CODE", "")).strip()
        if not zipcode:
            zipcode = "00000"

    except (KeyError, TypeError, ValueError):
        return None

    # Map Redfin fields to canonical schema
    # For Redfin, we have full data; use reasonable defaults for estimated fields
    full_bath = int(bathrooms)
    half_bath = 1 if (bathrooms - full_bath) >= 0.25 else 0

    # Estimate OverallQual (1-10) from property characteristics
    # Rule: larger homes, newer builds, and more bathrooms suggest higher quality
    overall_qual = min(
        10, max(1, int(2 + bathrooms + (year_built - 1900) / 20)))

    # Estimate OverallCond (1-9) — assume reasonable maintenance
    overall_cond = 7

    # Estimate HouseStyle from bedrooms
    if bedrooms <= 2:
        house_style = "1Story"
    elif bedrooms == 3:
        house_style = "SLvl"
    else:
        house_style = "2Story"

    # Estimate TotRmsAbvGrd from bedrooms + sqft_living
    tot_rms = max(4, min(14, int(bedrooms + 2 + sqft_living / 350)))

    # Estimate garage (typical for single-family)
    garage_cars = 2 if sqft_living > 1500 else 1
    garage_area = garage_cars * 240

    # Estimate fireplaces (typical for single-family)
    fireplaces = 1 if sqft_living > 1500 else 0

    year_remod = year_built  # No renovation data; use year built

    property_type = classify_property_type({
        "OverallQual": overall_qual,
        "GrLivArea": sqft_living,
        "TotRmsAbvGrd": tot_rms,
        "BedroomAbvGr": bedrooms,
        "HouseStyle": house_style,
        "CensusMedianValue": None,
        "OwnerOccupiedRate": None,
    })

    return {
        "LotArea": sqft_lot,
        "OverallQual": overall_qual,
        "OverallCond": overall_cond,
        "YearBuilt": year_built,
        "YearRemodAdd": year_remod,
        "GrLivArea": sqft_living,
        "FullBath": full_bath,
        "HalfBath": half_bath,
        "BedroomAbvGr": bedrooms,
        "TotRmsAbvGrd": tot_rms,
        "Fireplaces": fireplaces,
        "GarageCars": garage_cars,
        "GarageArea": garage_area,
        "PropertyType": property_type,
        "HouseStyle": house_style,
        "NeighborhoodScore": None,  # filled in after KNN fitting
        "CensusMedianValue": None,  # filled by Census API at inference
        "MedianIncomeK": None,
        "OwnerOccupiedRate": None,
        "Neighborhood": zipcode,
        "SalePrice": price,
        "_lat": lat,
        "_lon": lon,
        "_source": f"redfin-{state}",
    }


def _load_redfin_nationwide(
    data_dir: Path,
    min_price: float,
    max_bedrooms: int,
    state_filter: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Load all or a subset of Redfin CSV files from data/raw/HousingPriceUSA/.

    Each file is named like AK-1.csv, AK-2.csv, etc. (state abbreviation + partition).
    Filters for single-family homes and returns the canonicalized row list.

    Args:
        data_dir: Path to HousingPriceUSA directory
        min_price: Minimum price threshold
        max_bedrooms: Maximum bedroom count
        state_filter: Optional list of state abbreviations to load (e.g. ["WA", "CA"])
                     If None, loads all states.
    """
    csv_files = sorted(data_dir.glob("*.csv"))
    if state_filter:
        csv_files = [f for f in csv_files if f.stem.split(
            "-")[0].upper() in state_filter]

    print(f"  [redfin] Found {len(csv_files):,} file(s) to load")

    all_rows: list[dict[str, Any]] = []
    total_raw = 0
    failed_files = []

    for csv_file in csv_files:
        state = csv_file.stem.split("-")[0].upper()
        try:
            df = pd.read_csv(csv_file, dtype=str)
            total_raw += len(df)
            dropped_filter = 0
            dropped_bad = 0

            for _, row in df.iterrows():
                mapped = _map_redfin_row(
                    row, state=state, min_price=min_price, max_bedrooms=max_bedrooms)
                if mapped is None:
                    dropped_filter += 1
                else:
                    all_rows.append(mapped)

            print(
                f"    {state}: {len(df):,} raw → {len(df) - dropped_filter:,} mapped "
                f"({dropped_filter} filtered)"
            )
        except Exception as exc:
            failed_files.append((csv_file.name, str(exc)))
            print(f"    {state}: ERROR: {exc}")

    if failed_files:
        print(f"  [redfin] {len(failed_files)} file(s) failed to load")

    print(
        f"  [redfin] {total_raw:,} raw rows → {len(all_rows):,} single-family rows mapped")
    return all_rows


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_INTERNAL_KEYS = {"_lat", "_lon", "_source"}


def _strip_internal(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k not in _INTERNAL_KEYS}


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(_strip_internal(row)) + "\n")
    print(f"  Wrote {len(rows):,} rows → {path}")


def _write_report(
    report: dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"  Wrote report  → {path}")


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(rows: list[dict[str, Any]], scorer: NeighborhoodScoreService | None) -> None:
    kc_rows = [r for r in rows if r.get("_source") == "king-county"]
    ames_rows = [r for r in rows if r.get("_source") == "ames"]
    redfin_rows = [r for r in rows if r.get(
        "_source", "").startswith("redfin-")]
    scored_rows = [r for r in rows if r.get("NeighborhoodScore") is not None]

    prices = [r["SalePrice"] for r in rows]
    prop_types = {}
    for r in rows:
        pt = r.get("PropertyType", "unknown")
        prop_types[pt] = prop_types.get(pt, 0) + 1

    print()
    print("=" * 58)
    print("  INGEST SUMMARY")
    print("=" * 58)
    print(f"  Total rows          : {len(rows):,}")
    print(f"  King County rows    : {len(kc_rows):,}")
    print(f"  Ames rows           : {len(ames_rows):,}")
    print(f"  Redfin rows         : {len(redfin_rows):,}")
    print(f"  Rows with NeighScore: {len(scored_rows):,}")
    if prices:
        print(
            f"  Price range         : ${min(prices):,.0f} – ${max(prices):,.0f}")
        print(f"  Median price        : ${float(np.median(prices)):,.0f}")
    print(f"  PropertyType dist   : {prop_types}")
    if scorer:
        diag = scorer.diagnostics()
        print(
            f"  Scorer ref points   : {diag.get('reference_point_count', 0):,}")
        print(
            f"  Scorer k / decay    : {diag.get('k')} / {diag.get('decay_km')} km")
    print("  Census features     : None (CensusMedianValue, MedianIncomeK,")
    print("                        OwnerOccupiedRate — filled by live API")
    print("=" * 58)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map raw CSV datasets to the canonical 20-feature training JSONL format.",
    )
    parser.add_argument(
        "--source",
        choices=["king-county", "ames", "both",
                 "redfin-nationwide", "redfin-all"],
        default="both",
        help=(
            "Data source(s) to ingest. Options:\n"
            "  king-county: Original King County WA data\n"
            "  ames: Ames, Iowa data\n"
            "  both: King County + Ames (default)\n"
            "  redfin-nationwide: All nationwide Redfin single-family homes\n"
            "  redfin-all: Nationwide + King County + Ames"
        ),
    )
    parser.add_argument(
        "--kc-csv",
        default="data/raw/Housing.csv",
        help="Path to King County dataset (default: data/raw/Housing.csv).",
    )
    parser.add_argument(
        "--ames-csv",
        default="data/raw/housing.csv",
        help="Path to Ames Iowa dataset (default: data/raw/housing.csv).",
    )
    parser.add_argument(
        "--redfin-dir",
        default="data/raw/HousingPriceUSA",
        help="Path to Redfin nationwide CSV directory (default: data/raw/HousingPriceUSA).",
    )
    parser.add_argument(
        "--redfin-states",
        nargs="+",
        default=None,
        help=(
            "Optional: List of state abbreviations to load from Redfin directory. "
            "E.g., --redfin-states WA CA OR to load only Washington, California, Oregon. "
            "If not specified, loads all states in the directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory for JSONL and report (default: data/processed).",
    )
    parser.add_argument(
        "--output-file",
        default="csv_training_data.jsonl",
        help="Output JSONL filename (default: csv_training_data.jsonl).",
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Directory for neighborhood_scorer.joblib (default: models).",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=10,
        help="KNN neighbour count for NeighborhoodScore (default: 10).",
    )
    parser.add_argument(
        "--knn-decay-km",
        type=float,
        default=8.0,
        help="Gaussian decay radius in km (default: 8.0).",
    )
    parser.add_argument(
        "--national-scorer-path",
        default=None,
        help=(
            "Path to a pre-built national NeighborhoodScoreService joblib "
            "(produced by scripts/seed_national_neighborhood_scorer.py). "
            "When provided, training rows are scored directly from the "
            "national ZCTA reference index instead of the KC-only LOO scorer. "
            "Recommended: models/neighborhood_scorer.joblib after running "
            "seed_national_neighborhood_scorer.py."
        ),
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=50_000.0,
        help="Drop rows with SalePrice below this threshold (default: 50000).",
    )
    parser.add_argument(
        "--max-bedrooms",
        type=int,
        default=10,
        help="Drop rows with more bedrooms than this (default: 10).",
    )
    parser.add_argument(
        "--min-output-rows",
        type=int,
        default=100,
        help="Exit with an error if fewer rows are produced (default: 100).",
    )
    parser.add_argument(
        "--zcta-cache",
        default="data/processed/zcta_census_stats.json",
        help=(
            "Path for caching downloaded ZCTA census stats "
            "(default: data/processed/zcta_census_stats.json). "
            "Delete this file to force a fresh Census API download."
        ),
    )
    parser.add_argument(
        "--skip-census-enrichment",
        action="store_true",
        help="Skip ZCTA census enrichment step (useful for offline/debug runs).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    output_path = output_dir / args.output_file
    report_path = output_dir / "csv_ingest_report.json"
    scorer_path = model_dir / "neighborhood_scorer.joblib"

    print("=" * 58)
    print("  CSV TRAINING DATA INGESTION")
    print(f"  source   : {args.source}")
    print(f"  output   : {output_path}")
    print(f"  scorer   : {scorer_path}")
    print("=" * 58)

    all_rows: list[dict[str, Any]] = []

    if args.source in ("king-county", "both", "redfin-all"):
        kc_path = Path(args.kc_csv)
        if not kc_path.exists():
            print(
                f"ERROR: King County CSV not found at {kc_path}", file=sys.stderr)
            sys.exit(1)
        all_rows.extend(
            _load_king_county(kc_path, min_price=args.min_price,
                              max_bedrooms=args.max_bedrooms)
        )

    if args.source in ("ames", "both", "redfin-all"):
        ames_path = Path(args.ames_csv)
        if not ames_path.exists():
            print(f"ERROR: Ames CSV not found at {ames_path}", file=sys.stderr)
            sys.exit(1)
        all_rows.extend(_load_ames(ames_path, min_price=args.min_price))

    if args.source in ("redfin-nationwide", "redfin-all"):
        redfin_dir = Path(args.redfin_dir)
        if not redfin_dir.exists():
            print(
                f"ERROR: Redfin directory not found at {redfin_dir}", file=sys.stderr)
            sys.exit(1)
        all_rows.extend(
            _load_redfin_nationwide(
                redfin_dir,
                min_price=args.min_price,
                max_bedrooms=args.max_bedrooms,
                state_filter=args.redfin_states,
            )
        )

    if len(all_rows) < args.min_output_rows:
        print(
            f"ERROR: Only {len(all_rows)} rows produced; "
            f"minimum required is {args.min_output_rows}.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── ZCTA Census enrichment ─────────────────────────────────────────────
    # Populate CensusMedianValue, MedianIncomeK, OwnerOccupiedRate for KC rows
    # by joining to Census ACS5 ZCTA data (keyed by the 5-digit zipcode that
    # the KC ingest stores as the Neighborhood field).
    # This makes the three census features non-null in training so the RF can
    # learn their relationship with price — the key cross-market signal that
    # generalises to any US address via the live Census API at inference time.
    census_enriched_count = 0
    if not args.skip_census_enrichment and args.source in ("king-county", "both"):
        kc_zipcodes = [
            str(r.get("Neighborhood", "")).strip().zfill(5)
            for r in all_rows
            if r.get("_source") == "king-county"
        ]
        if kc_zipcodes:
            zcta_cache_path = Path(
                args.zcta_cache) if args.zcta_cache else None
            zcta_stats = _fetch_zcta_census_stats(
                kc_zipcodes, cache_path=zcta_cache_path)
            if zcta_stats:
                census_enriched_count = _apply_census_enrichment(
                    all_rows, zcta_stats)
                print(
                    f"  [census] {census_enriched_count:,} / {len(kc_zipcodes):,} KC rows "
                    "enriched with ACS5 ZCTA census statistics."
                )
            else:
                print(
                    "  [census] WARNING: No ZCTA stats retrieved — "
                    "census features remain null. Check network access or use "
                    "--skip-census-enrichment to suppress this warning."
                )
    else:
        if args.skip_census_enrichment:
            print("  [census] Census enrichment skipped (--skip-census-enrichment).")

    # Fit scorer and assign NeighborhoodScores to all rows with lat/lon
    national_scorer_path = (
        Path(args.national_scorer_path) if args.national_scorer_path else None
    )
    scorer = _fit_and_assign_neighborhood_scores(
        all_rows,
        k=args.knn_k,
        decay_km=args.knn_decay_km,
        scorer_path=scorer_path,
        national_scorer_path=national_scorer_path,
    )

    _print_summary(all_rows, scorer)

    # Write training JSONL (internal _lat/_lon/_source stripped)
    _write_jsonl(all_rows, output_path)

    # Write report
    kc_count = sum(1 for r in all_rows if r.get("_source") == "king-county")
    ames_count = sum(1 for r in all_rows if r.get("_source") == "ames")
    redfin_count = sum(1 for r in all_rows if r.get(
        "_source", "").startswith("redfin-"))
    scored_count = sum(1 for r in all_rows if r.get(
        "NeighborhoodScore") is not None)
    prices = [r["SalePrice"] for r in all_rows]
    prop_types = {}
    for r in all_rows:
        pt = r.get("PropertyType", "unknown")
        prop_types[pt] = prop_types.get(pt, 0) + 1

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": args.source,
        "output": str(output_path),
        "total_rows": len(all_rows),
        "king_county_rows": kc_count,
        "ames_rows": ames_count,
        "redfin_nationwide_rows": redfin_count,
        "rows_with_neighborhood_score": scored_count,
        "price_min": float(min(prices)) if prices else None,
        "price_max": float(max(prices)) if prices else None,
        "price_median": float(np.median(prices)) if prices else None,
        "property_type_distribution": prop_types,
        "knn_k": args.knn_k,
        "knn_decay_km": args.knn_decay_km,
        "scorer_path": str(scorer_path),
        "scorer_diagnostics": scorer.diagnostics() if scorer else None,
        "census_features_present": census_enriched_count > 0,
        "census_enriched_rows": census_enriched_count,
        "census_note": (
            f"{census_enriched_count:,} KC rows enriched with ACS5 ZCTA stats "
            "(CensusMedianValue, MedianIncomeK, OwnerOccupiedRate). "
            "These features now have real training signal for cross-market generalization. "
            "The live Census API fills them at inference time for any US address."
            if census_enriched_count > 0
            else (
                "CensusMedianValue / MedianIncomeK / OwnerOccupiedRate are null "
                "(no census enrichment; use --zcta-cache or check network access). "
                "The live Census API fills them at inference time for any US address."
            )
        ),
        "kc_column_mapping": {
            "grade→OverallQual": "linear remap 1-13 → 1-10",
            "condition→OverallCond": "linear remap 1-5 → 1-9",
            "bathrooms→FullBath+HalfBath": "decimal split at 0.25",
            "floors→HouseStyle": "1→1Story, 1.5→SLvl, ≥2→2Story",
            "GarageCars": "estimated from grade",
            "GarageArea": "GarageCars × 240 sqft",
            "Fireplaces": "estimated from grade",
            "TotRmsAbvGrd": "estimated from bedrooms + sqft_above / 350",
            "Neighborhood": "zipcode string",
        },
    }
    _write_report(report, report_path)

    print(f"\nDone. {len(all_rows):,} rows ready for training.")
    print(
        f"  Train:  RAW_DATA_PATH={output_path} python scripts/train.py --min-rows=100")


if __name__ == "__main__":
    main()
