#!/usr/bin/env python3
"""
Seed NeighborhoodScoreService with national ZCTA centroids + Census ACS median
home values.

Run once (or annually after a new ACS vintage is released):

    python scripts/seed_national_neighborhood_scorer.py

    # With a Census API key (removes rate-limit on large single requests):
    python scripts/seed_national_neighborhood_scorer.py --census-api-key YOUR_KEY

This replaces models/neighborhood_scorer.joblib with a scorer backed by
reference points across every populated ZCTA in the United States, so any
US address receives a meaningful NeighborhoodScore at inference time —
not just King County WA addresses.

Data sources
------------
  1. Census 2022 ZCTA5 Gazetteer (population-weighted centroids):
       https://www2.census.gov/geo/docs/maps-data/data/gazetteer/
           2022_Gazetteer/2022_Gaz_zcta_national.zip
     Fields used: GEOID (ZCTA5), INTPTLAT, INTPTLONG
  2. ACS5 (2022) B25077_001E — median home value, all ZCTAs in one request:
       https://api.census.gov/data/2022/acs/acs5
           ?get=B25077_001E&for=zip+code+tabulation+area:*

Outputs
-------
  models/neighborhood_scorer.joblib       — updated national scorer
  data/processed/zcta_national_centroids.json  — cached merged ZCTA data
      (delete this file to force a fresh Census download next run)
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from house_price_prediction.application.services.neighborhood_score_service import (  # noqa: E402
    NeighborhoodScoreService,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GAZETTEER_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
    "2022_Gazetteer/2022_Gaz_zcta_national.zip"
)
ACS5_URL = "https://api.census.gov/data/2022/acs/acs5"
CENSUS_ACS_NULL = -666_666_666

DEFAULT_CACHE_PATH = ROOT / "data/processed/zcta_national_centroids.json"
DEFAULT_OUTPUT_PATH = ROOT / "models/neighborhood_scorer.joblib"

# Larger decay for national coverage:
#   - urban ZCTAs span ~2-5 km → neighbours within 5-10 km
#   - suburban ZCTAs span ~5-10 km → within 10-20 km
#   - rural ZCTAs span 20-100 km → decay must reach ~30 km to avoid fallback
DECAY_KM = 20.0
KNN_K = 10

US_SANITY_CHECKS = [
    ("KC (Seattle area)",    47.50, -122.25),
    ("Manhattan NYC",        40.75,  -73.98),
    ("Miami FL",             25.77,  -80.19),
    ("Chicago IL",           41.88,  -87.63),
    ("Austin TX",            30.27,  -97.74),
    ("Denver CO",            39.74, -104.98),
    ("Rural Montana",        46.87, -110.36),
    ("Phoenix AZ",           33.45, -112.07),
    ("Boston MA",            42.36,  -71.06),
    ("San Francisco CA",     37.77, -122.42),
    ("Rural Mississippi",    32.73,  -89.70),
    ("Honolulu HI",          21.31, -157.80),
    ("Anchorage AK",         61.22, -149.90),
]


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def _fetch_zcta_gazetteer() -> dict[str, tuple[float, float]]:
    """Download ZCTA5 Gazetteer ZIP and return {zcta5: (lat, lon)}."""
    print(f"  Downloading ZCTA Gazetteer …  {GAZETTEER_URL}")
    with httpx.Client(follow_redirects=True, timeout=120.0) as client:
        resp = client.get(
            GAZETTEER_URL,
            headers={"User-Agent": "house-price-prediction/0.1"},
        )
    resp.raise_for_status()

    centroids: dict[str, tuple[float, float]] = {}
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        txt_name = next(n for n in zf.namelist() if n.endswith(".txt"))
        with zf.open(txt_name) as raw_f:
            text = io.TextIOWrapper(raw_f, encoding="utf-8")
            # The Gazetteer uses fixed-width columns padded with trailing spaces;
            # strip all field names so DictReader key lookups work correctly.
            header_line = text.readline()
            stripped_headers = [h.strip() for h in header_line.split("\t")]
            reader = csv.DictReader(text, fieldnames=stripped_headers, delimiter="\t")
            for row in reader:
                zcta = (row.get("GEOID") or "").strip().zfill(5)
                try:
                    lat = float((row.get("INTPTLAT") or "").strip())
                    lon = float((row.get("INTPTLONG") or "").strip())
                except ValueError:
                    continue
                if zcta:
                    centroids[zcta] = (lat, lon)

    print(f"  Loaded {len(centroids):,} ZCTA centroids from Gazetteer.")
    return centroids


def _fetch_zcta_median_values(census_api_key: str | None) -> dict[str, float]:
    """Fetch ACS5 B25077_001E (median home value) for all ZCTAs."""
    print("  Fetching ACS5 B25077 median home values for all ZCTAs …")
    params: dict[str, str] = {
        "get": "B25077_001E",
        "for": "zip code tabulation area:*",
    }
    if census_api_key:
        params["key"] = census_api_key

    with httpx.Client(timeout=180.0) as client:
        resp = client.get(
            ACS5_URL,
            params=params,
            headers={"User-Agent": "house-price-prediction/0.1"},
        )
    resp.raise_for_status()

    rows = resp.json()
    if len(rows) < 2:
        raise RuntimeError("ACS5 returned no ZCTA median home value data.")

    headers = rows[0]
    value_idx = headers.index("B25077_001E")
    zcta_idx = headers.index("zip code tabulation area")

    values: dict[str, float] = {}
    for r in rows[1:]:
        zcta = str(r[zcta_idx]).strip().zfill(5)
        raw = r[value_idx]
        if raw is None:
            continue
        try:
            v = int(raw)
        except (ValueError, TypeError):
            continue
        # -666666666 is the Census suppression sentinel
        if v <= 0 or v == CENSUS_ACS_NULL:
            continue
        values[zcta] = float(v)

    print(f"  Loaded {len(values):,} valid ZCTA median home values from ACS5.")
    return values


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Seed national NeighborhoodScoreService from Census ZCTA data. "
            "Replaces models/neighborhood_scorer.joblib."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to write the scorer joblib (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help=(
            f"Path for caching merged ZCTA centroid+value data "
            f"(default: {DEFAULT_CACHE_PATH}). "
            "Delete this file to force a fresh Census download."
        ),
    )
    parser.add_argument(
        "--census-api-key",
        default=None,
        help="Optional Census API key (not required; increases rate limit).",
    )
    parser.add_argument(
        "--decay-km",
        type=float,
        default=DECAY_KM,
        help=f"Gaussian decay distance in km (default: {DECAY_KM}).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=KNN_K,
        help=f"KNN neighbour count (default: {KNN_K}).",
    )
    args = parser.parse_args()

    print("\n=== Seeding national NeighborhoodScoreService ===\n")

    # ------------------------------------------------------------------
    # 1. Load or fetch ZCTA data
    # ------------------------------------------------------------------
    zcta_data: dict[str, dict[str, float]] = {}

    if args.cache.exists():
        print(f"  Loading merged ZCTA data from cache: {args.cache}")
        with args.cache.open() as f:
            zcta_data = json.load(f)
        print(f"  Cache has {len(zcta_data):,} ZCTAs.")
    else:
        centroids = _fetch_zcta_gazetteer()
        median_values = _fetch_zcta_median_values(args.census_api_key)

        # Merge: only ZCTAs with both centroid AND valid home value
        for zcta, (lat, lon) in centroids.items():
            if zcta in median_values:
                zcta_data[zcta] = {
                    "lat": lat,
                    "lon": lon,
                    "median_value": median_values[zcta],
                }

        print(f"\n  Merged: {len(zcta_data):,} ZCTAs have both centroid and median value.")

        # Cache to disk
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        with args.cache.open("w") as f:
            json.dump(zcta_data, f)
        print(f"  Cached to {args.cache}")

    if not zcta_data:
        print("ERROR: No ZCTA data available; cannot seed scorer.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Fit scorer
    # ------------------------------------------------------------------
    lats = [v["lat"] for v in zcta_data.values()]
    lons = [v["lon"] for v in zcta_data.values()]
    values = [v["median_value"] for v in zcta_data.values()]

    print(f"\n  Fitting NeighborhoodScoreService(k={args.k}, decay_km={args.decay_km}) …")
    print(f"  Reference points: {len(lats):,}")
    print(
        f"  Home-value range: "
        f"${min(values):,.0f} – ${max(values):,.0f}  "
        f"(median ${sorted(values)[len(values) // 2]:,.0f})"
    )

    svc = NeighborhoodScoreService(k=args.k, decay_km=args.decay_km)
    svc.fit(lats, lons, values)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    svc.save(args.output)
    print(f"  Saved scorer to {args.output}")

    # ------------------------------------------------------------------
    # 3. Sanity check across US cities
    # ------------------------------------------------------------------
    print("\n--- Sanity check: NeighborhoodScore for representative US locations ---")
    print(f"  (0 = low tier / $50k, 100 = high tier / $1.2M, scale is linear)")
    print()
    for name, lat, lon in US_SANITY_CHECKS:
        score = svc.score(lat, lon)
        print(f"  {name:28s}: {score:5.1f}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
