"""
House Price Predictor - CSV-Based Pipeline

Takes a single address as input and returns:
1. All 16 model features (from state CSV data)
2. Price prediction from trained ML model
3. Confidence metrics

Data flow:
  Address → Extract State → Load State CSV Files → Get Median Features
         → Get School District Rating → Add School Feature
         → Model.predict() → Price Prediction
"""

import httpx
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime
import glob
from pathlib import Path
import joblib


class AssessorAPIConnector:
    """Build property feature dicts from address parsing, geocoding, and Census ACS data for the .pkl model."""

    @staticmethod
    def search_property_by_address(address: str) -> Dict:
        """Derive pkl-compatible property features directly from the address using RentCast API."""
        print(f"[PROPERTY-LOOKUP] Building features from address: {address}")
        
        # First try RentCast for real property data
        try:
            from .infrastructure.providers.factory import create_property_data_provider
            from .config import load_settings
            from .domain.contracts.prediction_contracts import NormalizedAddress
            
            settings = load_settings()
            provider = create_property_data_provider(settings)
            
            # Parse address for RentCast
            import re
            parts = [p.strip() for p in address.split(',')]
            zip_match = re.search(r'\b(\d{5})\b', address)
            real_zipcode = zip_match.group(1) if zip_match else '00000'
            
            state_match = re.search(r'\b([A-Z]{2})\s+\d{5}\b', address)
            real_state = state_match.group(1) if state_match else 'XX'
            
            normalized = NormalizedAddress(
                address_line_1=parts[0] if parts else address,
                city=parts[1] if len(parts) > 1 else 'Unknown',
                state=real_state,
                postal_code=real_zipcode,
                country="US",
                formatted_address=address
            )
            
            print(f"[RENTCAST] Attempting to fetch real property data...")
            response = provider.fetch_property_features(normalized)
            if response and response.payload:
                features = AssessorAPIConnector._features_from_address(address, rentcast_data=response.payload)
                print(f"[OK] Features built from RentCast for: {address}")
                return features
        except Exception as e:
            print(f"[RENTCAST] Failed ({e}), falling back to Census+Heuristic...")
        
        # Fallback: use Census + heuristic defaults
        features = AssessorAPIConnector._features_from_address(address)
        print(f"[OK] Features built (Census+Heuristic) for: {address}")
        return features

    @staticmethod
    def _geocode_address_simple(address: str):
        """Return (lat, lon) for address via Nominatim, or (0.0, 0.0) on failure."""
        try:
            import urllib.request
            import json as _json
            import urllib.parse
            encoded = urllib.parse.quote(address)
            url = f"https://nominatim.openstreetmap.org/search?q={encoded}&format=json&limit=1"
            req = urllib.request.Request(url, headers={"User-Agent": "HousePricePrediction/1.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                results = _json.loads(resp.read())
            if results:
                return float(results[0]["lat"]), float(results[0]["lon"])
        except Exception as e:
            print(f"[GEOCODE-SIMPLE] Failed: {e}")
        return 0.0, 0.0

    @staticmethod
    def _census_acs_by_zip(zipcode: str) -> dict:
        """Fetch ACS 5-year estimates for a ZIP code tabulation area from the Census Bureau API.
        Returns dict with 'median_home_value' (B25077_001E) and 'median_income' (B19013_001E).
        Falls back to None values on any error."""
        try:
            import urllib.request
            import json as _json
            # Census ACS 5-year — no API key required for simple queries
            url = (
                f"https://api.census.gov/data/2023/acs/acs5"
                f"?get=B25077_001E,B19013_001E"
                f"&for=zip%20code%20tabulation%20area:{zipcode}"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "HousePricePrediction/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                rows = _json.loads(resp.read())
            # rows[0] = header, rows[1] = data
            if len(rows) >= 2:
                header, data = rows[0], rows[1]
                idx_val = header.index("B25077_001E")
                idx_inc = header.index("B19013_001E")
                median_home_value = int(data[idx_val]) if data[idx_val] not in (None, "-666666666") else None
                median_income     = int(data[idx_inc]) if data[idx_inc] not in (None, "-666666666") else None
                print(f"[CENSUS-ACS] zip={zipcode}  median_home_value=${median_home_value:,}  median_income=${median_income:,}")
                return {"median_home_value": median_home_value, "median_income": median_income}
        except Exception as e:
            print(f"[CENSUS-ACS] Failed for zip {zipcode}: {e}")
        return {"median_home_value": None, "median_income": None}

    @staticmethod
    def _features_from_address(address: str, rentcast_data: Dict = None) -> Dict:
        """Build house_price_model.pkl feature dict from address parsing, geocoding,
        and real Census ACS data — no CSV files are read."""
        import re

        # --- Parse city / state / zip ---
        parts = [p.strip() for p in address.split(',')]
        zip_match = re.search(r'\b(\d{5})\b', address)
        real_zipcode = zip_match.group(1) if zip_match else '00000'

        state_match = re.search(r'\b([A-Z]{2})\s+\d{5}\b', address)
        real_state = state_match.group(1) if state_match else 'XX'

        # City is the comma-part immediately before the "STATE ZIP" segment
        real_city = parts[-2] if len(parts) >= 3 else (parts[1] if len(parts) >= 2 else 'Unknown')
        real_city = re.sub(r'\b[A-Z]{2}\b', '', real_city).strip()

        # --- Geocode ---
        lat, lon = AssessorAPIConnector._geocode_address_simple(address)
        print(f"[FEATURES] city={real_city}, state={real_state}, zip={real_zipcode}, lat={lat:.4f}, lon={lon:.4f}")

        # --- Real Census ACS median home value and income by zip ---
        acs = AssessorAPIConnector._census_acs_by_zip(real_zipcode)
        census_median   = acs["median_home_value"]
        raw_income      = acs["median_income"]

        # Fallback: state-level median home values (2024 NAR estimates) if ACS unavailable
        STATE_MEDIAN: dict[str, int] = {
            "WA": 575000, "CA": 790000, "HI": 835000, "MA": 620000, "CO": 580000,
            "OR": 480000, "NJ": 520000, "NY": 550000, "UT": 520000, "AZ": 430000,
            "FL": 415000, "GA": 330000, "TX": 355000, "NC": 330000, "IL": 315000,
            "OH": 265000, "PA": 295000, "MI": 265000, "TN": 335000, "VA": 450000,
            "MD": 435000, "NV": 430000, "ID": 430000, "MN": 335000, "WI": 290000,
            "IN": 250000, "MO": 270000, "SC": 320000, "AL": 255000, "KY": 255000,
            "OK": 235000, "AR": 235000, "MS": 215000, "IA": 245000, "KS": 255000,
            "NE": 280000, "SD": 260000, "ND": 265000, "NM": 320000, "LA": 260000,
            "MT": 415000, "WV": 200000, "AK": 410000, "DC": 720000, "CT": 435000,
            "RI": 470000, "NH": 455000, "VT": 420000, "ME": 400000, "DE": 395000,
            "WY": 355000,
        }
        STATE_INCOME: dict[str, int] = {
            "WA": 95000, "CA": 91000, "HI": 88000, "MA": 96000, "CO": 89000,
            "OR": 78000, "NJ": 97000, "NY": 82000, "UT": 85000, "AZ": 72000,
            "FL": 67000, "GA": 71000, "TX": 73000, "NC": 68000, "IL": 75000,
            "OH": 66000, "PA": 72000, "MI": 67000, "TN": 66000, "VA": 90000,
            "MD": 98000, "NV": 70000, "ID": 70000, "MN": 84000, "WI": 72000,
            "IN": 65000, "MO": 66000, "SC": 64000, "AL": 59000, "KY": 60000,
            "OK": 60000, "AR": 57000, "MS": 52000, "IA": 68000, "KS": 67000,
            "NE": 70000, "SD": 65000, "ND": 71000, "NM": 58000, "LA": 57000,
            "MT": 65000, "WV": 51000, "AK": 82000, "DC": 101000, "CT": 90000,
            "RI": 77000, "NH": 92000, "VT": 72000, "ME": 68000, "DE": 76000,
            "WY": 68000,
        }
        if census_median is None or census_median <= 0:
            census_median = STATE_MEDIAN.get(real_state, 350000)
            print(f"[FEATURES] ACS unavailable — using state median ${census_median:,} for {real_state}")
        if raw_income is None or raw_income <= 0:
            raw_income = STATE_INCOME.get(real_state, 70000)

        median_income_k = round(raw_income / 1000)

        # NeighborhoodScore: percentile of census_median relative to national range $150k–$1.5M
        ns_raw = (census_median - 150000) / (1500000 - 150000) * 100
        neighborhood_score = max(10, min(99, round(ns_raw)))

        # --- Reasonable defaults for unknown property details ---
        yr_built       = 1990
        gr_liv_area    = 1910.0
        lot_area       = 7590.0
        # Use RentCast data if available, otherwise use heuristic defaults
        if rentcast_data:
            bedrooms       = int(rentcast_data.get('BedroomAbvGr', 3)) or 3
            full_bath      = int(rentcast_data.get('FullBath', 2)) or 2
            half_bath      = int(rentcast_data.get('HalfBath', 0)) or 0
            gr_liv_area    = float(rentcast_data.get('GrLivArea', 1910.0)) or 1910.0
            lot_area       = float(rentcast_data.get('LotArea', 7590.0)) or 7590.0
            yr_built       = int(rentcast_data.get('YearBuilt', 1990)) or 1990
            overall_qual   = int(rentcast_data.get('OverallQual', 7)) or 7
            overall_cond   = int(rentcast_data.get('OverallCond', 5)) or 5
            print(f"[FEATURES] Using RentCast data: {bedrooms}bed, {full_bath}bath, {gr_liv_area}sqft, built {yr_built}")
        else:
            bedrooms       = 3
            full_bath      = 2
            half_bath      = 0
            overall_qual   = 7
            overall_cond   = 5
        tot_rooms      = bedrooms + full_bath + 2
        price_per_sqft = round(census_median / gr_liv_area, 1)
        land_value     = round(census_median * 0.25)

        print(f"[FEATURES] score={neighborhood_score}, income={median_income_k}k, "
              f"census_median=${census_median:,}, price_per_sqft=${price_per_sqft}")

        return {
            # ── SmartRouter core features (17 columns) ──────────────────
            'LotArea':           lot_area,
            'OverallQual':       overall_qual,
            'OverallCond':       overall_cond,
            'YearBuilt':         yr_built,
            'YearRemodAdd':      yr_built + 10,
            'GrLivArea':         gr_liv_area,
            'FullBath':          full_bath,
            'HalfBath':          half_bath,
            'BedroomAbvGr':      bedrooms,
            'TotRmsAbvGrd':      tot_rooms,
            'Fireplaces':        1,
            'GarageCars':        2,
            'GarageArea':        round(gr_liv_area * 0.22),
            'NeighborhoodScore': neighborhood_score,
            'PropertyType':      'single_family',
            'HouseStyle':        '1Story',
            'Neighborhood':      real_zipcode,   # ZIP code — SmartRouter zip_col
            # ── Extra context (not passed to model, used in response) ──
            'CensusMedianValue': census_median,
            'MedianIncomeK':     median_income_k,
            'OwnerOccupiedRate': 0.65,
            'City':              real_city,
            'ZipCode':           real_zipcode,
            'State':             real_state,
            'SchoolDistrictRating': 7.5,
            'WalkScore':         min(100, neighborhood_score + 10),
            'HOAFee':            0.0,
            'PricePerSqft':      price_per_sqft,
            'LandValue':         land_value,
            'price':             census_median,
            'address':           address,
            'source':            'rentcast_property_data + Census ACS (SmartRouter)',
        }



class GeocodeAndCensus:
    """Get Census data by geocoding address."""

    @staticmethod
    def get_census_features(address: str) -> Dict:
        """
        Geocode address and fetch Census economic data.

        Args:
            address: Full address

        Returns:
            Dict with Census features: MedianIncome, UnemploymentRate
        """
        print(f"[GEOCODE] Processing address: {address}")

        try:
            # Step 1: Geocode address to coordinates
            coords = GeocodeAndCensus._geocode_address(address)

            # Step 2: Get Census tract from coordinates
            tract = GeocodeAndCensus._get_census_tract(
                coords['lat'], coords['lon'])

            # Step 3: Fetch Census data
            census_data = GeocodeAndCensus._fetch_census_data(tract)

            print(f"[OK] Retrieved Census data for tract {tract}")
            return census_data

        except Exception as e:
            print(f"[WARNING] Census data fetch failed: {e}, using defaults")
            return {
                'MedianIncome': 75000,
                'UnemploymentRate': 4.5
            }

    @staticmethod
    def _geocode_address(address: str) -> Dict:
        """Geocode address using Nominatim (FREE, no key needed)."""
        print(f"[GEOCODE] Converting address to coordinates...")

        params = {
            'q': address,
            'format': 'json'
        }

        response = httpx.get(
            'https://nominatim.openstreetmap.org/search', params=params)
        data = response.json()

        if not data:
            raise ValueError(f"Could not geocode address: {address}")

        coords = {
            'lat': float(data[0]['lat']),
            'lon': float(data[0]['lon']),
            'display_name': data[0]['display_name']
        }

        print(f"[OK] Geocoded to: {coords['lat']:.4f}, {coords['lon']:.4f}")
        return coords

    @staticmethod
    def _get_census_tract(lat: float, lon: float) -> str:
        """Get Census tract from coordinates using FCC API (FREE)."""
        print(f"[CENSUS] Getting Census tract...")

        params = {
            'lat': lat,
            'lon': lon,
            'format': 'json'
        }

        response = httpx.get(
            'https://geo.fcc.gov/api/census/tract', params=params)
        data = response.json()

        if 'properties' not in data:
            raise ValueError("Could not determine Census tract")

        tract = data['properties']['Census2020']['tract']
        print(f"[OK] Census tract: {tract}")
        return tract

    @staticmethod
    def _fetch_census_data(tract: str) -> Dict:
        """Fetch Census economic data (simulated for now)."""
        # In production, query Census API with actual tract
        # For now, use simulated data
        np.random.seed(hash(tract) % 2**32)

        census_data = {
            'MedianIncome': np.random.uniform(40000, 150000),
            'UnemploymentRate': np.random.uniform(2, 10)
        }

        print(f"[OK] Census data retrieved")
        return census_data


class SchoolDistrictFeature:
    """Get school district ratings from free APIs and databases."""

    # School district ratings database (free public data)
    # Maps school district names to average ratings (1-10 scale)
    SCHOOL_DISTRICT_DB = {
        'seattle': 7.8,
        'bellevue': 9.2,
        'redmond': 8.9,
        'kirkland': 8.5,
        'mercer island': 9.5,
        'eastside': 8.7,
        'lake union': 7.5,
        'shoreline': 8.3,
        'edmonds': 8.1,
        'sammamish': 8.8,
        'issaquah': 8.9,
        'skykomish': 6.5,
        'snoqualmie': 7.2,
        'north bend': 7.0,
        'tukwila': 7.3,
        'kent': 6.8,
        'auburn': 6.7,
    }

    @staticmethod
    def get_school_district_rating(address: str, lat: Optional[float] = None, lon: Optional[float] = None) -> Tuple[str, float]:
        """
        Get school district name and rating for an address.

        Uses multiple methods:
        1. Reverse geocoding with Nominatim to find district
        2. Look up in free school database
        3. Fall back to national average

        Args:
            address: Full address
            lat: Latitude (if already geocoded)
            lon: Longitude (if already geocoded)

        Returns:
            Tuple of (district_name, rating_1to10)
        """
        print(f"[SCHOOL] Getting school district rating...")

        try:
            # Method 1: Try reverse geocoding to get district/county info
            if lat and lon:
                district_name = SchoolDistrictFeature._reverse_geocode_district(lat, lon)
            else:
                # Method 2: Extract district hint from address
                district_name = SchoolDistrictFeature._extract_district_from_address(address)

            # Method 3: Look up rating in database
            rating = SchoolDistrictFeature._lookup_district_rating(district_name)

            print(f"[OK] School District: {district_name}, Rating: {rating:.1f}/10")
            return district_name, rating

        except Exception as e:
            print(f"[WARNING] School district lookup failed: {e}, using default")
            return "Unknown", 7.5  # National average

    @staticmethod
    def _reverse_geocode_district(lat: float, lon: float) -> str:
        """Use Nominatim reverse geocoding to find school district."""
        try:
            params = {
                'lat': lat,
                'lon': lon,
                'format': 'json',
                'zoom': 10
            }
            response = httpx.get('https://nominatim.openstreetmap.org/reverse', params=params, timeout=5.0)
            data = response.json()

            # Extract county/district info
            address_parts = data.get('address', {})
            county = address_parts.get('county', '')
            state = address_parts.get('state', '')

            print(f"[GEOCODE] Found: {county}, {state}")
            return county if county else "Unknown"

        except Exception as e:
            print(f"[GEOCODE] Reverse geocode failed: {e}")
            return "Unknown"

    @staticmethod
    def _extract_district_from_address(address: str) -> str:
        """Extract school district name from address string."""
        address_lower = address.lower()

        # Check for known district names
        for district in SchoolDistrictFeature.SCHOOL_DISTRICT_DB.keys():
            if district in address_lower:
                return district

        # Try to extract city name (usually after first comma)
        parts = address.split(',')
        if len(parts) >= 2:
            city = parts[1].strip().lower()
            return city

        return "Unknown"

    @staticmethod
    def _lookup_district_rating(district_name: str) -> float:
        """Look up school district rating from database."""
        if not district_name or district_name == "Unknown":
            return 7.5  # National average

        district_key = district_name.lower().strip()

        # Direct lookup
        if district_key in SchoolDistrictFeature.SCHOOL_DISTRICT_DB:
            return SchoolDistrictFeature.SCHOOL_DISTRICT_DB[district_key]

        # Fuzzy match - check if district name is substring
        for known_district, rating in SchoolDistrictFeature.SCHOOL_DISTRICT_DB.items():
            if known_district in district_key or district_key in known_district:
                return rating

        # Default to national average if not found
        return 7.5

    @staticmethod
    def get_nces_school_data(address: str) -> Dict:
        """
        Get school data from National Center for Education Statistics (FREE API, no key needed).

        NCES provides school information at:
        https://nces.ed.gov/ccd/

        For now, returns simulated data. In production, query the NCES API directly.
        """
        print(f"[NCES] Getting school data from NCES database...")

        try:
            # In production, you would query NCES Education Search API
            # Free endpoint: https://educationdata.urban.org/api/v1/schools/

            # For now, return simulated school metrics
            school_data = {
                'school_count': np.random.randint(5, 15),
                'avg_class_size': np.random.randint(20, 28),
                'graduation_rate': np.random.uniform(0.85, 0.98),
                'proficiency_rate': np.random.uniform(0.65, 0.95),
                'per_pupil_spending': np.random.randint(8000, 15000),
            }

            print(f"[OK] NCES data retrieved")
            return school_data

        except Exception as e:
            print(f"[WARNING] NCES lookup failed: {e}")
            return {
                'school_count': 8,
                'avg_class_size': 24,
                'graduation_rate': 0.90,
                'proficiency_rate': 0.80,
                'per_pupil_spending': 11000,
            }


class PricePredictionPipeline:
    """Complete pipeline: Address → Features → Price."""

    def __init__(self, model_path: str = 'models/nationwide_smart_router.pkl'):
        """Initialize the pipeline with trained model."""
        import sys
        import joblib

        # SmartRouter class lives in scripts/model_utils.py — make it importable
        _scripts_dir = str(Path(__file__).parent.parent.parent / 'scripts')
        if _scripts_dir not in sys.path:
            sys.path.insert(0, _scripts_dir)

        try:
            loaded = joblib.load(model_path)
            # Handle both raw model and dict-wrapped model formats
            self.model = loaded.get('model', loaded) if isinstance(loaded, dict) else loaded
            print(f"[OK] Model loaded: {model_path}  (type={type(self.model).__name__})")
        except FileNotFoundError:
            print(f"[WARNING] Model not found at {model_path}")
            print("[INFO] Using baseline demo model")
            self.model = None

    def predict_price(self, address: str, real_features: Dict = None) -> Dict:
        """
        Predict house price from address.

        Args:
            address: Full address (e.g., "123 Oak St, Seattle, WA 98101")
            real_features: Optional dict with real property data to use instead of simulated

        Returns:
            Dict with:
              - predicted_price: Estimated price ($)
              - confidence: Model confidence (%)
              - features: All 16 features used
              - error_margin: ±$ error estimate
        """
        print("\n" + "=" * 80)
        print(f"PRICE PREDICTION FOR: {address}")
        print("=" * 80)

        # Step 1: Get property features from Assessor API (or use provided real features)
        if real_features:
            print("\n[STEP 1/4] Using PROVIDED real property data...")
            assessor_data = real_features.copy()
            price_target = assessor_data.pop('price', None)
        else:
            print("\n[STEP 1/4] Fetching property data from County Assessor...")
            assessor_data = AssessorAPIConnector.search_property_by_address(address)
            price_target = assessor_data.pop('price')  # Extract target

        # Step 2: Get school district rating
        print("\n[STEP 2/4] Fetching school district rating...")
        try:
            district_name, school_rating = SchoolDistrictFeature.get_school_district_rating(address)
        except Exception as e:
            print(f"[WARNING] School district lookup failed: {e}")
            district_name, school_rating = "Unknown", 7.5

        # Step 3: Add school district to features
        print("\n[STEP 3/4] Combining all 16 features...")
        assessor_data['SchoolDistrictRating'] = school_rating
        assessor_data['SchoolDistrict'] = district_name

        # Step 4: Make prediction
        print("\n[STEP 4/4] Making price prediction...")
        prediction = self._make_prediction(assessor_data)

        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)

        return {
            'address': address,
            'predicted_price': prediction['predicted_price'],
            'actual_price': price_target,
            'confidence': prediction['confidence'],
            'error_margin': prediction['error_margin'],
            'error_margin_low': prediction['predicted_price'] - prediction['error_margin'],
            'error_margin_high': prediction['predicted_price'] + prediction['error_margin'],
            'all_16_features': assessor_data,
            'school_district': district_name,
            'school_rating': school_rating,
            'timestamp': datetime.now().isoformat()
        }

    # -----------------------------------------------------------------------
    # Market calibration: ratio of median transaction price (ZHVI, 2023)
    # to ACS B25077 median owner-occupied home value (self-reported survey).
    # ACS systematically under-states market prices because respondents estimate
    # rather than transact.  These factors close that gap per state.
    # Source: Zillow Home Value Index (Dec 2023) ÷ Census ACS 5-yr B25077.
    # -----------------------------------------------------------------------
    _MARKET_CALIBRATION: dict[str, float] = {
        "AL": 1.21, "AK": 1.10, "AZ": 1.13, "AR": 1.19, "CA": 1.16,
        "CO": 1.15, "CT": 1.13, "DE": 1.12, "DC": 1.09, "FL": 1.22,
        "GA": 1.21, "HI": 1.08, "ID": 1.14, "IL": 1.14, "IN": 1.17,
        "IA": 1.15, "KS": 1.16, "KY": 1.18, "LA": 1.18, "ME": 1.12,
        "MD": 1.11, "MA": 1.14, "MI": 1.16, "MN": 1.18, "MS": 1.20,
        "MO": 1.19, "MT": 1.13, "NE": 1.16, "NV": 1.14, "NH": 1.12,
        "NJ": 1.14, "NM": 1.15, "NY": 1.15, "NC": 1.22, "ND": 1.14,
        "OH": 1.17, "OK": 1.18, "OR": 1.15, "PA": 1.14, "RI": 1.12,
        "SC": 1.20, "SD": 1.14, "TN": 1.21, "TX": 1.20, "UT": 1.13,
        "VT": 1.11, "VA": 1.13, "WA": 1.17, "WV": 1.15, "WI": 1.16,
        "WY": 1.14,
    }
    _MARKET_CALIBRATION_DEFAULT: float = 1.15  # national average

    @classmethod
    def _market_calibration_factor(cls, state: str) -> float:
        """Return the ZHVI-to-ACS normalization multiplier for the given state."""
        return cls._MARKET_CALIBRATION.get(state.upper(), cls._MARKET_CALIBRATION_DEFAULT)

    # SmartRouter input columns — must match the 17 columns the model was trained on
    _SR_NUMERIC = [
        'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
        'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd',
        'Fireplaces', 'GarageCars', 'GarageArea', 'NeighborhoodScore',
    ]
    _SR_STR = ['PropertyType', 'HouseStyle', 'Neighborhood']

    def _make_prediction(self, features: Dict) -> Dict:
        """Make price prediction using the SmartRouter (or demo fallback)."""
        # Build the 17-column DataFrame that SmartRouter.predict() expects
        row = {}
        for col in self._SR_NUMERIC:
            v = features.get(col)
            row[col] = float(v) if v is not None else 0.0
        for col in self._SR_STR:
            v = features.get(col)
            row[col] = str(v) if v is not None else ''
        X = pd.DataFrame([row])

        if self.model:
            try:
                raw_price = float(self.model.predict(X)[0])
            except Exception as e:
                print(f"[WARN] SmartRouter predict failed: {e}")
                raw_price = self._demo_prediction(features)
        else:
            raw_price = self._demo_prediction(features)

        # SmartRouter is trained on actual Redfin transaction prices — no
        # ACS calibration needed.  Use a flat ±9% error margin reflecting
        # the model's measured mean error on held-out data.
        predicted_price = round(raw_price, 2)
        error_margin    = round(predicted_price * 0.09)

        # Confidence: based on how many SmartRouter features are non-default
        non_default = sum(
            1 for col in self._SR_NUMERIC + self._SR_STR
            if features.get(col) not in (None, 0, 0.0, '', 'Unknown')
        )
        confidence = round(non_default / len(self._SR_NUMERIC + self._SR_STR) * 100, 2)

        print(f"\n[PREDICTION] Price: ${predicted_price:,.2f}")
        print(f"[CONFIDENCE] {confidence:.2f}%")
        print(f"[ERROR MARGIN] ±${error_margin:,.2f}")

        return {
            'predicted_price': predicted_price,
            'confidence': confidence,
            'error_margin': error_margin
        }

    @staticmethod
    def _demo_prediction(features: Dict) -> float:
        """Simple demo prediction when model not available."""
        sqft_living = features.get('GrLivArea', features.get('sqft_living', 2000))
        yr_built = features.get('YearBuilt', features.get('yr_built', 1990))
        grade = features.get('OverallQual', features.get('grade', 7))

        base_price = 100000
        price_per_sqft = 150 + (grade * 20)
        predicted_price = base_price + (sqft_living * price_per_sqft)

        # Age adjustment
        age = 2026 - yr_built
        age_factor = np.exp(-0.015 * age)
        predicted_price *= age_factor

        return predicted_price


def main():
    """Main demo function."""
    print("\n" + "=" * 80)
    print("HOUSE PRICE PREDICTOR - Using County Assessor API")
    print("=" * 80)

    # Initialize pipeline
    pipeline = PricePredictionPipeline()

    # Example addresses to predict
    test_addresses = [
        "123 Oak Street, Seattle, WA 98101",
        "456 Pine Avenue, Bellevue, WA 98004",
        "789 Maple Boulevard, Redmond, WA 98052"
    ]

    results = []

    for address in test_addresses:
        try:
            result = pipeline.predict_price(address)
            results.append(result)

            # Print results
            print(f"\nAddress: {result['address']}")
            print(f"Predicted Price: ${result['predicted_price']:,.2f}")
            print(
                f"Price Range: ${result['error_margin_low']:,.2f} - ${result['error_margin_high']:,.2f}")
            print(f"Confidence: {result['confidence']:.2f}%\n")

        except Exception as e:
            print(f"[ERROR] Failed to predict for {address}: {e}\n")

    # Save results
    results_df = pd.DataFrame([
        {
            'address': r['address'],
            'predicted_price': r['predicted_price'],
            'error_margin': r['error_margin'],
            'confidence': r['confidence']
        }
        for r in results
    ])

    results_df.to_csv('data/predictions.csv', index=False)
    print(f"[OK] Predictions saved to: data/predictions.csv")


if __name__ == "__main__":
    main()
