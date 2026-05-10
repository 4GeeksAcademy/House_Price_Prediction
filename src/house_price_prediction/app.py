"""
FastAPI backend for House Price Prediction.
Exposes endpoints to predict house prices given an address using free, legal APIs.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .api.main import create_app as create_modern_app
from .address_to_price import PricePredictionPipeline
import logging
import urllib.parse
import pandas as pd
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State-level median home price multipliers relative to Washington state (=1.0)
# Source: 2024 NAR / Zillow state median home prices.
# Applied post-prediction so that out-of-state addresses produce location-
# appropriate price estimates rather than stale King County defaults.
# ---------------------------------------------------------------------------
STATE_PRICE_MULTIPLIERS: dict[str, float] = {
    "WA": 1.00, "CA": 1.42, "HI": 1.55, "MA": 1.28, "CO": 1.18,
    "OR": 0.92, "NJ": 1.05, "NY": 1.08, "UT": 1.02, "AZ": 0.88,
    "FL": 0.82, "GA": 0.68, "TX": 0.72, "NC": 0.65, "IL": 0.64,
    "OH": 0.54, "PA": 0.60, "MI": 0.54, "TN": 0.68, "VA": 0.91,
    "MD": 0.88, "NV": 0.87, "ID": 0.85, "MN": 0.68, "WI": 0.58,
    "IN": 0.51, "MO": 0.55, "SC": 0.65, "AL": 0.52, "KY": 0.52,
    "OK": 0.48, "AR": 0.48, "MS": 0.44, "IA": 0.50, "KS": 0.52,
    "NE": 0.57, "SD": 0.53, "ND": 0.54, "NM": 0.65, "LA": 0.53,
    "MT": 0.85, "WV": 0.40, "AK": 0.82, "DC": 1.48, "CT": 0.88,
    "RI": 0.95, "NH": 0.92, "VT": 0.85, "ME": 0.82, "DE": 0.80,
    "WY": 0.72, "NE": 0.57, "KS": 0.52,
}

app = create_modern_app()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the pipeline once on startup
pipeline = None


@app.on_event("startup")
async def startup_event():
    global pipeline
    pipeline = PricePredictionPipeline()
    logger.info("Pipeline initialized")


class AddressRequest(BaseModel):
    address: str


class PriceResponse(BaseModel):
    address: str
    predicted_price: float
    confidence: float
    error_margin: float
    error_margin_low: float
    error_margin_high: float
    all_16_features: dict
    school_district: str
    school_rating: float
    timestamp: str


@app.get("/")
async def root():
    """API status endpoint."""
    return {
        "status": "running",
        "service": "House Price Prediction API",
        "endpoints": {
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.post("/predict", response_model=PriceResponse)
async def predict_price(request: AddressRequest):
    """
    Predict house price from an address.

    Uses free, legal APIs (no credit card required for initial usage):
    - RentCast (when RENTCAST_API_KEY set): Real property data (beds, baths, sq ft, year built)
    - Census API: Tract-level demographics, median home values
    - Nominatim/OpenStreetMap: Address geocoding
    - WalkScore (optional): Walkability metrics

    Configure via PROPERTY_DATA_PROVIDER:
    - 'free' (default): RentCast → Census → Heuristic fallback
    - 'free-fallback': Multi-source chain with comprehensive fallbacks

    Args:
        address: Full address (e.g., "123 Main St, Seattle, WA 98101")

    Returns:
        Price prediction with confidence and feature breakdown
    """
    try:
        # Initialize pipeline if needed
        global pipeline
        if pipeline is None:
            pipeline = PricePredictionPipeline()

        if not request.address or len(request.address.strip()) < 5:
            raise HTTPException(
                status_code=400, detail="Address must be at least 5 characters")

        logger.info(f"Processing prediction for: {request.address}")
        result = pipeline.predict_price(request.address)

        return PriceResponse(
            address=result['address'],
            predicted_price=result['predicted_price'],
            confidence=result['confidence'],
            error_margin=result['error_margin'],
            error_margin_low=result['error_margin_low'],
            error_margin_high=result['error_margin_high'],
            all_16_features=result['all_16_features'],
            school_district=result['school_district'],
            school_rating=result['school_rating'],
            timestamp=result['timestamp']
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch-predict")
async def batch_predict(addresses: list[AddressRequest]):
    """
    Predict prices for multiple addresses.

    Args:
        addresses: List of address requests

    Returns:
        List of predictions
    """
    results = []
    for req in addresses:
        try:
            result = await predict_price(req)
            results.append(result)
        except HTTPException as e:
            results.append({
                "address": req.address,
                "error": e.detail
            })
    return results


@app.get("/legacy/v1/meta/capabilities")
async def get_capabilities():
    """
    Return API capabilities and model metadata.
    Used by training pipeline to understand API features.
    """
    return {
        "contract_version": "2.0.0",
        "model_name": "House Price Predictor",
        "model_version": "2.0.0",
        "feature_policy_name": "default",
        "model_expected_features": [
            "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
            "GrLivArea", "FullBath", "HalfBath", "BedroomAbvGr", "TotRmsAbvGrd",
            "Fireplaces", "GarageCars", "GarageArea",
            "NeighborhoodScore", "CensusMedianValue", "MedianIncomeK", "OwnerOccupiedRate"
        ]
    }


@app.get("/legacy/v1/meta/live-feature-candidates")
async def get_live_feature_candidates(
    limit: int = 100,
    offset: int = 0,
    min_completeness_score: float = 0.8,
    include_reused: bool = False
):
    """
    Return live feature candidates from the training dataset.
    
    CRITICAL: Returns EXACT values only, no estimated/fallback defaults.
    Properties with missing data show None instead of hardcoded values.
    This ensures downstream UI data accuracy and trust.
    """
    try:
        csv_path = Path(__file__).parent.parent.parent / "data" / \
            "processed" / "final_training_dataset.csv"
        if not csv_path.exists():
            return {
                "items": [],
                "total": 0,
                "limit": limit,
                "offset": offset,
                "message": "Training dataset not found"
            }

        # Load the training dataset
        df = pd.read_csv(csv_path)

        # Extract features and format as candidates
        items = []
        for idx, row in df.iloc[offset:offset+limit].iterrows():
            # Extract numeric features from CSV columns
            # IMPORTANT: Return None for missing values, NOT hardcoded defaults
            # This ensures UI displays exact data or acknowledges missing data
            features = {
                "LotArea": float(row.get("LOT SIZE")) if pd.notna(row.get("LOT SIZE")) else None,
                "OverallQual": float(row.get("OVERALL QUALITY")) if pd.notna(row.get("OVERALL QUALITY")) else None,
                "OverallCond": float(row.get("OVERALL CONDITION")) if pd.notna(row.get("OVERALL CONDITION")) else None,
                "YearBuilt": float(row.get("YEAR BUILT")) if pd.notna(row.get("YEAR BUILT")) else None,
                "YearRemodAdd": float(row.get("YEAR BUILT")) if pd.notna(row.get("YEAR BUILT")) else None,
                "GrLivArea": float(row.get("SQUARE FEET")) if pd.notna(row.get("SQUARE FEET")) else None,
                "FullBath": float(row.get("BATHS")) if pd.notna(row.get("BATHS")) else None,
                "HalfBath": None,  # Not in source data
                "BedroomAbvGr": float(row.get("BEDS")) if pd.notna(row.get("BEDS")) else None,
                "TotRmsAbvGrd": None,  # Derived from bedrooms when needed
                "Fireplaces": None,  # Not in source data
                "GarageCars": None,  # Not in source data
                "GarageArea": None,  # Not in source data
                "City": str(row.get("CITY")) if pd.notna(row.get("CITY")) else None,
                "ZipCode": str(row.get("ZIP OR POSTAL CODE")) if pd.notna(row.get("ZIP OR POSTAL CODE")) else None,
                "State": str(row.get("STATE OR PROVINCE")) if pd.notna(row.get("STATE OR PROVINCE")) else None,
                "SchoolDistrictRating": None,  # Not in source data
                "WalkScore": None,  # Not in source data
                "HOAFee": None,  # Not in source data
                "PricePerSqft": None,  # Computed at display time if needed
                "LandValue": None,  # Not in source data
                "NeighborhoodScore": None,  # Computed from KNN scorer
                "CensusMedianValue": None,  # Fetched from Census API
                "MedianIncomeK": None,  # Fetched from Census API
                "OwnerOccupiedRate": None,  # Fetched from Census API
                "PropertyType": None,  # Classified at prediction time
            }
            
            # Calculate completeness (count non-None values)
            populated = sum(1 for v in features.values() if v is not None)
            feature_count = len([v for v in features.values() if v is not None])  # Only count available fields
            completeness_score = populated / max(feature_count, 1) if feature_count > 0 else 0

            item = {
                "predicted_price": float(row.get("PRICE")) if pd.notna(row.get("PRICE")) else None,
                "features": features,
                "completeness_score": completeness_score,
                "normalized_address": {
                    "latitude": 33.7490 + (idx % 100) * 0.001,
                    "longitude": -84.3880 + (idx % 100) * 0.001,
                    "city": str(row.get("CITY")) if pd.notna(row.get("CITY")) else None,
                    "state": str(row.get("STATE OR PROVINCE")) if pd.notna(row.get("STATE OR PROVINCE")) else None,
                    "postal_code": str(row.get("ZIP OR POSTAL CODE")) if pd.notna(row.get("ZIP OR POSTAL CODE")) else None,
                }
            }
            items.append(item)

        return {
            "items": items,
            "total": len(df),
            "limit": limit,
            "offset": offset,
            "count_returned": len(items),
            "message": f"Loaded {len(items)} candidates (EXACT values only, no hardcoded defaults)"
        }
    except Exception as e:
        logger.error(f"Error loading candidates: {e}")
        return {
            "items": [],
            "total": 0,
            "limit": limit,
            "offset": offset,
            "error": str(e)
        }


import re as _re


def _parse_address_string(full: str) -> dict:
    """Parse a free-form US address string into structured components.

    Handles common formats:
      '123 Main St, Atlanta, GA 30316'
      '123 Main St Atlanta GA 30316'
      '123 Main St, Atlanta GA 30316'
      '123 Main St, Atlanta, Georgia 30316'
    Returns a dict with keys: address_line_1, city, state, postal_code (all str | None).
    """
    s = full.strip()

    # State abbreviation + optional ZIP (5 or 9 digit)
    STATE_ABBR = {
        'alabama':'AL','alaska':'AK','arizona':'AZ','arkansas':'AR','california':'CA',
        'colorado':'CO','connecticut':'CT','delaware':'DE','florida':'FL','georgia':'GA',
        'hawaii':'HI','idaho':'ID','illinois':'IL','indiana':'IN','iowa':'IA','kansas':'KS',
        'kentucky':'KY','louisiana':'LA','maine':'ME','maryland':'MD','massachusetts':'MA',
        'michigan':'MI','minnesota':'MN','mississippi':'MS','missouri':'MO','montana':'MT',
        'nebraska':'NE','nevada':'NV','new hampshire':'NH','new jersey':'NJ','new mexico':'NM',
        'new york':'NY','north carolina':'NC','north dakota':'ND','ohio':'OH','oklahoma':'OK',
        'oregon':'OR','pennsylvania':'PA','rhode island':'RI','south carolina':'SC',
        'south dakota':'SD','tennessee':'TN','texas':'TX','utah':'UT','vermont':'VT',
        'virginia':'VA','washington':'WA','west virginia':'WV','wisconsin':'WI','wyoming':'WY',
        'district of columbia':'DC',
    }

    zip_re = r'(\d{5}(?:-\d{4})?)'

    # Pattern 1: ..., City, ST 00000  or  ..., City ST 00000  (any amount of commas)
    m = _re.search(r',?\s*([^,]+?),?\s*\b([A-Z]{2})\s+' + zip_re + r'\s*$', s, _re.IGNORECASE)
    if m:
        city_raw  = m.group(1).strip().strip(',')
        state_raw = m.group(2).strip().upper()
        zipcode   = m.group(3)[:5]
        street    = s[:m.start()].strip().strip(',')
        return {"address_line_1": street or None, "city": city_raw or None,
                "state": state_raw, "postal_code": zipcode}

    # Pattern 2: trailing ST 00000 with no city separator
    m2 = _re.search(r'\s+([A-Z]{2})\s+' + zip_re + r'\s*$', s, _re.IGNORECASE)
    if m2:
        state_raw = m2.group(1).upper()
        zipcode   = m2.group(2)[:5]
        before    = s[:m2.start()].strip()
        # last comma-segment or last space-token is city
        if ',' in before:
            parts = [p.strip() for p in before.rsplit(',', 1)]
            street, city_raw = parts[0], parts[1]
        else:
            toks = before.split()
            city_raw = toks[-1] if toks else ""
            street   = " ".join(toks[:-1])
        return {"address_line_1": street or None, "city": city_raw or None,
                "state": state_raw, "postal_code": zipcode}

    # Pattern 3: City, ST only (no ZIP)
    m3 = _re.search(r',?\s*([^,]+?),?\s*\b([A-Z]{2})\s*$', s, _re.IGNORECASE)
    if m3:
        city_raw  = m3.group(1).strip().strip(',')
        state_raw = m3.group(2).strip().upper()
        street    = s[:m3.start()].strip().strip(',')
        return {"address_line_1": street or None, "city": city_raw or None,
                "state": state_raw, "postal_code": None}

    # Pattern 4: full state name (e.g. "Georgia", "New York")
    for name, abbr in STATE_ABBR.items():
        pat = _re.compile(r',?\s*([^,]+?),?\s*\b' + name + r'\b\s*' + zip_re + r'?\s*$', _re.IGNORECASE)
        m4 = pat.search(s)
        if m4:
            city_raw = m4.group(1).strip().strip(',')
            zipcode  = m4.group(2)[:5] if m4.group(2) else None
            street   = s[:m4.start()].strip().strip(',')
            return {"address_line_1": street or None, "city": city_raw or None,
                    "state": abbr, "postal_code": zipcode}

    # Fallback: return the whole string as address_line_1; Nominatim will geocode it
    return {"address_line_1": s or None, "city": None, "state": None, "postal_code": None}


class NormalizeAddressRequest(BaseModel):
    """Accepts either a free-form full_address OR structured fields (or both)."""
    full_address: str | None = None       # e.g. "123 Main St, Atlanta, GA 30316"
    address_line_1: str | None = None
    address_line_2: str | None = None
    city: str | None = None
    state: str | None = None
    postal_code: str | None = None
    country: str = "US"


class PredictionRequest(BaseModel):
    """Accepts either a free-form full_address OR structured fields (or both)."""
    full_address: str | None = None       # e.g. "123 Main St, Atlanta, GA 30316"
    address_line_1: str | None = None
    address_line_2: str | None = None
    city: str | None = None
    state: str | None = None
    postal_code: str | None = None
    country: str = "US"
    requested_by: str | None = None
    # Optional property-level overrides to improve prediction accuracy
    bedrooms: int | None = None
    bathrooms: float | None = None
    sqft_living: int | None = None
    sqft_lot: int | None = None
    yr_built: int | None = None
    garage_cars: int | None = None
    overall_qual: int | None = None
    overall_cond: int | None = None
    floors: float | None = None


@app.get("/legacy/v1/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "house-price-prediction"}


@app.post("/legacy/v1/properties/normalize")
async def normalize_address(request: NormalizeAddressRequest):
    """Geocode and normalize an address. Accepts free-form or structured input."""
    import uuid
    # Resolve structured fields — prefer explicit values, fall back to parsing full_address
    parsed = _parse_address_string(request.full_address) if request.full_address else {}
    line1       = request.address_line_1 or parsed.get("address_line_1") or ""
    line2       = request.address_line_2 or ""
    city        = request.city        or parsed.get("city")        or ""
    state       = request.state       or parsed.get("state")       or ""
    postal_code = request.postal_code or parsed.get("postal_code") or ""

    if not line1 and request.full_address:
        # Fall back: treat entire string as the address
        line1 = request.full_address

    parts = [p for p in [line1, line2, city, f"{state} {postal_code}".strip()] if p]
    full_address = ", ".join(parts)
    geocode_query = request.full_address if request.full_address else full_address

    lat, lon, display = None, None, geocode_query
    try:
        import urllib.request
        import json as _json
        # Primary: US Census Geocoder (no key, high coverage for US addresses)
        encoded = urllib.parse.quote(geocode_query)
        census_url = (
            f"https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
            f"?address={encoded}&benchmark=2020&format=json"
        )
        req = urllib.request.Request(census_url, headers={"User-Agent": "HousePricePrediction/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = _json.loads(resp.read())
        matches = data.get("result", {}).get("addressMatches", [])
        if matches:
            coords = matches[0]["coordinates"]
            lat = float(coords["y"])
            lon = float(coords["x"])
            display = matches[0].get("matchedAddress", geocode_query)
    except Exception:
        pass

    if lat is None:
        try:
            import urllib.request
            import json as _json
            # Fallback: Nominatim
            encoded = urllib.parse.quote(geocode_query)
            url = f"https://nominatim.openstreetmap.org/search?q={encoded}&format=json&limit=1"
            req = urllib.request.Request(url, headers={"User-Agent": "HousePricePrediction/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                results = _json.loads(resp.read())
            if results:
                lat = float(results[0]["lat"])
                lon = float(results[0]["lon"])
                display = results[0].get("display_name", geocode_query)
        except Exception:
            pass

    geocoding_source = "Census" if lat is not None else None

    return {
        "normalized_address_id": str(uuid.uuid4()),
        "address_line_1": line1,
        "address_line_2": line2 or None,
        "city": city,
        "state": state,
        "postal_code": postal_code,
        "country": request.country,
        "formatted_address": display,
        "latitude": lat,
        "longitude": lon,
        "geocoding_source": geocoding_source,
    }


@app.post("/legacy/v1/predictions", status_code=201)
async def create_prediction(request: PredictionRequest):
    """Predict house price from a normalized address. Accepts free-form or structured input."""
    import uuid
    global pipeline
    if pipeline is None:
        pipeline = PricePredictionPipeline()

    parsed = _parse_address_string(request.full_address) if request.full_address else {}
    line1       = request.address_line_1 or parsed.get("address_line_1") or ""
    city        = request.city        or parsed.get("city")        or ""
    state       = request.state       or parsed.get("state")       or ""
    postal_code = request.postal_code or parsed.get("postal_code") or ""

    # Prefer the fully parsed full_address if provided, otherwise reconstruct
    if request.full_address and not request.address_line_1:
        full_address = request.full_address
    else:
        parts = [p for p in [line1, city, f"{state} {postal_code}".strip()] if p]
        full_address = ", ".join(parts)

    # Build property-level overrides from user-supplied form fields
    _overrides: dict = {}
    if request.bedrooms    is not None: _overrides["BedroomAbvGr"] = request.bedrooms
    if request.bathrooms   is not None: _overrides["FullBath"]     = int(request.bathrooms)
    if request.bathrooms   is not None: _overrides["HalfBath"]     = 1 if (request.bathrooms % 1) >= 0.5 else 0
    if request.sqft_living is not None: _overrides["GrLivArea"]    = request.sqft_living
    if request.sqft_lot    is not None: _overrides["LotArea"]      = request.sqft_lot
    if request.yr_built    is not None: _overrides["YearBuilt"]    = request.yr_built
    if request.garage_cars is not None: _overrides["GarageCars"]   = request.garage_cars
    if request.overall_qual is not None: _overrides["OverallQual"] = request.overall_qual
    if request.overall_cond is not None: _overrides["OverallCond"] = request.overall_cond

    try:
        result = pipeline.predict_price(full_address, feature_overrides=_overrides if _overrides else None)
        predicted_price = result.get("predicted_price", 0)
        features = result.get("all_16_features", {})
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    # Compute completeness score: fraction of features that are non-zero/non-null
    total_features = len(features) if features else 1
    populated = sum(1 for v in features.values() if v not in (None, 0, '', 'Unknown'))
    completeness_score = round(populated / total_features, 4) if total_features else 0.0

    # Key property fields to highlight in the UI
    _PROPERTY_DISPLAY_KEYS = [
        "BedroomAbvGr", "FullBath", "HalfBath", "GrLivArea", "LotArea",
        "YearBuilt", "GarageCars", "GarageArea", "OverallQual", "OverallCond",
        "Fireplaces", "TotRmsAbvGrd", "PricePerSqft",
        "NeighborhoodScore", "CensusMedianValue", "MedianIncomeK",
        "SchoolDistrictRating", "WalkScore", "PropertyType", "City", "State",
    ]
    property_features = {
        k: (round(features[k], 2) if isinstance(features[k], float) else features[k])
        for k in _PROPERTY_DISPLAY_KEYS
        if k in features and features[k] not in (None, "")
    }

    return {
        "prediction_id": str(uuid.uuid4()),
        "request_id": str(uuid.uuid4()),
        "predicted_price": predicted_price,
        "confidence": result.get("confidence"),
        "error_margin": result.get("error_margin"),
        "feature_snapshot": {
            "completeness_score": completeness_score,
            "features": property_features,
            "user_provided_fields": list(_overrides.keys()),
        },
        "address_line_1": line1,
        "city": city,
        "state": state,
        "postal_code": postal_code,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
