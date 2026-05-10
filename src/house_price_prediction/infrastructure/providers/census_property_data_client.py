from __future__ import annotations

from datetime import UTC, datetime

import httpx

from house_price_prediction.domain.contracts.prediction_contracts import (
    NormalizedAddress,
    ProviderResponseContract,
)
from house_price_prediction.infrastructure.providers.base import PropertyDataProvider
from house_price_prediction.infrastructure.providers.property_type_classifier import (
    classify_property_type,
)
from house_price_prediction.infrastructure.providers.resilient import NonRetryableProviderError
from house_price_prediction.telemetry import get_logger

logger = get_logger(__name__)


class CensusPropertyDataClient:
    _STATE_FIPS_TO_ABBR: dict[str, str] = {
        "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
        "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI",
        "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
        "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
        "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
        "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
        "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
        "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
        "54": "WV", "55": "WI", "56": "WY",
    }
    _STATE_MEDIAN_HOME_VALUE: dict[str, int] = {
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
    _STATE_MEDIAN_INCOME: dict[str, int] = {
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
    _DEFAULT_OWNER_OCCUPIED_RATE: float = 0.65

    def __init__(
        self,
        fallback_provider: PropertyDataProvider | None = None,
        geocoder_base_url: str = "https://geocoding.geo.census.gov/geocoder",
        census_api_base_url: str = "https://api.census.gov/data/2022/acs/acs5",
        timeout_seconds: float = 10.0,
    ) -> None:
        self._fallback_provider = fallback_provider
        self._geocoder_base_url = geocoder_base_url.rstrip("/")
        self._census_api_base_url = census_api_base_url.rstrip("/")
        # Per-request timeout; the outer ResilientPropertyDataProvider enforces
        # an overall wall-clock budget.  Census makes two sequential HTTP calls
        # (tract lookup + ACS fetch), so each should have its own timeout.
        self._timeout_seconds = timeout_seconds

    def fetch_property_features(
        self,
        normalized_address: NormalizedAddress,
    ) -> ProviderResponseContract:
        fallback_response = (
            self._fallback_provider.fetch_property_features(normalized_address)
            if self._fallback_provider is not None
            else None
        )

        try:
            if normalized_address.latitude is None or normalized_address.longitude is None:
                raise NonRetryableProviderError("Coordinates are required for census context enrichment.")

            geography = self._lookup_census_tract(
                latitude=normalized_address.latitude,
                longitude=normalized_address.longitude,
            )
            census_context = self._fetch_census_context(geography)
            derived_features = self._derive_features(census_context, geography)

            used_backfill = fallback_response is not None
            payload = dict(fallback_response.payload) if used_backfill else {}
            payload.update(derived_features)
            payload["Neighborhood"] = normalized_address.postal_code or payload.get("Neighborhood")
            payload["feature_source"] = (
                "census_context_with_backfill" if used_backfill else "census_context"
            )
            payload["feature_provenance"] = self._build_feature_provenance(
                fallback_response=fallback_response,
                geography=geography,
                used_census=True,
            )

            return ProviderResponseContract(
                provider_name=("census_context_with_backfill" if fallback_response is not None else "census_context"),
                status="success",
                payload=payload,
                fetched_at=datetime.now(UTC),
            )
        except Exception as census_exc:
            if fallback_response is not None:
                logger.warning(
                    "census_property_enrichment_failed reason=%s fallback=%s "
                    "lat=%s lon=%s — using heuristic data instead",
                    type(census_exc).__name__,
                    fallback_response.provider_name,
                    normalized_address.latitude,
                    normalized_address.longitude,
                )
                payload = dict(fallback_response.payload)
                payload["Neighborhood"] = normalized_address.postal_code or payload.get("Neighborhood")
                payload["feature_provenance"] = self._build_feature_provenance(
                    fallback_response=fallback_response,
                    geography=None,
                    used_census=False,
                )
                return ProviderResponseContract(
                    provider_name=f"{fallback_response.provider_name}_fallback",
                    status="success",
                    payload=payload,
                    fetched_at=datetime.now(UTC),
                )
            raise

    def _lookup_census_tract(self, latitude: float, longitude: float) -> dict[str, str]:
        response = httpx.get(
            f"{self._geocoder_base_url}/geographies/coordinates",
            params={
                "x": longitude,
                "y": latitude,
                "benchmark": "Public_AR_Current",
                "vintage": "Current_Current",
                "layers": "Census Tracts",
                "format": "json",
            },
            headers={"User-Agent": "house-price-prediction-backend/0.1"},
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        geographies = response.json().get("result", {}).get("geographies", {})
        tract_rows = geographies.get("Census Tracts") or geographies.get("2020 Census Tracts") or []
        if not tract_rows:
            raise RuntimeError("Census geography lookup returned no tract information.")
        tract = tract_rows[0]
        return {
            "state": str(tract.get("STATE") or tract.get("STATEFP")),
            "county": str(tract.get("COUNTY") or tract.get("COUNTYFP")),
            "tract": str(tract.get("TRACT") or tract.get("TRACTCE")),
            "name": str(tract.get("NAME") or tract.get("BASENAME") or "Unknown Tract"),
        }

    def _fetch_census_context(self, geography: dict[str, str]) -> dict[str, str]:
        # Core housing variables
        #   B25077_001E  median home value
        #   B25035_001E  median year structure built
        #   B25018_001E  median number of rooms
        # Extended neighborhood-quality variables
        #   B19013_001E  median household income
        #   B25064_001E  median gross rent
        #   B25003_001E  total housing units (all tenures)
        #   B25003_002E  owner-occupied housing units
        #   B25071_001E  gross rent as % of household income (rent-burden)
        #   B01003_001E  total population
        acs_fields = (
            "NAME"
            ",B25077_001E,B25035_001E,B25018_001E"
            ",B19013_001E,B25064_001E"
            ",B25003_001E,B25003_002E"
            ",B25071_001E,B01003_001E"
        )
        response = httpx.get(
            self._census_api_base_url,
            params={
                "get": acs_fields,
                "for": f"tract:{geography['tract']}",
                "in": f"state:{geography['state']} county:{geography['county']}",
            },
            headers={"User-Agent": "house-price-prediction-backend/0.1"},
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        rows = response.json()
        if len(rows) < 2:
            raise RuntimeError("ACS returned no tract-level housing context.")
        headers = rows[0]
        values = rows[1]
        return dict(zip(headers, values, strict=True))

    def _derive_features(
        self,
        census_context: dict[str, str],
        geography: dict[str, str],
    ) -> dict[str, int | float | str | None]:
        """
        Enrich property features with Census data.
        
        CRITICAL: This function must ONLY provide Census-sourced enrichment
        (CensusMedianValue, MedianIncomeK, OwnerOccupiedRate). It must NOT derive
        or estimate physical property features (bedrooms, sqft, lot size, etc.).
        
        Physical features must come from actual property data sources (real databases,
        user input, or third-party APIs). Using Census demographic data to guess
        property characteristics degrades data accuracy and breaks UI trust.
        """
        state = self._resolve_state_abbr(geography.get("state"))

        # Core economic signals from Census tract, with state-level fallback when
        # tract-level ACS values are unavailable.
        median_home_value = self._safe_int(census_context.get("B25077_001E"))
        median_income = self._safe_int(census_context.get("B19013_001E"))
        median_rent = self._safe_int(census_context.get("B25064_001E"))
        total_units = self._safe_int(census_context.get("B25003_001E"))
        owner_units = self._safe_int(census_context.get("B25003_002E"))
        rent_burden_pct = self._safe_float(census_context.get("B25071_001E"))
        tract_population = self._safe_int(census_context.get("B01003_001E"))

        # Compute owner occupancy rate (neighborhood signal only, not for individual property)
        owner_rate: float | None = (
            owner_units / total_units if total_units and owner_units else None
        )

        median_home_value_source = "tract_acs"
        if median_home_value is None or median_home_value <= 0:
            median_home_value = self._STATE_MEDIAN_HOME_VALUE.get(state)
            median_home_value_source = (
                "state_fallback" if median_home_value is not None else "unavailable"
            )

        median_income_source = "tract_acs"
        if median_income is None or median_income <= 0:
            median_income = self._STATE_MEDIAN_INCOME.get(state)
            median_income_source = (
                "state_fallback" if median_income is not None else "unavailable"
            )

        owner_rate_source = "tract_acs"
        if owner_rate is None:
            owner_rate = self._DEFAULT_OWNER_OCCUPIED_RATE
            owner_rate_source = "state_default"

        # Persons-per-unit is a neighborhood density signal, NOT an individual property feature
        persons_per_unit: float | None = (
            tract_population / total_units if tract_population and total_units else None
        )

        # Return ONLY Census enrichment data; physical features come from fallback provider
        enrichment: dict[str, int | float | str | None] = {
            # ── Census-sourced neighborhood economic context ──────────────
            "CensusMedianValue": median_home_value,
            "MedianIncomeK": round(median_income / 1000.0, 1) if median_income else None,
            "OwnerOccupiedRate": round(owner_rate, 3) if owner_rate is not None else None,
            # NeighborhoodScore is computed at inference time by NeighborhoodScoreService (KNN);
            # Census provider does not compute it.
            "NeighborhoodScore": None,
            "census_median_value_source": median_home_value_source,
            "census_median_income_source": median_income_source,
            "census_owner_occupancy_source": owner_rate_source,
            # ── metadata for observability only (not consumed by model) ────
            "census_median_income": median_income,
            "census_median_rent": median_rent,
            "census_owner_occupancy_rate": round(owner_rate, 3) if owner_rate is not None else None,
            "census_rent_burden_pct": rent_burden_pct,
            "census_tract_population": tract_population,
            "census_persons_per_unit": round(persons_per_unit, 2) if persons_per_unit is not None else None,
        }

        return enrichment

    @classmethod
    def _resolve_state_abbr(cls, state: str | None) -> str:
        raw_state = (state or "").strip().upper()
        if len(raw_state) == 2 and raw_state.isalpha():
            return raw_state
        if raw_state.isdigit():
            return cls._STATE_FIPS_TO_ABBR.get(raw_state.zfill(2), raw_state)
        try:
            numeric_state = int(float(raw_state))
            return cls._STATE_FIPS_TO_ABBR.get(f"{numeric_state:02d}", raw_state)
        except (TypeError, ValueError):
            pass
        return raw_state

    @staticmethod
    def _build_feature_provenance(
        fallback_response: ProviderResponseContract | None,
        geography: dict[str, str] | None,
        used_census: bool,
    ) -> dict[str, object]:
        providers = ["census_context"] if used_census else []
        fallback_provider_name = fallback_response.provider_name if fallback_response is not None else None
        if fallback_provider_name is not None:
            providers.append(fallback_provider_name)

        provenance: dict[str, object] = {
            "strategy": "census_context" if used_census else "fallback_only",
            "providers": providers,
            "used_census": used_census,
            "used_backfill": fallback_response is not None,
        }
        if fallback_provider_name is not None:
            provenance["backfill_provider"] = fallback_provider_name
        if geography is not None:
            provenance["census_geography"] = geography
        return provenance

    @staticmethod
    def _safe_int(value: str | None) -> int | None:
        if value is None or value.startswith("-"):
            return None
        try:
            return int(float(value))
        except ValueError:
            return None

    @staticmethod
    def _safe_float(value: str | None) -> float | None:
        if value is None or value.startswith("-"):
            return None
        try:
            return float(value)
        except ValueError:
            return None

    @staticmethod
    def _clamp(value: int, minimum: int, maximum: int) -> int:
        return max(minimum, min(value, maximum))