from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx

from house_price_prediction.domain.contracts.prediction_contracts import (
    NormalizedAddress,
    ProviderResponseContract,
)
from house_price_prediction.infrastructure.providers.property_type_classifier import (
    classify_property_type,
)
from house_price_prediction.infrastructure.providers.resilient import NonRetryableProviderError


class RentcastPropertyDataClient:
    """True property-data API client.

    Uses RentCast's address search endpoint to pull real property facts
    (beds, baths, living area, lot area, year built) instead of inferring them.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.rentcast.io/v1",
        timeout_seconds: float = 10.0,
    ) -> None:
        if not api_key or not api_key.strip():
            raise NonRetryableProviderError("Missing RENTCAST_API_KEY for true property API provider.")
        self._api_key = api_key.strip()
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    def fetch_property_features(
        self,
        normalized_address: NormalizedAddress,
    ) -> ProviderResponseContract:
        search_address = normalized_address.formatted_address or self._compose_address(normalized_address)

        response = httpx.get(
            f"{self._base_url}/properties",
            params={"address": search_address},
            headers={
                "Accept": "application/json",
                "X-Api-Key": self._api_key,
                "User-Agent": "house-price-prediction-backend/0.1",
            },
            timeout=self._timeout_seconds,
        )

        if response.status_code in {401, 403}:
            raise NonRetryableProviderError("RentCast API authentication failed (check RENTCAST_API_KEY).")
        response.raise_for_status()

        record = self._pick_record(response.json())
        if record is None:
            raise NonRetryableProviderError(
                f"RentCast returned no property record for address '{search_address}'."
            )

        payload = self._map_record_to_features(record, normalized_address)

        return ProviderResponseContract(
            provider_name="rentcast_property_data",
            status="success",
            payload=payload,
            fetched_at=datetime.now(UTC),
        )

    @staticmethod
    def _compose_address(normalized_address: NormalizedAddress) -> str:
        parts = [
            normalized_address.address_line_1,
            normalized_address.city,
            normalized_address.state,
            normalized_address.postal_code,
            normalized_address.country,
        ]
        return ", ".join(str(p).strip() for p in parts if p)

    @staticmethod
    def _pick_record(data: Any) -> dict[str, Any] | None:
        if isinstance(data, list):
            return data[0] if data else None
        if isinstance(data, dict):
            if isinstance(data.get("results"), list) and data["results"]:
                first = data["results"][0]
                return first if isinstance(first, dict) else None
            if isinstance(data.get("data"), list) and data["data"]:
                first = data["data"][0]
                return first if isinstance(first, dict) else None
            return data
        return None

    @staticmethod
    def _num(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _first_num(cls, record: dict[str, Any], keys: tuple[str, ...]) -> float | None:
        for key in keys:
            if key in record:
                parsed = cls._num(record.get(key))
                if parsed is not None:
                    return parsed
        return None

    @classmethod
    def _map_record_to_features(
        cls,
        record: dict[str, Any],
        normalized_address: NormalizedAddress,
    ) -> dict[str, Any]:
        bedrooms = cls._first_num(record, ("bedrooms", "beds", "bedroomsTotal"))
        bathrooms = cls._first_num(record, ("bathrooms", "baths", "fullBathrooms"))
        living_area = cls._first_num(record, ("squareFootage", "sqft", "livingArea", "buildingArea"))
        lot_area = cls._first_num(record, ("lotSize", "lotSizeSquareFeet", "lotSizeSqFt", "lotArea"))
        year_built = cls._first_num(record, ("yearBuilt", "builtYear"))
        stories = cls._first_num(record, ("stories", "levels"))
        garage_spaces = cls._first_num(record, ("garageSpaces", "garageSpacesTotal"))

        full_bath = int(bathrooms) if bathrooms is not None else None
        half_bath = None
        if bathrooms is not None and full_bath is not None:
            frac = bathrooms - float(full_bath)
            if frac >= 0.49:
                half_bath = 1
            else:
                half_bath = 0

        tot_rooms = None
        if bedrooms is not None:
            tot_rooms = max(int(round(bedrooms)) + 2, 5)

        payload: dict[str, Any] = {
            "LotArea": int(round(lot_area)) if lot_area is not None else None,
            "OverallQual": 6,
            "OverallCond": 6,
            "YearBuilt": int(round(year_built)) if year_built is not None else None,
            "YearRemodAdd": int(round(year_built)) if year_built is not None else None,
            "GrLivArea": int(round(living_area)) if living_area is not None else None,
            "BasementSF": None,
            "FullBath": full_bath,
            "HalfBath": half_bath,
            "BedroomAbvGr": int(round(bedrooms)) if bedrooms is not None else None,
            "TotRmsAbvGrd": tot_rooms,
            "Fireplaces": None,
            "GarageCars": int(round(garage_spaces)) if garage_spaces is not None else None,
            "GarageArea": int(round(garage_spaces * 280)) if garage_spaces is not None else None,
            "Waterfront": None,
            "ViewScore": None,
            "HouseStyle": "2Story" if stories is not None and stories >= 2 else "1Story",
            "Neighborhood": normalized_address.postal_code,
            # Context fields intentionally left None here; Census/other enrichers can augment.
            "CensusMedianValue": None,
            "MedianIncomeK": None,
            "OwnerOccupiedRate": None,
            "NeighborhoodScore": None,
            "feature_source": "true_api",
            "feature_provenance": {
                "strategy": "rentcast_true_api",
                "providers": ["rentcast_property_data"],
                "address_used": normalized_address.formatted_address,
                "raw_fields_present": sorted(record.keys()),
            },
        }
        payload["PropertyType"] = classify_property_type(payload)
        return payload