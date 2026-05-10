from __future__ import annotations

from datetime import UTC, datetime

from house_price_prediction.domain.contracts.prediction_contracts import (
    NormalizedAddress,
    ProviderResponseContract,
)
from house_price_prediction.infrastructure.providers.census_property_data_client import (
    CensusPropertyDataClient,
)


class StubFallbackPropertyProvider:
    def fetch_property_features(self, normalized_address: NormalizedAddress) -> ProviderResponseContract:
        return ProviderResponseContract(
            provider_name="stub_fallback",
            status="success",
            payload={
                "LotArea": 15000,
                "OverallQual": 9,
                "Neighborhood": normalized_address.postal_code,
                "feature_source": "stub_fallback",
            },
            fetched_at=datetime.now(UTC),
        )


def test_census_provider_backfills_state_context_when_tract_acs_is_missing() -> None:
    client = CensusPropertyDataClient(fallback_provider=StubFallbackPropertyProvider())

    def fake_lookup_census_tract(latitude: float, longitude: float) -> dict[str, str]:
        return {"state": "DC", "county": "001", "tract": "000100", "name": "Test Tract"}

    def fake_fetch_census_context(geography: dict[str, str]) -> dict[str, str]:
        return {}

    client._lookup_census_tract = fake_lookup_census_tract  # type: ignore[method-assign]
    client._fetch_census_context = fake_fetch_census_context  # type: ignore[method-assign]

    normalized_address = NormalizedAddress(
        address_line_1="1600 Pennsylvania Avenue NW",
        address_line_2=None,
        city="Washington",
        state="DC",
        postal_code="20500",
        country="US",
        formatted_address="1600 Pennsylvania Avenue NW, Washington, DC 20500, US",
        latitude=38.8977,
        longitude=-77.0365,
        geocoding_source="free",
    )

    response = client.fetch_property_features(normalized_address)

    payload = response.payload
    assert payload["CensusMedianValue"] == 720000
    assert payload["MedianIncomeK"] == 101.0
    assert payload["OwnerOccupiedRate"] == 0.65
    assert payload["census_median_value_source"] == "state_fallback"
    assert payload["census_median_income_source"] == "state_fallback"
    assert payload["census_owner_occupancy_source"] == "state_default"
    assert payload["Neighborhood"] == "20500"
    assert payload["feature_source"] == "census_context_with_backfill"

    provenance = payload["feature_provenance"]
    assert provenance["used_census"] is True
    assert provenance["used_backfill"] is True
    assert provenance["census_geography"]["state"] == "DC"
