from __future__ import annotations

from datetime import UTC, datetime

from house_price_prediction.domain.contracts.prediction_contracts import (
    AddressPayload,
    GeocodingResultContract,
    NormalizedAddress,
    ProviderResponseContract,
)

# Approximate geographic centroid for each US state (lat, lon).
# Used as last-resort fallback coordinates when live geocoding fails so that
# Census property enrichment can still attempt a region-level ACS lookup.
_STATE_CENTROIDS: dict[str, tuple[float, float]] = {
    "AL": (32.806671, -86.791130), "AK": (61.370716, -152.404419),
    "AZ": (33.729759, -111.431221), "AR": (34.969704, -92.373123),
    "CA": (36.116203, -119.681564), "CO": (39.059811, -105.311104),
    "CT": (41.597782, -72.755371), "DE": (39.318523, -75.507141),
    "FL": (27.766279, -81.686783), "GA": (33.040619, -83.643074),
    "HI": (21.094318, -157.498337), "ID": (44.240459, -114.478828),
    "IL": (40.349457, -88.986137), "IN": (39.849426, -86.258278),
    "IA": (42.011539, -93.210526), "KS": (38.526600, -96.726486),
    "KY": (37.668140, -84.670067), "LA": (31.169960, -91.867805),
    "ME": (44.693947, -69.381927), "MD": (39.063946, -76.802101),
    "MA": (42.230171, -71.530106), "MI": (43.326618, -84.536095),
    "MN": (45.694454, -93.900192), "MS": (32.741646, -89.678696),
    "MO": (38.456085, -92.288368), "MT": (46.921925, -110.454353),
    "NE": (41.125370, -98.268082), "NV": (38.313515, -117.055374),
    "NH": (43.452492, -71.563896), "NJ": (40.298904, -74.521011),
    "NM": (34.840515, -106.248482), "NY": (42.165726, -74.948051),
    "NC": (35.630066, -79.806419), "ND": (47.528912, -99.784012),
    "OH": (40.388783, -82.764915), "OK": (35.565342, -96.928917),
    "OR": (44.572021, -122.070938), "PA": (40.590752, -77.209755),
    "RI": (41.680893, -71.511780), "SC": (33.856892, -80.945007),
    "SD": (44.299782, -99.438828), "TN": (35.747845, -86.692345),
    "TX": (31.054487, -97.563461), "UT": (40.150032, -111.862434),
    "VT": (44.045876, -72.710686), "VA": (37.769337, -78.169968),
    "WA": (47.400902, -121.490494), "WV": (38.491226, -80.954453),
    "WI": (44.268543, -89.616508), "WY": (42.755966, -107.302490),
    "DC": (38.897438, -77.026817),
}
# Fall back to centre of contiguous US when state is unknown
_US_CENTER: tuple[float, float] = (39.5, -98.35)


class FakeGeocodingClient:
    def normalize(self, address_payload: AddressPayload) -> GeocodingResultContract:
        normalized_line_1 = " ".join(address_payload.address_line_1.strip().upper().split())
        normalized_city = " ".join(address_payload.city.strip().upper().split())
        normalized_state = address_payload.state.strip().upper()
        normalized_postal_code = address_payload.postal_code.strip().upper()
        normalized_country = address_payload.country.strip().upper()
        normalized_line_2 = (
            " ".join(address_payload.address_line_2.strip().upper().split())
            if address_payload.address_line_2
            else None
        )

        # Provide state-centroid coordinates so that Census property enrichment
        # can still attempt a region-level ACS lookup even when live geocoding
        # services are unavailable.
        lat, lon = _STATE_CENTROIDS.get(normalized_state, _US_CENTER)

        normalized_address = NormalizedAddress(
            address_line_1=normalized_line_1,
            address_line_2=normalized_line_2,
            city=normalized_city,
            state=normalized_state,
            postal_code=normalized_postal_code,
            country=normalized_country,
            formatted_address=(
                f"{normalized_line_1}, {normalized_city}, {normalized_state} "
                f"{normalized_postal_code}, {normalized_country}"
            ),
            latitude=lat,
            longitude=lon,
            geocoding_source="fake",
        )
        return GeocodingResultContract(
            normalized_address=normalized_address,
            provider_response=ProviderResponseContract(
                provider_name="fake_geocoding",
                status="success",
                payload={
                    "query": {
                        "address_line_1": address_payload.address_line_1,
                        "address_line_2": address_payload.address_line_2,
                        "city": address_payload.city,
                        "state": address_payload.state,
                        "postal_code": address_payload.postal_code,
                        "country": address_payload.country,
                    },
                    "normalized": normalized_address.model_dump(),
                },
                fetched_at=datetime.now(UTC),
            ),
        )