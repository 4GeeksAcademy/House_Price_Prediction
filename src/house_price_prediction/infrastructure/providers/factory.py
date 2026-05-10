from __future__ import annotations

from house_price_prediction.config import Settings
from house_price_prediction.infrastructure.providers.base import (
    GeocodingProvider,
    PropertyDataProvider,
)
from house_price_prediction.infrastructure.providers.census_geocoding_client import (
    CensusGeocodingClient,
)
from house_price_prediction.infrastructure.providers.census_property_data_client import (
    CensusPropertyDataClient,
)
from house_price_prediction.infrastructure.providers.fallback_geocoding_provider import (
    FallbackGeocodingProvider,
)
from house_price_prediction.infrastructure.providers.fallback_property_data_provider import (
    FallbackPropertyDataProvider,
)
from house_price_prediction.infrastructure.providers.fake_geocoding_client import (
    FakeGeocodingClient,
)
from house_price_prediction.infrastructure.providers.fake_property_data_client import (
    FakePropertyDataClient,
)
from house_price_prediction.infrastructure.providers.heuristic_property_data_client import (
    HeuristicPropertyDataClient,
)
from house_price_prediction.infrastructure.providers.nominatim_geocoding_client import (
    NominatimGeocodingClient,
)
from house_price_prediction.infrastructure.providers.rentcast_property_data_client import (
    RentcastPropertyDataClient,
)
from house_price_prediction.infrastructure.providers.resilient import (
    ResilientGeocodingProvider,
    ResilientPropertyDataProvider,
)
from house_price_prediction.infrastructure.providers.walk_score_enrichment_client import (
    WalkScoreEnrichmentClient,
)


def create_property_data_provider(settings: Settings) -> PropertyDataProvider:
    provider_name = settings.property_data_provider.strip().lower()
    if provider_name == "fake":
        return ResilientPropertyDataProvider(
            provider_name=provider_name,
            delegate=FakePropertyDataClient(),
            timeout_seconds=settings.provider_timeout_seconds,
            max_retries=settings.provider_max_retries,
        )
    if provider_name == "free":
        # Production chain: RentCast (real property) → Census (tract context) → Heuristic fallback
        fallback_provider: PropertyDataProvider
        if settings.rentcast_api_key:
            fallback_provider = FallbackPropertyDataProvider(
                providers=(
                    RentcastPropertyDataClient(
                        api_key=settings.rentcast_api_key,
                        base_url=settings.rentcast_api_base_url,
                        timeout_seconds=settings.provider_timeout_seconds,
                    ),
                    CensusPropertyDataClient(
                        fallback_provider=HeuristicPropertyDataClient()
                    ),
                    HeuristicPropertyDataClient(),
                )
            )
        else:
            # No RentCast: Census → Heuristic
            fallback_provider = FallbackPropertyDataProvider(
                providers=(
                    CensusPropertyDataClient(
                        fallback_provider=HeuristicPropertyDataClient()
                    ),
                    HeuristicPropertyDataClient(),
                )
            )

        free_chain: PropertyDataProvider = fallback_provider
        if settings.walkscore_api_key:
            free_chain = WalkScoreEnrichmentClient(
                free_chain,
                settings.walkscore_api_key,
                timeout_seconds=settings.provider_timeout_seconds,
            )
        return ResilientPropertyDataProvider(
            provider_name=provider_name,
            delegate=free_chain,
            timeout_seconds=settings.provider_timeout_seconds,
            max_retries=settings.provider_max_retries,
        )
    if provider_name == "free-fallback":
        census_backfill: PropertyDataProvider
        if settings.rentcast_api_key:
            census_backfill = FallbackPropertyDataProvider(
                providers=(
                    RentcastPropertyDataClient(
                        api_key=settings.rentcast_api_key,
                        base_url=settings.rentcast_api_base_url,
                        timeout_seconds=settings.provider_timeout_seconds,
                    ),
                    HeuristicPropertyDataClient(),
                )
            )
        else:
            census_backfill = HeuristicPropertyDataClient()

        providers: list[PropertyDataProvider] = [
            CensusPropertyDataClient(fallback_provider=census_backfill),
            HeuristicPropertyDataClient(),
            FakePropertyDataClient(),
        ]
        fallback_chain: PropertyDataProvider = FallbackPropertyDataProvider(
            providers=tuple(providers)
        )
        if settings.walkscore_api_key:
            fallback_chain = WalkScoreEnrichmentClient(
                fallback_chain,
                settings.walkscore_api_key,
                timeout_seconds=settings.provider_timeout_seconds,
            )
        return ResilientPropertyDataProvider(
            provider_name=provider_name,
            delegate=fallback_chain,
            timeout_seconds=settings.provider_timeout_seconds,
            max_retries=settings.provider_max_retries,
        )
    raise ValueError(f"Unsupported property data provider '{settings.property_data_provider}'.")


def create_geocoding_provider(settings: Settings) -> GeocodingProvider:
    provider_name = settings.geocoding_provider.strip().lower()
    if provider_name == "fake":
        return ResilientGeocodingProvider(
            provider_name=provider_name,
            delegate=FakeGeocodingClient(),
            timeout_seconds=settings.provider_timeout_seconds,
            max_retries=settings.provider_max_retries,
        )
    if provider_name == "free":
        # Nominatim → Census → Fake (state centroid) for resilience
        return ResilientGeocodingProvider(
            provider_name=provider_name,
            delegate=FallbackGeocodingProvider(
                providers=(
                    NominatimGeocodingClient(),
                    CensusGeocodingClient(),
                    FakeGeocodingClient(),  # Fallback to state centroid if primary sources fail
                )
            ),
            timeout_seconds=settings.provider_timeout_seconds,
            max_retries=settings.provider_max_retries,
        )
    if provider_name == "free-fallback":
        return ResilientGeocodingProvider(
            provider_name=provider_name,
            delegate=FallbackGeocodingProvider(
                providers=(
                    NominatimGeocodingClient(),
                    CensusGeocodingClient(),
                    FakeGeocodingClient(),
                )
            ),
            timeout_seconds=settings.provider_timeout_seconds,
            max_retries=settings.provider_max_retries,
        )
    raise ValueError(f"Unsupported geocoding provider '{settings.geocoding_provider}'.")