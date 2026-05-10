"""Properties API router — endpoints for fetching and managing property data."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from house_price_prediction.api.dependencies import get_brain, get_settings
from house_price_prediction.api.guardrails import RequestGuardrailError, validate_address_payload
from house_price_prediction.application.services.prediction_orchestrator import Brain
from house_price_prediction.config import Settings
from house_price_prediction.domain.contracts.prediction_contracts import (
    AddressPayload,
    NormalizedAddress,
    PropertyFeaturesResponse,
)
from house_price_prediction.infrastructure.providers.resilient import ProviderExecutionError

router = APIRouter(prefix="/v1/properties", tags=["properties"])


@router.post("/normalize", response_model=NormalizedAddress)
async def normalize_address(
    payload: AddressPayload,
    orchestrator: Brain = Depends(get_brain),
    settings: Settings = Depends(get_settings),
) -> NormalizedAddress:
    """Geocode and normalize an address using the configured geocoding provider."""
    try:
        validate_address_payload(payload, settings)
        return orchestrator.normalize_address(payload)
    except RequestGuardrailError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    except ProviderExecutionError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc


@router.get("/{normalized_address_id}/features", response_model=PropertyFeaturesResponse)
async def get_property_features(
    normalized_address_id: UUID,
    orchestrator: Brain = Depends(get_brain),
) -> PropertyFeaturesResponse:
    """
    Fetch exact property features for a normalized address.
    
    This endpoint queries real property data sources (Census enrichment, property databases, etc.)
    and returns the actual property characteristics at that location, including:
    - Physical features (bedrooms, bathrooms, square footage, lot size)
    - Quality metrics (overall quality, condition, year built)
    - Amenities (garage, fireplace, basement, waterfront)
    - Neighborhood signals (Census median values, income, owner occupancy rate)
    
    The returned features include a `feature_source` and `feature_provenance` indicating
    where each feature came from (real data source vs. derived/estimated).
    
    If the address has insufficient data for property features, returns 404.
    """
    features = orchestrator.get_property_features(normalized_address_id)
    if features is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No property features found for normalized address {normalized_address_id}."
        )
    return features

