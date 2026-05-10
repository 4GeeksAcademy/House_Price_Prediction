"""Meta API router — endpoints for metadata and miscellaneous info."""

from fastapi import APIRouter, Depends

from house_price_prediction.api.dependencies import get_brain
from house_price_prediction.application.services.prediction_orchestrator import Brain
from house_price_prediction.domain.contracts.prediction_contracts import ApiCapabilitiesResponse

router = APIRouter(prefix="/v1/meta", tags=["meta"])


@router.get("/capabilities", response_model=ApiCapabilitiesResponse)
async def get_api_capabilities(
    orchestrator: Brain = Depends(get_brain),
) -> ApiCapabilitiesResponse:
    """Return API capabilities and model metadata for frontend ML contract."""
    return orchestrator.get_api_capabilities()


@router.get("/live-feature-candidates")
async def get_live_feature_candidates(
    limit: int = 100,
    offset: int = 0,
    min_completeness_score: float = 0.8,
    include_reused: bool = False,
    orchestrator: Brain = Depends(get_brain),
):
    """
    Return live feature candidates from recorded predictions.

    CRITICAL: Returns EXACT values only, no estimated/fallback defaults.
    Properties with missing data show None instead of hardcoded values.
    This ensures downstream UI data accuracy and trust.
    """
    return orchestrator.get_live_feature_candidates(
        limit=limit,
        offset=offset,
        min_completeness_score=min_completeness_score,
        include_reused=include_reused,
    )


