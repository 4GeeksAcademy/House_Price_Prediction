"""Dashboard API router — endpoints for UI bootstrap and status."""

from fastapi import APIRouter, Depends

from house_price_prediction.api.dependencies import get_brain, get_settings
from house_price_prediction.application.services.prediction_orchestrator import Brain
from house_price_prediction.config import Settings
from house_price_prediction.domain.contracts.prediction_contracts import (
    DashboardBootstrapResponse,
)

router = APIRouter(prefix="/v1/dashboard", tags=["dashboard"])


@router.get("/bootstrap", response_model=DashboardBootstrapResponse)
async def bootstrap_dashboard(
    orchestrator: Brain = Depends(get_brain),
    settings: Settings = Depends(get_settings),
) -> DashboardBootstrapResponse:
    """
    Bootstrap UI state with runtime metadata, provider policy, and recent predictions.
    
    Returns a single-call response containing:
    - Runtime metadata (model, environment, version)
    - Provider policy configuration
    - Recent predictions (last 5)
    - Event summary for observability
    - Link templates for frontend navigation
    """
    return orchestrator.get_dashboard_bootstrap()
