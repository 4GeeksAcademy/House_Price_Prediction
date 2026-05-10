"""Policies API router — endpoints for feature policy management."""

from fastapi import APIRouter, Depends, HTTPException, status

from house_price_prediction.api.dependencies import get_brain
from house_price_prediction.application.services.prediction_orchestrator import Brain
from house_price_prediction.domain.contracts.prediction_contracts import (
    FeaturePolicyCatalogResponse,
    FeaturePolicySimulationRequest,
    FeaturePolicySimulationResponse,
)

router = APIRouter(prefix="/v1/policies", tags=["policies"])


@router.get("/feature", response_model=FeaturePolicyCatalogResponse)
@router.get("/feature/catalog", response_model=FeaturePolicyCatalogResponse)
async def get_feature_policies(
    orchestrator: Brain = Depends(get_brain),
) -> FeaturePolicyCatalogResponse:
    """Get the catalog of available feature policies."""
    return orchestrator.get_feature_policy_catalog()


@router.post("/feature/simulate", response_model=FeaturePolicySimulationResponse)
async def simulate_feature_policies(
    request: FeaturePolicySimulationRequest,
    orchestrator: Brain = Depends(get_brain),
) -> FeaturePolicySimulationResponse:
    """Simulate feature policies for a given address."""
    try:
        return orchestrator.simulate_feature_policies(request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc

