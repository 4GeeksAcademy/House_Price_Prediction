from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from house_price_prediction.api.dependencies import get_brain, get_settings
from house_price_prediction.api.guardrails import RequestGuardrailError, validate_address_payload
from house_price_prediction.application.services.prediction_orchestrator import Brain
from house_price_prediction.config import Settings
from house_price_prediction.domain.contracts.prediction_contracts import (
    AddressBaselineResponse,
    AddressBaselineRequest,
    BaselineScenarioCatalogResponse,
    FullAuditRequest,
    FullAuditResponse,
    ScenarioBatchPipelineRequest,
    ScenarioBatchPipelineResponse,
)
from house_price_prediction.infrastructure.model_runtime.predictor import (
    ModelInferenceError,
    ModelNotReadyError,
)
from house_price_prediction.infrastructure.providers.resilient import ProviderExecutionError

router = APIRouter(prefix="/v1/validation", tags=["validation"])


@router.get("/scenarios", response_model=BaselineScenarioCatalogResponse)
def list_validation_scenarios(
    brain: Brain = Depends(get_brain),
) -> BaselineScenarioCatalogResponse:
    return brain.get_baseline_scenarios()


@router.post(
    "/address-baseline",
    response_model=AddressBaselineResponse,
    status_code=status.HTTP_200_OK,
)
def generate_address_baseline(
    payload: AddressBaselineRequest,
    brain: Brain = Depends(get_brain),
    settings: Settings = Depends(get_settings),
) -> AddressBaselineResponse:
    try:
        validate_address_payload(payload, settings)
        return brain.generate_address_baseline(
            payload,
            expectations=payload.expectations,
            policy_name_override=payload.preferred_policy_name,
            feature_overrides=payload.feature_overrides,
        )
    except RequestGuardrailError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except ModelInferenceError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    except ProviderExecutionError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@router.post(
    "/full-audit",
    response_model=FullAuditResponse,
    status_code=status.HTTP_200_OK,
)
def run_full_audit(
    payload: FullAuditRequest,
    brain: Brain = Depends(get_brain),
    settings: Settings = Depends(get_settings),
) -> FullAuditResponse:
    try:
        validate_address_payload(payload, settings)
        return brain.run_full_audit(payload)
    except RequestGuardrailError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except ModelInferenceError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    except ProviderExecutionError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@router.post(
    "/run-scenario-batch",
    response_model=ScenarioBatchPipelineResponse,
    status_code=status.HTTP_200_OK,
)
def run_scenario_batch(
    payload: ScenarioBatchPipelineRequest,
    brain: Brain = Depends(get_brain),
) -> ScenarioBatchPipelineResponse:
    """Run one or more registered scenarios through the full audit pipeline.

    Pass ``scenario_ids`` to select a subset; omit (or send ``null``) to run
    all registered scenarios.
    """
    return brain.run_scenario_batch(payload)

