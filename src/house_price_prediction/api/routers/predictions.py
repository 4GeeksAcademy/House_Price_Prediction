from __future__ import annotations

from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status

from house_price_prediction.api.dependencies import (
    get_brain,
    get_settings,
)
from house_price_prediction.api.guardrails import (
    RequestGuardrailError,
    validate_address_payload,
)
from house_price_prediction.application.services.prediction_orchestrator import (
    Brain,
)
from house_price_prediction.config import Settings
from house_price_prediction.domain.contracts.prediction_contracts import (
    PredictionDetailResponse,
    PredictionListResponse,
    PredictionRequestPayload,
    PredictionResponse,
    PredictionTraceResponse,
    PredictionWorkflowEventsResponse,
)
from house_price_prediction.infrastructure.model_runtime.predictor import (
    ModelInferenceError,
    ModelNotReadyError,
)
from house_price_prediction.infrastructure.providers.resilient import ProviderExecutionError

router = APIRouter(prefix="/v1/predictions", tags=["predictions"])


@router.post("", response_model=PredictionResponse, status_code=status.HTTP_201_CREATED)
def create_prediction(
    payload: PredictionRequestPayload,
    orchestrator: Brain = Depends(get_brain),
    settings: Settings = Depends(get_settings),
    x_correlation_id: str | None = Header(default=None),
) -> PredictionResponse:
    try:
        validate_address_payload(payload, settings)
        return orchestrator.create_prediction(payload, correlation_id=x_correlation_id)
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


@router.get("/{prediction_id}", response_model=PredictionDetailResponse)
def get_prediction(
    prediction_id: UUID,
    orchestrator: Brain = Depends(get_brain),
) -> PredictionDetailResponse:
    prediction = orchestrator.get_prediction_detail(prediction_id)
    if prediction is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found.")
    return prediction


@router.get("/{prediction_id}/trace", response_model=PredictionTraceResponse)
def get_prediction_trace(
    prediction_id: UUID,
    orchestrator: Brain = Depends(get_brain),
) -> PredictionTraceResponse:
    trace = orchestrator.get_prediction_trace(prediction_id)
    if trace is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction trace not found.")
    return trace


@router.get("/{prediction_id}/events", response_model=PredictionWorkflowEventsResponse)
def get_prediction_workflow_events(
    prediction_id: UUID,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    event_name: str | None = Query(default=None),
    sort: Literal["asc", "desc"] = Query(default="asc"),
    orchestrator: Brain = Depends(get_brain),
) -> PredictionWorkflowEventsResponse:
    events = orchestrator.get_prediction_workflow_events(
        prediction_id=prediction_id,
        limit=limit,
        offset=offset,
        event_name=event_name,
        sort=sort,
    )
    if events is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found.")
    return events


@router.get("", response_model=PredictionListResponse)
def list_predictions(
    limit: int = Query(default=10, ge=1, le=50),
    offset: int = Query(default=0, ge=0),
    orchestrator: Brain = Depends(get_brain),
) -> PredictionListResponse:
    return orchestrator.list_recent_predictions(limit=limit, offset=offset)