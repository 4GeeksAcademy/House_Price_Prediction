from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from house_price_prediction.domain.contracts.prediction_contracts import (
    NormalizedAddress,
    PredictionDetailResponse,
    PredictionRequestPayload,
    ProviderResponseContract,
)


class PredictionWorkflowCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    correlation_id: UUID
    submitted_at: datetime
    payload: PredictionRequestPayload


class PredictionWorkflowResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    prediction_id: UUID
    correlation_id: UUID
    status: str
    predicted_price: float
    currency: str
    confidence_score: float | None
    model_name: str
    model_version: str
    feature_snapshot_id: UUID
    normalized_address: NormalizedAddress
    submitted_at: datetime
    generated_at: datetime
    was_reused: bool
    source_prediction_id: UUID | None
    selected_feature_policy_name: str | None = None
    selected_feature_policy_version: str | None = None
    key_features: dict[str, Any] = Field(default_factory=dict)
    feature_source: str | None = None


class NormalizationStageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalized_address: NormalizedAddress
    geocoding_provider_response: ProviderResponseContract


class ReuseStageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalized_address_id: UUID
    model_registry_id: UUID
    reusable_prediction: PredictionDetailResponse | None


WorkflowEventName = Literal[
    "prediction_received",
    "address_normalized",
    "reuse_candidate_evaluated",
    "property_enrichment_completed",
    "feature_vector_created",
    "prediction_completed",
    "prediction_failed",
]


class WorkflowEventContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    event_name: WorkflowEventName
    payload: dict[str, Any]
    occurred_at: datetime