from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Feature override validation constants (shared across multiple request types)
# ---------------------------------------------------------------------------

_NUMERIC_OVERRIDE_FEATURES: frozenset[str] = frozenset({
    "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
    "GrLivArea", "FullBath", "HalfBath", "BedroomAbvGr", "TotRmsAbvGrd",
    "Fireplaces", "GarageCars", "GarageArea", "NeighborhoodScore",
    "CensusMedianValue", "MedianIncomeK", "OwnerOccupiedRate",
})

_NUMERIC_OVERRIDE_BOUNDS: dict[str, tuple[float, float]] = {
    "LotArea": (100, 2_000_000),
    "OverallQual": (1, 10),
    "OverallCond": (1, 10),
    "YearBuilt": (1800, 2030),
    "YearRemodAdd": (1800, 2030),
    "GrLivArea": (100, 30_000),
    "FullBath": (0, 20),
    "HalfBath": (0, 10),
    "BedroomAbvGr": (0, 20),
    "TotRmsAbvGrd": (1, 30),
    "Fireplaces": (0, 10),
    "GarageCars": (0, 10),
    "GarageArea": (0, 10_000),
    "NeighborhoodScore": (0.0, 100.0),
    "CensusMedianValue": (10_000, 5_000_000),
    "MedianIncomeK": (5.0, 1_000.0),
    "OwnerOccupiedRate": (0.0, 1.0),
}


def _validate_feature_overrides_dict(overrides: dict[str, Any] | None) -> list[str]:
    """Return a list of validation error messages for the given feature_overrides dict.

    Returns an empty list when all values are valid.
    """
    if overrides is None:
        return []
    errors: list[str] = []
    if len(overrides) > 50:
        return ["feature_overrides may not contain more than 50 keys."]
    for key, value in overrides.items():
        if key not in _NUMERIC_OVERRIDE_FEATURES:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(
                f"feature_overrides[{key!r}] must be a number, got {type(value).__name__!r}."
            )
            continue
        if math.isnan(value) or math.isinf(value):
            errors.append(f"feature_overrides[{key!r}] must be a finite number.")
            continue
        lo, hi = _NUMERIC_OVERRIDE_BOUNDS[key]
        if not (lo <= value <= hi):
            errors.append(
                f"feature_overrides[{key!r}] = {value} is outside allowed range [{lo}, {hi}]."
            )
    return errors


class AddressPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    address_line_1: str = Field(min_length=1, max_length=120)
    address_line_2: str | None = Field(default=None, max_length=120)
    city: str = Field(min_length=1, max_length=80)
    state: str = Field(min_length=2, max_length=30)
    postal_code: str = Field(min_length=3, max_length=20)
    country: str = Field(default="US", min_length=2, max_length=60)

    @field_validator("address_line_1", "city", mode="before")
    @classmethod
    def strip_whitespace(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("state", mode="before")
    @classmethod
    def normalize_state(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip().upper()
        return v

    @field_validator("postal_code", mode="before")
    @classmethod
    def normalize_postal_code(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("country", mode="before")
    @classmethod
    def normalize_country(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip().upper()
        return v


class PredictionRequestPayload(AddressPayload):
    requested_by: str | None = Field(default=None, max_length=120)
    preferred_policy_name: str | None = Field(
        default=None,
        max_length=80,
        description=(
            "Optional feature policy to apply for this specific request. "
            "When set, overrides the server-configured default and any state-level policy routing. "
            "Must match a known policy name from GET /v1/policies/feature."
        ),
    )
    feature_overrides: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional map of known feature values to inject before feature assembly. "
            "Keys must match expected model feature names (e.g. BedroomAbvGr, GrLivArea). "
            "Overrides provider-fetched values for matching keys. "
            "Prediction reuse is disabled whenever this field is non-null."
        ),
    )

    @model_validator(mode="after")
    def validate_feature_overrides(self) -> "PredictionRequestPayload":
        errors = _validate_feature_overrides_dict(self.feature_overrides)
        if errors:
            raise ValueError("; ".join(errors))
        return self


class NormalizedAddress(BaseModel):
    model_config = ConfigDict(extra="forbid")

    address_line_1: str
    address_line_2: str | None = None
    city: str
    state: str
    postal_code: str
    country: str
    formatted_address: str
    latitude: float | None = None
    longitude: float | None = None
    geocoding_source: str | None = None


class FeatureVectorContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    model_name: str
    model_version: str
    features: dict[str, Any]
    completeness_score: float
    feature_policy_name: str | None = None
    feature_policy_version: str | None = None
    weight_total: float | None = None


class ProviderResponseContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_name: str
    status: Literal["success", "failed"]
    payload: dict[str, Any]
    fetched_at: datetime


class ProviderResponseSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_name: str
    status: Literal["success", "failed"]
    fetched_at: datetime
    feature_source: str | None = None
    feature_provenance: dict[str, Any] | None = None
    payload_preview: dict[str, Any]


class FeatureSnapshotSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature_snapshot_id: UUID
    model_name: str
    model_version: str
    completeness_score: float
    feature_count: int
    populated_feature_count: int
    features: dict[str, Any]


class GeocodingResultContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalized_address: NormalizedAddress
    provider_response: ProviderResponseContract


PredictionStatus = Literal["received", "completed", "failed"]


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    prediction_id: UUID
    correlation_id: UUID
    status: PredictionStatus
    predicted_price: float
    currency: str = "USD"
    confidence_score: float | None = None
    model_name: str
    model_version: str
    feature_snapshot_id: UUID
    normalized_address: NormalizedAddress
    submitted_at: datetime
    generated_at: datetime
    was_reused: bool = False
    source_prediction_id: UUID | None = None
    selected_feature_policy_name: str | None = None
    selected_feature_policy_version: str | None = None
    key_features: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Key property features used for this prediction, e.g. bedrooms, "
            "living area sq ft, lot area, bathrooms, year built. "
            "Only populated features (non-null) are included."
        ),
    )
    exact_house_features: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Caller-supplied exact property facts for this prediction. "
            "These are the only home facts the UI should treat as factual."
        ),
    )
    actual_house_features: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Full populated feature payload used by model inference for this property. "
            "Includes all non-null model features, not just buyer-facing highlights."
        ),
    )
    feature_source: str | None = Field(
        default=None,
        description=(
            "Data source used to build the feature vector: 'census_context', "
            "'census_context_with_backfill', 'heuristic', or 'fake'. "
            "Useful for auditing data quality in production."
        ),
    )
    feature_provenance: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Metadata describing where features came from (provider payload lineage, "
            "fallback markers, and source-specific notes)."
        ),
    )


class PredictionDetailResponse(PredictionResponse):
    feature_snapshot: FeatureSnapshotSummary
    provider_responses: list[ProviderResponseSummary]
    error_message: str | None = None


class PropertyFeaturesResponse(BaseModel):
    """Response containing exact property features for a location.
    
    Includes all physical property characteristics and neighborhood signals
    from real data sources (Census enrichment, property databases, etc.).
    Features include `feature_source` and `feature_provenance` indicating
    data origin (real vs. estimated/fallback).
    """
    model_config = ConfigDict(extra="forbid")

    normalized_address_id: UUID
    features: dict[str, Any] = Field(
        ...,
        description="Exact property features (bedrooms, sqft, lot size, Quality, etc.)",
    )
    feature_source: str | None = Field(
        default=None,
        description="Data source for features: 'census_context', 'property_database', 'fake', etc.",
    )
    feature_provenance: dict[str, Any] | None = Field(
        default=None,
        description="Metadata about feature origins and data lineage.",
    )
    fetched_at: datetime
    completeness_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of expected features populated (0.0 to 1.0).",
    )


class PredictionTraceNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    prediction_id: UUID
    generated_at: datetime
    was_reused: bool
    source_prediction_id: UUID | None = None


class WorkflowEventItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_name: str
    payload: dict[str, Any]
    occurred_at: datetime


class PredictionTraceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    prediction_id: UUID
    source_prediction_id: UUID | None = None
    root_prediction_id: UUID
    was_reused: bool
    normalized_address: NormalizedAddress
    feature_snapshot: FeatureSnapshotSummary
    provider_responses: list[ProviderResponseSummary]
    trace_nodes: list[PredictionTraceNode]
    workflow_events: list[WorkflowEventItem] = Field(default_factory=list)


class PredictionListItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    prediction_id: UUID
    status: PredictionStatus
    predicted_price: float
    currency: str = "USD"
    model_name: str
    model_version: str
    normalized_address: NormalizedAddress
    submitted_at: datetime
    generated_at: datetime
    was_reused: bool = False
    source_prediction_id: UUID | None = None
    requested_by: str | None = None
    feature_source: str | None = None
    selected_feature_policy_name: str | None = None
    selected_feature_policy_version: str | None = None
    key_features: dict[str, Any] = Field(default_factory=dict)
    exact_house_features: dict[str, Any] = Field(default_factory=dict)


class PredictionListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[PredictionListItem]
    total: int = 0
    limit: int = 10
    offset: int = 0


class FeaturePolicyDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    description: str
    emphasis_features: list[str]


class FeaturePolicyCatalogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_policy_name: str
    default_policy_version: str
    state_overrides: dict[str, str]
    policies: list[FeaturePolicyDescriptor]


class FeaturePolicySimulationRequest(AddressPayload):
    policy_names: list[str] | None = None


class FeaturePolicySimulationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_name: str
    policy_version: str
    predicted_price: float
    completeness_score: float
    weight_total: float | None = None


class FeaturePolicySimulationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalized_address: NormalizedAddress
    provider_name: str
    simulations: list[FeaturePolicySimulationItem]


class FeatureBoundExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minimum: float
    maximum: float

    @model_validator(mode="after")
    def minimum_must_not_exceed_maximum(self) -> FeatureBoundExpectation:
        if self.minimum > self.maximum:
            raise ValueError(
                f"minimum ({self.minimum}) must not exceed maximum ({self.maximum})."
            )
        return self


class BaselineScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    label: str
    description: str
    payload: AddressPayload
    min_completeness_score: float = 0.95
    required_features: list[str] = Field(default_factory=list)
    feature_bounds: dict[str, FeatureBoundExpectation] = Field(default_factory=dict)


class BaselineScenarioCatalogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenarios: list[BaselineScenario]


class BaselineLocationObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalized_address: NormalizedAddress
    geocoding_provider: str
    geocoding_payload_preview: dict[str, Any]


class BaselineFeatureObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_features: list[str]
    pulled_features: list[str]
    missing_features: list[str]
    pulled_feature_count: int
    missing_feature_count: int
    completeness_score: float
    weight_total: float | None = None
    selected_feature_policy_name: str | None = None
    selected_feature_policy_version: str | None = None
    feature_values: dict[str, Any] = Field(default_factory=dict)
    key_feature_values: dict[str, Any] = Field(default_factory=dict)


class BaselineCheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_name: str
    status: Literal["passed", "failed", "skipped"]
    message: str


class BaselineAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overall_status: Literal["passed", "failed", "not_evaluated"]
    checks: list[BaselineCheckResult]


class BaselineExpectationsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_completeness_score: float | None = None
    required_features: list[str] = Field(default_factory=list)
    feature_bounds: dict[str, FeatureBoundExpectation] = Field(default_factory=dict)

    @field_validator("min_completeness_score")
    @classmethod
    def completeness_score_must_be_in_unit_interval(
        cls, v: float | None
    ) -> float | None:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(
                f"min_completeness_score must be between 0.0 and 1.0, got {v}."
            )
        return v


class AddressBaselineRequest(AddressPayload):
    expectations: BaselineExpectationsInput | None = None
    preferred_policy_name: str | None = Field(
        default=None,
        max_length=80,
        description=(
            "Optional feature policy to apply for this baseline. "
            "When set, overrides the server default and any state-level routing. "
            "Must match a known policy name from GET /v1/policies/feature."
        ),
    )
    feature_overrides: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional map of known feature values to inject before feature assembly. "
            "Keys must match expected model feature names (e.g. BedroomAbvGr, GrLivArea). "
            "Overrides provider-fetched values for matching keys."
        ),
    )

    @model_validator(mode="after")
    def validate_feature_overrides(self) -> "AddressBaselineRequest":
        errors = _validate_feature_overrides_dict(self.feature_overrides)
        if errors:
            raise ValueError("; ".join(errors))
        return self


class BaselineValueObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predicted_price: float
    model_name: str
    model_version: str
    property_provider: str
    property_feature_source: str | None = None
    property_feature_provenance: dict[str, Any] | None = None


class AddressBaselineResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: str
    evaluated_at: datetime
    input_address: AddressPayload
    location: BaselineLocationObservation
    features: BaselineFeatureObservation
    value: BaselineValueObservation
    assessment: BaselineAssessment


class FullAuditRequest(PredictionRequestPayload):
    expectations: BaselineExpectationsInput | None = None


class FullAuditResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: str
    audited_at: datetime
    baseline: AddressBaselineResponse
    prediction: PredictionResponse
    prediction_detail: PredictionDetailResponse | None = None
    prediction_trace: PredictionTraceResponse | None = None
    prediction_events: PredictionWorkflowEventsResponse | None = None
    issues: list[str]


class ScenarioPipelineResult(BaseModel):
    """Result of running one registered scenario through the full audit pipeline."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    label: str
    category: str
    pipeline_status: Literal["pass", "fail", "error"]
    issues: list[str]
    completeness_score: float | None = None
    predicted_price: float | None = None
    key_feature_values: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None


class ScenarioBatchPipelineRequest(BaseModel):
    """Request to run a subset (or all) registered scenarios through the pipeline."""

    model_config = ConfigDict(extra="forbid")

    scenario_ids: list[str] | None = None


class ScenarioBatchPipelineResponse(BaseModel):
    """Aggregated result of a scenario batch pipeline run."""

    model_config = ConfigDict(extra="forbid")

    contract_version: str
    started_at: datetime
    completed_at: datetime
    total: int
    passed: int
    failed: int
    errors: int
    results: list[ScenarioPipelineResult]


class PredictionWorkflowEventsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    prediction_id: UUID
    total_count: int
    limit: int
    offset: int
    event_name: str | None = None
    sort: Literal["asc", "desc"] = "asc"
    events: list[WorkflowEventItem]


class DashboardEventSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: UUID
    total_events: int
    latest_event_name: str | None = None
    latest_event_at: datetime | None = None
    recent_event_names: list[str]


class RuntimeSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app_name: str
    environment: str
    model_name: str
    model_version: str
    model_available: bool
    mock_predictor_enabled: bool


class ProviderPolicySummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    property_data_provider: str
    geocoding_provider: str
    provider_timeout_seconds: float
    provider_max_retries: int
    prediction_reuse_max_age_hours: int
    feature_policy_name: str
    feature_policy_version: str
    feature_policy_state_overrides: dict[str, str]


class UiLinkTemplates(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predictions_list: str
    prediction_detail: str
    prediction_trace: str
    prediction_create: str
    address_normalize: str


class DashboardBootstrapResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    contract_version: str
    runtime: RuntimeSummary
    provider_policy: ProviderPolicySummary
    recent_predictions: list[PredictionListItem]
    event_summary: DashboardEventSummary | None = None
    links: UiLinkTemplates


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    environment: str
    app_name: str
    model_name: str
    model_version: str
    model_available: bool
    mock_predictor_enabled: bool
    property_data_provider: str
    geocoding_provider: str
    provider_timeout_seconds: float
    provider_max_retries: int
    prediction_reuse_max_age_hours: int
    provider_response_cache_max_age_hours: int
    feature_policy_name: str
    feature_policy_version: str
    feature_policy_state_override_count: int
    walkscore_enrichment_active: bool = False
    live_mode_ready: bool
    live_mode_issues: list[str] = Field(default_factory=list)


class ApiEndpointDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    method: str
    path: str
    purpose: str


class ApiExamplePayloads(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_request: AddressPayload
    normalization_request: AddressPayload
    baseline_request: AddressPayload


class ApiCapabilitiesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: str
    generated_at: datetime
    runtime: RuntimeSummary
    provider_policy: ProviderPolicySummary
    model_expected_features: list[str]
    live_mode_ready: bool
    live_mode_issues: list[str] = Field(default_factory=list)
    endpoints: list[ApiEndpointDescriptor]
    examples: ApiExamplePayloads


class LiveFeatureCandidateItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: UUID
    request_id: UUID
    submitted_at: datetime
    generated_at: datetime
    predicted_price: float
    completeness_score: float
    was_reused: bool
    model_name: str
    model_version: str
    selected_feature_policy_name: str | None = None
    selected_feature_policy_version: str | None = None
    feature_source: str | None = None
    provider_name: str | None = None
    normalized_address: NormalizedAddress
    features: dict[str, Any]


class LiveFeatureCandidatesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: str
    generated_at: datetime
    total: int
    limit: int
    offset: int
    min_completeness_score: float
    include_reused: bool
    items: list[LiveFeatureCandidateItem]