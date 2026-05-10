from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy.orm import sessionmaker

from house_price_prediction.application.contracts.orchestration_contracts import (
    PredictionWorkflowCommand,
)
from house_price_prediction.application.services.data_orchestration_service import (
    DataOrchestrationLayer,
)
from house_price_prediction.application.services.feature_assembly_service import (
    FeatureAssemblyService,
)
from house_price_prediction.application.services.feature_policy_registry import (
    list_feature_policy_definitions,
)
from house_price_prediction.application.services.property_enrichment_service import (
    PropertyEnrichmentService,
)
from house_price_prediction.config import Settings
from house_price_prediction.application.services.prediction_validator import (
    PredictionValidator,
)
from house_price_prediction.application.services.scenario_registry import (
    get_all_scenarios,
    get_scenarios_by_ids,
)
from house_price_prediction.domain.contracts.prediction_contracts import (
    ApiCapabilitiesResponse,
    ApiEndpointDescriptor,
    ApiExamplePayloads,
    AddressBaselineResponse,
    AddressPayload,
    BaselineExpectationsInput,
    BaselineScenario,
    BaselineScenarioCatalogResponse,
    DashboardEventSummary,
    DashboardBootstrapResponse,
    FeaturePolicyCatalogResponse,
    FeaturePolicyDescriptor,
    FeaturePolicySimulationRequest,
    FeaturePolicySimulationResponse,
    FullAuditRequest,
    FullAuditResponse,
    LiveFeatureCandidatesResponse,
    NormalizedAddress,
    PredictionDetailResponse,
    PredictionListResponse,
    PredictionRequestPayload,
    PredictionResponse,
    ProviderPolicySummary,
    PredictionTraceResponse,
    PredictionWorkflowEventsResponse,
    RuntimeSummary,
    ScenarioBatchPipelineRequest,
    ScenarioBatchPipelineResponse,
    ScenarioPipelineResult,
    UiLinkTemplates,
)
from house_price_prediction.infrastructure.db.repositories import PredictionRepository
from house_price_prediction.infrastructure.model_runtime.predictor import PredictionRuntime
from house_price_prediction.infrastructure.providers.base import GeocodingProvider
from house_price_prediction.application.services.neighborhood_score_service import (
    NeighborhoodScoreService,
)
from house_price_prediction.telemetry import correlation_scope, get_logger

logger = get_logger(__name__)


class Brain:
    def __init__(
        self,
        session_factory: sessionmaker,
        feature_assembly_service: FeatureAssemblyService,
        prediction_runtime: PredictionRuntime,
        property_enrichment_service: PropertyEnrichmentService,
        geocoding_provider: GeocodingProvider,
        prediction_reuse_max_age_hours: int,
        provider_response_cache_max_age_hours: int,
        settings: Settings,
        neighborhood_scorer: NeighborhoodScoreService | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._prediction_runtime = prediction_runtime
        self._settings = settings
        self._data_orchestration_layer = DataOrchestrationLayer(
            session_factory=session_factory,
            feature_assembly_service=feature_assembly_service,
            prediction_runtime=prediction_runtime,
            property_enrichment_service=property_enrichment_service,
            geocoding_provider=geocoding_provider,
            prediction_reuse_max_age_hours=prediction_reuse_max_age_hours,
            provider_response_cache_max_age_hours=provider_response_cache_max_age_hours,
            neighborhood_scorer=neighborhood_scorer,
        )

    def normalize_address(self, payload: AddressPayload) -> NormalizedAddress:
        return self._data_orchestration_layer.normalize_address(payload)

    def get_feature_policy_catalog(self) -> FeaturePolicyCatalogResponse:
        policies = [
            FeaturePolicyDescriptor(
                name=policy.name,
                version=policy.version,
                description=policy.description,
                emphasis_features=list(policy.emphasis_features),
            )
            for policy in list_feature_policy_definitions()
        ]
        return FeaturePolicyCatalogResponse(
            default_policy_name=self._settings.feature_policy_name,
            default_policy_version=self._settings.feature_policy_version,
            state_overrides=self._settings.feature_policy_state_overrides or {},
            policies=policies,
        )

    def simulate_feature_policies(
        self,
        payload: FeaturePolicySimulationRequest,
    ) -> FeaturePolicySimulationResponse:
        return self._data_orchestration_layer.simulate_feature_policies(
            payload=payload,
            policy_names=payload.policy_names,
        )

    def get_baseline_scenarios(self) -> BaselineScenarioCatalogResponse:
        # Start with the static registry (always present, no traffic dependency).
        scenarios: list[BaselineScenario] = [
            BaselineScenario(
                scenario_id=reg.scenario_id,
                label=reg.label,
                description=reg.description,
                payload=reg.payload,
                min_completeness_score=reg.expectations.min_completeness_score or 0.0,
                required_features=reg.expectations.required_features,
                feature_bounds=reg.expectations.feature_bounds,
            )
            for reg in get_all_scenarios()
        ]

        # Append live-derived scenarios from recent traffic (last 10).
        # Guard against a cold or unavailable DB — return only registry scenarios
        # rather than letting a transient DB error break the catalog endpoint.
        existing_ids = {s.scenario_id for s in scenarios}
        try:
            recent = self.list_recent_predictions(limit=10)
        except Exception:  # noqa: BLE001
            logger.warning("Could not load recent predictions for scenario catalog; using registry only.")
            recent = None

        if recent is not None:
            for item in recent.items:
                addr = item.normalized_address
                live_id = f"live-{str(item.prediction_id)[:8]}"
                if live_id not in existing_ids:
                    scenarios.append(
                        BaselineScenario(
                            scenario_id=live_id,
                            label=addr.formatted_address,
                            description="Derived from recent live prediction traffic.",
                            payload=AddressPayload(
                                address_line_1=addr.address_line_1,
                                address_line_2=addr.address_line_2,
                                city=addr.city,
                                state=addr.state,
                                postal_code=addr.postal_code,
                                country=addr.country,
                            ),
                            min_completeness_score=0.0,
                            required_features=[],
                            feature_bounds={},
                        )
                    )
                    existing_ids.add(live_id)

        return BaselineScenarioCatalogResponse(scenarios=scenarios)

    def run_scenario_batch(
        self,
        request: ScenarioBatchPipelineRequest,
    ) -> ScenarioBatchPipelineResponse:
        """Run each registered scenario through the full audit pipeline.

        Scenarios are executed sequentially so that DB writes from each run
        are immediately durable for the next.  Failed individual scenarios are
        captured as ``error`` results rather than aborting the batch.
        """
        started_at = datetime.now(UTC)

        if request.scenario_ids is not None:
            targets = get_scenarios_by_ids(request.scenario_ids)
        else:
            targets = get_all_scenarios()

        results: list[ScenarioPipelineResult] = []
        for scenario in targets:
            try:
                audit = self.run_full_audit(
                    FullAuditRequest(
                        address_line_1=scenario.payload.address_line_1,
                        address_line_2=scenario.payload.address_line_2,
                        city=scenario.payload.city,
                        state=scenario.payload.state,
                        postal_code=scenario.payload.postal_code,
                        country=scenario.payload.country,
                        expectations=scenario.expectations,
                    )
                )
                pipeline_status: str = "fail" if audit.issues else "pass"
                raw_kfv = audit.baseline.features.key_feature_values
                results.append(
                    ScenarioPipelineResult(
                        scenario_id=scenario.scenario_id,
                        label=scenario.label,
                        category=scenario.category,
                        pipeline_status=pipeline_status,
                        issues=audit.issues,
                        completeness_score=audit.baseline.features.completeness_score,
                        predicted_price=audit.prediction.predicted_price,
                        key_feature_values={k: v for k, v in raw_kfv.items() if v is not None},
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Scenario %s failed with error: %s", scenario.scenario_id, exc)
                results.append(
                    ScenarioPipelineResult(
                        scenario_id=scenario.scenario_id,
                        label=scenario.label,
                        category=scenario.category,
                        pipeline_status="error",
                        issues=[],
                        error_message=str(exc),
                    )
                )

        completed_at = datetime.now(UTC)
        passed = sum(1 for r in results if r.pipeline_status == "pass")
        failed = sum(1 for r in results if r.pipeline_status == "fail")
        errors = sum(1 for r in results if r.pipeline_status == "error")

        return ScenarioBatchPipelineResponse(
            contract_version="2026-04-16",
            started_at=started_at,
            completed_at=completed_at,
            total=len(results),
            passed=passed,
            failed=failed,
            errors=errors,
            results=results,
        )

    def generate_address_baseline(
        self,
        payload: AddressPayload,
        expectations: BaselineExpectationsInput | None = None,
        policy_name_override: str | None = None,
        feature_overrides: dict | None = None,
    ) -> AddressBaselineResponse:
        return self._data_orchestration_layer.generate_address_baseline(
            payload=payload,
            expectations=expectations,
            policy_name_override=policy_name_override,
            feature_overrides=feature_overrides,
        )

    def run_full_audit(self, payload: FullAuditRequest) -> FullAuditResponse:
        # Pass per-request policy and feature overrides to baseline so both the
        # baseline and the subsequent prediction use the same feature assembly path.
        baseline = self.generate_address_baseline(
            payload=AddressPayload(
                address_line_1=payload.address_line_1,
                address_line_2=payload.address_line_2,
                city=payload.city,
                state=payload.state,
                postal_code=payload.postal_code,
                country=payload.country,
            ),
            expectations=payload.expectations,
            policy_name_override=payload.preferred_policy_name,
            feature_overrides=payload.feature_overrides,
        )

        prediction = self.create_prediction(payload)
        prediction_detail = self.get_prediction_detail(prediction.prediction_id)
        prediction_trace = self.get_prediction_trace(prediction.prediction_id)
        prediction_events = self.get_prediction_workflow_events(
            prediction_id=prediction.prediction_id,
            limit=50,
            offset=0,
            sort="asc",
        )

        issues: list[str] = []
        if baseline.assessment.overall_status == "failed":
            issues.append("Baseline expectation checks failed.")
        # Only surface raw missing-feature count when there are no expectations to
        # contextualise it.  If expectations were provided and the baseline passed,
        # missing features have already been accepted via the completeness threshold
        # or required-feature checks — reporting them again here would produce a
        # false "fail" in the batch runner for scenarios that deliberately tolerate
        # some missing features.
        if baseline.assessment.overall_status == "not_evaluated" and baseline.features.missing_feature_count > 0:
            issues.append(
                f"Missing model features detected: {baseline.features.missing_feature_count} "
                f"({', '.join(baseline.features.missing_features[:5])}"
                f"{'...' if len(baseline.features.missing_features) > 5 else ''})."
            )
        if prediction.status != "completed":
            issues.append(f"Prediction status is {prediction.status}.")
        if prediction_detail is None:
            issues.append("Prediction detail could not be loaded after creation.")
        if prediction_trace is None:
            issues.append("Prediction trace could not be loaded after creation.")
        if prediction_events is None or prediction_events.total_count == 0:
            issues.append("Prediction workflow events were not recorded.")

        return FullAuditResponse(
            contract_version="2026-04-16",
            audited_at=datetime.now(UTC),
            baseline=baseline,
            prediction=prediction,
            prediction_detail=prediction_detail,
            prediction_trace=prediction_trace,
            prediction_events=prediction_events,
            issues=issues,
        )

    def get_prediction_detail(self, prediction_id: UUID) -> PredictionDetailResponse | None:
        with self._session_factory() as session:
            repository = PredictionRepository(session)
            return repository.get_prediction_detail(prediction_id)

    def list_recent_predictions(self, limit: int = 10, offset: int = 0) -> PredictionListResponse:
        with self._session_factory() as session:
            repository = PredictionRepository(session)
            return repository.list_recent_predictions(limit=limit, offset=offset)

    def get_prediction_trace(self, prediction_id: UUID) -> PredictionTraceResponse | None:
        with self._session_factory() as session:
            repository = PredictionRepository(session)
            return repository.get_prediction_trace(prediction_id)

    def get_prediction_workflow_events(
        self,
        prediction_id: UUID,
        limit: int = 100,
        offset: int = 0,
        event_name: str | None = None,
        sort: str = "asc",
    ) -> PredictionWorkflowEventsResponse | None:
        with self._session_factory() as session:
            repository = PredictionRepository(session)
            return repository.get_prediction_workflow_events(
                prediction_id=prediction_id,
                limit=limit,
                offset=offset,
                event_name=event_name,
                sort=sort,
            )

    def get_live_feature_candidates(
        self,
        limit: int = 100,
        offset: int = 0,
        min_completeness_score: float = 0.0,
        include_reused: bool = False,
    ) -> LiveFeatureCandidatesResponse:
        with self._session_factory() as session:
            repository = PredictionRepository(session)
            return repository.list_live_feature_candidates(
                limit=limit,
                offset=offset,
                min_completeness_score=min_completeness_score,
                include_reused=include_reused,
            )

    def get_dashboard_bootstrap(self, limit: int = 5) -> DashboardBootstrapResponse:
        recent_predictions = self.list_recent_predictions(limit=limit)
        event_summary = None
        if recent_predictions.items:
            latest_prediction_id = recent_predictions.items[0].prediction_id
            latest_prediction_events = self.get_prediction_workflow_events(
                prediction_id=latest_prediction_id,
                limit=3,
                offset=0,
                sort="desc",
            )
            if latest_prediction_events is not None:
                latest_event = (
                    latest_prediction_events.events[0]
                    if latest_prediction_events.events
                    else None
                )
                event_summary = DashboardEventSummary(
                    prediction_id=latest_prediction_id,
                    total_events=latest_prediction_events.total_count,
                    latest_event_name=latest_event.event_name if latest_event else None,
                    latest_event_at=latest_event.occurred_at if latest_event else None,
                    recent_event_names=[
                        item.event_name for item in latest_prediction_events.events
                    ],
                )

        return DashboardBootstrapResponse(
            status="ok",
            contract_version="2026-04-16",
            runtime=RuntimeSummary(
                app_name=self._settings.app_name,
                environment=self._settings.app_env,
                model_name=self._settings.model_name,
                model_version=self._settings.model_version,
                model_available=self._prediction_runtime.is_available(),
                mock_predictor_enabled=self._settings.enable_mock_predictor,
            ),
            provider_policy=ProviderPolicySummary(
                property_data_provider=self._settings.property_data_provider,
                geocoding_provider=self._settings.geocoding_provider,
                provider_timeout_seconds=self._settings.provider_timeout_seconds,
                provider_max_retries=self._settings.provider_max_retries,
                prediction_reuse_max_age_hours=self._settings.prediction_reuse_max_age_hours,
                feature_policy_name=self._settings.feature_policy_name,
                feature_policy_version=self._settings.feature_policy_version,
                feature_policy_state_overrides=self._settings.feature_policy_state_overrides or {},
            ),
            recent_predictions=recent_predictions.items,
            event_summary=event_summary,
            links=UiLinkTemplates(
                predictions_list="/v1/predictions",
                prediction_detail="/v1/predictions/{prediction_id}",
                prediction_trace="/v1/predictions/{prediction_id}/trace",
                prediction_create="/v1/predictions",
                address_normalize="/v1/properties/normalize",
            ),
        )

    def get_api_capabilities(self) -> ApiCapabilitiesResponse:
        live_mode_issues: list[str] = []
        if self._settings.enable_mock_predictor:
            live_mode_issues.append("Mock predictor is enabled; set ENABLE_MOCK_PREDICTOR=false.")
        if self._settings.geocoding_provider.strip().lower() == "fake":
            live_mode_issues.append("Geocoding provider is fake; use a live geocoder provider.")
        if self._settings.property_data_provider.strip().lower() == "fake":
            live_mode_issues.append("Property data provider is fake; use a live property provider.")
        if not self._prediction_runtime.is_available():
            live_mode_issues.append("Model artifact is unavailable for inference.")

        endpoints = [
            ApiEndpointDescriptor(
                name="health",
                method="GET",
                path="/v1/health",
                purpose="Runtime and live-readiness status.",
            ),
            ApiEndpointDescriptor(
                name="normalize",
                method="POST",
                path="/v1/properties/normalize",
                purpose="Normalize and geocode a user-entered address.",
            ),
            ApiEndpointDescriptor(
                name="create_prediction",
                method="POST",
                path="/v1/predictions",
                purpose="Create a prediction using live providers and model inference.",
            ),
            ApiEndpointDescriptor(
                name="prediction_detail",
                method="GET",
                path="/v1/predictions/{prediction_id}",
                purpose="Inspect provider responses, features, and final prediction details.",
            ),
            ApiEndpointDescriptor(
                name="address_baseline",
                method="POST",
                path="/v1/validation/address-baseline",
                purpose="Run a non-persistent baseline check for a single address.",
            ),
            ApiEndpointDescriptor(
                name="full_audit",
                method="POST",
                path="/v1/validation/full-audit",
                purpose="End-to-end audit with baseline, prediction, trace, and issues summary.",
            ),
            ApiEndpointDescriptor(
                name="scenario_batch",
                method="POST",
                path="/v1/validation/run-scenario-batch",
                purpose="Automated batch pipeline checks for registered scenarios.",
            ),
            ApiEndpointDescriptor(
                name="api_capabilities",
                method="GET",
                path="/v1/meta/capabilities",
                purpose="Self-discovery contract for frontend and ML integration.",
            ),
            ApiEndpointDescriptor(
                name="live_feature_candidates",
                method="GET",
                path="/v1/meta/live-feature-candidates",
                purpose=(
                    "Clean feature rows from live traffic for feature engineering and "
                    "training-candidate dataset building."
                ),
            ),
        ]

        sample_address = AddressPayload(
            address_line_1="1600 Pennsylvania Ave NW",
            city="Washington",
            state="DC",
            postal_code="20500",
            country="US",
        )

        return ApiCapabilitiesResponse(
            contract_version="2026-04-17",
            generated_at=datetime.now(UTC),
            runtime=RuntimeSummary(
                app_name=self._settings.app_name,
                environment=self._settings.app_env,
                model_name=self._settings.model_name,
                model_version=self._settings.model_version,
                model_available=self._prediction_runtime.is_available(),
                mock_predictor_enabled=self._settings.enable_mock_predictor,
            ),
            provider_policy=ProviderPolicySummary(
                property_data_provider=self._settings.property_data_provider,
                geocoding_provider=self._settings.geocoding_provider,
                provider_timeout_seconds=self._settings.provider_timeout_seconds,
                provider_max_retries=self._settings.provider_max_retries,
                prediction_reuse_max_age_hours=self._settings.prediction_reuse_max_age_hours,
                feature_policy_name=self._settings.feature_policy_name,
                feature_policy_version=self._settings.feature_policy_version,
                feature_policy_state_overrides=self._settings.feature_policy_state_overrides or {},
            ),
            model_expected_features=list(self._prediction_runtime.expected_feature_names()),
            live_mode_ready=len(live_mode_issues) == 0,
            live_mode_issues=live_mode_issues,
            endpoints=endpoints,
            examples=ApiExamplePayloads(
                prediction_request=sample_address,
                normalization_request=sample_address,
                baseline_request=sample_address,
            ),
        )

    def create_prediction(
        self,
        payload: PredictionRequestPayload,
        correlation_id: str | None = None,
    ) -> PredictionResponse:
        request_id = uuid4()
        correlation_uuid = self._parse_correlation_id(correlation_id)
        submitted_at = datetime.now(UTC)

        with correlation_scope(str(correlation_uuid)):
            logger.info("prediction_request_start request_id=%s", request_id)
            workflow_result = self._data_orchestration_layer.execute_prediction_workflow(
                PredictionWorkflowCommand(
                    request_id=request_id,
                    correlation_id=correlation_uuid,
                    submitted_at=submitted_at,
                    payload=payload,
                )
            )

            # Validate prediction quality against Census context
            validator = PredictionValidator()
            validation_result = validator.validate_prediction(
                predicted_price=workflow_result.predicted_price,
                actual_house_features=workflow_result.actual_house_features,
                feature_source=workflow_result.feature_source,
            )
            
            # Update confidence score with validation result
            # Keep existing confidence_score if higher, use validation result if lower
            final_confidence = validation_result["confidence_score"]
            if workflow_result.confidence_score is not None:
                final_confidence = min(workflow_result.confidence_score, final_confidence)
            
            # Store validation notes in feature_provenance for audit trail
            feature_provenance = dict(workflow_result.feature_provenance or {})
            feature_provenance["validation_result"] = {
                "is_valid": validation_result["is_valid"],
                "confidence_score": final_confidence,
                "validation_notes": validation_result["validation_notes"],
                "anomaly_flags": validation_result["anomaly_flags"],
            }

            return PredictionResponse(
                request_id=workflow_result.request_id,
                prediction_id=workflow_result.prediction_id,
                correlation_id=workflow_result.correlation_id,
                status=workflow_result.status,
                predicted_price=workflow_result.predicted_price,
                currency=workflow_result.currency,
                confidence_score=final_confidence,
                model_name=workflow_result.model_name,
                model_version=workflow_result.model_version,
                feature_snapshot_id=workflow_result.feature_snapshot_id,
                normalized_address=workflow_result.normalized_address,
                submitted_at=workflow_result.submitted_at,
                generated_at=workflow_result.generated_at,
                was_reused=workflow_result.was_reused,
                source_prediction_id=workflow_result.source_prediction_id,
                selected_feature_policy_name=workflow_result.selected_feature_policy_name,
                selected_feature_policy_version=workflow_result.selected_feature_policy_version,
                key_features=workflow_result.key_features,
                exact_house_features=workflow_result.exact_house_features,
                actual_house_features=workflow_result.actual_house_features,
                feature_source=workflow_result.feature_source,
                feature_provenance=feature_provenance,
            )

    @staticmethod
    def _parse_correlation_id(correlation_id: str | None) -> UUID:
        if correlation_id is None:
            return uuid4()
        try:
            return UUID(correlation_id)
        except ValueError:
            return uuid4()