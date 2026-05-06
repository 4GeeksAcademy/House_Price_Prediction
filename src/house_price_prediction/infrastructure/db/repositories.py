from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from house_price_prediction.domain.contracts.prediction_contracts import (
    FeatureSnapshotSummary,
    FeatureVectorContract,
    LiveFeatureCandidateItem,
    LiveFeatureCandidatesResponse,
    NormalizedAddress,
    PredictionDetailResponse,
    PredictionListItem,
    PredictionListResponse,
    PredictionRequestPayload,
    PredictionTraceNode,
    PredictionTraceResponse,
    PredictionWorkflowEventsResponse,
    ProviderResponseContract,
    ProviderResponseSummary,
    WorkflowEventItem,
)
from house_price_prediction.feature_schema import KEY_BUYER_FEATURES
from house_price_prediction.infrastructure.db.models import (
    FeatureSnapshotModel,
    ModelRegistryModel,
    NormalizedAddressModel,
    PredictionModel,
    PredictionRequestModel,
    ProviderResponseModel,
    WorkflowEventModel,
)


def _extract_key_features(features: dict[str, Any]) -> dict[str, Any]:
    """Return only the populated buyer-relevant features from a raw feature dict."""
    return {k: features[k] for k in KEY_BUYER_FEATURES if features.get(k) is not None}


class PredictionRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create_prediction_request(
        self,
        request_id: UUID,
        correlation_id: UUID,
        normalized_address_id: UUID,
        payload: PredictionRequestPayload,
        normalized_address: NormalizedAddress,
        submitted_at: datetime,
        feature_policy_name: str,
        feature_policy_version: str,
    ) -> None:
        self._session.add(
            PredictionRequestModel(
                id=str(request_id),
                correlation_id=str(correlation_id),
                normalized_address_id=str(normalized_address_id),
                requested_by=payload.requested_by,
                address_line_1=payload.address_line_1,
                address_line_2=payload.address_line_2,
                city=payload.city,
                state=payload.state,
                postal_code=payload.postal_code,
                country=payload.country,
                normalized_address=normalized_address.formatted_address,
                feature_policy_name=feature_policy_name,
                feature_policy_version=feature_policy_version,
                status="received",
                submitted_at=submitted_at,
            )
        )

    def get_or_create_normalized_address(self, normalized_address: NormalizedAddress) -> UUID:
        existing = self._session.scalar(
            select(NormalizedAddressModel).where(
                NormalizedAddressModel.formatted_address == normalized_address.formatted_address
            )
        )
        if existing is not None:
            if normalized_address.latitude is not None:
                existing.latitude = normalized_address.latitude
            if normalized_address.longitude is not None:
                existing.longitude = normalized_address.longitude
            if normalized_address.geocoding_source is not None:
                existing.geocoding_source = normalized_address.geocoding_source
            return UUID(existing.id)

        normalized_address_id = uuid4()
        self._session.add(
            NormalizedAddressModel(
                id=str(normalized_address_id),
                address_line_1=normalized_address.address_line_1,
                address_line_2=normalized_address.address_line_2,
                city=normalized_address.city,
                state=normalized_address.state,
                postal_code=normalized_address.postal_code,
                country=normalized_address.country,
                formatted_address=normalized_address.formatted_address,
                latitude=normalized_address.latitude,
                longitude=normalized_address.longitude,
                geocoding_source=normalized_address.geocoding_source,
            )
        )
        return normalized_address_id

    def find_recent_property_response_for_address(
        self,
        normalized_address: NormalizedAddress,
        max_age_hours: int,
    ) -> ProviderResponseContract | None:
        if max_age_hours <= 0:
            return None

        normalized_address_id = self._session.scalar(
            select(NormalizedAddressModel.id).where(
                NormalizedAddressModel.formatted_address == normalized_address.formatted_address
            )
        )
        if normalized_address_id is None:
            return None

        cutoff = datetime.now(UTC) - timedelta(hours=max_age_hours)
        rows = (
            self._session.query(ProviderResponseModel)
            .join(PredictionRequestModel, PredictionRequestModel.id == ProviderResponseModel.request_id)
            .filter(PredictionRequestModel.normalized_address_id == normalized_address_id)
            .filter(ProviderResponseModel.status == "success")
            .filter(ProviderResponseModel.fetched_at >= cutoff)
            .order_by(ProviderResponseModel.fetched_at.desc())
            .limit(50)
            .all()
        )

        # Keep only responses that look like full property feature payloads.
        # All three key features must be present — a partial match could be a
        # geocoding response leaking into the property-response table.
        required_feature_keys = {"LotArea", "OverallQual", "GrLivArea"}
        for row in rows:
            payload = row.payload
            if not isinstance(payload, dict):
                continue
            if not required_feature_keys.issubset(payload.keys()):
                continue
            return ProviderResponseContract(
                provider_name=row.provider_name,
                status=row.status,
                payload=payload,
                fetched_at=row.fetched_at,
            )
        return None

    def register_model_version(
        self,
        model_name: str,
        model_version: str,
        feature_columns: list[str],
    ) -> UUID:
        existing = self._session.scalar(
            select(ModelRegistryModel).where(
                ModelRegistryModel.model_name == model_name,
                ModelRegistryModel.model_version == model_version,
            )
        )
        if existing is not None:
            self._session.query(ModelRegistryModel).filter(
                ModelRegistryModel.model_name == model_name,
                ModelRegistryModel.id != existing.id,
            ).update({ModelRegistryModel.is_active: False}, synchronize_session=False)
            existing.feature_columns = feature_columns
            existing.is_active = True
            return UUID(existing.id)

        self._session.query(ModelRegistryModel).filter(
            ModelRegistryModel.model_name == model_name
        ).update({ModelRegistryModel.is_active: False}, synchronize_session=False)

        model_registry_id = uuid4()
        self._session.add(
            ModelRegistryModel(
                id=str(model_registry_id),
                model_name=model_name,
                model_version=model_version,
                feature_columns=feature_columns,
                is_active=True,
            )
        )
        return model_registry_id

    def update_request_status(
        self,
        request_id: UUID,
        status: str,
        error_message: str | None = None,
    ) -> None:
        request = self._session.get(PredictionRequestModel, str(request_id))
        if request is None:
            return
        request.status = status
        request.error_message = error_message

    def create_feature_snapshot(self, feature_vector: FeatureVectorContract) -> UUID:
        feature_snapshot_id = uuid4()
        self._session.add(
            FeatureSnapshotModel(
                id=str(feature_snapshot_id),
                request_id=str(feature_vector.request_id),
                model_name=feature_vector.model_name,
                model_version=feature_vector.model_version,
                completeness_score=feature_vector.completeness_score,
                features=feature_vector.features,
            )
        )
        return feature_snapshot_id

    def create_provider_response(
        self,
        request_id: UUID,
        provider_response: ProviderResponseContract,
    ) -> UUID:
        provider_response_id = uuid4()
        self._session.add(
            ProviderResponseModel(
                id=str(provider_response_id),
                request_id=str(request_id),
                provider_name=provider_response.provider_name,
                status=provider_response.status,
                payload=provider_response.payload,
                fetched_at=provider_response.fetched_at,
            )
        )
        return provider_response_id

    def create_workflow_event(
        self,
        request_id: UUID,
        event_name: str,
        payload: dict[str, object],
        occurred_at: datetime,
    ) -> UUID:
        workflow_event_id = uuid4()
        self._session.add(
            WorkflowEventModel(
                id=str(workflow_event_id),
                request_id=str(request_id),
                event_name=event_name,
                payload=payload,
                occurred_at=occurred_at,
            )
        )
        return workflow_event_id

    def create_prediction(
        self,
        request_id: UUID,
        model_registry_id: UUID,
        feature_snapshot_id: UUID,
        predicted_price: float,
        model_name: str,
        model_version: str,
        confidence_score: float | None = None,
        was_reused: bool = False,
        source_prediction_id: UUID | None = None,
    ) -> UUID:
        prediction_id = uuid4()
        self._session.add(
            PredictionModel(
                id=str(prediction_id),
                request_id=str(request_id),
                model_registry_id=str(model_registry_id),
                source_prediction_id=str(source_prediction_id) if source_prediction_id else None,
                feature_snapshot_id=str(feature_snapshot_id),
                predicted_price=predicted_price,
                currency="USD",
                confidence_score=confidence_score,
                model_name=model_name,
                model_version=model_version,
                was_reused=was_reused,
            )
        )
        return prediction_id

    def get_prediction_detail(self, prediction_id: UUID) -> PredictionDetailResponse | None:
        prediction = self._session.get(PredictionModel, str(prediction_id))
        if prediction is None:
            return None

        request = self._session.get(PredictionRequestModel, prediction.request_id)
        feature_snapshot = self._session.get(FeatureSnapshotModel, prediction.feature_snapshot_id)
        if request is None or feature_snapshot is None:
            return None

        normalized_address = self._build_normalized_address(request)
        trace_feature_snapshot = feature_snapshot
        trace_request_id = UUID(request.id)
        trace_prediction = self._resolve_trace_prediction(prediction)
        if trace_prediction is not None:
            trace_request_id = UUID(trace_prediction.request_id)
            source_feature_snapshot = self._session.get(
                FeatureSnapshotModel, trace_prediction.feature_snapshot_id
            )
            if source_feature_snapshot is not None:
                trace_feature_snapshot = source_feature_snapshot

        provider_responses = self.get_provider_responses(trace_request_id)
        trace_features = (
            trace_feature_snapshot.features
            if isinstance(trace_feature_snapshot.features, dict)
            else {}
        )
        return PredictionDetailResponse(
            request_id=UUID(request.id),
            prediction_id=UUID(prediction.id),
            correlation_id=UUID(request.correlation_id),
            status=request.status,
            predicted_price=prediction.predicted_price,
            currency=prediction.currency,
            confidence_score=prediction.confidence_score,
            model_name=prediction.model_name,
            model_version=prediction.model_version,
            feature_snapshot_id=UUID(feature_snapshot.id),
            feature_snapshot=self._build_feature_snapshot_summary(trace_feature_snapshot),
            provider_responses=[
                self._build_provider_response_summary(provider_response)
                for provider_response in provider_responses
            ],
            normalized_address=normalized_address,
            submitted_at=request.submitted_at,
            generated_at=prediction.generated_at,
            was_reused=prediction.was_reused,
            source_prediction_id=UUID(prediction.source_prediction_id)
            if prediction.source_prediction_id is not None
            else None,
            selected_feature_policy_name=request.feature_policy_name,
            selected_feature_policy_version=request.feature_policy_version,
            error_message=request.error_message,
            key_features=_extract_key_features(trace_features),
        )

    def list_recent_predictions(self, limit: int = 10, offset: int = 0) -> PredictionListResponse:
        # Count query (same filter, no pagination) — one DB round-trip.
        total: int = self._session.scalar(
            select(func.count(PredictionModel.id))
            .join(PredictionRequestModel, PredictionRequestModel.id == PredictionModel.request_id)
            .where(PredictionRequestModel.status == "completed")
        ) or 0

        # Primary query: join PredictionRequestModel + NormalizedAddressModel in one shot
        # so we never call session.get() per-row for those two tables.
        rows = self._session.execute(
            select(PredictionModel, PredictionRequestModel, NormalizedAddressModel)
            .join(PredictionRequestModel, PredictionRequestModel.id == PredictionModel.request_id)
            .join(
                NormalizedAddressModel,
                NormalizedAddressModel.id == PredictionRequestModel.normalized_address_id,
            )
            .where(PredictionRequestModel.status == "completed")
            .order_by(PredictionModel.generated_at.desc())
            .limit(limit)
            .offset(offset)
        ).all()

        if not rows:
            return PredictionListResponse(items=[], total=total, limit=limit, offset=offset)

        # Batch-resolve the "trace" request ID for reused predictions.
        # For reused rows, the provider responses live on source_prediction_id's request.
        # Collect all source_prediction_ids we need to resolve in one query.
        reused_source_ids = [
            prediction.source_prediction_id
            for prediction, _, _ in rows
            if prediction.was_reused and prediction.source_prediction_id is not None
        ]
        source_prediction_request_map: dict[str, str] = {}  # source_pred_id → request_id
        if reused_source_ids:
            source_rows = self._session.execute(
                select(PredictionModel.id, PredictionModel.request_id).where(
                    PredictionModel.id.in_(reused_source_ids)
                )
            ).all()
            source_prediction_request_map = {row.id: row.request_id for row in source_rows}

        # Collect all request IDs whose provider responses we need (batch load).
        trace_request_ids: dict[str, str] = {}  # prediction.id → effective_request_id
        for prediction, request, _ in rows:
            if prediction.was_reused and prediction.source_prediction_id is not None:
                effective = source_prediction_request_map.get(
                    prediction.source_prediction_id, request.id
                )
            else:
                effective = request.id
            trace_request_ids[prediction.id] = effective

        # Batch-load all provider responses for the relevant request IDs in one query.
        all_trace_request_ids = list(set(trace_request_ids.values()))
        provider_rows = self._session.execute(
            select(ProviderResponseModel)
            .where(ProviderResponseModel.request_id.in_(all_trace_request_ids))
            .order_by(ProviderResponseModel.fetched_at.asc())
        ).scalars().all()

        provider_by_request: dict[str, list[ProviderResponseContract]] = {}
        for row in provider_rows:
            contract = ProviderResponseContract(
                provider_name=row.provider_name,
                status=row.status,
                payload=row.payload,
                fetched_at=row.fetched_at,
            )
            provider_by_request.setdefault(row.request_id, []).append(contract)

        # Batch-load feature snapshots so we can surface key_features on every list item.
        all_feature_snapshot_ids = [prediction.feature_snapshot_id for prediction, _, _ in rows]
        snapshot_rows = self._session.execute(
            select(FeatureSnapshotModel).where(
                FeatureSnapshotModel.id.in_(all_feature_snapshot_ids)
            )
        ).scalars().all()
        snapshot_by_id: dict[str, dict[str, Any]] = {
            row.id: (row.features if isinstance(row.features, dict) else {})
            for row in snapshot_rows
        }

        # Build list items — zero extra DB queries from here.
        items: list[PredictionListItem] = []
        for prediction, request, norm_addr in rows:
            effective_request_id = trace_request_ids[prediction.id]
            provider_responses = provider_by_request.get(effective_request_id, [])
            feature_source = self._find_feature_source(provider_responses)

            raw_features = snapshot_by_id.get(prediction.feature_snapshot_id, {})
            key_features = _extract_key_features(raw_features)

            normalized_address = NormalizedAddress(
                address_line_1=norm_addr.address_line_1,
                address_line_2=norm_addr.address_line_2,
                city=norm_addr.city,
                state=norm_addr.state,
                postal_code=norm_addr.postal_code,
                country=norm_addr.country,
                formatted_address=request.normalized_address,
                latitude=norm_addr.latitude,
                longitude=norm_addr.longitude,
                geocoding_source=norm_addr.geocoding_source,
            )

            items.append(
                PredictionListItem(
                    request_id=UUID(request.id),
                    prediction_id=UUID(prediction.id),
                    status=request.status,
                    predicted_price=prediction.predicted_price,
                    currency=prediction.currency,
                    model_name=prediction.model_name,
                    model_version=prediction.model_version,
                    normalized_address=normalized_address,
                    submitted_at=request.submitted_at,
                    generated_at=prediction.generated_at,
                    was_reused=prediction.was_reused,
                    source_prediction_id=UUID(prediction.source_prediction_id)
                    if prediction.source_prediction_id is not None
                    else None,
                    requested_by=request.requested_by,
                    feature_source=feature_source,
                    selected_feature_policy_name=request.feature_policy_name,
                    selected_feature_policy_version=request.feature_policy_version,
                    key_features=key_features,
                )
            )

        return PredictionListResponse(items=items, total=total, limit=limit, offset=offset)

    def get_prediction_trace(self, prediction_id: UUID) -> PredictionTraceResponse | None:
        prediction = self._session.get(PredictionModel, str(prediction_id))
        if prediction is None:
            return None

        request = self._session.get(PredictionRequestModel, prediction.request_id)
        if request is None:
            return None

        trace_predictions = self._resolve_trace_chain(prediction)
        root_prediction = trace_predictions[-1]
        root_request = self._session.get(PredictionRequestModel, root_prediction.request_id)
        root_feature_snapshot = self._session.get(
            FeatureSnapshotModel, root_prediction.feature_snapshot_id
        )
        if root_request is None or root_feature_snapshot is None:
            return None

        return PredictionTraceResponse(
            request_id=UUID(request.id),
            prediction_id=UUID(prediction.id),
            source_prediction_id=UUID(prediction.source_prediction_id)
            if prediction.source_prediction_id is not None
            else None,
            root_prediction_id=UUID(root_prediction.id),
            was_reused=prediction.was_reused,
            normalized_address=self._build_normalized_address(request),
            feature_snapshot=self._build_feature_snapshot_summary(root_feature_snapshot),
            provider_responses=[
                self._build_provider_response_summary(provider_response)
                for provider_response in self.get_provider_responses(UUID(root_request.id))
            ],
            trace_nodes=[self._build_prediction_trace_node(item) for item in trace_predictions],
            workflow_events=self._list_workflow_events(request_id=UUID(request.id), limit=100)[0],
        )

    def find_reusable_prediction(
        self,
        normalized_address_id: UUID,
        model_registry_id: UUID,
        max_age_hours: int,
        feature_policy_name: str,
        feature_policy_version: str,
    ) -> PredictionDetailResponse | None:
        if max_age_hours <= 0:
            return None

        cutoff = datetime.now(UTC) - timedelta(hours=max_age_hours)
        statement = (
            select(PredictionModel.id)
            .join(PredictionRequestModel, PredictionRequestModel.id == PredictionModel.request_id)
            .where(PredictionRequestModel.normalized_address_id == str(normalized_address_id))
            .where(PredictionRequestModel.status == "completed")
            .where(PredictionRequestModel.feature_policy_name == feature_policy_name)
            .where(PredictionRequestModel.feature_policy_version == feature_policy_version)
            .where(PredictionModel.model_registry_id == str(model_registry_id))
            .where(PredictionModel.generated_at >= cutoff)
            .order_by(PredictionModel.generated_at.desc())
            .limit(1)
        )
        prediction_id = self._session.scalar(statement)
        if prediction_id is None:
            return None
        return self.get_prediction_detail(UUID(prediction_id))

    def get_provider_responses(self, request_id: UUID) -> list[ProviderResponseContract]:
        provider_rows = (
            self._session.query(ProviderResponseModel)
            .filter(ProviderResponseModel.request_id == str(request_id))
            .order_by(ProviderResponseModel.fetched_at.asc())
            .all()
        )
        return [
            ProviderResponseContract(
                provider_name=row.provider_name,
                status=row.status,
                payload=row.payload,
                fetched_at=row.fetched_at,
            )
            for row in provider_rows
        ]

    def get_request_id_by_correlation_id(self, correlation_id: UUID) -> UUID | None:
        request_id = self._session.scalar(
            select(PredictionRequestModel.id)
            .where(PredictionRequestModel.correlation_id == str(correlation_id))
            .order_by(PredictionRequestModel.submitted_at.desc())
            .limit(1)
        )
        return UUID(request_id) if request_id is not None else None

    def get_request_status(self, request_id: UUID) -> str | None:
        request = self._session.get(PredictionRequestModel, str(request_id))
        return request.status if request is not None else None

    def get_request_id_for_prediction(self, prediction_id: UUID) -> UUID | None:
        prediction = self._session.get(PredictionModel, str(prediction_id))
        if prediction is None:
            return None
        return UUID(prediction.request_id)

    def get_prediction_workflow_events(
        self,
        prediction_id: UUID,
        limit: int = 100,
        offset: int = 0,
        event_name: str | None = None,
        sort: str = "asc",
    ) -> PredictionWorkflowEventsResponse | None:
        request_id = self.get_request_id_for_prediction(prediction_id)
        if request_id is None:
            return None

        events, total_count = self._list_workflow_events(
            request_id=request_id,
            limit=limit,
            offset=offset,
            event_name=event_name,
            sort=sort,
        )
        return PredictionWorkflowEventsResponse(
            request_id=request_id,
            prediction_id=prediction_id,
            total_count=total_count,
            limit=limit,
            offset=offset,
            event_name=event_name,
            sort=sort,
            events=events,
        )

    def list_live_feature_candidates(
        self,
        limit: int = 100,
        offset: int = 0,
        min_completeness_score: float = 0.0,
        include_reused: bool = False,
    ) -> LiveFeatureCandidatesResponse:
        base_query = (
            select(PredictionModel.id)
            .join(PredictionRequestModel, PredictionRequestModel.id == PredictionModel.request_id)
            .join(FeatureSnapshotModel, FeatureSnapshotModel.id == PredictionModel.feature_snapshot_id)
            .where(PredictionRequestModel.status == "completed")
            .where(FeatureSnapshotModel.completeness_score >= min_completeness_score)
        )
        if not include_reused:
            base_query = base_query.where(PredictionModel.was_reused.is_(False))

        total = self._session.scalar(select(func.count()).select_from(base_query.subquery())) or 0

        rows_query = (
            select(PredictionModel, PredictionRequestModel, FeatureSnapshotModel, NormalizedAddressModel)
            .join(PredictionRequestModel, PredictionRequestModel.id == PredictionModel.request_id)
            .join(FeatureSnapshotModel, FeatureSnapshotModel.id == PredictionModel.feature_snapshot_id)
            .join(
                NormalizedAddressModel,
                NormalizedAddressModel.id == PredictionRequestModel.normalized_address_id,
            )
            .where(PredictionRequestModel.status == "completed")
            .where(FeatureSnapshotModel.completeness_score >= min_completeness_score)
            .order_by(PredictionModel.generated_at.desc())
            .limit(limit)
            .offset(offset)
        )
        if not include_reused:
            rows_query = rows_query.where(PredictionModel.was_reused.is_(False))

        rows = self._session.execute(rows_query).all()

        if not rows:
            return LiveFeatureCandidatesResponse(
                contract_version="2026-04-17",
                generated_at=datetime.now(UTC),
                total=int(total),
                limit=limit,
                offset=offset,
                min_completeness_score=min_completeness_score,
                include_reused=include_reused,
                items=[],
            )

        request_ids = [request.id for _, request, _, _ in rows]
        provider_rows = self._session.execute(
            select(ProviderResponseModel)
            .where(ProviderResponseModel.request_id.in_(request_ids))
            .order_by(ProviderResponseModel.fetched_at.asc())
        ).scalars().all()

        provider_by_request: dict[str, list[ProviderResponseContract]] = {}
        for row in provider_rows:
            provider_by_request.setdefault(row.request_id, []).append(
                ProviderResponseContract(
                    provider_name=row.provider_name,
                    status=row.status,
                    payload=row.payload,
                    fetched_at=row.fetched_at,
                )
            )

        items: list[LiveFeatureCandidateItem] = []
        for prediction, request, feature_snapshot, norm_addr in rows:
            provider_responses = provider_by_request.get(request.id, [])
            feature_source = self._find_feature_source(provider_responses)

            latest_property_provider_name: str | None = None
            for provider_response in reversed(provider_responses):
                payload = provider_response.payload
                if isinstance(payload, dict) and any(
                    key in payload for key in ("LotArea", "OverallQual", "GrLivArea")
                ):
                    latest_property_provider_name = provider_response.provider_name
                    break

            normalized_address = NormalizedAddress(
                address_line_1=norm_addr.address_line_1,
                address_line_2=norm_addr.address_line_2,
                city=norm_addr.city,
                state=norm_addr.state,
                postal_code=norm_addr.postal_code,
                country=norm_addr.country,
                formatted_address=request.normalized_address,
                latitude=norm_addr.latitude,
                longitude=norm_addr.longitude,
                geocoding_source=norm_addr.geocoding_source,
            )

            items.append(
                LiveFeatureCandidateItem(
                    prediction_id=UUID(prediction.id),
                    request_id=UUID(request.id),
                    submitted_at=request.submitted_at,
                    generated_at=prediction.generated_at,
                    predicted_price=prediction.predicted_price,
                    completeness_score=feature_snapshot.completeness_score,
                    was_reused=prediction.was_reused,
                    model_name=prediction.model_name,
                    model_version=prediction.model_version,
                    selected_feature_policy_name=request.feature_policy_name,
                    selected_feature_policy_version=request.feature_policy_version,
                    feature_source=feature_source,
                    provider_name=latest_property_provider_name,
                    normalized_address=normalized_address,
                    features=dict(feature_snapshot.features),
                )
            )

        return LiveFeatureCandidatesResponse(
            contract_version="2026-04-17",
            generated_at=datetime.now(UTC),
            total=int(total),
            limit=limit,
            offset=offset,
            min_completeness_score=min_completeness_score,
            include_reused=include_reused,
            items=items,
        )

    def _list_workflow_events(
        self,
        request_id: UUID,
        limit: int | None = None,
        offset: int = 0,
        event_name: str | None = None,
        sort: str = "asc",
    ) -> tuple[list[WorkflowEventItem], int]:
        filters = [WorkflowEventModel.request_id == str(request_id)]
        if event_name:
            filters.append(WorkflowEventModel.event_name == event_name)

        total_count = int(
            self._session.scalar(
                select(func.count(WorkflowEventModel.id)).where(*filters)
            ) or 0
        )

        order_col = (
            WorkflowEventModel.occurred_at.desc()
            if sort == "desc"
            else WorkflowEventModel.occurred_at.asc()
        )
        stmt = (
            select(WorkflowEventModel)
            .where(*filters)
            .order_by(order_col)
            .offset(offset)
        )
        if limit is not None:
            stmt = stmt.limit(limit)

        rows = self._session.scalars(stmt).all()
        return [
            WorkflowEventItem(
                event_name=row.event_name,
                payload=row.payload,
                occurred_at=row.occurred_at,
            )
            for row in rows
        ], total_count

    def count_normalized_addresses(self) -> int:
        return int(self._session.scalar(select(func.count()).select_from(NormalizedAddressModel)) or 0)

    def get_model_registry_entry(self, model_name: str, model_version: str) -> ModelRegistryModel | None:
        return self._session.scalar(
            select(ModelRegistryModel).where(
                ModelRegistryModel.model_name == model_name,
                ModelRegistryModel.model_version == model_version,
            )
        )

    @staticmethod
    def _build_feature_snapshot_summary(
        feature_snapshot: FeatureSnapshotModel,
    ) -> FeatureSnapshotSummary:
        features = dict(feature_snapshot.features)
        feature_count = len(features)
        populated_feature_count = sum(value is not None for value in features.values())
        return FeatureSnapshotSummary(
            feature_snapshot_id=UUID(feature_snapshot.id),
            model_name=feature_snapshot.model_name,
            model_version=feature_snapshot.model_version,
            completeness_score=feature_snapshot.completeness_score,
            feature_count=feature_count,
            populated_feature_count=populated_feature_count,
            features=features,
        )

    @staticmethod
    def _build_provider_response_summary(
        provider_response: ProviderResponseContract,
    ) -> ProviderResponseSummary:
        payload = provider_response.payload
        payload_preview = {
            key: value
            for key, value in payload.items()
            if key not in {"result", "query", "normalized", "features"}
        }
        return ProviderResponseSummary(
            provider_name=provider_response.provider_name,
            status=provider_response.status,
            fetched_at=provider_response.fetched_at,
            feature_source=payload.get("feature_source"),
            feature_provenance=payload.get("feature_provenance"),
            payload_preview=payload_preview,
        )

    @staticmethod
    def _find_feature_source(
        provider_responses: list[ProviderResponseContract],
    ) -> str | None:
        for provider_response in reversed(provider_responses):
            feature_source = provider_response.payload.get("feature_source")
            if isinstance(feature_source, str):
                return feature_source
        return None

    def _resolve_trace_prediction(
        self,
        prediction: PredictionModel,
    ) -> PredictionModel | None:
        current = prediction
        visited_ids = {current.id}
        while current.source_prediction_id is not None:
            source_prediction = self._session.get(PredictionModel, current.source_prediction_id)
            if source_prediction is None or source_prediction.id in visited_ids:
                break
            current = source_prediction
            visited_ids.add(current.id)
        return current

    def _resolve_trace_chain(
        self,
        prediction: PredictionModel,
    ) -> list[PredictionModel]:
        chain = [prediction]
        current = prediction
        visited_ids = {current.id}
        while current.source_prediction_id is not None:
            source_prediction = self._session.get(PredictionModel, current.source_prediction_id)
            if source_prediction is None or source_prediction.id in visited_ids:
                break
            chain.append(source_prediction)
            current = source_prediction
            visited_ids.add(current.id)
        return chain

    def _build_prediction_trace_node(
        self,
        prediction: PredictionModel,
    ) -> PredictionTraceNode:
        return PredictionTraceNode(
            request_id=UUID(prediction.request_id),
            prediction_id=UUID(prediction.id),
            generated_at=prediction.generated_at,
            was_reused=prediction.was_reused,
            source_prediction_id=UUID(prediction.source_prediction_id)
            if prediction.source_prediction_id is not None
            else None,
        )

    def _build_normalized_address(self, request: PredictionRequestModel) -> NormalizedAddress:
        normalized_address_row = self._session.get(
            NormalizedAddressModel, request.normalized_address_id
        )
        if normalized_address_row is not None:
            # Prefer geocoder-normalized field values stored in the normalized_addresses table.
            return NormalizedAddress(
                address_line_1=normalized_address_row.address_line_1,
                address_line_2=normalized_address_row.address_line_2,
                city=normalized_address_row.city,
                state=normalized_address_row.state,
                postal_code=normalized_address_row.postal_code,
                country=normalized_address_row.country,
                formatted_address=request.normalized_address,
                latitude=normalized_address_row.latitude,
                longitude=normalized_address_row.longitude,
                geocoding_source=normalized_address_row.geocoding_source,
            )
        # Fallback: reconstruct from raw request fields when the normalized address
        # record is unexpectedly missing (should not occur in normal operation).
        return NormalizedAddress(
            address_line_1=" ".join(request.address_line_1.strip().upper().split()),
            address_line_2=(
                " ".join(request.address_line_2.strip().upper().split())
                if request.address_line_2
                else None
            ),
            city=" ".join(request.city.strip().upper().split()),
            state=request.state.strip().upper(),
            postal_code=request.postal_code.strip().upper(),
            country=request.country.strip().upper(),
            formatted_address=request.normalized_address,
            latitude=None,
            longitude=None,
            geocoding_source=None,
        )