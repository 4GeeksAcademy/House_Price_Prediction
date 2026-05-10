from __future__ import annotations

from datetime import UTC, datetime
from dataclasses import replace
from pathlib import Path
from time import sleep
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import text

from house_price_prediction.api.main import create_app
from house_price_prediction.config import Settings
from house_price_prediction.domain.contracts.prediction_contracts import (
    GeocodingResultContract,
    NormalizedAddress,
    ProviderResponseContract,
)
from house_price_prediction.infrastructure.providers.census_property_data_client import (
    CensusPropertyDataClient,
)
from house_price_prediction.infrastructure.db.repositories import PredictionRepository
from house_price_prediction.infrastructure.db.session import init_database
from house_price_prediction.infrastructure.providers.fallback_geocoding_provider import (
    FallbackGeocodingProvider,
)
from house_price_prediction.infrastructure.providers.fallback_property_data_provider import (
    FallbackPropertyDataProvider,
)
from house_price_prediction.infrastructure.providers.fake_geocoding_client import FakeGeocodingClient
from house_price_prediction.infrastructure.providers.fake_property_data_client import (
    FakePropertyDataClient,
)
from house_price_prediction.infrastructure.providers.heuristic_property_data_client import (
    HeuristicPropertyDataClient,
)
from house_price_prediction.infrastructure.providers.resilient import (
    ProviderExecutionError,
    ResilientPropertyDataProvider,
)
from house_price_prediction.data import load_dataset
from house_price_prediction.model import load_model_artifact, save_model_artifact, train_and_save_model


class FailingPropertyProvider:
    def fetch_property_features(self, normalized_address) -> ProviderResponseContract:
        raise RuntimeError("boom")


class SlowPropertyProvider:
    def fetch_property_features(self, normalized_address) -> ProviderResponseContract:
        sleep(0.05)
        return ProviderResponseContract(
            provider_name="slow",
            status="success",
            payload={"LotArea": 1000},
            fetched_at=datetime.now(UTC),
        )


class FailingGeocoder:
    def normalize(self, address_payload):
        raise RuntimeError("no match")


class AlwaysFailPropertyProvider:
    def fetch_property_features(self, normalized_address) -> ProviderResponseContract:
        raise RuntimeError("upstream unavailable")


def build_test_settings(tmp_path: Path) -> Settings:
    return Settings(
        raw_data_path=Path("data/raw/housing.csv"),
        target_column="SalePrice",
        model_path=tmp_path / "model.joblib",
        test_size=0.2,
        random_state=42,
        app_name="House Price Prediction API Test",
        app_env="test",
        api_host="127.0.0.1",
        api_port=8001,
        database_url=f"sqlite:///{tmp_path / 'backend_test.db'}",
        model_name="house-price-random-forest",
        model_version="test-version",
        enable_mock_predictor=True,
        property_data_provider="fake",
        geocoding_provider="fake",
        prediction_reuse_max_age_hours=24,
        provider_timeout_seconds=3.0,
        provider_max_retries=2,
    )


def test_prediction_endpoint_persists_and_fetches_result(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "123 Main St",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33101",
                "country": "US",
                "requested_by": "team@example.com",
            },
        )

        assert response.status_code == 201
        payload = response.json()
        assert payload["status"] == "completed"
        assert payload["predicted_price"] > 0
        assert payload["model_version"] == "test-version"

        fetch_response = client.get(f"/v1/predictions/{payload['prediction_id']}")
        assert fetch_response.status_code == 200
        fetch_payload = fetch_response.json()
        assert fetch_payload["request_id"] == payload["request_id"]
        assert fetch_payload["predicted_price"] == payload["predicted_price"]
        assert fetch_payload["normalized_address"]["city"] == "MIAMI"
        assert fetch_payload["normalized_address"]["geocoding_source"] == "fake"
        assert fetch_payload["feature_snapshot"]["feature_count"] >= 1
        assert fetch_payload["provider_responses"][1]["feature_source"] == "fake"
        assert fetch_payload["provider_responses"][1]["feature_provenance"]["strategy"] == "deterministic_fake"
        assert fetch_payload["source_prediction_id"] is None


def test_neighborhood_score_injected_from_national_scorer(tmp_path: Path):
    """NeighborhoodScore must be non-null in the feature snapshot for any geocoded address.

    The national NeighborhoodScoreService (30k+ ZCTA reference points) is loaded
    at app startup and injected into the enriched payload right before feature
    assembly.  This test verifies the injection end-to-end using the real scorer
    artifact and the fake geocoding/property providers.
    """
    from house_price_prediction.application.services.neighborhood_score_service import (
        NeighborhoodScoreService,
    )
    from house_price_prediction.infrastructure.providers.fake_geocoding_client import (
        FakeGeocodingClient,
    )

    scorer_path = Path("models/neighborhood_scorer.joblib")
    if not scorer_path.exists():
        import pytest
        pytest.skip("neighborhood_scorer.joblib not found — run seed_national_neighborhood_scorer.py first")

    settings = build_test_settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "350 5th Ave",
                "city": "New York",
                "state": "NY",
                "postal_code": "10118",
                "country": "US",
            },
        )
        assert response.status_code == 201
        prediction_id = response.json()["prediction_id"]

        detail = client.get(f"/v1/predictions/{prediction_id}").json()
        features = detail["feature_snapshot"]["features"]
        # National scorer must have injected a real score, not null
        neighborhood_score = features.get("NeighborhoodScore")
        assert neighborhood_score is not None, (
            "NeighborhoodScore must not be null — "
            "national scorer should have injected a value from lat/lon"
        )
        assert 0.0 <= neighborhood_score <= 100.0, (
            f"NeighborhoodScore {neighborhood_score} outside [0, 100]"
        )


def test_health_endpoint_reports_runtime_state(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.get("/v1/health")

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["mock_predictor_enabled"] is True
        assert payload["model_available"] is True
        assert payload["property_data_provider"] == "fake"
        assert payload["geocoding_provider"] == "fake"
        assert payload["provider_timeout_seconds"] == 3.0
        assert payload["provider_max_retries"] == 2
        assert payload["provider_response_cache_max_age_hours"] == 24
        assert payload["feature_policy_name"] == "balanced-v1"
        assert payload["feature_policy_version"] == "v1"
        assert payload["feature_policy_state_override_count"] == 0
        assert payload["live_mode_ready"] is False
        assert any("Mock predictor" in issue for issue in payload["live_mode_issues"])
        assert any("Geocoding provider is fake" in issue for issue in payload["live_mode_issues"])
        assert any("Property data provider is fake" in issue for issue in payload["live_mode_issues"])


def test_health_endpoint_warns_when_free_provider_timeout_is_too_short(tmp_path: Path):
    """When free/free-fallback providers are configured, the health endpoint must
    add a live_mode_issues entry when provider_timeout_seconds < 20.0.
    Census ACS enrichment makes two sequential HTTP calls and can easily take
    20+ seconds; a shorter timeout will cause silent provider failures.
    """
    settings = replace(
        build_test_settings(tmp_path),
        property_data_provider="free-fallback",
        geocoding_provider="free-fallback",
        # provider_timeout_seconds defaults to 3.0 in build_test_settings — well below 20.0
    )

    with TestClient(create_app(settings)) as client:
        response = client.get("/v1/health")

    assert response.status_code == 200
    payload = response.json()
    timeout_issues = [i for i in payload["live_mode_issues"] if "provider_timeout_seconds" in i]
    assert len(timeout_issues) == 1, (
        f"Expected exactly one timeout warning; got: {payload['live_mode_issues']}"
    )
    assert "20" in timeout_issues[0], "Warning should mention the minimum recommended timeout"


def test_health_endpoint_no_timeout_warning_when_timeout_is_adequate(tmp_path: Path):
    """No timeout warning should appear when provider_timeout_seconds >= 20.0,
    even with free/free-fallback providers configured.
    """
    settings = replace(
        build_test_settings(tmp_path),
        property_data_provider="free-fallback",
        geocoding_provider="free-fallback",
        provider_timeout_seconds=25.0,
    )

    with TestClient(create_app(settings)) as client:
        response = client.get("/v1/health")

    assert response.status_code == 200
    payload = response.json()
    timeout_issues = [i for i in payload["live_mode_issues"] if "provider_timeout_seconds" in i]
    assert len(timeout_issues) == 0, (
        f"No timeout warning expected for 25s timeout; got: {payload['live_mode_issues']}"
    )



    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        for address_line_1 in ["123 Main St", "456 Oak Ave"]:
            response = client.post(
                "/v1/predictions",
                json={
                    "address_line_1": address_line_1,
                    "city": "Miami",
                    "state": "FL",
                    "postal_code": "33101",
                    "country": "US",
                },
            )
            assert response.status_code == 201

        list_response = client.get("/v1/predictions?limit=2")

    assert list_response.status_code == 200
    payload = list_response.json()
    assert len(payload["items"]) == 2
    assert payload["items"][0]["feature_source"] == "fake"
    assert payload["items"][0]["normalized_address"]["country"] == "US"


def test_dashboard_bootstrap_returns_runtime_policy_and_recent_predictions(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        create_response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "123 Main St",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33101",
                "country": "US",
            },
        )
        assert create_response.status_code == 201

        response = client.get("/v1/dashboard/bootstrap?limit=1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["contract_version"] == "2026-04-16"
    assert payload["runtime"]["app_name"] == settings.app_name
    assert payload["runtime"]["model_version"] == settings.model_version
    assert payload["provider_policy"]["geocoding_provider"] == settings.geocoding_provider
    assert payload["provider_policy"]["property_data_provider"] == settings.property_data_provider
    assert payload["provider_policy"]["feature_policy_name"] == settings.feature_policy_name
    assert payload["provider_policy"]["feature_policy_version"] == settings.feature_policy_version
    assert payload["provider_policy"]["feature_policy_state_overrides"] == settings.feature_policy_state_overrides
    assert payload["links"]["prediction_detail"] == "/v1/predictions/{prediction_id}"
    assert len(payload["recent_predictions"]) == 1
    assert payload["recent_predictions"][0]["normalized_address"]["city"] == "MIAMI"
    assert payload["event_summary"]["prediction_id"] == payload["recent_predictions"][0]["prediction_id"]
    assert payload["event_summary"]["total_events"] >= 1
    assert payload["event_summary"]["latest_event_name"] is not None
    assert isinstance(payload["event_summary"]["recent_event_names"], list)


def test_api_capabilities_endpoint_exposes_frontend_ml_contract(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.get("/v1/meta/capabilities")

    assert response.status_code == 200
    payload = response.json()
    assert payload["contract_version"] == "2026-04-17"
    assert payload["runtime"]["app_name"] == settings.app_name
    assert payload["runtime"]["mock_predictor_enabled"] is True
    assert payload["provider_policy"]["property_data_provider"] == settings.property_data_provider
    assert payload["provider_policy"]["geocoding_provider"] == settings.geocoding_provider
    assert isinstance(payload["model_expected_features"], list)
    assert "LotArea" in payload["model_expected_features"]
    assert payload["live_mode_ready"] is False
    assert any("Mock predictor" in issue for issue in payload["live_mode_issues"])
    endpoint_names = {item["name"] for item in payload["endpoints"]}
    assert "create_prediction" in endpoint_names
    assert "api_capabilities" in endpoint_names
    assert "live_feature_candidates" in endpoint_names
    assert payload["examples"]["prediction_request"]["address_line_1"] == "1600 Pennsylvania Ave NW"


def test_live_feature_candidates_endpoint_returns_feature_rows(tmp_path: Path):
    settings = replace(
        build_test_settings(tmp_path),
        prediction_reuse_max_age_hours=0,
    )

    with TestClient(create_app(settings)) as client:
        for address in ["100 Alpha St", "200 Beta St", "300 Gamma St"]:
            response = client.post(
                "/v1/predictions",
                json={
                    "address_line_1": address,
                    "city": "Miami",
                    "state": "FL",
                    "postal_code": "33101",
                    "country": "US",
                },
            )
            assert response.status_code == 201

        export_response = client.get(
            "/v1/meta/live-feature-candidates?limit=10&offset=0&min_completeness_score=0.8"
        )

    assert export_response.status_code == 200
    payload = export_response.json()
    assert payload["contract_version"] == "2026-04-17"
    assert payload["limit"] == 10
    assert payload["offset"] == 0
    assert payload["include_reused"] is False
    assert payload["total"] >= 3
    assert len(payload["items"]) >= 3

    first = payload["items"][0]
    assert first["was_reused"] is False
    assert first["completeness_score"] >= 0.8
    assert first["features"]["LotArea"] is not None
    assert first["features"]["OverallQual"] is not None
    assert first["normalized_address"]["state"] == "FL"


def test_live_feature_candidates_include_reused_toggle(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        first = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "77 Reuse Lane",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33101",
                "country": "US",
            },
        )
        second = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "77 Reuse Lane",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33101",
                "country": "US",
            },
        )
        assert first.status_code == 201
        assert second.status_code == 201
        assert first.json()["was_reused"] is False
        assert second.json()["was_reused"] is True

        no_reused = client.get("/v1/meta/live-feature-candidates?include_reused=false")
        with_reused = client.get("/v1/meta/live-feature-candidates?include_reused=true")

    assert no_reused.status_code == 200
    assert with_reused.status_code == 200
    no_reused_payload = no_reused.json()
    with_reused_payload = with_reused.json()

    assert no_reused_payload["include_reused"] is False
    assert with_reused_payload["include_reused"] is True
    assert all(item["was_reused"] is False for item in no_reused_payload["items"])
    assert any(item["was_reused"] is True for item in with_reused_payload["items"])


def test_reused_prediction_detail_surfaces_source_lineage(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        first_response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "1600 Pennsylvania Ave NW",
                "city": "Washington",
                "state": "DC",
                "postal_code": "20500",
                "country": "US",
            },
        )
        second_response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "1600 Pennsylvania Ave NW",
                "city": "Washington",
                "state": "DC",
                "postal_code": "20500",
                "country": "US",
            },
        )

        assert first_response.status_code == 201
        assert second_response.status_code == 201
        second_payload = second_response.json()
        assert second_payload["was_reused"] is True
        assert second_payload["source_prediction_id"] == first_response.json()["prediction_id"]

        detail_response = client.get(f"/v1/predictions/{second_payload['prediction_id']}")

    assert detail_response.status_code == 200
    detail_payload = detail_response.json()
    assert detail_payload["was_reused"] is True
    assert detail_payload["source_prediction_id"] == first_response.json()["prediction_id"]
    assert len(detail_payload["provider_responses"]) == 2
    assert detail_payload["provider_responses"][1]["feature_source"] == "fake"


def test_prediction_trace_endpoint_returns_lineage_nodes(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        first_response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "1600 Pennsylvania Ave NW",
                "city": "Washington",
                "state": "DC",
                "postal_code": "20500",
                "country": "US",
            },
        )
        second_response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "1600 Pennsylvania Ave NW",
                "city": "Washington",
                "state": "DC",
                "postal_code": "20500",
                "country": "US",
            },
        )

        trace_response = client.get(
            f"/v1/predictions/{second_response.json()['prediction_id']}/trace"
        )

    assert first_response.status_code == 201
    assert second_response.status_code == 201
    assert trace_response.status_code == 200
    trace_payload = trace_response.json()
    assert trace_payload["was_reused"] is True
    assert trace_payload["root_prediction_id"] == first_response.json()["prediction_id"]
    assert len(trace_payload["trace_nodes"]) == 2
    assert trace_payload["trace_nodes"][0]["prediction_id"] == second_response.json()["prediction_id"]
    assert trace_payload["trace_nodes"][1]["prediction_id"] == first_response.json()["prediction_id"]
    assert trace_payload["provider_responses"][1]["provider_name"] == "fake_property_data"


def test_prediction_persists_provider_response(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "456 Oak Ave",
                "city": "Orlando",
                "state": "FL",
                "postal_code": "32801",
                "country": "US",
            },
        )

        assert response.status_code == 201
        payload = response.json()

    session_factory = init_database(settings.database_url)
    with session_factory() as session:
        repository = PredictionRepository(session)
        provider_responses = repository.get_provider_responses(payload["request_id"])

    assert len(provider_responses) == 2
    assert provider_responses[0].provider_name == "fake_geocoding"
    assert provider_responses[0].status == "success"
    assert "normalized" in provider_responses[0].payload
    assert provider_responses[1].provider_name == "fake_property_data"
    assert provider_responses[1].status == "success"
    assert "LotArea" in provider_responses[1].payload
    assert provider_responses[1].payload["feature_provenance"]["strategy"] == "deterministic_fake"


def test_model_artifact_preserves_feature_metadata(tmp_path: Path):
    model_path = tmp_path / "artifact.joblib"

    save_model_artifact(
        model={"kind": "dummy-model"},
        model_path=model_path,
        feature_columns=["LotArea", "OverallQual", "Neighborhood"],
        target_column="SalePrice",
        model_name="house-price-random-forest",
        model_version="2026.04",
    )

    artifact = load_model_artifact(model_path)

    assert artifact.model == {"kind": "dummy-model"}
    assert artifact.metadata.feature_columns == ("LotArea", "OverallQual", "Neighborhood")
    assert artifact.metadata.target_column == "SalePrice"
    assert artifact.metadata.model_version == "2026.04"


def test_repeated_requests_reuse_canonical_normalized_address_and_register_model(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    responses = []
    with TestClient(create_app(settings)) as client:
        for _ in range(2):
            response = client.post(
                "/v1/predictions",
                json={
                    "address_line_1": " 789 Pine Road ",
                    "city": "Tampa",
                    "state": "fl",
                    "postal_code": "33602",
                    "country": "us",
                },
            )
            assert response.status_code == 201
            responses.append(response.json())

    session_factory = init_database(settings.database_url)
    with session_factory() as session:
        repository = PredictionRepository(session)
        assert repository.count_normalized_addresses() == 1

        registry_entry = repository.get_model_registry_entry(
            model_name=settings.model_name,
            model_version=settings.model_version,
        )

    assert registry_entry is not None
    assert registry_entry.is_active is True
    assert "LotArea" in registry_entry.feature_columns
    assert responses[0]["was_reused"] is False
    assert responses[1]["was_reused"] is True
    assert responses[0]["predicted_price"] == responses[1]["predicted_price"]


def test_reuse_can_be_disabled_with_zero_hour_freshness_window(tmp_path: Path):
    settings = replace(build_test_settings(tmp_path), prediction_reuse_max_age_hours=0)

    responses = []
    with TestClient(create_app(settings)) as client:
        for _ in range(2):
            response = client.post(
                "/v1/predictions",
                json={
                    "address_line_1": "22 Sunset Blvd",
                    "city": "Miami",
                    "state": "FL",
                    "postal_code": "33101",
                    "country": "US",
                },
            )
            assert response.status_code == 201
            responses.append(response.json())

    assert responses[0]["was_reused"] is False
    assert responses[1]["was_reused"] is False


def test_property_enrichment_cache_is_used_when_prediction_reuse_is_disabled(tmp_path: Path):
    settings = replace(
        build_test_settings(tmp_path),
        prediction_reuse_max_age_hours=0,
        provider_response_cache_max_age_hours=24,
    )

    with TestClient(create_app(settings)) as client:
        first = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "22 Sunset Blvd",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33101",
                "country": "US",
            },
        )
        assert first.status_code == 201

        second = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "22 Sunset Blvd",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33101",
                "country": "US",
            },
        )
        assert second.status_code == 201

        first_id = first.json()["prediction_id"]
        second_id = second.json()["prediction_id"]

        first_events = client.get(f"/v1/predictions/{first_id}/events?sort=asc")
        second_events = client.get(f"/v1/predictions/{second_id}/events?sort=asc")
        assert first_events.status_code == 200
        assert second_events.status_code == 200

        first_property_event = next(
            event
            for event in first_events.json()["events"]
            if event["event_name"] == "property_enrichment_completed"
        )
        second_property_event = next(
            event
            for event in second_events.json()["events"]
            if event["event_name"] == "property_enrichment_completed"
        )

        assert first_property_event["payload"]["provider_cache_hit"] is False
        assert second_property_event["payload"]["provider_cache_hit"] is True

        second_detail = client.get(f"/v1/predictions/{second_id}")
        assert second_detail.status_code == 200
        provider_names = [item["provider_name"] for item in second_detail.json()["provider_responses"]]
        assert any(name.endswith("_cache") for name in provider_names)


def test_reuse_does_not_cross_feature_policy_boundaries(tmp_path: Path):
    base_settings = build_test_settings(tmp_path)
    first_policy_settings = replace(
        base_settings,
        feature_policy_name="balanced-v1",
    )
    second_policy_settings = replace(
        base_settings,
        feature_policy_name="quality-first-v1",
    )

    with TestClient(create_app(first_policy_settings)) as first_client:
        first_response = first_client.post(
            "/v1/predictions",
            json={
                "address_line_1": "55 Policy Way",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33101",
                "country": "US",
            },
        )

    with TestClient(create_app(second_policy_settings)) as second_client:
        second_response = second_client.post(
            "/v1/predictions",
            json={
                "address_line_1": "55 Policy Way",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33101",
                "country": "US",
            },
        )

    assert first_response.status_code == 201
    assert second_response.status_code == 201
    assert first_response.json()["was_reused"] is False
    assert second_response.json()["was_reused"] is False
    assert first_response.json()["selected_feature_policy_name"] == "balanced-v1"
    assert second_response.json()["selected_feature_policy_name"] == "quality-first-v1"


def test_normalize_route_uses_geocoding_provider(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/properties/normalize",
            json={
                "address_line_1": " 123 main st ",
                "city": "miami",
                "state": "fl",
                "postal_code": "33101",
                "country": "us",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["address_line_1"] == "123 MAIN ST"
    assert payload["city"] == "MIAMI"
    assert payload["formatted_address"] == "123 MAIN ST, MIAMI, FL 33101, US"


def test_prediction_maps_provider_failure_to_bad_gateway(tmp_path: Path):
    settings = build_test_settings(tmp_path)
    failing_provider = ResilientPropertyDataProvider(
        provider_name="failing",
        delegate=FailingPropertyProvider(),
        timeout_seconds=0.1,
        max_retries=1,
    )

    with TestClient(create_app(settings, property_data_provider=failing_provider)) as client:
        response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "500 Brickell Ave",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33131",
                "country": "US",
            },
        )

    assert response.status_code == 502
    assert "Provider 'failing' failed" in response.json()["detail"]


def test_prediction_persists_failed_provider_snapshot_and_failed_request_status(tmp_path: Path):
    settings = build_test_settings(tmp_path)
    failing_provider = ResilientPropertyDataProvider(
        provider_name="failing",
        delegate=FailingPropertyProvider(),
        timeout_seconds=0.1,
        max_retries=1,
    )
    correlation_id = "11111111-1111-4111-8111-111111111111"

    with TestClient(create_app(settings, property_data_provider=failing_provider)) as client:
        response = client.post(
            "/v1/predictions",
            headers={"x-correlation-id": correlation_id},
            json={
                "address_line_1": "500 Brickell Ave",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33131",
                "country": "US",
            },
        )

    assert response.status_code == 502

    session_factory = init_database(settings.database_url)
    with session_factory() as session:
        repository = PredictionRepository(session)
        request_id = repository.get_request_id_by_correlation_id(correlation_id)

        assert request_id is not None
        assert repository.get_request_status(request_id) == "failed"

        provider_responses = repository.get_provider_responses(request_id)

    assert len(provider_responses) == 2
    assert provider_responses[0].provider_name == "fake_geocoding"
    assert provider_responses[0].status == "success"
    assert provider_responses[1].provider_name == "failing"
    assert provider_responses[1].status == "failed"
    assert provider_responses[1].payload["stage"] == "provider_execution"


def test_normalize_route_maps_provider_failure_to_bad_gateway(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(
        create_app(
            settings,
            geocoding_provider=type(
                "AlwaysFailGeocoder",
                (),
                {
                    "normalize": lambda self, payload: (_ for _ in ()).throw(
                        ProviderExecutionError("failing-geocoder", "Provider 'failing-geocoder' failed: no match")
                    )
                },
            )(),
        )
    ) as client:
        response = client.post(
            "/v1/properties/normalize",
            json={
                "address_line_1": "123 main st",
                "city": "miami",
                "state": "fl",
                "postal_code": "33101",
                "country": "us",
            },
        )

    assert response.status_code == 502
    assert "failing-geocoder" in response.json()["detail"]


def test_resilient_provider_times_out(tmp_path: Path):
    settings = build_test_settings(tmp_path)
    slow_provider = ResilientPropertyDataProvider(
        provider_name="slow",
        delegate=SlowPropertyProvider(),
        timeout_seconds=0.01,
        max_retries=0,
    )

    with TestClient(create_app(settings, property_data_provider=slow_provider)) as client:
        response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "100 Ocean Dr",
                "city": "Miami Beach",
                "state": "FL",
                "postal_code": "33139",
                "country": "US",
            },
        )

    assert response.status_code == 502
    assert "timed out" in response.json()["detail"]


def test_fallback_geocoder_uses_next_provider():
    provider = FallbackGeocodingProvider(
        providers=(FailingGeocoder(), FakeGeocodingClient())
    )

    result = provider.normalize(
        address_payload=type(
            "Payload",
            (),
            {
                "address_line_1": "1 Main St",
                "address_line_2": None,
                "city": "Miami",
                "state": "FL",
                "postal_code": "33101",
                "country": "US",
            },
        )()
    )

    assert isinstance(result, GeocodingResultContract)
    assert result.normalized_address.formatted_address == "1 MAIN ST, MIAMI, FL 33101, US"
    assert result.normalized_address.geocoding_source == "fake"


def test_census_property_provider_falls_back_without_coordinates():
    provider = CensusPropertyDataClient(fallback_provider=FailingPropertyProvider())
    normalized_address = NormalizedAddress(
        address_line_1="1 MAIN ST",
        city="MIAMI",
        state="FL",
        postal_code="33101",
        country="US",
        formatted_address="1 MAIN ST, MIAMI, FL 33101, US",
    )

    try:
        provider.fetch_property_features(normalized_address)
        raised = False
    except RuntimeError:
        raised = True

    assert raised is True


def test_census_property_provider_uses_fake_backfill_without_coordinates():
    provider = CensusPropertyDataClient(fallback_provider=FakePropertyDataClient())
    normalized_address = NormalizedAddress(
        address_line_1="1 MAIN ST",
        city="MIAMI",
        state="FL",
        postal_code="33101",
        country="US",
        formatted_address="1 MAIN ST, MIAMI, FL 33101, US",
    )

    response = provider.fetch_property_features(normalized_address)

    assert response.status == "success"
    assert response.provider_name.endswith("fallback")
    assert "LotArea" in response.payload
    assert response.payload["feature_provenance"]["used_backfill"] is True
    assert response.payload["feature_provenance"]["used_census"] is False


def test_property_fallback_uses_next_provider():
    provider = FallbackPropertyDataProvider(
        providers=(AlwaysFailPropertyProvider(), HeuristicPropertyDataClient(), FakePropertyDataClient())
    )
    normalized_address = NormalizedAddress(
        address_line_1="1 MAIN ST",
        city="MIAMI",
        state="FL",
        postal_code="33101",
        country="US",
        formatted_address="1 MAIN ST, MIAMI, FL 33101, US",
    )

    response = provider.fetch_property_features(normalized_address)

    assert response.status == "success"
    assert response.provider_name == "heuristic_property_data"
    assert response.payload["LotArea"] >= 5000
    assert response.payload["feature_source"] == "heuristic"
    assert response.payload["feature_provenance"]["strategy"] == "heuristic"


def test_census_property_provider_annotates_census_and_backfill_provenance():
    provider = CensusPropertyDataClient(fallback_provider=HeuristicPropertyDataClient())

    provider._lookup_census_tract = lambda latitude, longitude: {
        "state": "11",
        "county": "001",
        "tract": "980000",
        "name": "Downtown, Sample",
    }
    provider._fetch_census_context = lambda geography: {
        "B25077_001E": "650000",
        "B25035_001E": "1998",
        "B25018_001E": "7.1",
    }

    normalized_address = NormalizedAddress(
        address_line_1="1 MAIN ST",
        city="WASHINGTON",
        state="DC",
        postal_code="20500",
        country="US",
        formatted_address="1 MAIN ST, WASHINGTON, DC 20500, US",
        latitude=38.8976,
        longitude=-77.0365,
    )

    response = provider.fetch_property_features(normalized_address)

    assert response.provider_name == "census_context_with_backfill"
    assert response.payload["feature_source"] == "census_context_with_backfill"
    assert response.payload["feature_provenance"]["used_census"] is True
    assert response.payload["feature_provenance"]["used_backfill"] is True
    assert response.payload["feature_provenance"]["backfill_provider"] == "heuristic_property_data"


def test_prediction_guardrail_rejects_po_box(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "P.O. Box 123",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33101",
                "country": "US",
            },
        )

    assert response.status_code == 422
    assert "PO boxes" in response.json()["detail"]


def test_prediction_guardrail_rejects_non_us_for_free_providers(tmp_path: Path):
    settings = replace(
        build_test_settings(tmp_path),
        property_data_provider="free-fallback",
        geocoding_provider="free-fallback",
    )

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "111 Wellington St",
                "city": "Ottawa",
                "state": "ON",
                "postal_code": "K1A0A9",
                "country": "CA",
            },
        )

    assert response.status_code == 422
    assert "only US addresses" in response.json()["detail"]


def test_normalize_guardrail_rejects_po_box(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/properties/normalize",
            json={
                "address_line_1": "PO Box 88",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33101",
                "country": "US",
            },
        )

    assert response.status_code == 422
    assert "PO boxes" in response.json()["detail"]


def test_non_mock_prediction_path_with_trained_artifact(tmp_path: Path):
    data_path = tmp_path / "housing.csv"
    pd = __import__("pandas")

    training_data = pd.DataFrame(
        {
            "LotArea": [8500, 9600, 11250, 9550, 14260, 14115],
            "OverallQual": [7, 6, 7, 7, 8, 5],
            "OverallCond": [5, 8, 5, 5, 5, 5],
            "YearBuilt": [2003, 1976, 2001, 1915, 2000, 1993],
            "YearRemodAdd": [2003, 1976, 2002, 1970, 2000, 1995],
            "GrLivArea": [1710, 1262, 1786, 1717, 2198, 1362],
            "FullBath": [2, 2, 2, 1, 2, 1],
            "HalfBath": [1, 0, 1, 0, 1, 1],
            "BedroomAbvGr": [3, 3, 3, 3, 4, 3],
            "TotRmsAbvGrd": [8, 6, 6, 7, 9, 7],
            "Fireplaces": [0, 1, 1, 1, 1, 0],
            "GarageCars": [2, 2, 2, 3, 3, 2],
            "GarageArea": [548, 460, 608, 642, 836, 480],
            "Neighborhood": ["CollgCr", "Veenker", "CollgCr", "Crawfor", "NoRidge", "Mitchel"],
            "HouseStyle": ["2Story", "1Story", "2Story", "2Story", "2Story", "1Story"],
            "SalePrice": [208500, 181500, 223500, 140000, 250000, 143000],
        }
    )
    training_data.to_csv(data_path, index=False)

    settings = replace(
        build_test_settings(tmp_path),
        raw_data_path=data_path,
        model_path=tmp_path / "real_model.joblib",
        database_url=f"sqlite:///{tmp_path / 'non_mock_test.db'}",
        model_version="real-test-version",
        enable_mock_predictor=False,
    )

    train_and_save_model(settings)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "123 Main St",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33101",
                "country": "US",
            },
        )

    assert response.status_code == 201
    payload = response.json()
    assert payload["predicted_price"] > 0
    assert payload["model_version"] == "real-test-version"


def test_prediction_persists_brain_workflow_events(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "300 Biscayne Blvd",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33131",
                "country": "US",
            },
        )

    assert response.status_code == 201
    request_id = response.json()["request_id"]

    session_factory = init_database(settings.database_url)
    with session_factory() as session:
        rows = session.execute(
            text(
                """
                SELECT event_name
                FROM workflow_events
                WHERE request_id = :request_id
                ORDER BY occurred_at ASC
                """
            ),
            {"request_id": request_id},
        ).fetchall()

    event_names = [row[0] for row in rows]
    assert "prediction_received" in event_names
    assert "address_normalized" in event_names
    assert "reuse_candidate_evaluated" in event_names
    assert "property_enrichment_completed" in event_names
    assert "feature_vector_created" in event_names
    assert "prediction_completed" in event_names


def test_prediction_workflow_events_endpoint_returns_event_stream(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        create_response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "500 Brickell Ave",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33131",
                "country": "US",
            },
        )
        assert create_response.status_code == 201
        prediction_id = create_response.json()["prediction_id"]

        events_response = client.get(f"/v1/predictions/{prediction_id}/events")

    assert events_response.status_code == 200
    payload = events_response.json()
    assert payload["prediction_id"] == prediction_id
    assert payload["total_count"] >= 4
    assert payload["limit"] == 100
    assert payload["offset"] == 0
    assert payload["event_name"] is None
    assert len(payload["events"]) >= 4
    event_names = [item["event_name"] for item in payload["events"]]
    assert "prediction_received" in event_names
    assert "prediction_completed" in event_names


def test_prediction_workflow_events_endpoint_supports_filter_and_pagination(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        create_response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "900 Bayfront Pkwy",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33132",
                "country": "US",
            },
        )
        assert create_response.status_code == 201
        prediction_id = create_response.json()["prediction_id"]

        filtered_response = client.get(
            f"/v1/predictions/{prediction_id}/events",
            params={"event_name": "prediction_completed", "limit": 1, "offset": 0},
        )

    assert filtered_response.status_code == 200
    payload = filtered_response.json()
    assert payload["event_name"] == "prediction_completed"
    assert payload["limit"] == 1
    assert payload["offset"] == 0
    assert payload["sort"] == "asc"
    assert payload["total_count"] >= 1
    assert len(payload["events"]) == 1
    assert payload["events"][0]["event_name"] == "prediction_completed"


def test_prediction_workflow_events_endpoint_supports_desc_sort(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        create_response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "77 Harbor Dr",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33132",
                "country": "US",
            },
        )
        assert create_response.status_code == 201
        prediction_id = create_response.json()["prediction_id"]

        events_response = client.get(
            f"/v1/predictions/{prediction_id}/events",
            params={"sort": "desc", "limit": 1},
        )

    assert events_response.status_code == 200
    payload = events_response.json()
    assert payload["sort"] == "desc"
    assert len(payload["events"]) == 1
    assert payload["events"][0]["event_name"] == "prediction_completed"


def test_prediction_workflow_events_endpoint_returns_404_for_unknown_prediction(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.get(f"/v1/predictions/{uuid4()}/events")

    assert response.status_code == 404
    assert response.json()["detail"] == "Prediction not found."


def test_prediction_trace_includes_workflow_events(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        create_response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "101 Ocean Dr",
                "city": "Miami Beach",
                "state": "FL",
                "postal_code": "33139",
                "country": "US",
            },
        )
        assert create_response.status_code == 201
        prediction_id = create_response.json()["prediction_id"]

        trace_response = client.get(f"/v1/predictions/{prediction_id}/trace")

    assert trace_response.status_code == 200
    payload = trace_response.json()
    assert "workflow_events" in payload
    assert len(payload["workflow_events"]) >= 1
    assert payload["workflow_events"][0]["event_name"] == "prediction_received"


def test_feature_policy_catalog_endpoint_returns_runtime_defaults_and_definitions(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.get("/v1/policies/feature")

    assert response.status_code == 200
    payload = response.json()
    assert payload["default_policy_name"] == settings.feature_policy_name
    assert payload["default_policy_version"] == settings.feature_policy_version
    assert payload["state_overrides"] == {}

    policy_names = [item["name"] for item in payload["policies"]]
    assert "balanced-v1" in policy_names
    assert "quality-first-v1" in policy_names
    assert "land-first-v1" in policy_names
    assert "market-context-v1" in policy_names


def test_feature_policy_simulation_endpoint_returns_requested_policy_results(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/policies/feature/simulate",
            json={
                "address_line_1": "1200 Brickell Bay Dr",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33131",
                "country": "US",
                "policy_names": ["balanced-v1", "quality-first-v1"],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["normalized_address"]["city"] == "MIAMI"
    assert payload["provider_name"] == "fake_property_data"
    assert len(payload["simulations"]) == 2

    returned_policy_names = [item["policy_name"] for item in payload["simulations"]]
    assert returned_policy_names == ["balanced-v1", "quality-first-v1"]
    for item in payload["simulations"]:
        assert item["predicted_price"] > 0
        assert 0 <= item["completeness_score"] <= 1
        assert item["weight_total"] is not None


def test_feature_policy_simulation_endpoint_rejects_unknown_policy_name(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/policies/feature/simulate",
            json={
                "address_line_1": "1200 Brickell Bay Dr",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33131",
                "country": "US",
                "policy_names": ["unknown-v1"],
            },
        )

    assert response.status_code == 422
    assert "Unknown feature policy name" in response.json()["detail"]


def test_validation_scenarios_endpoint_returns_baseline_catalog(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.get("/v1/validation/scenarios")

    assert response.status_code == 200
    payload = response.json()
    assert "scenarios" in payload
    assert isinstance(payload["scenarios"], list)


def test_address_baseline_endpoint_returns_location_feature_and_value_contract(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/validation/address-baseline",
            json={
                "address_line_1": "500 Brickell Ave",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33131",
                "country": "US",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["contract_version"] == "2026-04-16"
    assert payload["location"]["normalized_address"]["city"] == "MIAMI"
    assert payload["location"]["geocoding_provider"] == "fake_geocoding"
    assert payload["features"]["pulled_feature_count"] >= 1
    assert payload["features"]["missing_feature_count"] >= 0
    assert payload["features"]["pulled_feature_count"] + payload["features"]["missing_feature_count"] == len(payload["features"]["expected_features"])
    assert "BedroomAbvGr" in payload["features"]["key_feature_values"]
    assert "TotRmsAbvGrd" in payload["features"]["key_feature_values"]
    assert "GrLivArea" in payload["features"]["key_feature_values"]
    assert payload["value"]["predicted_price"] > 0
    assert payload["value"]["property_provider"] == "fake_property_data"
    assert payload["assessment"]["overall_status"] == "not_evaluated"


def test_address_baseline_with_scenario_returns_assessment_checks(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/validation/address-baseline",
            json={
                "address_line_1": "500 Brickell Ave",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33131",
                "country": "US",
                "expectations": {
                    "min_completeness_score": 0.98,
                    "required_features": ["BedroomAbvGr", "TotRmsAbvGrd", "GrLivArea", "LotArea"],
                    "feature_bounds": {
                        "BedroomAbvGr": {"minimum": 1, "maximum": 8},
                        "TotRmsAbvGrd": {"minimum": 2, "maximum": 14},
                        "GrLivArea": {"minimum": 500, "maximum": 6000},
                        "LotArea": {"minimum": 1000, "maximum": 50000},
                    },
                },
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["assessment"]["checks"]) >= 1
    check_names = [item["check_name"] for item in payload["assessment"]["checks"]]
    assert "completeness_threshold" in check_names
    assert any(name.startswith("required_feature:") for name in check_names)
    assert any(name.startswith("feature_bounds:") for name in check_names)


def test_full_audit_endpoint_returns_aggregated_report(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/validation/full-audit",
            json={
                "address_line_1": "500 Brickell Ave",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33131",
                "country": "US",
                "requested_by": "audit@example.com",
                "expectations": {
                    "min_completeness_score": 0.98,
                    "required_features": ["BedroomAbvGr", "TotRmsAbvGrd", "GrLivArea", "LotArea"],
                },
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["contract_version"] == "2026-04-16"
    assert payload["prediction"]["status"] == "completed"
    assert payload["baseline"]["assessment"]["overall_status"] in {"passed", "failed"}
    assert isinstance(payload["issues"], list)


def test_run_scenario_batch_all_registered_scenarios_pass(tmp_path: Path):
    """
    POST /v1/validation/run-scenario-batch with no filter should execute all
    registered scenarios and return a well-formed batched result.  In test
    mode (fake providers) every scenario is expected to pass because the
    expectations are calibrated to the fake provider output range.
    """
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post("/v1/validation/run-scenario-batch", json={})

    assert response.status_code == 200
    payload = response.json()

    assert payload["contract_version"] == "2026-04-16"
    assert isinstance(payload["total"], int) and payload["total"] > 0
    assert payload["errors"] == 0, f"Unexpected errors in batch: {payload.get('results')}"
    assert payload["failed"] == 0, (
        f"Unexpected failures in batch: "
        f"{[r for r in payload['results'] if r['pipeline_status'] == 'fail']}"
    )
    assert payload["passed"] == payload["total"]

    for result in payload["results"]:
        assert result["pipeline_status"] == "pass"
        assert result["completeness_score"] is not None
        assert result["predicted_price"] is not None
        assert result["issues"] == []
        assert isinstance(result["key_feature_values"], dict)


def test_run_scenario_batch_single_scenario_by_id(tmp_path: Path):
    """Filtering by scenario_id should return only the specified scenario."""
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/validation/run-scenario-batch",
            json={"scenario_ids": ["college-town-ia"]},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["results"][0]["scenario_id"] == "college-town-ia"
    assert payload["results"][0]["pipeline_status"] == "pass"


# ---------------------------------------------------------------------------
# Regression tests for bugs fixed in the audit pass
# ---------------------------------------------------------------------------

def test_baseline_with_null_completeness_score_does_not_raise(tmp_path: Path):
    """Regression for Bug 1: passing expectations with min_completeness_score=None
    must not raise TypeError when the completeness check is skipped.
    The baseline should still evaluate required_features checks.
    """
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/validation/address-baseline",
            json={
                "address_line_1": "100 E Ohio St",
                "city": "Chicago",
                "state": "IL",
                "postal_code": "60611",
                "country": "US",
                "expectations": {
                    # No completeness threshold — only required-features check
                    "min_completeness_score": None,
                    "required_features": ["BedroomAbvGr", "GrLivArea"],
                    "feature_bounds": {},
                },
            },
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    # No completeness_threshold check should exist in checks
    check_names = [c["check_name"] for c in payload["assessment"]["checks"]]
    assert "completeness_threshold" not in check_names
    # Required feature checks must be present
    assert any(n.startswith("required_feature:") for n in check_names)
    # Overall status must be a valid value, not an error
    assert payload["assessment"]["overall_status"] in {"passed", "failed"}


def test_full_audit_passing_baseline_with_missing_features_has_no_issues(tmp_path: Path):
    """Regression for Bug 2: when expectations are provided and the baseline
    assessment passes (threshold met), run_full_audit must NOT append an extra
    'Missing model features' issue entry just because some features were absent.
    The batch runner would falsely mark those scenarios as 'fail'.
    """
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        # Use a low completeness threshold that will definitely pass,
        # ensuring the baseline assessment is "passed" even if some features miss.
        response = client.post(
            "/v1/validation/full-audit",
            json={
                "address_line_1": "300 W Grand Ave",
                "city": "Chicago",
                "state": "IL",
                "postal_code": "60654",
                "country": "US",
                "expectations": {
                    "min_completeness_score": 0.01,  # near-zero: will definitely pass
                    "required_features": [],
                    "feature_bounds": {},
                },
            },
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["baseline"]["assessment"]["overall_status"] == "passed"
    # A passing baseline must not generate a spurious "Missing model features" issue
    for issue in payload["issues"]:
        assert "Missing model features" not in issue, (
            f"Spurious missing-feature issue found despite passing baseline: {issue}"
        )


def test_full_audit_not_evaluated_baseline_surfaces_missing_feature_count(tmp_path: Path):
    """When no expectations are given (not_evaluated), missing features should
    be surfaced in issues so the caller knows the data quality state.
    """
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        # Send no expectations at all
        response = client.post(
            "/v1/validation/full-audit",
            json={
                "address_line_1": "413 Duff Ave",
                "city": "Ames",
                "state": "IA",
                "postal_code": "50010",
                "country": "US",
            },
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["baseline"]["assessment"]["overall_status"] == "not_evaluated"
    # Fake provider returns all features → missing count is 0 → no issue appended.
    # This asserts the contract: if 0 missing, no issue; if > 0, issue IS present.
    missing_count = payload["baseline"]["features"]["missing_feature_count"]
    has_missing_issue = any("Missing model features" in i for i in payload["issues"])
    if missing_count == 0:
        assert not has_missing_issue
    else:
        assert has_missing_issue


def test_baseline_contract_version_is_consistent(tmp_path: Path):
    """contract_version in baseline and full-audit responses must match."""
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        baseline_resp = client.post(
            "/v1/validation/address-baseline",
            json={
                "address_line_1": "1 Infinite Loop",
                "city": "Cupertino",
                "state": "CA",
                "postal_code": "95014",
                "country": "US",
            },
        )
        audit_resp = client.post(
            "/v1/validation/full-audit",
            json={
                "address_line_1": "1 Infinite Loop",
                "city": "Cupertino",
                "state": "CA",
                "postal_code": "95014",
                "country": "US",
            },
        )

    assert baseline_resp.status_code == 200
    assert audit_resp.status_code == 200
    assert baseline_resp.json()["contract_version"] == audit_resp.json()["contract_version"]
    assert baseline_resp.json()["contract_version"] == "2026-04-16"


def test_init_database_is_migration_first_by_default(tmp_path: Path):
    database_url = f"sqlite:///{tmp_path / 'migration_first.db'}"
    session_factory = init_database(database_url)

    with session_factory() as session:
        table_rows = session.execute(
            text(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                """
            )
        ).fetchall()

    table_names = {row[0] for row in table_rows}
    assert "prediction_requests" not in table_names
    assert "workflow_events" not in table_names


def test_load_dataset_supports_jsonl(tmp_path: Path):
    dataset_path = tmp_path / "live_feature_store.jsonl"
    dataset_path.write_text(
        '{"LotArea": 9000, "OverallQual": 7, "SalePrice": 250000}\n'
        '{"LotArea": 10000, "OverallQual": 8, "SalePrice": 310000}\n',
        encoding="utf-8",
    )

    df = load_dataset(dataset_path)

    assert list(df.columns) == ["LotArea", "OverallQual", "SalePrice"]
    assert len(df) == 2
    assert float(df.iloc[1]["SalePrice"]) == 310000.0


def test_load_dataset_rejects_unknown_file_type(tmp_path: Path):
    dataset_path = tmp_path / "features.unsupported"
    dataset_path.write_text("noop", encoding="utf-8")

    try:
        _ = load_dataset(dataset_path)
    except ValueError as exc:
        assert "Unsupported dataset file type" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported dataset file type")


def test_predictions_list_response_includes_pagination_metadata(tmp_path: Path):
    """PredictionListResponse must carry total, limit, and offset fields."""
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        for address_line_1 in ["1 Alpha Rd", "2 Beta Rd", "3 Gamma Rd"]:
            r = client.post(
                "/v1/predictions",
                json={
                    "address_line_1": address_line_1,
                    "city": "Denver",
                    "state": "CO",
                    "postal_code": "80201",
                    "country": "US",
                },
            )
            assert r.status_code == 201

        # First page: limit=2, offset=0
        first_page = client.get("/v1/predictions?limit=2&offset=0")
    assert first_page.status_code == 200
    payload = first_page.json()
    assert payload["total"] == 3
    assert payload["limit"] == 2
    assert payload["offset"] == 0
    assert len(payload["items"]) == 2


def test_predictions_list_offset_returns_correct_page(tmp_path: Path):
    """offset parameter must skip the right number of rows."""
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        ids_in_order: list[str] = []
        for address_line_1 in ["10 First St", "20 Second St", "30 Third St"]:
            r = client.post(
                "/v1/predictions",
                json={
                    "address_line_1": address_line_1,
                    "city": "Austin",
                    "state": "TX",
                    "postal_code": "78701",
                    "country": "US",
                },
            )
            assert r.status_code == 201
            ids_in_order.append(r.json()["prediction_id"])

        all_items = client.get("/v1/predictions?limit=50&offset=0").json()["items"]
        second_page = client.get("/v1/predictions?limit=2&offset=2").json()

    # Results are newest-first, so third (last submitted) is at index 0 in all_items.
    assert second_page["offset"] == 2
    assert len(second_page["items"]) == 1
    assert second_page["items"][0]["prediction_id"] == all_items[2]["prediction_id"]


def test_fresh_prediction_has_null_confidence_score(tmp_path: Path):
    """confidence_score is now computed for fresh predictions using validation results.

    This provides a quick quality metric for API consumers.
    """
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        r = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "77 Oak Lane",
                "city": "Portland",
                "state": "OR",
                "postal_code": "97201",
                "country": "US",
            },
        )
        assert r.status_code == 201
        prediction_id = r.json()["prediction_id"]
        # confidence_score is computed in the prediction response
        prediction_response = r.json()
        assert prediction_response["confidence_score"] is not None
        assert 0.0 <= prediction_response["confidence_score"] <= 1.0

        detail = client.get(f"/v1/predictions/{prediction_id}")
    assert detail.status_code == 200
    # completeness_score still reaches the caller via the feature_snapshot
    assert detail.json()["feature_snapshot"]["completeness_score"] > 0.0


# ---------------------------------------------------------------------------
# Per-request preferred_policy_name and feature_overrides
# ---------------------------------------------------------------------------

def test_preferred_policy_name_overrides_server_default(tmp_path: Path):
    """preferred_policy_name in the request payload must override the server-
    level policy and be reflected in selected_feature_policy_name on the
    response and in the workflow event stream.
    """
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "88 Policy Blvd",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33101",
                "country": "US",
                "preferred_policy_name": "quality-first-v1",
            },
        )
        assert response.status_code == 201
        payload = response.json()
        assert payload["selected_feature_policy_name"] == "quality-first-v1"
        prediction_id = payload["prediction_id"]

        events = client.get(f"/v1/predictions/{prediction_id}/events").json()

    feature_vector_event = next(
        (e for e in events["events"] if e["event_name"] == "feature_vector_created"), None
    )
    assert feature_vector_event is not None
    assert feature_vector_event["payload"]["selected_feature_policy_name"] == "quality-first-v1"


def test_preferred_policy_name_unknown_returns_422(tmp_path: Path):
    """An unrecognised preferred_policy_name must be rejected with 422 before
    any DB writes or external calls happen.
    """
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "99 Bad Policy Lane",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33101",
                "country": "US",
                "preferred_policy_name": "does-not-exist-v99",
            },
        )

    assert response.status_code == 422
    assert "Unknown feature policy" in response.json()["detail"]
    assert "does-not-exist-v99" in response.json()["detail"]


def test_feature_overrides_are_applied_to_feature_vector(tmp_path: Path):
    """feature_overrides must inject the caller-supplied values into the feature
    vector.  We override LotArea with a sentinel value and confirm it surfaces
    in the stored feature snapshot.
    """
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "42 Override Ave",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33101",
                "country": "US",
                "feature_overrides": {"LotArea": 99999, "BedroomAbvGr": 7},
            },
        )
        assert response.status_code == 201
        prediction_id = response.json()["prediction_id"]

        detail = client.get(f"/v1/predictions/{prediction_id}").json()
        events = client.get(f"/v1/predictions/{prediction_id}/events").json()

    # Feature snapshot must carry the overridden values
    assert detail["feature_snapshot"]["features"]["LotArea"] == 99999
    assert detail["feature_snapshot"]["features"]["BedroomAbvGr"] == 7

    # The feature_vector_created event must record the override metadata
    fv_event = next(
        e for e in events["events"] if e["event_name"] == "feature_vector_created"
    )
    assert fv_event["payload"]["feature_override_count"] == 2
    assert "LotArea" in fv_event["payload"]["feature_override_keys"]
    assert "BedroomAbvGr" in fv_event["payload"]["feature_override_keys"]


def test_feature_overrides_bypass_prediction_reuse(tmp_path: Path):
    """When feature_overrides is present, prediction reuse must be skipped even
    if the same address was previously predicted.  Each override set is treated
    as a distinct prediction request.
    """
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        # Establish a base prediction for this address
        base = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "100 Reuse Bypass Ct",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33101",
                "country": "US",
            },
        )
        assert base.status_code == 201
        assert base.json()["was_reused"] is False

        # Second request — same address, WITH feature_overrides → must NOT reuse
        override_resp = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "100 Reuse Bypass Ct",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33101",
                "country": "US",
                "feature_overrides": {"LotArea": 55555},
            },
        )
        assert override_resp.status_code == 201
        assert override_resp.json()["was_reused"] is False

        # Third request — same address, no overrides → SHOULD reuse
        reused = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "100 Reuse Bypass Ct",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33101",
                "country": "US",
            },
        )
        assert reused.status_code == 201
        assert reused.json()["was_reused"] is True


def test_policy_simulation_rejects_empty_policy_names_list(tmp_path: Path):
    """Passing policy_names=[] must return 422 — an empty list is ambiguous and
    should not silently fall back to simulating all policies.  Callers must
    either omit the field or pass null to request all-policy simulation.
    """
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/policies/feature/simulate",
            json={
                "address_line_1": "1 Empty List St",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33101",
                "country": "US",
                "policy_names": [],
            },
        )

    assert response.status_code == 422
    assert "policy_names must not be an empty list" in response.json()["detail"]


def test_policy_simulation_null_policy_names_simulates_all_policies(tmp_path: Path):
    """Omitting policy_names (null) must simulate every registered policy."""
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/policies/feature/simulate",
            json={
                "address_line_1": "200 All Policies Dr",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33101",
                "country": "US",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    returned_names = {item["policy_name"] for item in payload["simulations"]}
    assert "balanced-v1" in returned_names
    assert "quality-first-v1" in returned_names
    assert "land-first-v1" in returned_names
    assert "market-context-v1" in returned_names


def test_feature_overrides_workflow_event_marks_bypass_reuse(tmp_path: Path):
    """reuse_candidate_evaluated workflow event must flag bypass_reuse_by_feature_overrides
    when feature_overrides is provided.
    """
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "55 Bypass Flag Rd",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33101",
                "country": "US",
                "feature_overrides": {"GarageArea": 800},
            },
        )
        assert response.status_code == 201
        prediction_id = response.json()["prediction_id"]

        events = client.get(f"/v1/predictions/{prediction_id}/events").json()

    reuse_event = next(
        e for e in events["events"] if e["event_name"] == "reuse_candidate_evaluated"
    )
    assert reuse_event["payload"]["bypass_reuse_by_feature_overrides"] is True


def test_policy_simulation_rejects_blank_only_policy_names_list(tmp_path: Path):
    """Passing policy_names=[""] (all-blank entries) must return 422 — the
    normalisation step strips and lowercases entries, so blank strings are dropped.
    The caller should receive an error rather than an empty-simulation 200.
    """
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/policies/feature/simulate",
            json={
                "address_line_1": "1 Blank Policy Ln",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33101",
                "country": "US",
                "policy_names": ["", "   "],
            },
        )

    assert response.status_code == 422
    assert "No valid policy names remain" in response.json()["detail"]


def test_baseline_inverted_feature_bounds_returns_422(tmp_path: Path):
    """Sending expectations with minimum > maximum must return 422 at input
    validation time, not silently produce a 'failed' check at runtime.
    """
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/validation/address-baseline",
            json={
                "address_line_1": "1 Bad Bounds St",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33101",
                "country": "US",
                "expectations": {
                    "feature_bounds": {
                        "LotArea": {"minimum": 50000, "maximum": 100},
                    }
                },
            },
        )

    assert response.status_code == 422


def test_address_baseline_accepts_preferred_policy_name(tmp_path: Path):
    """POST /v1/validation/address-baseline should honour preferred_policy_name
    and reflect the selected policy in the response's feature observation.
    """
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response_balanced = client.post(
            "/v1/validation/address-baseline",
            json={
                "address_line_1": "100 Policy Test Ave",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33101",
                "country": "US",
                "preferred_policy_name": "balanced-v1",
            },
        )
        response_market = client.post(
            "/v1/validation/address-baseline",
            json={
                "address_line_1": "100 Policy Test Ave",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33101",
                "country": "US",
                "preferred_policy_name": "market-context-v1",
            },
        )

    assert response_balanced.status_code == 200
    assert response_market.status_code == 200
    balanced_payload = response_balanced.json()
    market_payload = response_market.json()
    assert balanced_payload["features"]["selected_feature_policy_name"] == "balanced-v1"
    assert market_payload["features"]["selected_feature_policy_name"] == "market-context-v1"
    # market-context-v1 assigns higher weights to census signals, so its weight_total
    # is larger than balanced-v1's uniform weight_total.
    assert (
        market_payload["features"]["weight_total"]
        > balanced_payload["features"]["weight_total"]
    )


def test_address_baseline_feature_overrides_applied(tmp_path: Path):
    """feature_overrides on the baseline request should change the assembled
    feature vector and therefore the predicted price.
    """
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        baseline = client.post(
            "/v1/validation/address-baseline",
            json={
                "address_line_1": "77 Override Blvd",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33101",
                "country": "US",
            },
        )
        overridden = client.post(
            "/v1/validation/address-baseline",
            json={
                "address_line_1": "77 Override Blvd",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33101",
                "country": "US",
                "feature_overrides": {"OverallQual": 10, "GrLivArea": 4000},
            },
        )

    assert baseline.status_code == 200
    assert overridden.status_code == 200
    assert overridden.json()["value"]["predicted_price"] > baseline.json()["value"]["predicted_price"]


def test_completeness_score_out_of_range_returns_422(tmp_path: Path):
    """min_completeness_score > 1.0 must be rejected by the input validator."""
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/validation/address-baseline",
            json={
                "address_line_1": "1 Bad Score St",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33101",
                "country": "US",
                "expectations": {"min_completeness_score": 1.5},
            },
        )

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# key_features data-flow tests
# ---------------------------------------------------------------------------

from house_price_prediction.feature_schema import KEY_BUYER_FEATURES


def test_prediction_response_includes_key_features(tmp_path: Path):
    """POST /v1/predictions must return key_features populated with buyer-relevant
    property attributes — none of the values should be None.
    """
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "10 Key Feature Ave",
                "city": "Denver",
                "state": "CO",
                "postal_code": "80201",
                "country": "US",
            },
        )

    assert response.status_code == 201
    payload = response.json()
    kf = payload["key_features"]
    assert isinstance(kf, dict), "key_features must be a dict"
    assert len(kf) > 0, "key_features must not be empty on a successful prediction"
    # Every returned value must be non-None (None values must be filtered out)
    for name, value in kf.items():
        assert value is not None, f"key_features[{name!r}] should not be None"
    # At minimum, the common structural features must be present via fake provider
    for required in ("BedroomAbvGr", "GrLivArea", "LotArea", "OverallQual"):
        assert required in kf, f"Expected {required!r} in key_features"


def test_prediction_list_items_include_key_features(tmp_path: Path):
    """GET /v1/predictions list items must each carry a non-empty key_features dict."""
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        client.post(
            "/v1/predictions",
            json={
                "address_line_1": "20 List Feature Blvd",
                "city": "Denver",
                "state": "CO",
                "postal_code": "80201",
                "country": "US",
            },
        )
        list_resp = client.get("/v1/predictions?limit=5&offset=0")

    assert list_resp.status_code == 200
    items = list_resp.json()["items"]
    assert len(items) > 0
    for item in items:
        kf = item.get("key_features")
        assert isinstance(kf, dict), f"key_features missing or wrong type on list item {item['prediction_id']}"
        assert len(kf) > 0, f"key_features empty on list item {item['prediction_id']}"
        for name, value in kf.items():
            assert value is not None, f"None value in key_features[{name!r}] on list item"


def test_prediction_detail_includes_key_features(tmp_path: Path):
    """GET /v1/predictions/{id} must carry key_features identical to what the
    POST response returned.
    """
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        create_resp = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "30 Detail Feature Ct",
                "city": "Denver",
                "state": "CO",
                "postal_code": "80201",
                "country": "US",
            },
        )
        assert create_resp.status_code == 201
        prediction_id = create_resp.json()["prediction_id"]
        create_kf = create_resp.json()["key_features"]

        detail_resp = client.get(f"/v1/predictions/{prediction_id}")

    assert detail_resp.status_code == 200
    detail_kf = detail_resp.json()["key_features"]
    assert isinstance(detail_kf, dict)
    assert len(detail_kf) > 0
    # Detail key_features must match what was returned by the original POST
    assert detail_kf == create_kf


def test_reused_prediction_response_has_key_features(tmp_path: Path):
    """When a prediction is reused (was_reused=True), key_features must still be
    populated — they come from the original prediction's feature snapshot.
    """
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        first = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "40 Reuse Feature Ln",
                "city": "Denver",
                "state": "CO",
                "postal_code": "80201",
                "country": "US",
            },
        )
        assert first.status_code == 201
        assert first.json()["was_reused"] is False
        original_kf = first.json()["key_features"]

        second = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "40 Reuse Feature Ln",
                "city": "Denver",
                "state": "CO",
                "postal_code": "80201",
                "country": "US",
            },
        )

    assert second.status_code == 201
    assert second.json()["was_reused"] is True
    reused_kf = second.json()["key_features"]
    assert isinstance(reused_kf, dict)
    assert len(reused_kf) > 0, "key_features must not be empty on a reused prediction"
    # Reused prediction must carry the same feature values as the original
    assert reused_kf == original_kf


def test_key_features_only_contains_known_buyer_feature_names(tmp_path: Path):
    """key_features must only contain names drawn from KEY_BUYER_FEATURES."""
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "50 Schema Check St",
                "city": "Denver",
                "state": "CO",
                "postal_code": "80201",
                "country": "US",
            },
        )

    assert response.status_code == 201
    kf = response.json()["key_features"]
    allowed = set(KEY_BUYER_FEATURES)
    for name in kf:
        assert name in allowed, f"Unexpected feature {name!r} in key_features"


def test_feature_overrides_reflected_in_key_features(tmp_path: Path):
    """When BedroomAbvGr is overridden, key_features on the response must carry
    the caller-supplied value, not the provider's default.
    """
    settings = build_test_settings(tmp_path)
    sentinel = 9  # unlikely to match the fake provider's default

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "60 Override Check Dr",
                "city": "Denver",
                "state": "CO",
                "postal_code": "80201",
                "country": "US",
                "feature_overrides": {"BedroomAbvGr": sentinel},
            },
        )

    assert response.status_code == 201
    kf = response.json()["key_features"]
    assert kf.get("BedroomAbvGr") == sentinel, (
        f"Expected BedroomAbvGr={sentinel} in key_features after override, got {kf.get('BedroomAbvGr')}"
    )


def test_exact_house_features_are_persisted_on_prediction_responses(tmp_path: Path):
    """Exact caller-supplied home facts must be preserved separately from inferred model inputs."""
    settings = build_test_settings(tmp_path)
    overrides = {"BedroomAbvGr": 4, "FullBath": 2, "LotArea": 7200}

    with TestClient(create_app(settings)) as client:
        create_response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "61 Exact Facts Way",
                "city": "Denver",
                "state": "CO",
                "postal_code": "80201",
                "country": "US",
                "feature_overrides": overrides,
            },
        )

        assert create_response.status_code == 201
        create_payload = create_response.json()
        assert create_payload["exact_house_features"] == overrides

        prediction_id = create_payload["prediction_id"]
        detail_response = client.get(f"/v1/predictions/{prediction_id}")
        list_response = client.get("/v1/predictions", params={"limit": 20})

    assert detail_response.status_code == 200
    detail_payload = detail_response.json()
    assert detail_payload["exact_house_features"] == overrides

    assert list_response.status_code == 200
    list_payload = list_response.json()
    matching_items = [item for item in list_payload["items"] if item["prediction_id"] == prediction_id]
    assert len(matching_items) == 1
    assert matching_items[0]["exact_house_features"] == overrides


# ---------------------------------------------------------------------------
# feature_source data-flow tests
# ---------------------------------------------------------------------------

def test_prediction_response_includes_feature_source(tmp_path: Path):
    """POST /v1/predictions must return a non-empty feature_source string so that
    API consumers can tell which data source backed the prediction (census, heuristic, fake…).
    """
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "11 Source St",
                "city": "Denver",
                "state": "CO",
                "postal_code": "80201",
                "country": "US",
            },
        )

    assert response.status_code == 201
    payload = response.json()
    assert "feature_source" in payload
    # In test mode the fake provider is used; feature_source must be "fake".
    assert payload["feature_source"] == "fake"


def test_reused_prediction_carries_feature_source(tmp_path: Path):
    """When a prediction is reused (was_reused=True), feature_source must still be
    populated — it is carried from the original prediction's feature snapshot.
    """
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        first = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "12 Reuse Source Ave",
                "city": "Denver",
                "state": "CO",
                "postal_code": "80201",
                "country": "US",
            },
        )
        assert first.status_code == 201
        assert first.json()["was_reused"] is False

        second = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "12 Reuse Source Ave",
                "city": "Denver",
                "state": "CO",
                "postal_code": "80201",
                "country": "US",
            },
        )

    assert second.status_code == 201
    assert second.json()["was_reused"] is True
    assert second.json()["feature_source"] == first.json()["feature_source"]


def test_reused_prediction_carries_ui_feature_payload(tmp_path: Path):
    """Reused predictions must preserve the same UI-facing feature payload as the
    original response so the frontend does not lose feature details on cache hits.
    """
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        first = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "13 Reuse UI Payload Ave",
                "city": "Denver",
                "state": "CO",
                "postal_code": "80201",
                "country": "US",
            },
        )
        assert first.status_code == 201
        assert first.json()["was_reused"] is False

        second = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "13 Reuse UI Payload Ave",
                "city": "Denver",
                "state": "CO",
                "postal_code": "80201",
                "country": "US",
            },
        )

    assert second.status_code == 201
    second_payload = second.json()
    first_payload = first.json()

    assert second_payload["was_reused"] is True
    assert isinstance(second_payload["actual_house_features"], dict)
    assert len(second_payload["actual_house_features"]) > 0
    assert second_payload["actual_house_features"] == first_payload["actual_house_features"]
    assert second_payload["feature_provenance"] == first_payload["feature_provenance"]


def test_reused_prediction_detail_carries_ui_feature_payload(tmp_path: Path):
    """Prediction detail responses must preserve the same UI-facing feature payload
    for reused predictions so drill-down views stay accurate.
    """
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        first = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "14 Reuse Detail Payload Ave",
                "city": "Denver",
                "state": "CO",
                "postal_code": "80201",
                "country": "US",
            },
        )
        assert first.status_code == 201

        second = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "14 Reuse Detail Payload Ave",
                "city": "Denver",
                "state": "CO",
                "postal_code": "80201",
                "country": "US",
            },
        )
        assert second.status_code == 201
        assert second.json()["was_reused"] is True

        prediction_id = second.json()["prediction_id"]
        detail = client.get(f"/v1/predictions/{prediction_id}")

    assert detail.status_code == 200
    detail_payload = detail.json()
    first_payload = first.json()

    assert detail_payload["was_reused"] is True
    assert detail_payload["actual_house_features"] == first_payload["actual_house_features"]
    assert detail_payload["feature_source"] == first_payload["feature_source"]
    # Reused predictions preserve the original feature_provenance structure
    assert "providers" in detail_payload["feature_provenance"]


# ---------------------------------------------------------------------------
# FakeGeocodingClient coordinates tests
# ---------------------------------------------------------------------------

def test_fake_geocoding_client_emits_state_centroid_coordinates():
    """FakeGeocodingClient must now return non-None latitude/longitude derived
    from the state centroid table so Census property enrichment can attempt a
    region-level ACS lookup even when live geocoders are unavailable.
    """
    from house_price_prediction.domain.contracts.prediction_contracts import AddressPayload

    client = FakeGeocodingClient()

    # Colorado centroid should be close to (39.06, -105.31)
    result = client.normalize(
        AddressPayload(
            address_line_1="1 Test St",
            city="Denver",
            state="CO",
            postal_code="80201",
            country="US",
        )
    )
    addr = result.normalized_address
    assert addr.latitude is not None, "latitude must not be None"
    assert addr.longitude is not None, "longitude must not be None"
    assert 36.0 < addr.latitude < 42.0, f"CO latitude out of expected range: {addr.latitude}"
    assert -110.0 < addr.longitude < -100.0, f"CO longitude out of expected range: {addr.longitude}"


def test_fake_geocoding_client_falls_back_to_us_center_for_unknown_state():
    """FakeGeocodingClient must fall back to the US geographic centre when the
    state code is not found in the centroid table (e.g. an empty or invalid state).
    """
    from house_price_prediction.domain.contracts.prediction_contracts import AddressPayload

    client = FakeGeocodingClient()

    result = client.normalize(
        AddressPayload(
            address_line_1="1 Unknown St",
            city="Anytown",
            state="XX",           # not a real US state code
            postal_code="00000",
            country="US",
        )
    )
    addr = result.normalized_address
    # US geographic centre: ~(39.5, -98.35)
    assert addr.latitude == 39.5
    assert addr.longitude == -98.35


# ---------------------------------------------------------------------------
# Provider cache strengthening tests
# ---------------------------------------------------------------------------

def test_provider_cache_does_not_return_geocoding_only_response(tmp_path: Path):
    """find_recent_property_response_for_address must only return responses that
    contain ALL three key property features (LotArea, OverallQual, GrLivArea).
    A payload missing any of them must be skipped — tightened from 'any 2 of 3'.
    """
    from house_price_prediction.infrastructure.db.session import init_database
    from house_price_prediction.domain.contracts.prediction_contracts import (
        AddressPayload,
        NormalizedAddress,
        PredictionRequestPayload,
    )
    from uuid import uuid4
    from datetime import UTC, datetime

    database_url = f"sqlite:///{tmp_path / 'cache_test.db'}"
    session_factory = init_database(database_url, create_schema=True)

    normalized_address = NormalizedAddress(
        address_line_1="99 Cache Test Rd",
        city="Denver",
        state="CO",
        postal_code="80201",
        country="US",
        formatted_address="99 CACHE TEST RD, DENVER, CO 80201, US",
        latitude=39.7,
        longitude=-104.9,
        geocoding_source="fake",
    )

    with session_factory() as session:
        repo = PredictionRepository(session)
        req_id = uuid4()
        addr_id = repo.get_or_create_normalized_address(normalized_address)
        _ = repo.register_model_version("m", "1", ["LotArea"])
        payload = PredictionRequestPayload(
            address_line_1="99 Cache Test Rd",
            city="Denver",
            state="CO",
            postal_code="80201",
            country="US",
        )
        repo.create_prediction_request(
            request_id=req_id,
            correlation_id=uuid4(),
            normalized_address_id=addr_id,
            payload=payload,
            normalized_address=normalized_address,
            submitted_at=datetime.now(UTC),
            feature_policy_name="balanced-v1",
            feature_policy_version="v1",
        )
        # Store a response that only has LotArea and OverallQual — missing GrLivArea.
        # This simulates the old "2 of 3" scenario — should NOT be returned now.
        from house_price_prediction.domain.contracts.prediction_contracts import ProviderResponseContract
        repo.create_provider_response(
            request_id=req_id,
            provider_response=ProviderResponseContract(
                provider_name="partial_provider",
                status="success",
                payload={"LotArea": 8000, "OverallQual": 7},
                fetched_at=datetime.now(UTC),
            ),
        )
        session.commit()
        result = repo.find_recent_property_response_for_address(
            normalized_address=normalized_address,
            max_age_hours=24,
        )

    # Must return None — partial payload must be rejected
    assert result is None, (
        "find_recent_property_response_for_address should not return a partial payload "
        f"missing GrLivArea, got: {result}"
    )


# ---------------------------------------------------------------------------
# Config & telemetry unit tests
# ---------------------------------------------------------------------------

def test_load_settings_production_safe_defaults():
    """Without any override env vars, load_settings must return production-safe values.

    The defaults must not enable the mock predictor or fall back to the 'fake'
    provider chains — both would silently serve placeholder data in production.
    The default timeout must be at or above the 20 s threshold below which the
    health endpoint emits a warning for free API providers.
    """
    import os
    from house_price_prediction.config import load_settings

    risky_keys = [
        "ENABLE_MOCK_PREDICTOR",
        "PROPERTY_DATA_PROVIDER",
        "GEOCODING_PROVIDER",
        "PROVIDER_TIMEOUT_SECONDS",
    ]
    saved = {k: os.environ.pop(k) for k in risky_keys if k in os.environ}
    load_settings.cache_clear()
    try:
        s = load_settings()
    finally:
        os.environ.update(saved)
        load_settings.cache_clear()

    assert s.enable_mock_predictor is False, (
        "enable_mock_predictor default must be False — True would mean every "
        "deployment without an explicit env var silently serves mock predictions."
    )
    assert s.property_data_provider in ["free", "free-fallback"], (
        f"property_data_provider default must be 'free' or 'free-fallback', got {s.property_data_provider!r}"
    )
    assert s.geocoding_provider in ["free", "free-fallback"], (
        f"geocoding_provider default must be 'free' or 'free-fallback', got {s.geocoding_provider!r}"
    )
    assert s.provider_timeout_seconds >= 20.0, (
        f"provider_timeout_seconds default must be >= 20.0 s for free providers, "
        f"got {s.provider_timeout_seconds}"
    )


def test_configure_logging_does_not_duplicate_handlers():
    """Repeated configure_logging() calls must install exactly one StreamHandler."""
    import logging
    from house_price_prediction.telemetry import configure_logging, CorrelationIdFilter

    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_filters = root.filters[:]

    # Strip any previously installed CorrelationId StreamHandler so the test
    # starts from a known clean state regardless of import-time side effects.
    root.handlers = [
        h for h in root.handlers
        if not (
            isinstance(h, logging.StreamHandler)
            and any(isinstance(f, CorrelationIdFilter) for f in h.filters)
        )
    ]
    try:
        configure_logging()
        configure_logging()
        configure_logging()

        our_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.StreamHandler)
            and any(isinstance(f, CorrelationIdFilter) for f in h.filters)
        ]
        assert len(our_handlers) == 1, (
            f"Expected exactly 1 CorrelationId StreamHandler after 3 configure_logging() calls, "
            f"got {len(our_handlers)}"
        )
    finally:
        root.handlers = saved_handlers
        root.filters = saved_filters


# ---------------------------------------------------------------------------
# Input validation: address normalization, postal code, feature overrides
# ---------------------------------------------------------------------------

def test_state_field_is_uppercased_in_prediction_request(tmp_path: Path):
    """state field sent lowercase is normalised to uppercase."""
    from house_price_prediction.domain.contracts.prediction_contracts import PredictionRequestPayload

    p = PredictionRequestPayload(
        address_line_1="123 Main St",
        city="Boston",
        state="ma",
        postal_code="02101",
        country="us",
    )
    assert p.state == "MA"
    assert p.country == "US"


def test_address_fields_are_stripped(tmp_path: Path):
    """Leading/trailing whitespace on address_line_1 and city is removed."""
    from house_price_prediction.domain.contracts.prediction_contracts import PredictionRequestPayload

    p = PredictionRequestPayload(
        address_line_1="  123 Main St  ",
        city="  Chicago  ",
        state="  IL  ",
        postal_code="  60601  ",
        country="US",
    )
    assert p.address_line_1 == "123 Main St"
    assert p.city == "Chicago"
    assert p.state == "IL"
    assert p.postal_code == "60601"


def test_invalid_us_zip_code_returns_422(tmp_path: Path):
    """A non-ZIP postal_code for a US address triggers a 422 from the guardrail."""
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        resp = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "1 Times Square",
                "city": "New York",
                "state": "NY",
                "postal_code": "INVALID",
                "country": "US",
            },
        )
    assert resp.status_code == 422, resp.text
    assert "postal_code" in resp.text.lower() or "zip" in resp.text.lower()


def test_valid_zip_plus_four_is_accepted(tmp_path: Path):
    """ZIP+4 format (12345-6789) should pass the ZIP code guardrail."""
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        resp = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "1 Main St",
                "city": "Anytown",
                "state": "TX",
                "postal_code": "75201-1234",
                "country": "US",
            },
        )
    assert resp.status_code == 201, resp.text


def test_feature_overrides_out_of_bounds_returns_422(tmp_path: Path):
    """feature_overrides with an out-of-range numeric value returns 422."""
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        resp = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "1 Main St",
                "city": "Austin",
                "state": "TX",
                "postal_code": "78701",
                "country": "US",
                "feature_overrides": {"OverallQual": 999},
            },
        )
    assert resp.status_code == 422, resp.text
    assert "OverallQual" in resp.text


def test_feature_overrides_wrong_type_returns_422(tmp_path: Path):
    """feature_overrides with a string value for a numeric feature returns 422."""
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        resp = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "1 Main St",
                "city": "Austin",
                "state": "TX",
                "postal_code": "78701",
                "country": "US",
                "feature_overrides": {"GrLivArea": "big"},
            },
        )
    assert resp.status_code == 422, resp.text
    assert "GrLivArea" in resp.text


def test_feature_overrides_valid_values_accepted(tmp_path: Path):
    """feature_overrides with in-bounds numeric values should succeed."""
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        resp = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "1 Main St",
                "city": "Seattle",
                "state": "WA",
                "postal_code": "98101",
                "country": "US",
                "feature_overrides": {
                    "OverallQual": 8,
                    "GrLivArea": 2000,
                    "BedroomAbvGr": 3,
                },
            },
        )
    assert resp.status_code == 201, resp.text
