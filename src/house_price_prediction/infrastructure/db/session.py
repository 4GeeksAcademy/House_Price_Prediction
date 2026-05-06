from __future__ import annotations

from sqlalchemy import inspect
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    pass


def _connect_args(database_url: str) -> dict[str, object]:
    if database_url.startswith("sqlite"):
        return {"check_same_thread": False}
    return {}


def _validate_required_schema(engine) -> None:
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    required_tables = {
        "prediction_requests",
        "predictions",
        "feature_snapshots",
        "provider_responses",
        "workflow_events",
        "model_registry",
        "normalized_addresses",
    }
    missing_tables = sorted(required_tables - existing_tables)
    if missing_tables:
        raise RuntimeError(
            "Database schema is out of date. Missing required tables: "
            + ", ".join(missing_tables)
            + ". Run Alembic migrations before starting the API."
        )

    request_columns = {column["name"] for column in inspector.get_columns("prediction_requests")}
    required_request_columns = {
        "feature_policy_name",
        "feature_policy_version",
        "normalized_address_id",
    }
    missing_request_columns = sorted(required_request_columns - request_columns)
    if missing_request_columns:
        raise RuntimeError(
            "Database schema is out of date. Missing prediction_requests columns: "
            + ", ".join(missing_request_columns)
            + ". Run Alembic migrations before starting the API."
        )

    prediction_columns = {column["name"] for column in inspector.get_columns("predictions")}
    required_prediction_columns = {
        "was_reused",
        "source_prediction_id",
    }
    missing_prediction_columns = sorted(required_prediction_columns - prediction_columns)
    if missing_prediction_columns:
        raise RuntimeError(
            "Database schema is out of date. Missing predictions columns: "
            + ", ".join(missing_prediction_columns)
            + ". Run Alembic migrations before starting the API."
        )


def init_database(
    database_url: str,
    create_schema: bool = False,
    validate_schema: bool = False,
) -> sessionmaker:
    from house_price_prediction.infrastructure.db import models  # noqa: F401

    engine = create_engine(
        database_url,
        future=True,
        connect_args=_connect_args(database_url),
    )
    if create_schema:
        Base.metadata.create_all(engine)
    if validate_schema:
        _validate_required_schema(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)