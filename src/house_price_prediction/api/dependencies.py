from __future__ import annotations

from collections.abc import Generator

from fastapi import Request
from sqlalchemy.orm import Session

from house_price_prediction.application.services.prediction_orchestrator import (
    Brain,
)
from house_price_prediction.config import Settings
from house_price_prediction.infrastructure.model_runtime.predictor import PredictionRuntime


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_prediction_runtime(request: Request) -> PredictionRuntime:
    return request.app.state.prediction_runtime


def get_brain(request: Request) -> Brain:
    return request.app.state.brain


def get_db_session(request: Request) -> Generator[Session, None, None]:
    session_factory = request.app.state.session_factory
    with session_factory() as session:
        yield session