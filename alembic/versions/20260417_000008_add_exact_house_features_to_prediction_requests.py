"""add exact_house_features to prediction_requests

Revision ID: 20260417_000008
Revises: 20260416_000007
Create Date: 2026-05-10 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260417_000008"
down_revision = "20260416_000007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "prediction_requests",
        sa.Column(
            "exact_house_features",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
    )


def downgrade() -> None:
    op.drop_column("prediction_requests", "exact_house_features")
