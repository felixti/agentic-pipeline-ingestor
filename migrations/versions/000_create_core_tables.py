"""Create core tables - jobs and pipelines.

Revision ID: 000
Revises: 
Create Date: 2026-02-17 14:00:00.000000

This migration must run BEFORE 001 as it creates the jobs table
that is referenced by other migrations.

"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "000"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create jobs and pipelines tables."""
    
    # Create pipelines table (referenced by jobs)
    op.create_table(
        "pipelines",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("config", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("is_active", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_by", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    
    op.create_index("idx_pipelines_name", "pipelines", ["name"])
    
    # Create jobs table (referenced by many other tables)
    op.create_table(
        "jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="created"),
        sa.Column("source_type", sa.String(length=50), nullable=False),
        sa.Column("source_uri", sa.String(length=500), nullable=True),
        sa.Column("file_name", sa.String(length=255), nullable=True),
        sa.Column("file_size", sa.BigInteger(), nullable=True),
        sa.Column("mime_type", sa.String(length=100), nullable=True),
        sa.Column("priority", sa.String(length=20), nullable=False, server_default="normal"),
        sa.Column("mode", sa.String(length=20), nullable=False, server_default="async"),
        sa.Column("external_id", sa.String(length=255), nullable=True),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("error_code", sa.String(length=50), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_retries", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("locked_by", sa.String(length=255), nullable=True),
        sa.Column("locked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("pipeline_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("pipeline_config", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["pipeline_id"], ["pipelines.id"],
            name="fk_jobs_pipeline_id",
            ondelete="SET NULL"
        ),
    )
    
    # Create indexes for jobs table
    op.create_index("idx_jobs_status", "jobs", ["status"])
    op.create_index("idx_jobs_source_type", "jobs", ["source_type"])
    op.create_index("idx_jobs_external_id", "jobs", ["external_id"])
    op.create_index("idx_jobs_locked_by", "jobs", ["locked_by"])
    op.create_index("idx_jobs_created_at", "jobs", ["created_at"])


def downgrade() -> None:
    """Drop jobs and pipelines tables."""
    op.drop_index("idx_jobs_created_at", table_name="jobs")
    op.drop_index("idx_jobs_locked_by", table_name="jobs")
    op.drop_index("idx_jobs_external_id", table_name="jobs")
    op.drop_index("idx_jobs_source_type", table_name="jobs")
    op.drop_index("idx_jobs_status", table_name="jobs")
    op.drop_table("jobs")
    
    op.drop_index("idx_pipelines_name", table_name="pipelines")
    op.drop_table("pipelines")
