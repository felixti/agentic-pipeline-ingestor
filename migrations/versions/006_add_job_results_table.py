"""Add job_results table for storing job processing results.

Revision ID: 006
Revises: 005
Create Date: 2026-02-24 11:30:00.000000

"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

# revision identifiers, used by Alembic.
revision: str = "006"
down_revision: str | None = "005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create job_results table.
    
    Creates:
    - job_results table with all result columns
    - Foreign key constraint to jobs.id with CASCADE delete
    - Unique constraint on job_id (one result per job)
    - Indexes on job_id and expires_at for efficient queries
    """
    op.create_table(
        "job_results",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("job_id", UUID(as_uuid=True), nullable=False),
        sa.Column("extracted_text", sa.Text(), nullable=True),
        sa.Column("output_data", JSONB(), nullable=True),
        sa.Column("metadata", JSONB(), nullable=False, server_default="{}"),
        sa.Column("quality_score", sa.Float(), nullable=True),
        sa.Column("processing_time_ms", sa.Integer(), nullable=True),
        sa.Column("output_uri", sa.String(length=500), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        
        # Primary key
        sa.PrimaryKeyConstraint("id"),
        
        # Foreign key constraint to jobs table with CASCADE delete
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["jobs.id"],
            name="fk_job_results_job_id",
            ondelete="CASCADE"
        ),
        
        # Unique constraint: one result per job
        sa.UniqueConstraint("job_id", name="uq_job_results_job_id"),
    )
    
    # Create indexes for efficient lookups
    # Index on job_id for foreign key lookups and result retrieval
    op.create_index(
        "idx_job_results_job_id",
        "job_results",
        ["job_id"]
    )
    
    # Index on expires_at for cleanup queries
    op.create_index(
        "idx_job_results_expires",
        "job_results",
        ["expires_at"]
    )
    
    # Index on created_at for sorting and pagination
    op.create_index(
        "idx_job_results_created",
        "job_results",
        ["created_at"]
    )


def downgrade() -> None:
    """Drop job_results table.
    
    Drops all indexes and the table in reverse order of creation.
    """
    # Drop indexes
    op.drop_index("idx_job_results_created", table_name="job_results")
    op.drop_index("idx_job_results_expires", table_name="job_results")
    op.drop_index("idx_job_results_job_id", table_name="job_results")
    
    # Drop table (automatically drops constraints)
    op.drop_table("job_results")
