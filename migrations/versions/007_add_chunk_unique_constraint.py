"""Add unique constraint to document_chunks table.

Ensures no duplicate chunks can be created when a job retries.
Uses ON CONFLICT DO UPDATE for graceful handling of existing chunks.

Revision ID: 007
Revises: 006
Create Date: 2026-02-24 11:35:00.000000

"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

# revision identifiers, used by Alembic.
revision: str = "007"
down_revision: str | None = "006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add unique constraint on (job_id, chunk_index) if not exists.
    
    This constraint prevents duplicate chunks when jobs are retried.
    The upsert logic in the EmbedStage handles conflicts gracefully
    by updating existing chunks rather than failing.
    """
    # Check if constraint already exists (idempotent)
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    
    # Get existing constraints
    constraints = inspector.get_unique_constraints("document_chunks")
    constraint_names = {c["name"] for c in constraints}
    
    if "uq_document_chunks_job_chunk" not in constraint_names:
        # Create unique constraint
        op.create_unique_constraint(
            "uq_document_chunks_job_chunk",
            "document_chunks",
            ["job_id", "chunk_index"]
        )


def downgrade() -> None:
    """Remove unique constraint from document_chunks table."""
    # Drop constraint if it exists
    op.drop_constraint(
        "uq_document_chunks_job_chunk",
        "document_chunks",
        type_="unique"
    )
