"""Add pgvector and pg_trgm extensions.

Revision ID: 002
Revises: 001
Create Date: 2026-02-18 01:00:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Enable pgvector and pg_trgm extensions for vector storage and fuzzy text search."""
    # pgvector extension - provides vector data type and similarity operators
    # Required for storing and querying document embeddings (1536-dim by default)
    # Supports L2, inner product, and cosine distance operations
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # pg_trgm extension - provides trigram matching for fuzzy text search
    # Required for similarity comparisons on text content
    # Supports GIN indexes for fast similarity and pattern matching
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")


def downgrade() -> None:
    """Drop pgvector and pg_trgm extensions."""
    # Drop pg_trgm first (no dependencies on vector)
    op.execute("DROP EXTENSION IF EXISTS pg_trgm;")
    
    # Drop pgvector extension
    op.execute("DROP EXTENSION IF EXISTS vector;")
