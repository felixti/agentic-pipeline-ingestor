"""Add cache tables for multi-layer caching.

Revision ID: 005
Revises: 004
Create Date: 2026-02-20 19:00:00.000000

"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

# revision identifiers, used by Alembic.
revision: str = "005"
down_revision: str | None = "004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create cache tables for multi-layer caching system.
    
    Creates:
    - embedding_cache: Stores text embeddings for reuse
    - query_cache: Stores query results with optional vector embedding
    - llm_response_cache: Stores LLM responses for common prompts
    - Indexes for efficient lookups
    """
    # Create embedding_cache table
    op.create_table(
        "embedding_cache",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("text_hash", sa.String(64), nullable=False),
        sa.Column("text_preview", sa.String(200), nullable=True),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("embedding", sa.Text(), nullable=False),  # Stored as vector type via SQL
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("accessed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("access_count", sa.Integer(), nullable=False, server_default="1"),
        
        # Primary key
        sa.PrimaryKeyConstraint("id"),
        
        # Unique constraint on text_hash + model combination
        sa.UniqueConstraint("text_hash", "model", name="uq_embedding_cache_text_model"),
    )
    
    # Create indexes for embedding_cache
    op.create_index(
        "idx_embedding_cache_text_hash",
        "embedding_cache",
        ["text_hash"]
    )
    op.create_index(
        "idx_embedding_cache_model",
        "embedding_cache",
        ["model"]
    )
    op.create_index(
        "idx_embedding_cache_accessed",
        "embedding_cache",
        ["accessed_at"]
    )
    
    # Create query_cache table
    op.create_table(
        "query_cache",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("query_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("query_embedding", sa.Text(), nullable=True),  # Stored as vector type via SQL
        sa.Column("result_json", JSONB(), nullable=False),
        sa.Column("strategy_config", JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ttl_seconds", sa.Integer(), nullable=False, server_default="3600"),
        
        # Primary key
        sa.PrimaryKeyConstraint("id"),
    )
    
    # Create indexes for query_cache
    op.create_index(
        "idx_query_cache_hash",
        "query_cache",
        ["query_hash"]
    )
    op.create_index(
        "idx_query_cache_created",
        "query_cache",
        ["created_at"]
    )
    
    # Create llm_response_cache table
    op.create_table(
        "llm_response_cache",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("prompt_hash", sa.String(64), nullable=False),
        sa.Column("prompt_preview", sa.Text(), nullable=True),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("response", sa.Text(), nullable=False),
        sa.Column("tokens_used", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        
        # Primary key
        sa.PrimaryKeyConstraint("id"),
        
        # Unique constraint on prompt_hash + model
        sa.UniqueConstraint("prompt_hash", "model", name="uq_llm_cache_prompt_model"),
    )
    
    # Create indexes for llm_response_cache
    op.create_index(
        "idx_llm_cache_prompt_hash",
        "llm_response_cache",
        ["prompt_hash"]
    )
    op.create_index(
        "idx_llm_cache_model",
        "llm_response_cache",
        ["model"]
    )
    op.create_index(
        "idx_llm_cache_created",
        "llm_response_cache",
        ["created_at"]
    )
    
    # Create HNSW index for semantic similarity search on query_cache
    # This enables efficient L3 semantic cache lookups
    op.execute(
        """
        CREATE INDEX idx_query_cache_embedding_hnsw 
        ON query_cache 
        USING hnsw (query_embedding::vector vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        WHERE query_embedding IS NOT NULL;
        """
    )


def downgrade() -> None:
    """Drop cache tables.
    
    Drops all cache tables and their indexes in reverse order.
    """
    # Drop HNSW index first
    op.execute("DROP INDEX IF EXISTS idx_query_cache_embedding_hnsw;")
    
    # Drop LLM response cache
    op.drop_index("idx_llm_cache_created", table_name="llm_response_cache")
    op.drop_index("idx_llm_cache_model", table_name="llm_response_cache")
    op.drop_index("idx_llm_cache_prompt_hash", table_name="llm_response_cache")
    op.drop_table("llm_response_cache")
    
    # Drop query cache
    op.drop_index("idx_query_cache_created", table_name="query_cache")
    op.drop_index("idx_query_cache_hash", table_name="query_cache")
    op.drop_table("query_cache")
    
    # Drop embedding cache
    op.drop_index("idx_embedding_cache_accessed", table_name="embedding_cache")
    op.drop_index("idx_embedding_cache_model", table_name="embedding_cache")
    op.drop_index("idx_embedding_cache_text_hash", table_name="embedding_cache")
    op.drop_table("embedding_cache")
