"""Add contextual retrieval support to document_chunks table.

Revision ID: 004
Revises: 003
Create Date: 2026-02-20 17:00:00.000000

"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID

# revision identifiers, used by Alembic.
revision: str = "004"
down_revision: str | None = "003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add contextual retrieval columns and tables.
    
    Creates:
    - Additional columns on document_chunks table for context storage
    - document_hierarchy table for hierarchical relationships
    - Indexes for efficient context lookups
    - Enable PostgreSQL ltree extension
    """
    # Enable ltree extension for hierarchical queries
    op.execute("CREATE EXTENSION IF NOT EXISTS ltree;")
    
    # Add contextual retrieval columns to document_chunks table
    op.add_column(
        "document_chunks",
        sa.Column(
            "parent_document_id",
            UUID(as_uuid=True),
            sa.ForeignKey("jobs.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    
    op.add_column(
        "document_chunks",
        sa.Column(
            "section_headers",
            ARRAY(sa.Text()),
            nullable=True,
        ),
    )
    
    op.add_column(
        "document_chunks",
        sa.Column(
            "document_metadata",
            JSONB(),
            nullable=False,
            server_default="{}",
        ),
    )
    
    op.add_column(
        "document_chunks",
        sa.Column(
            "context_type",
            sa.String(50),
            nullable=True,
        ),
    )
    
    op.add_column(
        "document_chunks",
        sa.Column(
            "enhanced_content",
            sa.Text(),
            nullable=True,
        ),
    )
    
    # Create index on parent_document_id for efficient parent lookups
    op.create_index(
        "idx_document_chunks_parent_doc",
        "document_chunks",
        ["parent_document_id"],
    )
    
    # Create index on context_type for filtering by strategy
    op.create_index(
        "idx_document_chunks_context_type",
        "document_chunks",
        ["context_type"],
    )
    
    # Create GIN index on section_headers for array operations
    op.create_index(
        "idx_document_chunks_section_headers",
        "document_chunks",
        ["section_headers"],
        postgresql_using="gin",
    )
    
    # Create GIN index on document_metadata for JSON queries
    op.create_index(
        "idx_document_chunks_doc_metadata",
        "document_chunks",
        ["document_metadata"],
        postgresql_using="gin",
    )
    
    # Create document_hierarchy table for hierarchical relationships
    op.create_table(
        "document_hierarchy",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_id", UUID(as_uuid=True), nullable=True),
        sa.Column("level", sa.Integer(), nullable=False),
        sa.Column("parent_id", UUID(as_uuid=True), nullable=True),
        sa.Column("path", sa.Text(), nullable=True),  # ltree path stored as text
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("node_metadata", JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        
        # Primary key
        sa.PrimaryKeyConstraint("id"),
        
        # Foreign key constraints
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["jobs.id"],
            name="fk_document_hierarchy_document_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["document_chunks.id"],
            name="fk_document_hierarchy_chunk_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["parent_id"],
            ["document_hierarchy.id"],
            name="fk_document_hierarchy_parent_id",
            ondelete="CASCADE",
        ),
    )
    
    # Create indexes on document_hierarchy
    op.create_index(
        "idx_document_hierarchy_document_id",
        "document_hierarchy",
        ["document_id"],
    )
    
    op.create_index(
        "idx_document_hierarchy_chunk_id",
        "document_hierarchy",
        ["chunk_id"],
        unique=True,
    )
    
    op.create_index(
        "idx_document_hierarchy_parent_id",
        "document_hierarchy",
        ["parent_id"],
    )
    
    op.create_index(
        "idx_document_hierarchy_level",
        "document_hierarchy",
        ["level"],
    )
    
    # Create GIST index on path for ltree queries (if ltree is available)
    op.execute(
        """
        CREATE INDEX idx_document_hierarchy_path 
        ON document_hierarchy 
        USING gist (path::ltree);
        """
    )


def downgrade() -> None:
    """Remove contextual retrieval columns and tables.
    
    Drops all indexes, tables, and columns added in upgrade.
    """
    # Drop document_hierarchy table indexes
    op.execute("DROP INDEX IF EXISTS idx_document_hierarchy_path;")
    op.drop_index("idx_document_hierarchy_level", table_name="document_hierarchy")
    op.drop_index("idx_document_hierarchy_parent_id", table_name="document_hierarchy")
    op.drop_index("idx_document_hierarchy_chunk_id", table_name="document_hierarchy")
    op.drop_index("idx_document_hierarchy_document_id", table_name="document_hierarchy")
    
    # Drop document_hierarchy table
    op.drop_table("document_hierarchy")
    
    # Drop document_chunks indexes
    op.drop_index("idx_document_chunks_doc_metadata", table_name="document_chunks")
    op.drop_index("idx_document_chunks_section_headers", table_name="document_chunks")
    op.drop_index("idx_document_chunks_context_type", table_name="document_chunks")
    op.drop_index("idx_document_chunks_parent_doc", table_name="document_chunks")
    
    # Drop document_chunks columns
    op.drop_column("document_chunks", "enhanced_content")
    op.drop_column("document_chunks", "context_type")
    op.drop_column("document_chunks", "document_metadata")
    op.drop_column("document_chunks", "section_headers")
    op.drop_column("document_chunks", "parent_document_id")
