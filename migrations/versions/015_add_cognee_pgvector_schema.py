"""Add Cognee pgvector schema

Revision ID: 015
Revises: 012, 013, 014
Create Date: 2026-02-28 20:30:00.000000

"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import UUID as postgresql_uuid

# revision identifiers, used by Alembic.
revision: str = "015"
down_revision: str | Sequence[str] | None = ("012", "013", "014")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create Cognee pgvector schema for vector storage and entity mirroring.
    
    Creates three tables:
    - cognee_vectors: Store chunk embeddings with ivfflat index for ANN search
    - cognee_documents: Document metadata storage
    - cognee_entities: Entity mirror from Neo4j for hybrid queries
    """
    # Ensure pgvector extension is available
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # Create cognee_vectors table for chunk embeddings
    # Stores vector embeddings with metadata for similarity search
    op.create_table(
        "cognee_vectors",
        sa.Column(
            "id",
            postgresql_uuid(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("chunk_id", sa.String(255), nullable=False),
        sa.Column("document_id", sa.String(255), nullable=False),
        sa.Column(
            "dataset_id",
            sa.String(255),
            nullable=False,
            server_default="default",
        ),
        # embedding column stored as vector type via raw SQL
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    
    # Add embedding column as VECTOR type using raw SQL
    # Using 1536 dimensions for OpenAI text-embedding-3-small
    op.execute(
        """
        ALTER TABLE cognee_vectors
        ADD COLUMN embedding VECTOR(1536);
        """
    )
    
    # Create indexes for cognee_vectors
    # Standard B-tree indexes for filtering
    op.create_index(
        "idx_cognee_vectors_chunk_id",
        "cognee_vectors",
        ["chunk_id"],
    )
    op.create_index(
        "idx_cognee_vectors_document_id",
        "cognee_vectors",
        ["document_id"],
    )
    op.create_index(
        "idx_cognee_vectors_dataset_id",
        "cognee_vectors",
        ["dataset_id"],
    )
    
    # IVFFlat index for approximate nearest neighbor search
    # lists=100: Number of inverted lists (partitions)
    # Good balance between build time, memory, and search quality
    # For ~1M vectors, 100 lists provides good recall with reasonable speed
    op.execute(
        """
        CREATE INDEX idx_cognee_vectors_embedding
        ON cognee_vectors
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
        """
    )
    
    # Create cognee_documents table for document metadata
    # Mirrors document info from Neo4j for hybrid queries
    op.create_table(
        "cognee_documents",
        sa.Column(
            "id",
            postgresql_uuid(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("document_id", sa.String(255), nullable=False, unique=True),
        sa.Column(
            "dataset_id",
            sa.String(255),
            nullable=False,
            server_default="default",
        ),
        sa.Column("title", sa.String(500), nullable=True),
        sa.Column("source_path", sa.Text(), nullable=True),
        sa.Column("content_hash", sa.String(64), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.Column(
            "chunk_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "entity_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    
    # Create indexes for cognee_documents
    op.create_index(
        "idx_cognee_documents_dataset_id",
        "cognee_documents",
        ["dataset_id"],
    )
    op.create_index(
        "idx_cognee_documents_document_id",
        "cognee_documents",
        ["document_id"],
    )
    
    # Create cognee_entities table for entity mirror
    # Mirrors entities from Neo4j for hybrid graph + vector queries
    op.create_table(
        "cognee_entities",
        sa.Column(
            "id",
            postgresql_uuid(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("entity_id", sa.String(255), nullable=False, unique=True),
        sa.Column("name", sa.String(500), nullable=False),
        sa.Column("type", sa.String(100), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("document_id", sa.String(255), nullable=True),
        sa.Column(
            "dataset_id",
            sa.String(255),
            nullable=False,
            server_default="default",
        ),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    
    # Create indexes for cognee_entities
    op.create_index(
        "idx_cognee_entities_name",
        "cognee_entities",
        ["name"],
    )
    op.create_index(
        "idx_cognee_entities_type",
        "cognee_entities",
        ["type"],
    )
    op.create_index(
        "idx_cognee_entities_document_id",
        "cognee_entities",
        ["document_id"],
    )


def downgrade() -> None:
    """Drop Cognee pgvector schema tables and indexes.
    
    Drops all tables and indexes in reverse order of creation.
    """
    # Drop cognee_entities indexes and table
    op.drop_index("idx_cognee_entities_document_id", table_name="cognee_entities")
    op.drop_index("idx_cognee_entities_type", table_name="cognee_entities")
    op.drop_index("idx_cognee_entities_name", table_name="cognee_entities")
    op.drop_table("cognee_entities")
    
    # Drop cognee_documents indexes and table
    op.drop_index("idx_cognee_documents_document_id", table_name="cognee_documents")
    op.drop_index("idx_cognee_documents_dataset_id", table_name="cognee_documents")
    op.drop_table("cognee_documents")
    
    # Drop cognee_vectors indexes and table
    op.execute("DROP INDEX IF EXISTS idx_cognee_vectors_embedding;")
    op.drop_index("idx_cognee_vectors_dataset_id", table_name="cognee_vectors")
    op.drop_index("idx_cognee_vectors_document_id", table_name="cognee_vectors")
    op.drop_index("idx_cognee_vectors_chunk_id", table_name="cognee_vectors")
    op.drop_table("cognee_vectors")
