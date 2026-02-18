"""Add document_chunks table with HNSW and GIN indexes.

Revision ID: 003
Revises: 002
Create Date: 2026-02-18 01:07:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

# revision identifiers, used by Alembic.
revision: str = '003'
down_revision: Union[str, None] = '002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create document_chunks table with vector support and search indexes.
    
    Creates:
    - document_chunks table with all columns
    - HNSW index for efficient vector similarity search (cosine distance)
    - GIN index for full-text search using tsvector
    - GIN index for trigram fuzzy matching
    - Standard B-tree indexes for job_id and content_hash
    - Unique composite constraint on (job_id, chunk_index)
    """
    # Create document_chunks table
    op.create_table(
        'document_chunks',
        sa.Column('id', UUID(as_uuid=True), nullable=False),
        sa.Column('job_id', UUID(as_uuid=True), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('content_hash', sa.String(64), nullable=True),
        sa.Column('embedding', sa.Text(), nullable=True),  # Stored as vector type via SQL
        sa.Column('metadata', JSONB(), nullable=False, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        
        # Primary key
        sa.PrimaryKeyConstraint('id'),
        
        # Foreign key constraint
        sa.ForeignKeyConstraint(
            ['job_id'], ['jobs.id'],
            name='fk_document_chunks_job_id',
            ondelete='CASCADE'
        ),
        
        # Unique constraint: one chunk_index per job
        sa.UniqueConstraint('job_id', 'chunk_index', name='uq_document_chunks_job_chunk'),
    )
    
    # Create standard B-tree indexes
    # Index on job_id for filtering chunks by job
    op.create_index(
        'idx_document_chunks_job_id',
        'document_chunks',
        ['job_id']
    )
    
    # Index on content_hash for deduplication lookups
    op.create_index(
        'idx_document_chunks_content_hash',
        'document_chunks',
        ['content_hash']
    )
    
    # Composite index for efficient job + chunk_index queries
    op.create_index(
        'idx_document_chunks_job_chunk',
        'document_chunks',
        ['job_id', 'chunk_index']
    )
    
    # HNSW index for vector similarity search (cosine distance)
    # m=16: Number of bi-directional links for each node (higher = more accurate, more memory)
    # ef_construction=64: Size of dynamic candidate list during construction
    op.execute(
        """
        CREATE INDEX idx_document_chunks_embedding_hnsw 
        ON document_chunks 
        USING hnsw (embedding::vector vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """
    )
    
    # GIN index for full-text search on content
    # Uses PostgreSQL tsvector with English text search configuration
    op.execute(
        """
        CREATE INDEX idx_document_chunks_content_tsvector 
        ON document_chunks 
        USING gin (to_tsvector('english', content));
        """
    )
    
    # GIN index for trigram fuzzy matching on content
    # Enables fast similarity comparisons and pattern matching
    op.execute(
        """
        CREATE INDEX idx_document_chunks_content_trgm 
        ON document_chunks 
        USING gin (content gin_trgm_ops);
        """
    )


def downgrade() -> None:
    """Drop document_chunks table and all associated indexes.
    
    Drops all indexes and the table in reverse order of creation.
    """
    # Drop GIN indexes first
    op.execute("DROP INDEX IF EXISTS idx_document_chunks_content_trgm;")
    op.execute("DROP INDEX IF EXISTS idx_document_chunks_content_tsvector;")
    
    # Drop HNSW index
    op.execute("DROP INDEX IF EXISTS idx_document_chunks_embedding_hnsw;")
    
    # Drop standard indexes
    op.drop_index('idx_document_chunks_job_chunk', table_name='document_chunks')
    op.drop_index('idx_document_chunks_content_hash', table_name='document_chunks')
    op.drop_index('idx_document_chunks_job_id', table_name='document_chunks')
    
    # Drop table (automatically drops unique constraint)
    op.drop_table('document_chunks')
