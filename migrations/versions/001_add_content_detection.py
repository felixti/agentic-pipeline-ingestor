"""Add content detection results table.

Revision ID: 001
Revises: 
Create Date: 2026-02-17 14:30:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create content_detection_results table."""
    op.create_table(
        'content_detection_results',
        sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('file_hash', sa.String(length=64), nullable=False),
        sa.Column('file_size', sa.BigInteger(), nullable=False),
        sa.Column('content_type', sa.String(length=20), nullable=False),
        sa.Column('confidence', sa.Numeric(precision=3, scale=2), nullable=False),
        sa.Column('recommended_parser', sa.String(length=50), nullable=False),
        sa.Column('alternative_parsers', postgresql.ARRAY(sa.String()), nullable=False, server_default='{}'),
        sa.Column('text_statistics', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('image_statistics', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('page_results', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('processing_time_ms', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('access_count', sa.Integer(), server_default='1', nullable=False),
        sa.Column('last_accessed_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('file_hash', name='uq_detection_file_hash')
    )
    
    # Create indexes
    op.create_index('idx_detection_hash', 'content_detection_results', ['file_hash'])
    op.create_index('idx_detection_type', 'content_detection_results', ['content_type'])
    op.create_index('idx_detection_expires', 'content_detection_results', ['expires_at'])
    op.create_index('idx_detection_created', 'content_detection_results', ['created_at'])
    
    # Create link table for jobs
    op.create_table(
        'job_detection_results',
        sa.Column('job_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('detection_result_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.ForeignKeyConstraint(['job_id'], ['jobs.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['detection_result_id'], ['content_detection_results.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('job_id', 'detection_result_id')
    )
    
    # Create index on job link
    op.create_index('idx_job_detection_result', 'job_detection_results', ['detection_result_id'])


def downgrade() -> None:
    """Drop content_detection_results table."""
    op.drop_index('idx_job_detection_result', table_name='job_detection_results')
    op.drop_table('job_detection_results')
    
    op.drop_index('idx_detection_created', table_name='content_detection_results')
    op.drop_index('idx_detection_expires', table_name='content_detection_results')
    op.drop_index('idx_detection_type', table_name='content_detection_results')
    op.drop_index('idx_detection_hash', table_name='content_detection_results')
    op.drop_table('content_detection_results')
