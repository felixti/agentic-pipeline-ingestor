from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID as postgresql_uuid

revision: str = "014"
down_revision: str | tuple[str, ...] | None = ("010", "011")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "chunk_embeddings",
        sa.Column("id", postgresql_uuid(as_uuid=True), primary_key=True),
        sa.Column(
            "chunk_id",
            postgresql_uuid(as_uuid=True),
            sa.ForeignKey("document_chunks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("embedding", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("chunk_id", "model_name", name="uq_chunk_model"),
    )

    op.execute(
        """
        CREATE INDEX idx_chunk_embeddings_embedding
        ON chunk_embeddings
        USING hnsw ((embedding::vector) vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """
    )

    op.execute(
        """
        INSERT INTO chunk_embeddings (id, chunk_id, model_name, dimensions, embedding, created_at)
        SELECT
            gen_random_uuid(),
            id,
            'text-embedding-3-small',
            1536,
            embedding::text,
            created_at
        FROM document_chunks
        WHERE embedding IS NOT NULL;
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_chunk_embeddings_embedding;")
    op.drop_table("chunk_embeddings")
