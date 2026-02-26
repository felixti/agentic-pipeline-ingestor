from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "013"
down_revision: str | Sequence[str] | None = ("010", "011")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE document_chunks
        ADD COLUMN sparse_embedding SPARSEVEC(30522);
        """
    )
    op.add_column(
        "document_chunks",
        sa.Column("sparse_model", sa.String(length=50), nullable=True),
    )

    op.execute(
        """
        CREATE INDEX idx_document_chunks_sparse_embedding
        ON document_chunks
        USING hnsw (sparse_embedding sparsevec_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        WHERE sparse_embedding IS NOT NULL;
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_document_chunks_sparse_embedding;")
    op.drop_column("document_chunks", "sparse_model")
    op.drop_column("document_chunks", "sparse_embedding")
