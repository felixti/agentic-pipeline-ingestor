from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "010"
down_revision: str | None = "009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "document_chunks",
        sa.Column("quality_score", sa.Float(), nullable=True),
    )
    op.execute(
        """
        CREATE INDEX idx_document_chunks_quality_score
        ON document_chunks (quality_score)
        WHERE quality_score IS NOT NULL;
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_document_chunks_quality_score;")
    op.drop_column("document_chunks", "quality_score")
