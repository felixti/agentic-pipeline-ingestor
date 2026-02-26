from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import UUID as postgresql_uuid

revision: str = "011"
down_revision: str | None = "009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "document_entities",
        sa.Column("id", postgresql_uuid(as_uuid=True), primary_key=True),
        sa.Column(
            "job_id",
            postgresql_uuid(as_uuid=True),
            sa.ForeignKey("jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "chunk_id",
            postgresql_uuid(as_uuid=True),
            sa.ForeignKey("document_chunks.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("entity_text", sa.Text(), nullable=False),
        sa.Column("entity_type", sa.String(50), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("start_pos", sa.Integer(), nullable=True),
        sa.Column("end_pos", sa.Integer(), nullable=True),
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_index("idx_document_entities_type", "document_entities", ["entity_type"])
    op.execute(
        """
        CREATE INDEX idx_document_entities_text
        ON document_entities USING gin (to_tsvector('english', entity_text));
        """
    )
    op.create_index("idx_document_entities_chunk", "document_entities", ["chunk_id"])


def downgrade() -> None:
    op.drop_index("idx_document_entities_chunk", table_name="document_entities")
    op.execute("DROP INDEX IF EXISTS idx_document_entities_text;")
    op.drop_index("idx_document_entities_type", table_name="document_entities")
    op.drop_table("document_entities")
