from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import UUID as postgresql_uuid

revision: str = "012"
down_revision: str | None = "011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "search_analytics",
        sa.Column("id", postgresql_uuid(as_uuid=True), primary_key=True),
        sa.Column(
            "query_text",
            sa.Text(),
            nullable=False,
        ),
        sa.Column("query_embedding_hash", sa.String(64), nullable=True),
        sa.Column("search_type", sa.String(20), nullable=False),  # semantic, text, hybrid
        sa.Column("filters", postgresql.JSONB(), nullable=True),
        sa.Column("result_count", sa.Integer(), nullable=False, default=0),
        sa.Column("top_k", sa.Integer(), nullable=False, default=10),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column(
            "user_id",
            postgresql_uuid(as_uuid=True),
            nullable=True,
        ),
        sa.Column(
            "job_id",
            postgresql_uuid(as_uuid=True),
            sa.ForeignKey("jobs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "clicked_results", postgresql.JSONB(), nullable=True
        ),  # List of clicked chunk IDs
        sa.Column(
            "relevance_feedback", sa.Integer(), nullable=True
        ),  # -1, 0, 1 for thumbs down/neutral/up
        sa.Column("metadata", postgresql.JSONB(), nullable=True, default={}),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Indexes for analytics queries
    op.create_index(
        "idx_search_analytics_created_at",
        "search_analytics",
        ["created_at"],
    )
    op.create_index(
        "idx_search_analytics_search_type",
        "search_analytics",
        ["search_type"],
    )
    op.create_index(
        "idx_search_analytics_user_id",
        "search_analytics",
        ["user_id"],
    )
    op.execute(
        """
        CREATE INDEX idx_search_analytics_query_text
        ON search_analytics USING gin (to_tsvector('english', query_text));
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_search_analytics_query_text;")
    op.drop_index("idx_search_analytics_user_id", table_name="search_analytics")
    op.drop_index("idx_search_analytics_search_type", table_name="search_analytics")
    op.drop_index("idx_search_analytics_created_at", table_name="search_analytics")
    op.drop_table("search_analytics")
