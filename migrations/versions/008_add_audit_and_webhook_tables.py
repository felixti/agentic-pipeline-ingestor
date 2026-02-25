"""Add audit logs, API keys, and webhook tables.

Revision ID: 008
Revises: 007
Create Date: 2026-02-25 20:00:00.000000

This migration adds supporting tables for:
- Audit logging
- API key management  
- Webhook subscriptions and deliveries

"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

# revision identifiers, used by Alembic.
revision: str = "008"
down_revision: str | None = "007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create audit logs, API keys, and webhook tables."""
    
    # Create audit_logs table
    op.create_table(
        "audit_logs",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=True),
        sa.Column("api_key_id", sa.String(length=255), nullable=True),
        sa.Column("action", sa.String(length=50), nullable=False),
        sa.Column("resource_type", sa.String(length=50), nullable=False),
        sa.Column("resource_id", sa.String(length=255), nullable=True),
        sa.Column("request_method", sa.String(length=10), nullable=True),
        sa.Column("request_path", sa.String(length=500), nullable=True),
        sa.Column("request_details", JSONB(), nullable=True),
        sa.Column("success", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("ip_address", sa.String(length=45), nullable=True),
        sa.Column("user_agent", sa.String(length=500), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    
    # Create indexes for audit_logs
    op.create_index("idx_audit_timestamp", "audit_logs", ["timestamp"])
    op.create_index("idx_audit_user_id", "audit_logs", ["user_id"])
    op.create_index("idx_audit_action", "audit_logs", ["action"])
    op.create_index("idx_audit_resource", "audit_logs", ["resource_type"])
    
    # Create api_keys table
    op.create_table(
        "api_keys",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("key_hash", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("permissions", sa.ARRAY(sa.String()), nullable=False, server_default="{}"),
        sa.Column("is_active", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_by", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("key_hash", name="uq_api_keys_key_hash"),
    )
    
    # Create indexes for api_keys
    op.create_index("idx_api_keys_hash", "api_keys", ["key_hash"])
    op.create_index("idx_api_keys_created", "api_keys", ["created_at"])
    
    # Create webhook_subscriptions table
    op.create_table(
        "webhook_subscriptions",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("url", sa.String(length=500), nullable=False),
        sa.Column("events", sa.ARRAY(sa.String()), nullable=False, server_default="{}"),
        sa.Column("secret", sa.String(length=255), nullable=True),
        sa.Column("is_active", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    
    # Create indexes for webhook_subscriptions
    op.create_index("idx_webhook_subs_user", "webhook_subscriptions", ["user_id"])
    op.create_index("idx_webhook_subs_created", "webhook_subscriptions", ["created_at"])
    
    # Create webhook_deliveries table
    op.create_table(
        "webhook_deliveries",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("subscription_id", UUID(as_uuid=True), nullable=False),
        sa.Column("event_type", sa.String(length=50), nullable=False),
        sa.Column("payload", JSONB(), nullable=False, server_default="{}"),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="5"),
        sa.Column("http_status", sa.Integer(), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_retry_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["subscription_id"],
            ["webhook_subscriptions.id"],
            name="fk_webhook_deliveries_subscription_id",
            ondelete="CASCADE"
        ),
    )
    
    # Create indexes for webhook_deliveries
    op.create_index("idx_webhook_deliv_subscription", "webhook_deliveries", ["subscription_id"])
    op.create_index("idx_webhook_deliv_event", "webhook_deliveries", ["event_type"])
    op.create_index("idx_webhook_deliv_status", "webhook_deliveries", ["status"])
    op.create_index("idx_webhook_deliv_created", "webhook_deliveries", ["created_at"])


def downgrade() -> None:
    """Drop audit logs, API keys, and webhook tables."""
    
    # Drop webhook deliveries indexes and table
    op.drop_index("idx_webhook_deliv_created", table_name="webhook_deliveries")
    op.drop_index("idx_webhook_deliv_status", table_name="webhook_deliveries")
    op.drop_index("idx_webhook_deliv_event", table_name="webhook_deliveries")
    op.drop_index("idx_webhook_deliv_subscription", table_name="webhook_deliveries")
    op.drop_table("webhook_deliveries")
    
    # Drop webhook subscriptions indexes and table
    op.drop_index("idx_webhook_subs_created", table_name="webhook_subscriptions")
    op.drop_index("idx_webhook_subs_user", table_name="webhook_subscriptions")
    op.drop_table("webhook_subscriptions")
    
    # Drop API keys indexes and table
    op.drop_index("idx_api_keys_created", table_name="api_keys")
    op.drop_index("idx_api_keys_hash", table_name="api_keys")
    op.drop_table("api_keys")
    
    # Drop audit logs indexes and table
    op.drop_index("idx_audit_resource", table_name="audit_logs")
    op.drop_index("idx_audit_action", table_name="audit_logs")
    op.drop_index("idx_audit_user_id", table_name="audit_logs")
    op.drop_index("idx_audit_timestamp", table_name="audit_logs")
    op.drop_table("audit_logs")
