"""Unit tests for the retention manager module.

This module tests the DataRetentionManager and related classes.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.retention.manager import (
    DEFAULT_RETENTION_RULES,
    DataRetentionManager,
    RetentionAction,
    RetentionPolicy,
    RetentionRule,
    get_retention_manager,
    set_retention_manager,
)

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_rule():
    """Create a sample retention rule."""
    return RetentionRule(
        name="test_rule",
        description="Test rule for unit tests",
        data_type="test_data",
        retention_days=30,
        action=RetentionAction.DELETE,
        priority=100,
    )


@pytest.fixture
def sample_policy(sample_rule):
    """Create a sample retention policy."""
    return RetentionPolicy(
        name="test_policy",
        description="Test policy for unit tests",
        rules=[sample_rule],
        default_action=RetentionAction.KEEP,
    )


@pytest.fixture
def mock_handler():
    """Create a mock retention handler."""
    handler = AsyncMock()
    handler.delete = AsyncMock(return_value=True)
    handler.archive = AsyncMock(return_value=True)
    handler.compress = AsyncMock(return_value=True)
    handler.anonymize = AsyncMock(return_value=True)
    return handler


@pytest.fixture
def manager(sample_policy, mock_handler):
    """Create a DataRetentionManager with mocked dependencies."""
    return DataRetentionManager(policy=sample_policy, handler=mock_handler)


@pytest.fixture
def expired_date():
    """Create a date in the past (expired)."""
    return datetime.utcnow() - timedelta(days=100)


@pytest.fixture
def future_date():
    """Create a date in the future (not expired)."""
    return datetime.utcnow() + timedelta(days=100)


# ============================================================================
# RetentionAction Enum Tests
# ============================================================================

@pytest.mark.unit
class TestRetentionAction:
    """Tests for RetentionAction enum."""

    def test_enum_values(self):
        """Test retention action enum values."""
        assert RetentionAction.DELETE == "delete"
        assert RetentionAction.ARCHIVE == "archive"
        assert RetentionAction.COMPRESS == "compress"
        assert RetentionAction.ANONYMIZE == "anonymize"
        assert RetentionAction.KEEP == "keep"

    def test_enum_comparison(self):
        """Test retention action comparison."""
        assert RetentionAction.DELETE == "delete"
        assert RetentionAction.DELETE != "archive"
        assert RetentionAction.KEEP.value == "keep"


# ============================================================================
# RetentionRule Tests
# ============================================================================

@pytest.mark.unit
class TestRetentionRule:
    """Tests for RetentionRule class."""

    def test_rule_creation(self):
        """Test creating a retention rule."""
        rule = RetentionRule(
            name="my_rule",
            description="My rule",
            data_type="documents",
            retention_days=90,
            action=RetentionAction.ARCHIVE,
            archive_location="cold_storage",
            conditions={"department": "hr"},
            priority=50,
        )

        assert rule.name == "my_rule"
        assert rule.data_type == "documents"
        assert rule.retention_days == 90
        assert rule.action == RetentionAction.ARCHIVE
        assert rule.archive_location == "cold_storage"
        assert rule.conditions == {"department": "hr"}
        assert rule.priority == 50

    def test_is_applicable_matching_type(self, sample_rule):
        """Test is_applicable with matching data type."""
        assert sample_rule.is_applicable("test_data") is True

    def test_is_applicable_non_matching_type(self, sample_rule):
        """Test is_applicable with non-matching data type."""
        assert sample_rule.is_applicable("other_data") is False

    def test_is_applicable_with_matching_conditions(self, sample_rule):
        """Test is_applicable with matching conditions."""
        sample_rule.conditions = {"status": "active"}
        assert sample_rule.is_applicable("test_data", {"status": "active"}) is True

    def test_is_applicable_with_non_matching_conditions(self, sample_rule):
        """Test is_applicable with non-matching conditions."""
        sample_rule.conditions = {"status": "active"}
        assert sample_rule.is_applicable("test_data", {"status": "inactive"}) is False

    def test_is_applicable_with_no_metadata(self, sample_rule):
        """Test is_applicable with no metadata provided."""
        sample_rule.conditions = {"status": "active"}
        # When no metadata is provided, conditions are not checked (rule applies)
        assert sample_rule.is_applicable("test_data", None) is True

    def test_get_expiration_date(self, sample_rule):
        """Test get_expiration_date calculation."""
        created_at = datetime(2024, 1, 1, 12, 0, 0)
        expiration = sample_rule.get_expiration_date(created_at)

        expected = datetime(2024, 1, 31, 12, 0, 0)
        assert expiration == expected

    def test_is_expired_true(self, sample_rule):
        """Test is_expired returns True for old data."""
        old_date = datetime.utcnow() - timedelta(days=100)
        assert sample_rule.is_expired(old_date) is True

    def test_is_expired_false(self, sample_rule):
        """Test is_expired returns False for recent data."""
        recent_date = datetime.utcnow() - timedelta(days=10)
        assert sample_rule.is_expired(recent_date) is False

    def test_is_expired_exact_boundary(self, sample_rule):
        """Test is_expired at exact boundary."""
        # Exactly at retention period should not be expired (using > comparison)
        from unittest.mock import patch
        fixed_now = datetime(2024, 6, 15, 12, 0, 0)
        with patch("src.retention.manager.datetime") as mock_datetime:
            mock_datetime.utcnow.return_value = fixed_now
            mock_datetime.__gt__ = datetime.__gt__
            mock_datetime.__add__ = datetime.__add__
            # Data created 30 days ago, exactly at boundary
            boundary_date = fixed_now - timedelta(days=30)
            assert sample_rule.is_expired(boundary_date) is False


# ============================================================================
# RetentionPolicy Tests
# ============================================================================

@pytest.mark.unit
class TestRetentionPolicy:
    """Tests for RetentionPolicy class."""

    def test_policy_creation(self, sample_rule):
        """Test creating a retention policy."""
        policy = RetentionPolicy(
            name="my_policy",
            description="My policy",
            rules=[sample_rule],
            default_action=RetentionAction.DELETE,
        )

        assert policy.name == "my_policy"
        assert policy.description == "My policy"
        assert len(policy.rules) == 1
        assert policy.default_action == RetentionAction.DELETE

    def test_get_applicable_rule_returns_matching(self, sample_policy, sample_rule):
        """Test get_applicable_rule returns matching rule."""
        result = sample_policy.get_applicable_rule("test_data")

        assert result == sample_rule

    def test_get_applicable_rule_no_match(self, sample_policy):
        """Test get_applicable_rule returns None when no match."""
        result = sample_policy.get_applicable_rule("nonexistent_data")

        assert result is None

    def test_get_applicable_rule_priority_order(self):
        """Test get_applicable_rule returns highest priority rule."""
        low_priority = RetentionRule(
            name="low",
            description="Low priority",
            data_type="data",
            retention_days=30,
            action=RetentionAction.DELETE,
            priority=10,
        )
        high_priority = RetentionRule(
            name="high",
            description="High priority",
            data_type="data",
            retention_days=60,
            action=RetentionAction.ARCHIVE,
            priority=100,
        )

        policy = RetentionPolicy(
            name="test",
            description="Test",
            rules=[low_priority, high_priority],
        )

        result = policy.get_applicable_rule("data")

        assert result.name == "high"

    def test_get_applicable_rule_with_metadata(self):
        """Test get_applicable_rule with metadata matching."""
        rule_with_condition = RetentionRule(
            name="conditional",
            description="Conditional rule",
            data_type="data",
            retention_days=30,
            action=RetentionAction.DELETE,
            conditions={"type": "sensitive"},
            priority=100,
        )
        general_rule = RetentionRule(
            name="general",
            description="General rule",
            data_type="data",
            retention_days=60,
            action=RetentionAction.KEEP,
            priority=50,
        )

        policy = RetentionPolicy(
            name="test",
            description="Test",
            rules=[general_rule, rule_with_condition],
        )

        # Should match conditional rule
        result = policy.get_applicable_rule("data", {"type": "sensitive"})
        assert result.name == "conditional"

        # Should match general rule
        result = policy.get_applicable_rule("data", {"type": "normal"})
        assert result.name == "general"


# ============================================================================
# DataRetentionManager Initialization Tests
# ============================================================================

@pytest.mark.unit
class TestDataRetentionManagerInitialization:
    """Tests for DataRetentionManager initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default policy."""
        manager = DataRetentionManager()

        assert manager.policy is not None
        assert manager.policy.name == "default"
        assert len(manager.policy.rules) == len(DEFAULT_RETENTION_RULES)
        assert manager.handler is None

    def test_init_with_custom_policy(self, sample_policy):
        """Test initialization with custom policy."""
        manager = DataRetentionManager(policy=sample_policy)

        assert manager.policy == sample_policy

    def test_init_with_handler(self, mock_handler):
        """Test initialization with handler."""
        manager = DataRetentionManager(handler=mock_handler)

        assert manager.handler is mock_handler

    def test_default_retention_rules(self):
        """Test default retention rules are defined."""
        assert len(DEFAULT_RETENTION_RULES) > 0

        # Check for expected rule types
        rule_names = [r.name for r in DEFAULT_RETENTION_RULES]
        assert "raw_files" in rule_names
        assert "processed_data" in rule_names
        assert "job_metadata" in rule_names
        assert "audit_logs" in rule_names
        assert "temporary_files" in rule_names


# ============================================================================
# Retention Rule Lookup Tests
# ============================================================================

@pytest.mark.unit
class TestRetentionRuleLookup:
    """Tests for retention rule lookup methods."""

    def test_get_retention_rule_finds_matching(self, manager, sample_rule):
        """Test get_retention_rule finds matching rule."""
        result = manager.get_retention_rule("test_data")

        assert result.name == sample_rule.name

    def test_get_retention_rule_returns_default_for_unknown(self, manager):
        """Test get_retention_rule returns default for unknown types."""
        result = manager.get_retention_rule("unknown_type")

        assert result.name == "default"
        assert result.data_type == "unknown_type"
        assert result.action == manager.policy.default_action

    def test_get_retention_rule_with_metadata(self, manager):
        """Test get_retention_rule with metadata."""
        # Create a rule with conditions
        conditional_rule = RetentionRule(
            name="conditional",
            description="Conditional",
            data_type="data",
            retention_days=30,
            action=RetentionAction.DELETE,
            conditions={"status": "failed"},
            priority=200,
        )
        manager.policy.rules.append(conditional_rule)

        result = manager.get_retention_rule("data", {"status": "failed"})
        assert result.name == "conditional"

    def test_is_data_expired_true(self, manager, expired_date):
        """Test is_data_expired returns True for expired data."""
        assert manager.is_data_expired("test_data", expired_date) is True

    def test_is_data_expired_false(self, manager, future_date):
        """Test is_data_expired returns False for non-expired data."""
        assert manager.is_data_expired("test_data", future_date) is False

    def test_get_expiration_date(self, manager):
        """Test get_expiration_date calculation."""
        created_at = datetime(2024, 6, 1, 12, 0, 0)
        expiration = manager.get_expiration_date("test_data", created_at)

        expected = datetime(2024, 7, 1, 12, 0, 0)  # 30 days later
        assert expiration == expected


# ============================================================================
# Policy Application Tests
# ============================================================================

@pytest.mark.unit
class TestPolicyApplication:
    """Tests for apply_retention_policy method."""

    @pytest.mark.asyncio
    async def test_apply_retention_policy_returns_summary(self, manager):
        """Test apply_retention_policy returns summary."""
        result = await manager.apply_retention_policy("my_bucket")

        assert result["bucket"] == "my_bucket"
        assert "processed" in result
        assert "deleted" in result
        assert "archived" in result
        assert "errors" in result

    @pytest.mark.asyncio
    async def test_apply_retention_policy_with_data_type(self, manager):
        """Test apply_retention_policy with data type filter."""
        result = await manager.apply_retention_policy("my_bucket", data_type="raw_files")

        assert result["bucket"] == "my_bucket"


# ============================================================================
# Data Item Processing Tests
# ============================================================================

@pytest.mark.unit
class TestDataItemProcessing:
    """Tests for process_data_item method."""

    @pytest.mark.asyncio
    async def test_process_not_expired_returns_true(self, manager, future_date):
        """Test processing non-expired data returns True (no action needed)."""
        result = await manager.process_data_item("data_1", "test_data", future_date)

        assert result is True

    @pytest.mark.asyncio
    async def test_process_delete_action(self, manager, mock_handler, expired_date):
        """Test processing delete action."""
        # Create rule with DELETE action
        delete_rule = RetentionRule(
            name="delete_rule",
            description="Delete rule",
            data_type="temp_data",
            retention_days=1,
            action=RetentionAction.DELETE,
            priority=100,
        )
        manager.policy.rules = [delete_rule]

        result = await manager.process_data_item("data_1", "temp_data", expired_date)

        assert result is True
        mock_handler.delete.assert_called_once_with("data_1", "temp_data")

    @pytest.mark.asyncio
    async def test_process_archive_action(self, manager, mock_handler, expired_date):
        """Test processing archive action."""
        archive_rule = RetentionRule(
            name="archive_rule",
            description="Archive rule",
            data_type="old_data",
            retention_days=1,
            action=RetentionAction.ARCHIVE,
            archive_location="cold_storage",
            priority=100,
        )
        manager.policy.rules = [archive_rule]

        result = await manager.process_data_item("data_1", "old_data", expired_date)

        assert result is True
        mock_handler.archive.assert_called_once_with("data_1", "old_data", "cold_storage")

    @pytest.mark.asyncio
    async def test_process_compress_action(self, manager, mock_handler, expired_date):
        """Test processing compress action."""
        compress_rule = RetentionRule(
            name="compress_rule",
            description="Compress rule",
            data_type="large_data",
            retention_days=1,
            action=RetentionAction.COMPRESS,
            priority=100,
        )
        manager.policy.rules = [compress_rule]

        result = await manager.process_data_item("data_1", "large_data", expired_date)

        assert result is True
        mock_handler.compress.assert_called_once_with("data_1", "large_data")

    @pytest.mark.asyncio
    async def test_process_anonymize_action(self, manager, mock_handler, expired_date):
        """Test processing anonymize action."""
        anonymize_rule = RetentionRule(
            name="anonymize_rule",
            description="Anonymize rule",
            data_type="user_data",
            retention_days=1,
            action=RetentionAction.ANONYMIZE,
            priority=100,
        )
        manager.policy.rules = [anonymize_rule]

        result = await manager.process_data_item("data_1", "user_data", expired_date)

        assert result is True
        mock_handler.anonymize.assert_called_once_with("data_1", "user_data")

    @pytest.mark.asyncio
    async def test_process_keep_action(self, manager, expired_date):
        """Test processing keep action (no-op)."""
        keep_rule = RetentionRule(
            name="keep_rule",
            description="Keep rule",
            data_type="important_data",
            retention_days=1,
            action=RetentionAction.KEEP,
            priority=100,
        )
        manager.policy.rules = [keep_rule]

        result = await manager.process_data_item("data_1", "important_data", expired_date)

        assert result is True
        # No handler calls expected

    @pytest.mark.asyncio
    async def test_process_unknown_action(self, manager, expired_date, mock_handler):
        """Test processing with a rule that has action handler returning False."""
        # Create a rule with DELETE action but make handler return False
        delete_rule = RetentionRule(
            name="delete_fail",
            description="Delete fail",
            data_type="data",
            retention_days=1,
            action=RetentionAction.DELETE,
            priority=100,
        )
        manager.policy.rules = [delete_rule]
        mock_handler.delete.return_value = False

        result = await manager.process_data_item("data_1", "data", expired_date)

        assert result is False

    @pytest.mark.asyncio
    async def test_process_handles_errors(self, manager, mock_handler, expired_date):
        """Test processing handles handler errors."""
        mock_handler.delete.side_effect = Exception("Delete failed")

        delete_rule = RetentionRule(
            name="delete_rule",
            description="Delete rule",
            data_type="temp_data",
            retention_days=1,
            action=RetentionAction.DELETE,
            priority=100,
        )
        manager.policy.rules = [delete_rule]

        result = await manager.process_data_item("data_1", "temp_data", expired_date)

        assert result is False


# ============================================================================
# Action Handler Tests
# ============================================================================

@pytest.mark.unit
class TestActionHandlers:
    """Tests for action handler registration and usage."""

    def test_register_action_handler(self, manager):
        """Test registering a custom action handler."""
        mock_handler = MagicMock()

        manager.register_action_handler(RetentionAction.DELETE, mock_handler)

        assert RetentionAction.DELETE in manager._action_handlers
        assert manager._action_handlers[RetentionAction.DELETE] == mock_handler

    @pytest.mark.asyncio
    async def test_custom_handler_called(self, manager, expired_date):
        """Test custom handler is called instead of default."""
        custom_handler = AsyncMock(return_value=True)
        manager.register_action_handler(RetentionAction.DELETE, custom_handler)

        delete_rule = RetentionRule(
            name="delete_rule",
            description="Delete rule",
            data_type="temp_data",
            retention_days=1,
            action=RetentionAction.DELETE,
            priority=100,
        )
        manager.policy.rules = [delete_rule]

        await manager.process_data_item("data_1", "temp_data", expired_date)

        custom_handler.assert_called_once_with("data_1", "temp_data")

    @pytest.mark.asyncio
    async def test_delete_data_with_handler(self, manager, mock_handler):
        """Test _delete_data with handler."""
        result = await manager._delete_data("data_1", "test_data")

        assert result is True
        mock_handler.delete.assert_called_once_with("data_1", "test_data")

    @pytest.mark.asyncio
    async def test_delete_data_no_handler(self, manager):
        """Test _delete_data without handler returns False."""
        manager.handler = None

        result = await manager._delete_data("data_1", "test_data")

        assert result is False

    @pytest.mark.asyncio
    async def test_archive_data_with_handler(self, manager, mock_handler):
        """Test _archive_data with handler."""
        result = await manager._archive_data("data_1", "test_data", "archive_location")

        assert result is True
        mock_handler.archive.assert_called_once_with("data_1", "test_data", "archive_location")

    @pytest.mark.asyncio
    async def test_archive_data_default_location(self, manager, mock_handler):
        """Test _archive_data with default location."""
        result = await manager._archive_data("data_1", "test_data", None)

        assert result is True
        mock_handler.archive.assert_called_once_with("data_1", "test_data", "archive")

    @pytest.mark.asyncio
    async def test_compress_data_with_handler(self, manager, mock_handler):
        """Test _compress_data with handler."""
        result = await manager._compress_data("data_1", "test_data")

        assert result is True
        mock_handler.compress.assert_called_once_with("data_1", "test_data")

    @pytest.mark.asyncio
    async def test_anonymize_data_with_handler(self, manager, mock_handler):
        """Test _anonymize_data with handler."""
        result = await manager._anonymize_data("data_1", "test_data")

        assert result is True
        mock_handler.anonymize.assert_called_once_with("data_1", "test_data")


# ============================================================================
# Policy Summary Tests
# ============================================================================

@pytest.mark.unit
class TestPolicySummary:
    """Tests for get_policy_summary method."""

    def test_get_policy_summary(self, manager, sample_rule):
        """Test getting policy summary."""
        summary = manager.get_policy_summary()

        assert summary["policy_name"] == manager.policy.name
        assert summary["description"] == manager.policy.description
        assert summary["default_action"] == manager.policy.default_action.value
        assert len(summary["rules"]) == len(manager.policy.rules)

        # Check rule details
        rule_summary = summary["rules"][0]
        assert rule_summary["name"] == sample_rule.name
        assert rule_summary["data_type"] == sample_rule.data_type
        assert rule_summary["retention_days"] == sample_rule.retention_days
        assert rule_summary["action"] == sample_rule.action.value

    def test_get_policy_summary_empty_rules(self):
        """Test getting policy summary with no rules."""
        policy = RetentionPolicy(name="empty", description="Empty", rules=[])
        manager = DataRetentionManager(policy=policy)

        summary = manager.get_policy_summary()

        assert summary["rules"] == []


# ============================================================================
# Global Instance Tests
# ============================================================================

@pytest.mark.unit
class TestGlobalInstance:
    """Tests for global retention manager instance."""

    def test_get_retention_manager_singleton(self):
        """Test get_retention_manager returns singleton."""
        # Reset global state
        set_retention_manager(None)

        manager1 = get_retention_manager()
        manager2 = get_retention_manager()

        assert manager1 is manager2

    def test_set_retention_manager(self, sample_policy):
        """Test set_retention_manager sets global instance."""
        custom_manager = DataRetentionManager(policy=sample_policy)

        set_retention_manager(custom_manager)

        assert get_retention_manager() is custom_manager

    def test_set_retention_manager_overwrites(self, manager):
        """Test set_retention_manager overwrites existing."""
        set_retention_manager(manager)

        new_manager = DataRetentionManager()
        set_retention_manager(new_manager)

        assert get_retention_manager() is new_manager


# ============================================================================
# Default Rules Tests
# ============================================================================

@pytest.mark.unit
class TestDefaultRules:
    """Tests for default retention rules."""

    def test_raw_files_rule(self):
        """Test raw files retention rule."""
        rule = next(r for r in DEFAULT_RETENTION_RULES if r.name == "raw_files")

        assert rule.data_type == "raw_file"
        assert rule.retention_days == 30
        assert rule.action == RetentionAction.DELETE

    def test_processed_data_rule(self):
        """Test processed data retention rule."""
        rule = next(r for r in DEFAULT_RETENTION_RULES if r.name == "processed_data")

        assert rule.data_type == "processed_data"
        assert rule.retention_days == 90
        assert rule.action == RetentionAction.ARCHIVE
        assert rule.archive_location == "cold_storage"

    def test_job_metadata_rule(self):
        """Test job metadata retention rule."""
        rule = next(r for r in DEFAULT_RETENTION_RULES if r.name == "job_metadata")

        assert rule.data_type == "job_metadata"
        assert rule.retention_days == 2555  # 7 years
        assert rule.action == RetentionAction.KEEP

    def test_audit_logs_rule(self):
        """Test audit logs retention rule."""
        rule = next(r for r in DEFAULT_RETENTION_RULES if r.name == "audit_logs")

        assert rule.data_type == "audit_log"
        assert rule.retention_days == 2555
        assert rule.action == RetentionAction.KEEP

    def test_lineage_records_rule(self):
        """Test lineage records retention rule."""
        rule = next(r for r in DEFAULT_RETENTION_RULES if r.name == "lineage_records")

        assert rule.data_type == "lineage_record"
        assert rule.retention_days == 2555
        assert rule.action == RetentionAction.KEEP

    def test_temporary_files_rule(self):
        """Test temporary files retention rule."""
        rule = next(r for r in DEFAULT_RETENTION_RULES if r.name == "temporary_files")

        assert rule.data_type == "temp_file"
        assert rule.retention_days == 1
        assert rule.action == RetentionAction.DELETE

    def test_failed_job_data_rule(self):
        """Test failed job data retention rule."""
        rule = next(r for r in DEFAULT_RETENTION_RULES if r.name == "failed_job_data")

        assert rule.data_type == "raw_file"
        assert rule.retention_days == 14
        assert rule.action == RetentionAction.DELETE
        assert rule.conditions == {"job_status": "failed"}
        assert rule.priority == 200  # Higher than raw_files

    def test_failed_job_rule_priority(self):
        """Test failed job rule has higher priority than raw files."""
        raw_rule = next(r for r in DEFAULT_RETENTION_RULES if r.name == "raw_files")
        failed_rule = next(r for r in DEFAULT_RETENTION_RULES if r.name == "failed_job_data")

        assert failed_rule.priority > raw_rule.priority
