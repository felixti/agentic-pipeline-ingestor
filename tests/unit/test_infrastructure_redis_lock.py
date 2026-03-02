"""Unit tests for Redis distributed locking.

Tests for the DistributedLock and CogneeWriteLock classes.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Skip tests if redis is not available
pytest.importorskip("redis.asyncio", reason="Redis not installed")

from src.infrastructure.redis.lock import (
    CogneeWriteLock,
    DistributedLock,
    LockAcquisitionError,
    distributed_lock,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client."""
    client = MagicMock()
    
    # Mock lock
    mock_lock = MagicMock()
    mock_lock.acquire = AsyncMock(return_value=True)
    mock_lock.release = AsyncMock()
    
    client.lock = MagicMock(return_value=mock_lock)
    client.ping = AsyncMock(return_value=True)
    
    return client


@pytest.fixture
def mock_failing_redis_client():
    """Create a mock Redis client that fails to acquire locks."""
    client = MagicMock()
    
    # Mock lock that fails to acquire
    mock_lock = MagicMock()
    mock_lock.acquire = AsyncMock(return_value=False)
    mock_lock.release = AsyncMock()
    
    client.lock = MagicMock(return_value=mock_lock)
    client.ping = AsyncMock(return_value=True)
    
    return client


# ============================================================================
# DistributedLock Tests
# ============================================================================

@pytest.mark.unit
class TestDistributedLock:
    """Tests for DistributedLock class."""

    @pytest.mark.asyncio
    async def test_lock_initialization(self, mock_redis_client):
        """Test lock initialization."""
        lock = DistributedLock(
            name="test_lock",
            redis_client=mock_redis_client,
            timeout=60,
            blocking=True,
            blocking_timeout=30.0,
        )

        assert lock.name == "test_lock"
        assert lock.timeout == 60
        assert lock.blocking is True
        assert lock.blocking_timeout == 30.0
        assert lock.is_acquired is False

    @pytest.mark.asyncio
    async def test_lock_acquire_success(self, mock_redis_client):
        """Test successful lock acquisition."""
        lock = DistributedLock(
            name="test_lock",
            redis_client=mock_redis_client,
        )

        result = await lock.acquire()

        assert result is True
        assert lock.is_acquired is True
        mock_redis_client.lock.assert_called_once_with("test_lock", timeout=60)

    @pytest.mark.asyncio
    async def test_lock_acquire_failure(self, mock_failing_redis_client):
        """Test lock acquisition failure with blocking."""
        lock = DistributedLock(
            name="test_lock",
            redis_client=mock_failing_redis_client,
            blocking=True,
            blocking_timeout=5.0,
        )

        with pytest.raises(LockAcquisitionError):
            await lock.acquire()

    @pytest.mark.asyncio
    async def test_lock_acquire_non_blocking(self, mock_failing_redis_client):
        """Test non-blocking lock acquisition failure."""
        lock = DistributedLock(
            name="test_lock",
            redis_client=mock_failing_redis_client,
            blocking=False,
        )

        result = await lock.acquire()

        assert result is False
        assert lock.is_acquired is False

    @pytest.mark.asyncio
    async def test_lock_release(self, mock_redis_client):
        """Test lock release."""
        lock = DistributedLock(
            name="test_lock",
            redis_client=mock_redis_client,
        )

        await lock.acquire()
        assert lock.is_acquired is True

        await lock.release()
        assert lock.is_acquired is False

    @pytest.mark.asyncio
    async def test_lock_release_not_acquired(self, mock_redis_client):
        """Test release when lock was not acquired (safe to call)."""
        lock = DistributedLock(
            name="test_lock",
            redis_client=mock_redis_client,
        )

        # Should not raise error
        await lock.release()
        assert lock.is_acquired is False

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_redis_client):
        """Test using lock as async context manager."""
        lock = DistributedLock(
            name="test_lock",
            redis_client=mock_redis_client,
        )

        async with lock:
            assert lock.is_acquired is True

        assert lock.is_acquired is False

    @pytest.mark.asyncio
    async def test_context_manager_exception(self, mock_redis_client):
        """Test lock is released even when exception occurs."""
        lock = DistributedLock(
            name="test_lock",
            redis_client=mock_redis_client,
        )

        with pytest.raises(ValueError):
            async with lock:
                assert lock.is_acquired is True
                raise ValueError("Test error")

        assert lock.is_acquired is False

    @pytest.mark.asyncio
    async def test_lock_gets_redis_client(self):
        """Test that lock gets Redis client if not provided."""
        lock = DistributedLock(name="test_lock")
        
        mock_client = MagicMock()
        mock_lock_instance = MagicMock()
        mock_lock_instance.acquire = AsyncMock(return_value=True)
        mock_lock_instance.release = AsyncMock()
        mock_client.lock = MagicMock(return_value=mock_lock_instance)

        with patch(
            "src.infrastructure.redis.client.get_redis_client",
            AsyncMock(return_value=mock_client),
        ):
            result = await lock.acquire()

        assert result is True


# ============================================================================
# CogneeWriteLock Tests
# ============================================================================

@pytest.mark.unit
class TestCogneeWriteLock:
    """Tests for CogneeWriteLock class."""

    @pytest.mark.asyncio
    async def test_cognee_lock_initialization(self, mock_redis_client):
        """Test Cognee write lock initialization."""
        lock = CogneeWriteLock(
            dataset_id="my_dataset",
            redis_client=mock_redis_client,
            timeout=300,
        )

        assert lock.dataset_id == "my_dataset"
        # Internal lock should have correct name
        assert lock._lock.name == "cognee_write_lock:my_dataset"

    @pytest.mark.asyncio
    async def test_cognee_lock_acquire_release(self, mock_redis_client):
        """Test Cognee lock acquire and release."""
        lock = CogneeWriteLock(
            dataset_id="my_dataset",
            redis_client=mock_redis_client,
        )

        result = await lock.acquire()
        assert result is True

        await lock.release()
        # Should complete without error

    @pytest.mark.asyncio
    async def test_cognee_lock_context_manager(self, mock_redis_client):
        """Test Cognee lock as context manager."""
        lock = CogneeWriteLock(
            dataset_id="my_dataset",
            redis_client=mock_redis_client,
        )

        async with lock:
            # Critical section
            pass

        # Should complete without error

    @pytest.mark.asyncio
    async def test_different_datasets_different_locks(self, mock_redis_client):
        """Test that different datasets have different locks."""
        lock1 = CogneeWriteLock(
            dataset_id="dataset_1",
            redis_client=mock_redis_client,
        )
        lock2 = CogneeWriteLock(
            dataset_id="dataset_2",
            redis_client=mock_redis_client,
        )

        assert lock1._lock.name != lock2._lock.name
        assert "dataset_1" in lock1._lock.name
        assert "dataset_2" in lock2._lock.name


# ============================================================================
# distributed_lock Context Manager Tests
# ============================================================================

@pytest.mark.unit
class TestDistributedLockContextManager:
    """Tests for the distributed_lock context manager function."""

    @pytest.mark.asyncio
    async def test_distributed_lock_context(self, mock_redis_client):
        """Test distributed_lock context manager."""
        async with distributed_lock(
            name="test_lock",
            redis_client=mock_redis_client,
            timeout=60,
        ) as lock:
            assert isinstance(lock, DistributedLock)
            assert lock.is_acquired is True

    @pytest.mark.asyncio
    async def test_distributed_lock_acquire_failure(self, mock_failing_redis_client):
        """Test distributed_lock raises on acquire failure."""
        with pytest.raises(LockAcquisitionError):
            async with distributed_lock(
                name="test_lock",
                redis_client=mock_failing_redis_client,
                blocking=True,
                blocking_timeout=1.0,
            ):
                pass  # Should not reach here

    @pytest.mark.asyncio
    async def test_distributed_lock_releases_on_exception(self, mock_redis_client):
        """Test lock is released even when exception occurs."""
        lock_holder = {"lock": None}

        with pytest.raises(ValueError):
            async with distributed_lock(
                name="test_lock",
                redis_client=mock_redis_client,
            ) as lock:
                lock_holder["lock"] = lock
                assert lock.is_acquired is True
                raise ValueError("Test error")

        # Lock should be released
        assert lock_holder["lock"].is_acquired is False
