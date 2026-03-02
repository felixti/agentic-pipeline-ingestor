"""Tests for database transaction handling.

This module tests the transaction handling fixes to ensure:
1. Sessions are properly committed before closing
2. Job existence is verified before creating chunks
3. Rollbacks happen on errors
4. No "closed transaction" errors occur
"""

import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4, UUID

from src.db.transaction import safe_transaction, verify_job_exists_simple
from src.db.repositories.document_chunk_repository import DocumentChunkRepository
from src.db.models import DocumentChunkModel, JobModel, JobStatus


@pytest.mark.asyncio
async def test_safe_transaction_commits_on_success():
    """Test that safe_transaction commits when operation succeeds."""
    # Create a mock session factory
    mock_session = AsyncMock(spec=AsyncSession)
    mock_factory = MagicMock(return_value=mock_session)
    
    async with safe_transaction(mock_factory) as session:
        # Perform some operation
        await session.execute("SELECT 1")
    
    # Verify commit was called
    mock_session.commit.assert_called_once()
    mock_session.close.assert_called_once()
    mock_session.rollback.assert_not_called()


@pytest.mark.asyncio
async def test_safe_transaction_rolls_back_on_error():
    """Test that safe_transaction rolls back when operation fails."""
    mock_session = AsyncMock(spec=AsyncSession)
    mock_factory = MagicMock(return_value=mock_session)
    
    with pytest.raises(ValueError, match="Test error"):
        async with safe_transaction(mock_factory) as session:
            raise ValueError("Test error")
    
    # Verify rollback was called, not commit
    mock_session.rollback.assert_called_once()
    mock_session.commit.assert_not_called()
    mock_session.close.assert_called_once()


@pytest.mark.asyncio
async def test_safe_transaction_rolls_back_on_sql_error():
    """Test that safe_transaction rolls back on SQLAlchemy errors."""
    mock_session = AsyncMock(spec=AsyncSession)
    mock_factory = MagicMock(return_value=mock_session)
    
    with pytest.raises(SQLAlchemyError):
        async with safe_transaction(mock_factory) as session:
            raise SQLAlchemyError("Database error")
    
    # Verify rollback was called
    mock_session.rollback.assert_called_once()
    mock_session.commit.assert_not_called()
    mock_session.close.assert_called_once()


@pytest.mark.asyncio
async def test_document_chunk_repository_fails_for_nonexistent_job():
    """Test that DocumentChunkRepository raises error for non-existent job."""
    # Create mock session
    mock_session = AsyncMock(spec=AsyncSession)
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None  # Job doesn't exist
    mock_session.execute.return_value = mock_result
    
    repo = DocumentChunkRepository(mock_session)
    
    # Create a chunk for a non-existent job
    non_existent_job_id = uuid4()
    chunk = DocumentChunkModel(
        id=uuid4(),
        job_id=non_existent_job_id,
        chunk_index=0,
        content="Test content",
        content_hash="abc123",
    )
    
    # Should fail because job doesn't exist
    with pytest.raises(ValueError, match=f"Job {non_existent_job_id} does not exist"):
        await repo.create(chunk)


@pytest.mark.asyncio
async def test_document_chunk_repository_succeeds_for_existing_job():
    """Test that DocumentChunkRepository succeeds when job exists."""
    # Create mock session
    mock_session = AsyncMock(spec=AsyncSession)
    mock_result = MagicMock()
    
    # Create a mock job that exists
    mock_job = MagicMock(spec=JobModel)
    mock_job.id = uuid4()
    mock_result.scalar_one_or_none.return_value = mock_job
    mock_session.execute.return_value = mock_result
    
    repo = DocumentChunkRepository(mock_session)
    
    chunk = DocumentChunkModel(
        id=uuid4(),
        job_id=mock_job.id,
        chunk_index=0,
        content="Test content",
        content_hash="abc123",
    )
    
    # Should succeed because job exists
    created = await repo.create(chunk)
    assert created.id == chunk.id
    mock_session.add.assert_called_once()


@pytest.mark.asyncio
async def test_document_chunk_repository_bulk_create_verifies_job():
    """Test that bulk_create verifies job exists."""
    mock_session = AsyncMock(spec=AsyncSession)
    mock_result = MagicMock()
    
    mock_job = MagicMock(spec=JobModel)
    mock_job.id = uuid4()
    mock_result.scalar_one_or_none.return_value = mock_job
    mock_session.execute.return_value = mock_result
    
    repo = DocumentChunkRepository(mock_session)
    
    chunks = [
        DocumentChunkModel(
            id=uuid4(),
            job_id=mock_job.id,
            chunk_index=i,
            content=f"Content {i}",
            content_hash=f"hash{i}",
        )
        for i in range(3)
    ]
    
    # Should succeed because job exists
    created = await repo.bulk_create(chunks)
    assert len(created) == 3


@pytest.mark.asyncio
async def test_document_chunk_repository_upsert_verifies_job():
    """Test that upsert_chunks verifies job exists."""
    mock_session = AsyncMock(spec=AsyncSession)
    mock_result = MagicMock()
    
    mock_job = MagicMock(spec=JobModel)
    mock_job.id = uuid4()
    mock_result.scalar_one_or_none.return_value = mock_job
    mock_session.execute.return_value = mock_result
    mock_session.execute.return_value.rowcount = 3
    
    repo = DocumentChunkRepository(mock_session)
    
    job_id = uuid4()
    # Update the mock to return the job for the specific job_id
    mock_result.scalar_one_or_none.return_value = mock_job
    
    chunks = [
        DocumentChunkModel(
            id=uuid4(),
            job_id=mock_job.id,  # Use the mock job's ID
            chunk_index=i,
            content=f"Content {i}",
            content_hash=f"hash{i}",
        )
        for i in range(3)
    ]
    
    # Mock the upsert execution
    upsert_result_mock = MagicMock()
    upsert_result_mock.rowcount = 3
    
    # Configure execute to return different results based on call
    call_count = [0]
    def side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call is job verification
            return mock_result
        else:
            # Subsequent calls are upsert
            return upsert_result_mock
    
    mock_session.execute.side_effect = side_effect
    
    # Should succeed because job exists
    upserted, inserted, updated = await repo.upsert_chunks(chunks)
    assert len(upserted) == 3


@pytest.mark.asyncio
async def test_verify_job_exists_simple_returns_true_for_existing_job():
    """Test that verify_job_exists_simple returns True for existing job."""
    mock_session = AsyncMock(spec=AsyncSession)
    mock_result = MagicMock()
    
    mock_job = MagicMock(spec=JobModel)
    mock_job.id = uuid4()
    mock_result.scalar_one_or_none.return_value = mock_job
    mock_session.execute.return_value = mock_result
    
    exists = await verify_job_exists_simple(mock_session, mock_job.id)
    assert exists is True


@pytest.mark.asyncio
async def test_verify_job_exists_simple_returns_false_for_nonexistent_job():
    """Test that verify_job_exists_simple returns False for non-existent job."""
    mock_session = AsyncMock(spec=AsyncSession)
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result
    
    non_existent_job_id = uuid4()
    exists = await verify_job_exists_simple(mock_session, non_existent_job_id)
    assert exists is False


@pytest.mark.asyncio
async def test_verify_job_exists_simple_handles_string_uuid():
    """Test that verify_job_exists_simple handles string UUIDs."""
    mock_session = AsyncMock(spec=AsyncSession)
    mock_result = MagicMock()
    
    mock_job = MagicMock(spec=JobModel)
    mock_job.id = uuid4()
    mock_result.scalar_one_or_none.return_value = mock_job
    mock_session.execute.return_value = mock_result
    
    exists = await verify_job_exists_simple(mock_session, str(mock_job.id))
    assert exists is True


class TestTransactionIsolation:
    """Tests for transaction isolation and visibility."""
    
    @pytest.mark.asyncio
    async def test_job_verification_uses_fresh_query(self):
        """Test that job verification queries the database directly."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        job_id = uuid4()
        exists = await verify_job_exists_simple(mock_session, job_id)
        
        assert exists is False
        mock_session.execute.assert_called_once()
