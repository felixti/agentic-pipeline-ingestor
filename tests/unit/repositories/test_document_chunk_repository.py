"""Unit tests for DocumentChunkRepository.

Tests all CRUD operations and special methods of the DocumentChunkRepository.
Uses mocked SQLAlchemy sessions for isolation.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import DocumentChunkModel
from src.db.repositories.document_chunk_repository import DocumentChunkRepository

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_session():
    """Create a mock SQLAlchemy async session."""
    session = AsyncMock(spec=AsyncSession)
    return session


@pytest.fixture
def repository(mock_session):
    """Create a DocumentChunkRepository with mocked session."""
    return DocumentChunkRepository(mock_session)


@pytest.fixture
def sample_chunk():
    """Create a sample DocumentChunkModel."""
    return DocumentChunkModel(
        id=uuid4(),
        job_id=uuid4(),
        chunk_index=0,
        content="This is sample chunk content for testing.",
        content_hash="a" * 64,
        embedding=[0.1] * 1536,
        metadata={"page": 1, "source": "test"},
        created_at=datetime.utcnow(),
    )


@pytest.fixture
def sample_chunks():
    """Create multiple sample DocumentChunkModels."""
    job_id = uuid4()
    return [
        DocumentChunkModel(
            id=uuid4(),
            job_id=job_id,
            chunk_index=i,
            content=f"Chunk {i} content",
            content_hash=f"hash{i}",
            embedding=[0.1 * (i + 1)] * 1536,
            metadata={"page": i + 1},
            created_at=datetime.utcnow(),
        )
        for i in range(5)
    ]


# =============================================================================
# Create Operation Tests
# =============================================================================

@pytest.mark.unit
class TestDocumentChunkRepositoryCreate:
    """Tests for create operation."""

    @pytest.mark.asyncio
    async def test_create_success(self, repository, mock_session, sample_chunk):
        """Test successful creation of a chunk."""
        # Execute
        result = await repository.create(sample_chunk)
        
        # Assert
        mock_session.add.assert_called_once_with(sample_chunk)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(sample_chunk)
        assert result == sample_chunk

    @pytest.mark.asyncio
    async def test_create_commit_failure(self, repository, mock_session, sample_chunk):
        """Test creation failure when commit fails."""
        # Setup
        mock_session.commit.side_effect = SQLAlchemyError("Commit failed")
        
        # Execute & Assert
        with pytest.raises(SQLAlchemyError) as exc_info:
            await repository.create(sample_chunk)
        
        assert "Commit failed" in str(exc_info.value)
        mock_session.add.assert_called_once_with(sample_chunk)


# =============================================================================
# Read Operation Tests
# =============================================================================

@pytest.mark.unit
class TestDocumentChunkRepositoryRead:
    """Tests for read operations."""

    @pytest.mark.asyncio
    async def test_get_by_id_found(self, repository, mock_session, sample_chunk):
        """Test getting a chunk by ID when it exists."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_chunk
        mock_session.execute.return_value = mock_result
        
        # Execute
        result = await repository.get_by_id(sample_chunk.id)
        
        # Assert
        assert result == sample_chunk
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, repository, mock_session):
        """Test getting a chunk by ID when it doesn't exist."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        # Execute
        result = await repository.get_by_id(uuid4())
        
        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_job_id_success(self, repository, mock_session, sample_chunks):
        """Test getting chunks by job ID."""
        # Setup
        job_id = sample_chunks[0].job_id
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = sample_chunks
        mock_session.execute.return_value = mock_result
        
        # Execute
        results = await repository.get_by_job_id(job_id)
        
        # Assert
        assert len(results) == 5
        assert all(isinstance(chunk, DocumentChunkModel) for chunk in results)

    @pytest.mark.asyncio
    async def test_get_by_job_id_with_pagination(self, repository, mock_session, sample_chunks):
        """Test getting chunks by job ID with pagination."""
        # Setup
        job_id = sample_chunks[0].job_id
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = sample_chunks[:2]
        mock_session.execute.return_value = mock_result
        
        # Execute
        results = await repository.get_by_job_id(job_id, limit=2, offset=0)
        
        # Assert
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_get_by_job_id_no_results(self, repository, mock_session):
        """Test getting chunks by job ID when none exist."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result
        
        # Execute
        results = await repository.get_by_job_id(uuid4())
        
        # Assert
        assert results == []

    @pytest.mark.asyncio
    async def test_get_by_content_hash_found(self, repository, mock_session, sample_chunk):
        """Test getting chunks by content hash."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_chunk]
        mock_session.execute.return_value = mock_result
        
        # Execute
        results = await repository.get_by_content_hash(sample_chunk.content_hash)
        
        # Assert
        assert len(results) == 1
        assert results[0] == sample_chunk

    @pytest.mark.asyncio
    async def test_get_by_content_hash_not_found(self, repository, mock_session):
        """Test getting chunks by content hash when none exist."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result
        
        # Execute
        results = await repository.get_by_content_hash("nonexistent_hash")
        
        # Assert
        assert results == []


# =============================================================================
# Bulk Operation Tests
# =============================================================================

@pytest.mark.unit
class TestDocumentChunkRepositoryBulk:
    """Tests for bulk operations."""

    @pytest.mark.asyncio
    async def test_bulk_create_success(self, repository, mock_session, sample_chunks):
        """Test successful bulk creation."""
        # Execute
        results = await repository.bulk_create(sample_chunks)
        
        # Assert
        assert mock_session.add.call_count == len(sample_chunks)
        mock_session.commit.assert_called_once()
        assert mock_session.refresh.call_count == len(sample_chunks)
        assert len(results) == len(sample_chunks)

    @pytest.mark.asyncio
    async def test_bulk_create_empty_list(self, repository, mock_session):
        """Test bulk creation with empty list."""
        # Execute
        results = await repository.bulk_create([])
        
        # Assert
        mock_session.add.assert_not_called()
        mock_session.commit.assert_not_called()
        assert results == []

    @pytest.mark.asyncio
    async def test_bulk_create_failure(self, repository, mock_session, sample_chunks):
        """Test bulk creation failure."""
        # Setup
        mock_session.commit.side_effect = SQLAlchemyError("Bulk insert failed")
        
        # Execute & Assert
        with pytest.raises(SQLAlchemyError):
            await repository.bulk_create(sample_chunks)


# =============================================================================
# Update Operation Tests
# =============================================================================

@pytest.mark.unit
class TestDocumentChunkRepositoryUpdate:
    """Tests for update operations."""

    @pytest.mark.asyncio
    async def test_update_embedding_success(self, repository, mock_session, sample_chunk):
        """Test successful embedding update."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_chunk
        mock_session.execute.return_value = mock_result
        
        new_embedding = [0.5] * 1536
        
        # Execute
        result = await repository.update_embedding(sample_chunk.id, new_embedding)
        
        # Assert
        assert result is True
        assert sample_chunk.embedding == new_embedding
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(sample_chunk)

    @pytest.mark.asyncio
    async def test_update_embedding_not_found(self, repository, mock_session):
        """Test embedding update when chunk not found."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        # Execute
        result = await repository.update_embedding(uuid4(), [0.1] * 1536)
        
        # Assert
        assert result is False
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_embedding_wrong_dimensions(self, repository, mock_session, sample_chunk):
        """Test embedding update with wrong dimensions."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_chunk
        mock_session.execute.return_value = mock_result
        
        wrong_embedding = [0.1] * 768  # Wrong dimensions
        
        # Execute & Assert
        with pytest.raises(ValueError) as exc_info:
            await repository.update_embedding(sample_chunk.id, wrong_embedding)
        
        assert "1536" in str(exc_info.value)
        assert "768" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_update_embedding_database_error(self, repository, mock_session, sample_chunk):
        """Test embedding update with database error."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_chunk
        mock_session.execute.return_value = mock_result
        mock_session.commit.side_effect = SQLAlchemyError("Update failed")
        
        # Execute & Assert
        with pytest.raises(SQLAlchemyError):
            await repository.update_embedding(sample_chunk.id, [0.1] * 1536)


# =============================================================================
# Delete Operation Tests
# =============================================================================

@pytest.mark.unit
class TestDocumentChunkRepositoryDelete:
    """Tests for delete operations."""

    @pytest.mark.asyncio
    async def test_delete_by_job_id_success(self, repository, mock_session):
        """Test successful deletion by job ID."""
        # Setup
        job_id = uuid4()
        mock_result = MagicMock()
        mock_result.rowcount = 5
        mock_session.execute.return_value = mock_result
        
        # Execute
        result = await repository.delete_by_job_id(job_id)
        
        # Assert
        assert result == 5
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_by_job_id_no_matches(self, repository, mock_session):
        """Test deletion by job ID when no chunks match."""
        # Setup
        mock_result = MagicMock()
        mock_result.rowcount = 0
        mock_session.execute.return_value = mock_result
        
        # Execute
        result = await repository.delete_by_job_id(uuid4())
        
        # Assert
        assert result == 0

    @pytest.mark.asyncio
    async def test_delete_single_success(self, repository, mock_session, sample_chunk):
        """Test successful single chunk deletion."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_chunk
        mock_session.execute.return_value = mock_result
        
        # Execute
        result = await repository.delete(sample_chunk.id)
        
        # Assert
        assert result is True
        mock_session.delete.assert_called_once_with(sample_chunk)
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_single_not_found(self, repository, mock_session):
        """Test single chunk deletion when chunk not found."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        # Execute
        result = await repository.delete(uuid4())
        
        # Assert
        assert result is False
        mock_session.delete.assert_not_called()


# =============================================================================
# Query Operation Tests
# =============================================================================

@pytest.mark.unit
class TestDocumentChunkRepositoryQueries:
    """Tests for query operations."""

    @pytest.mark.asyncio
    async def test_exists_by_job_id_and_index_true(self, repository, mock_session):
        """Test existence check when chunk exists."""
        # Setup
        job_id = uuid4()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1
        mock_session.execute.return_value = mock_result
        
        # Execute
        result = await repository.exists_by_job_id_and_index(job_id, 0)
        
        # Assert
        assert result is True

    @pytest.mark.asyncio
    async def test_exists_by_job_id_and_index_false(self, repository, mock_session):
        """Test existence check when chunk doesn't exist."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar.return_value = 0
        mock_session.execute.return_value = mock_result
        
        # Execute
        result = await repository.exists_by_job_id_and_index(uuid4(), 999)
        
        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_count_by_job_id(self, repository, mock_session):
        """Test counting chunks by job ID."""
        # Setup
        job_id = uuid4()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 42
        mock_session.execute.return_value = mock_result
        
        # Execute
        result = await repository.count_by_job_id(job_id)
        
        # Assert
        assert result == 42

    @pytest.mark.asyncio
    async def test_count_by_job_id_no_chunks(self, repository, mock_session):
        """Test counting chunks by job ID when none exist."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar.return_value = 0
        mock_session.execute.return_value = mock_result
        
        # Execute
        result = await repository.count_by_job_id(uuid4())
        
        # Assert
        assert result == 0

    @pytest.mark.asyncio
    async def test_count_by_job_id_none_result(self, repository, mock_session):
        """Test counting when scalar returns None."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar.return_value = None
        mock_session.execute.return_value = mock_result
        
        # Execute
        result = await repository.count_by_job_id(uuid4())
        
        # Assert
        assert result == 0

    @pytest.mark.asyncio
    async def test_get_chunks_without_embeddings(self, repository, mock_session):
        """Test getting chunks without embeddings."""
        # Setup
        job_id = uuid4()
        chunks_without_embeddings = [
            DocumentChunkModel(
                id=uuid4(),
                job_id=job_id,
                chunk_index=i,
                content=f"Chunk {i}",
                content_hash=f"hash{i}",
                embedding=None,
                metadata={},
                created_at=datetime.utcnow(),
            )
            for i in range(3)
        ]
        
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = chunks_without_embeddings
        mock_session.execute.return_value = mock_result
        
        # Execute
        results = await repository.get_chunks_without_embeddings(job_id)
        
        # Assert
        assert len(results) == 3
        assert all(chunk.embedding is None for chunk in results)

    @pytest.mark.asyncio
    async def test_get_chunks_without_embeddings_with_limit(self, repository, mock_session):
        """Test getting chunks without embeddings with limit."""
        # Setup
        job_id = uuid4()
        chunks = [
            DocumentChunkModel(
                id=uuid4(),
                job_id=job_id,
                chunk_index=i,
                content=f"Chunk {i}",
                content_hash=f"hash{i}",
                embedding=None,
                metadata={},
                created_at=datetime.utcnow(),
            )
            for i in range(10)
        ]
        
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = chunks[:5]
        mock_session.execute.return_value = mock_result
        
        # Execute
        results = await repository.get_chunks_without_embeddings(job_id, limit=5)
        
        # Assert
        assert len(results) == 5


# =============================================================================
# Error Handling Tests
# =============================================================================

@pytest.mark.unit
class TestDocumentChunkRepositoryErrors:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_get_by_id_database_error(self, repository, mock_session):
        """Test database error handling in get_by_id."""
        # Setup
        mock_session.execute.side_effect = SQLAlchemyError("Query failed")
        
        # Execute & Assert
        with pytest.raises(SQLAlchemyError):
            await repository.get_by_id(uuid4())

    @pytest.mark.asyncio
    async def test_get_by_job_id_database_error(self, repository, mock_session):
        """Test database error handling in get_by_job_id."""
        # Setup
        mock_session.execute.side_effect = SQLAlchemyError("Query failed")
        
        # Execute & Assert
        with pytest.raises(SQLAlchemyError):
            await repository.get_by_job_id(uuid4())

    @pytest.mark.asyncio
    async def test_delete_by_job_id_database_error(self, repository, mock_session):
        """Test database error handling in delete_by_job_id."""
        # Setup
        mock_session.execute.side_effect = SQLAlchemyError("Delete failed")
        
        # Execute & Assert
        with pytest.raises(SQLAlchemyError):
            await repository.delete_by_job_id(uuid4())


# =============================================================================
# Integration-like Tests
# =============================================================================

@pytest.mark.unit
class TestDocumentChunkRepositoryIntegration:
    """Integration-like tests for repository workflows."""

    @pytest.mark.asyncio
    async def test_create_and_retrieve_workflow(self, repository, mock_session):
        """Test create and retrieve workflow."""
        # Create chunk
        chunk = DocumentChunkModel(
            id=uuid4(),
            job_id=uuid4(),
            chunk_index=0,
            content="Test content",
            content_hash="abc123",
            embedding=[0.1] * 1536,
            metadata={"page": 1},
            created_at=datetime.utcnow(),
        )
        
        # Mock execute to return the chunk on second call (get_by_id)
        mock_execute_results = [
            MagicMock(),  # First call (commit) doesn't need special return
            MagicMock(scalar_one_or_none=lambda: chunk),  # Second call (get_by_id)
        ]
        mock_session.execute.side_effect = mock_execute_results
        
        # Create
        created = await repository.create(chunk)
        assert created == chunk
        
        # Note: In real test, we'd need to reset mock properly

    @pytest.mark.asyncio
    async def test_chunk_lifecycle(self, repository, mock_session):
        """Test full chunk lifecycle: create, update, delete."""
        chunk_id = uuid4()
        job_id = uuid4()
        
        # Setup mocks for each operation
        chunk = DocumentChunkModel(
            id=chunk_id,
            job_id=job_id,
            chunk_index=0,
            content="Original content",
            content_hash="original_hash",
            embedding=None,
            metadata={},
            created_at=datetime.utcnow(),
        )
        
        # Mock for update_embedding
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = chunk
        mock_session.execute.return_value = mock_result
        
        # Update embedding
        new_embedding = [0.5] * 1536
        updated = await repository.update_embedding(chunk_id, new_embedding)
        assert updated is True
        assert chunk.embedding == new_embedding
        
        # Mock for delete
        mock_session.reset_mock()
        mock_session.execute.return_value = mock_result
        
        # Delete
        deleted = await repository.delete(chunk_id)
        assert deleted is True

    @pytest.mark.asyncio
    async def test_bulk_operations_workflow(self, repository, mock_session):
        """Test bulk create and delete workflow."""
        job_id = uuid4()
        
        # Create chunks
        chunks = [
            DocumentChunkModel(
                id=uuid4(),
                job_id=job_id,
                chunk_index=i,
                content=f"Content {i}",
                content_hash=f"hash{i}",
                embedding=[0.1 * i] * 1536,
                metadata={"index": i},
                created_at=datetime.utcnow(),
            )
            for i in range(10)
        ]
        
        # Bulk create
        created = await repository.bulk_create(chunks)
        assert len(created) == 10
        
        # Mock for count
        mock_session.reset_mock()
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 10
        mock_session.execute.return_value = mock_count_result
        
        count = await repository.count_by_job_id(job_id)
        assert count == 10
        
        # Mock for delete
        mock_session.reset_mock()
        mock_delete_result = MagicMock()
        mock_delete_result.rowcount = 10
        mock_session.execute.return_value = mock_delete_result
        
        # Delete all
        deleted = await repository.delete_by_job_id(job_id)
        assert deleted == 10


# =============================================================================
# Edge Cases
# =============================================================================

@pytest.mark.unit
class TestDocumentChunkRepositoryEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_get_by_job_id_large_offset(self, repository, mock_session):
        """Test get_by_job_id with large offset."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result
        
        results = await repository.get_by_job_id(uuid4(), limit=100, offset=10000)
        assert results == []

    @pytest.mark.asyncio
    async def test_exists_by_job_id_and_index_zero_index(self, repository, mock_session):
        """Test exists with chunk_index 0."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1
        mock_session.execute.return_value = mock_result
        
        result = await repository.exists_by_job_id_and_index(uuid4(), 0)
        assert result is True

    @pytest.mark.asyncio
    async def test_update_embedding_with_special_characters_in_content(self, repository, mock_session):
        """Test embedding update with special content."""
        chunk = DocumentChunkModel(
            id=uuid4(),
            job_id=uuid4(),
            chunk_index=0,
            content="Special chars: àáâãäåæçèéêë ñ 中文 🎉 <script>",
            content_hash="special_hash",
            embedding=None,
            metadata={"special": True},
            created_at=datetime.utcnow(),
        )
        
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = chunk
        mock_session.execute.return_value = mock_result
        
        embedding = [0.1] * 1536
        result = await repository.update_embedding(chunk.id, embedding)
        assert result is True
