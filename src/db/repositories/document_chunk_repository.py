"""Repository for document chunk data access."""

from typing import Optional
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import DocumentChunkModel, JobModel
from src.observability.logging import get_logger

logger = get_logger(__name__)


class DocumentChunkRepository:
    """Repository for document chunk CRUD operations.

    Provides methods for creating, reading, updating, and deleting document chunks
    with their vector embeddings. Supports bulk operations and efficient querying
    by job ID, content hash, and chunk index.
    """

    def __init__(self, session: AsyncSession):
        """Initialize repository.

        Args:
            session: SQLAlchemy async session
        """
        self.session = session

    async def _verify_job_exists(self, job_id: UUID) -> bool:
        """Verify that the referenced job exists.
        
        Args:
            job_id: Job ID to verify
            
        Returns:
            True if job exists, False otherwise
        """
        result = await self.session.execute(
            select(JobModel).where(JobModel.id == job_id)
        )
        job = result.scalar_one_or_none()
        if job is None:
            logger.error("job_not_found_for_chunks", job_id=str(job_id))
            return False
        return True

    async def create(self, chunk: DocumentChunkModel) -> DocumentChunkModel:
        """Create a new document chunk.

        Args:
            chunk: DocumentChunkModel instance to create

        Returns:
            Created DocumentChunkModel instance with populated ID

        Raises:
            SQLAlchemyError: If database operation fails
            ValueError: If referenced job does not exist
        """
        # Verify job exists before creating chunk
        if not await self._verify_job_exists(chunk.job_id):
            raise ValueError(
                f"Cannot create chunk: Job {chunk.job_id} does not exist"
            )
        
        self.session.add(chunk)
        await self.session.flush()  # Flush to get ID without committing
        await self.session.refresh(chunk)
        return chunk

    async def get_by_id(self, chunk_id: UUID) -> DocumentChunkModel | None:
        """Get chunk by ID.

        Args:
            chunk_id: Chunk UUID

        Returns:
            DocumentChunkModel if found, None otherwise
        """
        result = await self.session.execute(
            select(DocumentChunkModel).where(DocumentChunkModel.id == chunk_id)
        )
        return result.scalar_one_or_none()

    async def get_by_job_id(
        self,
        job_id: UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> list[DocumentChunkModel]:
        """Get chunks by job ID with pagination.

        Chunks are ordered by chunk_index for consistent retrieval.

        Args:
            job_id: Job UUID
            limit: Maximum number of chunks to return (default: 100)
            offset: Number of chunks to skip (default: 0)

        Returns:
            List of DocumentChunkModel instances
        """
        result = await self.session.execute(
            select(DocumentChunkModel)
            .where(DocumentChunkModel.job_id == job_id)
            .order_by(DocumentChunkModel.chunk_index)
            .offset(offset)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_by_content_hash(self, content_hash: str) -> list[DocumentChunkModel]:
        """Get chunks by content hash.

        Used for deduplication to find chunks with identical content.

        Args:
            content_hash: SHA-256 hash of chunk content

        Returns:
            List of DocumentChunkModel instances with matching hash
        """
        result = await self.session.execute(
            select(DocumentChunkModel)
            .where(DocumentChunkModel.content_hash == content_hash)
            .order_by(DocumentChunkModel.created_at)
        )
        return list(result.scalars().all())

    async def bulk_create(
        self,
        chunks: list[DocumentChunkModel],
    ) -> list[DocumentChunkModel]:
        """Create multiple chunks in a single batch operation.

        More efficient than creating chunks individually when processing
        large documents with many chunks.

        Args:
            chunks: List of DocumentChunkModel instances to create

        Returns:
            List of created DocumentChunkModel instances with populated IDs

        Raises:
            SQLAlchemyError: If database operation fails
            ValueError: If referenced job does not exist
        """
        if not chunks:
            return []

        # Verify job exists (all chunks should belong to same job)
        job_id = chunks[0].job_id
        if not await self._verify_job_exists(job_id):
            raise ValueError(
                f"Cannot create chunks: Job {job_id} does not exist"
            )

        # Add all chunks to session
        for chunk in chunks:
            self.session.add(chunk)

        await self.session.flush()  # Flush to get IDs without committing

        # Refresh all chunks to get IDs and defaults
        for chunk in chunks:
            await self.session.refresh(chunk)

        return chunks

    async def update_embedding(
        self,
        chunk_id: UUID,
        embedding: list[float],
    ) -> bool:
        """Update the embedding vector for a chunk.

        Args:
            chunk_id: Chunk UUID
            embedding: List of float values representing the embedding vector

        Returns:
            True if update was successful, False if chunk not found

        Raises:
            ValueError: If embedding dimensions don't match expected size (1536)
            SQLAlchemyError: If database operation fails
        """
        chunk = await self.get_by_id(chunk_id)
        if not chunk:
            return False

        # Validate dimensions using the model's validation
        chunk.set_embedding(embedding)

        await self.session.flush()
        await self.session.refresh(chunk)

        return True

    async def delete_by_job_id(self, job_id: UUID) -> int:
        """Delete all chunks for a job.

        Args:
            job_id: Job UUID

        Returns:
            Number of chunks deleted
        """
        result = await self.session.execute(
            delete(DocumentChunkModel).where(DocumentChunkModel.job_id == job_id)
        )
        await self.session.flush()

        return result.rowcount or 0

    async def exists_by_job_id_and_index(
        self,
        job_id: UUID,
        chunk_index: int,
    ) -> bool:
        """Check if a chunk exists for a job at a specific index.

        Args:
            job_id: Job UUID
            chunk_index: Chunk index within the document

        Returns:
            True if chunk exists, False otherwise
        """
        result = await self.session.execute(
            select(func.count(DocumentChunkModel.id))
            .where(DocumentChunkModel.job_id == job_id)
            .where(DocumentChunkModel.chunk_index == chunk_index)
        )
        count = result.scalar()
        return count > 0 if count else False

    async def count_by_job_id(self, job_id: UUID) -> int:
        """Count chunks for a job.

        Args:
            job_id: Job UUID

        Returns:
            Number of chunks for the job
        """
        result = await self.session.execute(
            select(func.count(DocumentChunkModel.id)).where(DocumentChunkModel.job_id == job_id)
        )
        count = result.scalar()
        return count or 0

    async def get_chunks_without_embeddings(
        self,
        job_id: UUID,
        limit: int = 100,
    ) -> list[DocumentChunkModel]:
        """Get chunks that don't have embeddings yet.

        Useful for batch embedding generation pipelines.

        Args:
            job_id: Job UUID
            limit: Maximum number of chunks to return

        Returns:
            List of DocumentChunkModel instances without embeddings
        """
        result = await self.session.execute(
            select(DocumentChunkModel)
            .where(DocumentChunkModel.job_id == job_id)
            .where(DocumentChunkModel.embedding.is_(None))
            .order_by(DocumentChunkModel.chunk_index)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def delete(self, chunk_id: UUID) -> bool:
        """Delete a single chunk by ID.

        Args:
            chunk_id: Chunk UUID

        Returns:
            True if deleted, False if not found
        """
        chunk = await self.get_by_id(chunk_id)
        if not chunk:
            return False

        await self.session.delete(chunk)
        await self.session.flush()

        return True

    async def upsert_chunks(
        self,
        chunks: list[DocumentChunkModel],
    ) -> tuple[list[DocumentChunkModel], int, int]:
        """Upsert multiple chunks using ON CONFLICT DO UPDATE.

        Inserts chunks that don't exist and updates those that do, based on the
        unique constraint (job_id, chunk_index). This is idempotent and safe
        for retries - subsequent calls with the same chunks will update rather
        than fail with duplicate key errors.

        Args:
            chunks: List of DocumentChunkModel instances to upsert

        Returns:
            Tuple of (upserted chunks list, inserted count, updated count)
            Note: The returned chunks may not have IDs for updated records

        Raises:
            SQLAlchemyError: If database operation fails
            ValueError: If referenced job does not exist
        """
        from datetime import datetime
        from uuid import uuid4

        if not chunks:
            return [], 0, 0

        # Get job_id from first chunk
        job_id = chunks[0].job_id

        # Verify job exists before attempting upsert
        if not await self._verify_job_exists(job_id):
            raise ValueError(
                f"Cannot upsert chunks: Job {job_id} does not exist"
            )

        # Prepare data for bulk upsert
        # Convert embedding list to PostgreSQL vector format
        values = []
        for chunk in chunks:
            # Generate UUID if not present (bulk insert doesn't trigger defaults)
            if chunk.id is None:
                chunk.id = uuid4()

            # Set created_at if not present
            if chunk.created_at is None:
                chunk.created_at = datetime.utcnow()

            embedding_str = None
            if chunk.embedding is not None:
                # Format as PostgreSQL vector: [x,y,z]
                embedding_str = f"[{','.join(str(x) for x in chunk.embedding)}]"

            values.append(
                {
                    "id": chunk.id,
                    "job_id": chunk.job_id,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "content_hash": chunk.content_hash,
                    "embedding": embedding_str,
                    "chunk_metadata": chunk.chunk_metadata,
                    "quality_score": chunk.quality_score,
                    "created_at": chunk.created_at,
                }
            )

        # Build upsert statement
        # On conflict (job_id, chunk_index), update content, hash, embedding, and metadata
        stmt = insert(DocumentChunkModel).values(values)

        update_dict = {
            "content": stmt.excluded.content,
            "content_hash": stmt.excluded.content_hash,
            "embedding": stmt.excluded.embedding,
            "chunk_metadata": stmt.excluded.chunk_metadata,
            "quality_score": stmt.excluded.quality_score,
        }

        upsert_stmt = stmt.on_conflict_do_update(
            index_elements=["job_id", "chunk_index"],
            set_=update_dict,
        )

        try:
            result = await self.session.execute(upsert_stmt)
            await self.session.flush()

            # PostgreSQL doesn't give us separate insert/update counts easily
            # We return the chunks and estimate counts based on rowcount
            # rowcount reflects the total number of rows affected (inserted + updated)
            total_affected = result.rowcount or 0

            # Heuristic: if we have the same number of chunks as affected rows,
            # and we know some may have been updates, we estimate:
            # - Assume roughly half are updates in a retry scenario
            # - But we can't know for sure without an extra query
            estimated_inserted = total_affected
            estimated_updated = 0

            logger.info(
                "chunks_upserted",
                job_id=str(job_id),
                chunk_count=len(chunks),
                total_affected=total_affected,
            )

            return chunks, estimated_inserted, estimated_updated

        except SQLAlchemyError as e:
            logger.error(
                "upsert_chunks_failed",
                job_id=str(job_id),
                error=str(e),
                chunk_count=len(chunks),
            )
            raise
