from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import ChunkEmbeddingModel


class ChunkEmbeddingRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_embedding(self, chunk_id: UUID, model_name: str) -> list[float] | None:
        result = await self.session.execute(
            select(ChunkEmbeddingModel).where(
                ChunkEmbeddingModel.chunk_id == chunk_id,
                ChunkEmbeddingModel.model_name == model_name,
            )
        )
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return self._parse_embedding(row.embedding)

    async def set_embedding(
        self,
        chunk_id: UUID,
        model_name: str,
        embedding: list[float],
        dimensions: int,
    ) -> ChunkEmbeddingModel:
        embedding_value = self._to_vector_string(embedding)

        stmt = insert(ChunkEmbeddingModel).values(
            chunk_id=chunk_id,
            model_name=model_name,
            dimensions=dimensions,
            embedding=embedding_value,
        )
        upsert_stmt = stmt.on_conflict_do_update(
            index_elements=["chunk_id", "model_name"],
            set_={
                "dimensions": dimensions,
                "embedding": embedding_value,
            },
        )

        await self.session.execute(upsert_stmt)
        await self.session.commit()

        result = await self.session.execute(
            select(ChunkEmbeddingModel).where(
                ChunkEmbeddingModel.chunk_id == chunk_id,
                ChunkEmbeddingModel.model_name == model_name,
            )
        )
        return result.scalar_one()

    async def get_embeddings_for_search(
        self,
        model_name: str,
        query_embedding: list[float],
        top_k: int,
        min_similarity: float,
    ) -> list[tuple[UUID, float]]:
        vector_str = self._to_vector_string(query_embedding)

        stmt = text(
            """
            SELECT
                ce.chunk_id,
                1 - (ce.embedding::vector <=> CAST(:query_embedding AS vector)) AS similarity
            FROM chunk_embeddings ce
            WHERE ce.model_name = :model_name
              AND (1 - (ce.embedding::vector <=> CAST(:query_embedding AS vector))) >= :min_similarity
            ORDER BY ce.embedding::vector <=> CAST(:query_embedding AS vector)
            LIMIT :top_k
            """
        )

        result = await self.session.execute(
            stmt,
            {
                "model_name": model_name,
                "query_embedding": vector_str,
                "min_similarity": min_similarity,
                "top_k": top_k,
            },
        )

        return [(row[0], float(row[1])) for row in result.fetchall()]

    def _to_vector_string(self, embedding: list[float]) -> str:
        return f"[{','.join(str(x) for x in embedding)}]"

    def _parse_embedding(self, embedding: str) -> list[float]:
        values = embedding.strip("[]")
        if not values:
            return []
        return [float(v) for v in values.split(",")]
