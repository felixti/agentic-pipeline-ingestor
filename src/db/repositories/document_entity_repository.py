from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import DocumentEntityModel


class DocumentEntityRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, entity: DocumentEntityModel) -> DocumentEntityModel:
        self.session.add(entity)
        await self.session.commit()
        await self.session.refresh(entity)
        return entity

    async def bulk_create(self, entities: list[DocumentEntityModel]) -> None:
        if not entities:
            return
        self.session.add_all(entities)
        await self.session.commit()

    async def get_by_chunk(self, chunk_id: UUID) -> list[DocumentEntityModel]:
        result = await self.session.execute(
            select(DocumentEntityModel).where(DocumentEntityModel.chunk_id == chunk_id)
        )
        return list(result.scalars().all())

    async def search_by_entity(
        self,
        entity_text: str,
        entity_type: str | None = None,
    ) -> list[UUID]:
        query = select(DocumentEntityModel.chunk_id).where(
            DocumentEntityModel.entity_text.ilike(f"%{entity_text}%")
        )
        if entity_type:
            query = query.where(DocumentEntityModel.entity_type == entity_type)
        result = await self.session.execute(query)
        return [row[0] for row in result.all() if row[0] is not None]
