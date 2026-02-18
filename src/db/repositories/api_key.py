"""Repository for API key data access."""

import hashlib
import secrets
from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import asc, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import ApiKeyModel


class APIKeyRepository:
    """Repository for API key CRUD operations."""
    
    def __init__(self, session: AsyncSession):
        """Initialize repository.
        
        Args:
            session: SQLAlchemy async session
        """
        self.session = session
    
    @staticmethod
    def _hash_key(api_key: str) -> str:
        """Hash an API key for storage/comparison.
        
        Args:
            api_key: Raw API key
            
        Returns:
            Hashed key
        """
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    @staticmethod
    def generate_api_key(prefix: str = "sk") -> str:
        """Generate a new secure API key.
        
        Args:
            prefix: Key prefix
            
        Returns:
            Generated API key
        """
        random_part = secrets.token_urlsafe(32)
        random_part = random_part.replace("-", "").replace("_", "")[:32]
        return f"{prefix}_{random_part}"
    
    async def create(
        self,
        name: str,
        permissions: list[str] | None = None,
        created_by: str | None = None,
        expires_at: datetime | None = None,
    ) -> tuple[ApiKeyModel, str]:
        """Create a new API key.
        
        Args:
            name: Key name/description
            permissions: List of permissions
            created_by: User who created the key
            expires_at: Expiration timestamp
            
        Returns:
            Tuple of (ApiKeyModel, raw_api_key)
        """
        # Generate API key
        raw_key = self.generate_api_key()
        key_hash = self._hash_key(raw_key)
        
        api_key = ApiKeyModel(
            key_hash=key_hash,
            name=name,
            permissions=permissions or [],
            is_active=1,
            created_by=created_by,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
        )
        
        self.session.add(api_key)
        await self.session.commit()
        await self.session.refresh(api_key)
        
        return api_key, raw_key
    
    async def get_by_hash(
        self,
        key_hash: str,
        include_inactive: bool = False,
    ) -> Optional[ApiKeyModel]:
        """Get API key by hash.
        
        Args:
            key_hash: Hashed API key
            include_inactive: Whether to include inactive keys
            
        Returns:
            ApiKeyModel if found, None otherwise
        """
        query = select(ApiKeyModel).where(ApiKeyModel.key_hash == key_hash)
        
        if not include_inactive:
            query = query.where(ApiKeyModel.is_active == 1)
        
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def validate_key(
        self,
        api_key: str,
    ) -> Optional[ApiKeyModel]:
        """Validate an API key.
        
        Args:
            api_key: Raw API key
            
        Returns:
            ApiKeyModel if valid, None otherwise
        """
        key_hash = self._hash_key(api_key)
        key_model = await self.get_by_hash(key_hash)
        
        if key_model is None:
            return None
        
        # Check expiration
        if key_model.expires_at and datetime.utcnow() > key_model.expires_at:
            return None
        
        return key_model
    
    async def update_last_used(
        self,
        key_id: UUID,
    ) -> Optional[ApiKeyModel]:
        """Update last used timestamp.
        
        Args:
            key_id: API key ID
            
        Returns:
            Updated ApiKeyModel if found, None otherwise
        """
        result = await self.session.execute(
            select(ApiKeyModel).where(ApiKeyModel.id == key_id)
        )
        key = result.scalar_one_or_none()
        
        if key:
            key.last_used_at = datetime.utcnow()
            await self.session.commit()
            await self.session.refresh(key)
        
        return key
    
    async def list_keys(
        self,
        page: int = 1,
        limit: int = 20,
        include_inactive: bool = False,
    ) -> tuple[list[ApiKeyModel], int]:
        """List API keys.
        
        Args:
            page: Page number (1-indexed)
            limit: Items per page
            include_inactive: Whether to include inactive keys
            
        Returns:
            Tuple of (keys list, total count)
        """
        query = select(ApiKeyModel)
        count_query = select(func.count(ApiKeyModel.id))
        
        if not include_inactive:
            query = query.where(ApiKeyModel.is_active == 1)
            count_query = count_query.where(ApiKeyModel.is_active == 1)
        
        # Get total count
        count_result = await self.session.execute(count_query)
        total = count_result.scalar()
        
        # Apply sorting and pagination
        query = query.order_by(desc(ApiKeyModel.created_at))
        offset = (page - 1) * limit
        query = query.offset(offset).limit(limit)
        
        result = await self.session.execute(query)
        keys = result.scalars().all()
        
        return list(keys), total
    
    async def deactivate(
        self,
        key_id: str | UUID,
    ) -> bool:
        """Deactivate an API key.
        
        Args:
            key_id: API key ID
            
        Returns:
            True if deactivated, False if not found
        """
        if isinstance(key_id, str):
            try:
                key_id = UUID(key_id)
            except ValueError:
                return False
        
        result = await self.session.execute(
            select(ApiKeyModel).where(ApiKeyModel.id == key_id)
        )
        key = result.scalar_one_or_none()
        
        if not key:
            return False
        
        key.is_active = 0
        await self.session.commit()
        
        return True
    
    async def delete(
        self,
        key_id: str | UUID,
    ) -> bool:
        """Delete an API key.
        
        Args:
            key_id: API key ID
            
        Returns:
            True if deleted, False if not found
        """
        if isinstance(key_id, str):
            try:
                key_id = UUID(key_id)
            except ValueError:
                return False
        
        result = await self.session.execute(
            select(ApiKeyModel).where(ApiKeyModel.id == key_id)
        )
        key = result.scalar_one_or_none()
        
        if not key:
            return False
        
        await self.session.delete(key)
        await self.session.commit()
        
        return True
