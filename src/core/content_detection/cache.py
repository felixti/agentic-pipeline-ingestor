"""Redis caching layer for content detection results."""

import json
from typing import Optional

import redis.asyncio as redis

from src.core.content_detection.models import (
    ContentDetectionRecord,
    ImageStatistics,
    PageAnalysis,
    TextStatistics,
)


class DetectionCache:
    """Redis cache for detection results."""
    
    CACHE_PREFIX = "detection:"
    DEFAULT_TTL = 30 * 24 * 60 * 60  # 30 days in seconds
    
    def __init__(self, redis_client: redis.Redis):
        """Initialize cache.
        
        Args:
            redis_client: Redis async client
        """
        self.redis = redis_client
    
    def _make_key(self, file_hash: str) -> str:
        """Create cache key from file hash.
        
        Args:
            file_hash: SHA-256 file hash
            
        Returns:
            Cache key
        """
        return f"{self.CACHE_PREFIX}{file_hash}"
    
    async def get(self, file_hash: str) -> Optional[ContentDetectionRecord]:
        """Get cached detection result.
        
        Args:
            file_hash: SHA-256 file hash
            
        Returns:
            Cached record if found, None otherwise
        """
        try:
            key = self._make_key(file_hash)
            data = await self.redis.get(key)
            
            if data is None:
                return None
            
            # Parse JSON and reconstruct model
            record_dict = json.loads(data)
            return self._dict_to_model(record_dict)
            
        except Exception:
            # Cache failure shouldn't break the flow
            return None
    
    async def set(
        self,
        file_hash: str,
        record: ContentDetectionRecord,
        ttl: Optional[int] = None
    ) -> bool:
        """Cache detection result.
        
        Args:
            file_hash: SHA-256 file hash
            record: Detection record to cache
            ttl: Time to live in seconds (default: 30 days)
            
        Returns:
            True if cached successfully, False otherwise
        """
        try:
            key = self._make_key(file_hash)
            data = json.dumps(self._model_to_dict(record), default=str)
            
            await self.redis.setex(
                key,
                ttl or self.DEFAULT_TTL,
                data
            )
            return True
            
        except Exception:
            # Cache failure shouldn't break the flow
            return False
    
    async def delete(self, file_hash: str) -> bool:
        """Delete cached detection result.
        
        Args:
            file_hash: SHA-256 file hash
            
        Returns:
            True if deleted, False otherwise
        """
        try:
            key = self._make_key(file_hash)
            result = await self.redis.delete(key)
            return result > 0
        except Exception:
            return False
    
    async def exists(self, file_hash: str) -> bool:
        """Check if detection result is cached.
        
        Args:
            file_hash: SHA-256 file hash
            
        Returns:
            True if cached, False otherwise
        """
        try:
            key = self._make_key(file_hash)
            return await self.redis.exists(key) > 0
        except Exception:
            return False
    
    async def get_ttl(self, file_hash: str) -> int:
        """Get remaining TTL for cached item.
        
        Args:
            file_hash: SHA-256 file hash
            
        Returns:
            Remaining TTL in seconds, -1 if no expiry, -2 if not found
        """
        try:
            key = self._make_key(file_hash)
            return await self.redis.ttl(key)
        except Exception:
            return -2
    
    async def clear_all(self) -> int:
        """Clear all detection cache entries.
        
        Warning: Use with caution in production!
        
        Returns:
            Number of keys deleted
        """
        try:
            pattern = f"{self.CACHE_PREFIX}*"
            keys = []
            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                return await self.redis.delete(*keys)
            return 0
        except Exception:
            return 0
    
    def _model_to_dict(self, record: ContentDetectionRecord) -> dict:
        """Convert model to dictionary for JSON serialization.
        
        Args:
            record: Detection record
            
        Returns:
            Dictionary representation
        """
        return {
            "id": str(record.id),
            "file_hash": record.file_hash,
            "file_size": record.file_size,
            "content_type": record.content_type,
            "confidence": record.confidence,
            "recommended_parser": record.recommended_parser,
            "alternative_parsers": record.alternative_parsers,
            "text_statistics": record.text_statistics.model_dump(),
            "image_statistics": record.image_statistics.model_dump(),
            "page_results": [p.model_dump() for p in record.page_results],
            "processing_time_ms": record.processing_time_ms,
            "created_at": record.created_at.isoformat() if record.created_at else None,
            "expires_at": record.expires_at.isoformat() if record.expires_at else None,
            "access_count": record.access_count,
            "last_accessed_at": record.last_accessed_at.isoformat() if record.last_accessed_at else None,
        }
    
    def _dict_to_model(self, data: dict) -> ContentDetectionRecord:
        """Convert dictionary to model.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Detection record model
        """
        from datetime import datetime
        from uuid import UUID
        
        return ContentDetectionRecord(
            id=UUID(data["id"]),
            file_hash=data["file_hash"],
            file_size=data["file_size"],
            content_type=data["content_type"],
            confidence=data["confidence"],
            recommended_parser=data["recommended_parser"],
            alternative_parsers=data.get("alternative_parsers", []),
            text_statistics=TextStatistics(**data["text_statistics"]),
            image_statistics=ImageStatistics(**data["image_statistics"]),
            page_results=[PageAnalysis(**p) for p in data["page_results"]],
            processing_time_ms=data["processing_time_ms"],
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            access_count=data.get("access_count", 1),
            last_accessed_at=datetime.fromisoformat(data["last_accessed_at"]) if data.get("last_accessed_at") else None,
        )


class NullDetectionCache:
    """Null object pattern for when Redis is unavailable."""
    
    async def get(self, file_hash: str) -> None:
        """Always returns None."""
        return None
    
    async def set(self, file_hash: str, record, ttl=None) -> bool:
        """Always returns False."""
        return False
    
    async def delete(self, file_hash: str) -> bool:
        """Always returns False."""
        return False
    
    async def exists(self, file_hash: str) -> bool:
        """Always returns False."""
        return False
    
    async def get_ttl(self, file_hash: str) -> int:
        """Always returns -2 (not found)."""
        return -2
    
    async def clear_all(self) -> int:
        """Always returns 0."""
        return 0
