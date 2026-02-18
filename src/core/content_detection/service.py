"""Content detection service with caching integration."""

import hashlib
from pathlib import Path
from typing import Optional, Union

import structlog

from src.core.content_detection.analyzer import PDFContentAnalyzer
from src.core.content_detection.cache import DetectionCache, NullDetectionCache
from src.core.content_detection.models import ContentAnalysisResult, ContentDetectionRecord

logger = structlog.get_logger(__name__)


class ContentDetectionService:
    """Service for content detection with caching support."""
    
    def __init__(
        self,
        analyzer: Optional[PDFContentAnalyzer] = None,
        cache: Optional[DetectionCache] = None,
    ):
        """Initialize service.
        
        Args:
            analyzer: PDF analyzer instance
            cache: Redis cache instance (optional)
        """
        self.analyzer = analyzer or PDFContentAnalyzer()
        self.cache = cache or NullDetectionCache()
    
    async def detect(
        self,
        file_path: Union[str, Path],
        skip_cache: bool = False
    ) -> ContentAnalysisResult:
        """Detect content type of a PDF file with caching.
        
        Flow:
        1. Calculate file hash
        2. Check cache (if not skipped)
        3. If cache miss, analyze file
        4. Store result in cache
        
        Args:
            file_path: Path to PDF file
            skip_cache: Skip cache lookup and force re-analysis
            
        Returns:
            Content analysis result
        """
        from src.observability.metrics import DetectionMetrics
        
        path = Path(file_path)
        file_hash = self._calculate_file_hash(path)
        file_size = path.stat().st_size
        
        log = logger.bind(
            file_path=str(path),
            file_hash=file_hash,
            file_size=file_size,
            skip_cache=skip_cache
        )
        
        # Step 1: Check cache
        if not skip_cache:
            cached_result = await self._get_from_cache(file_hash)
            if cached_result:
                DetectionMetrics.record_cache_hit()
                log.info(
                    "content_detection.completed",
                    event_type="cache_hit",
                    content_type=cached_result.content_type,
                    confidence=cached_result.confidence,
                    file_hash=file_hash,
                    duration_ms=0,
                )
                return cached_result
            
            DetectionMetrics.record_cache_miss()
            log.info("content_detection.cache_miss", file_hash=file_hash)
        
        # Step 2: Analyze file
        log.info(
            "content_detection.analysis_started",
            event_type="analysis_started",
            file_hash=file_hash,
            file_size=file_size,
        )
        
        import time
        start_time = time.time()
        
        try:
            result = self.analyzer.analyze(path)
            duration_ms = result.processing_time_ms
            
            # Record metrics
            DetectionMetrics.record_detection(
                content_type=result.content_type.value,
                duration_seconds=duration_ms / 1000.0,
                file_size_bytes=file_size
            )
            
            log.info(
                "content_detection.completed",
                event_type="analysis_completed",
                file_hash=result.file_hash,
                content_type=result.content_type.value,
                confidence=result.confidence,
                recommended_parser=result.recommended_parser,
                duration_ms=duration_ms,
                total_pages=result.text_statistics.total_pages,
            )
            
            # Step 3: Store in cache
            cache_success = await self._store_in_cache(result)
            if cache_success:
                log.debug(
                    "content_detection.cache_stored",
                    event_type="cache_stored",
                    file_hash=result.file_hash,
                )
            
            return result
            
        except Exception as e:
            DetectionMetrics.record_error(error_type=type(e).__name__)
            log.error(
                "content_detection.failed",
                event_type="analysis_failed",
                file_hash=file_hash,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise
    
    async def detect_from_bytes(
        self,
        content: bytes,
        skip_cache: bool = False
    ) -> ContentAnalysisResult:
        """Detect content type from file bytes with caching.
        
        Args:
            content: PDF file content as bytes
            skip_cache: Skip cache lookup and force re-analysis
            
        Returns:
            Content analysis result
        """
        import tempfile
        
        # Calculate hash from content
        file_hash = hashlib.sha256(content).hexdigest()
        
        log = logger.bind(
            file_size=len(content),
            file_hash=file_hash,
            skip_cache=skip_cache
        )
        
        # Step 1: Check cache
        if not skip_cache:
            cached_result = await self._get_from_cache(file_hash)
            if cached_result:
                log.info(
                    "content_detection.cache_hit",
                    content_type=cached_result.content_type,
                    confidence=cached_result.confidence,
                )
                return cached_result
            
            log.info("content_detection.cache_miss")
        
        # Step 2: Save to temp file and analyze
        log.info("content_detection.analysis_started")
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        
        try:
            result = self.analyzer.analyze(tmp_path)
            
            log.info(
                "content_detection.analysis_completed",
                content_type=result.content_type,
                confidence=result.confidence,
                processing_time_ms=result.processing_time_ms,
            )
            
            # Step 3: Store in cache
            await self._store_in_cache(result)
            
            return result
            
        except Exception as e:
            log.error(
                "content_detection.analysis_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise
        finally:
            # Cleanup temp file
            tmp_path.unlink(missing_ok=True)
    
    async def _get_from_cache(
        self,
        file_hash: str
    ) -> Optional[ContentAnalysisResult]:
        """Get result from cache.
        
        Args:
            file_hash: File hash
            
        Returns:
            Cached result if found, None otherwise
        """
        try:
            record = await self.cache.get(file_hash)
            if record is None:
                return None
            
            # Convert ContentDetectionRecord to ContentAnalysisResult
            return ContentAnalysisResult(
                id=record.id,
                file_hash=record.file_hash,
                file_size=record.file_size,
                content_type=record.content_type,
                confidence=record.confidence,
                recommended_parser=record.recommended_parser,
                alternative_parsers=record.alternative_parsers,
                text_statistics=record.text_statistics,
                image_statistics=record.image_statistics,
                page_results=record.page_results,
                processing_time_ms=record.processing_time_ms,
                created_at=record.created_at,
            )
        except Exception as e:
            logger.warning(
                "content_detection.cache_read_error",
                file_hash=file_hash,
                error=str(e)
            )
            return None
    
    async def _store_in_cache(self, result: ContentAnalysisResult) -> bool:
        """Store result in cache.
        
        Args:
            result: Analysis result
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            # Convert to record format for caching
            record = ContentDetectionRecord(
                id=result.id,
                file_hash=result.file_hash,
                file_size=result.file_size,
                content_type=result.content_type,
                confidence=result.confidence,
                recommended_parser=result.recommended_parser,
                alternative_parsers=result.alternative_parsers,
                text_statistics=result.text_statistics,
                image_statistics=result.image_statistics,
                page_results=result.page_results,
                processing_time_ms=result.processing_time_ms,
                created_at=result.created_at,
            )
            
            success = await self.cache.set(result.file_hash, record)
            
            if success:
                logger.debug(
                    "content_detection.cache_stored",
                    file_hash=result.file_hash,
                    content_type=result.content_type
                )
            
            return success
            
        except Exception as e:
            logger.warning(
                "content_detection.cache_write_error",
                file_hash=result.file_hash,
                error=str(e)
            )
            return False
    
    @staticmethod
    def _calculate_file_hash(file_path: Path) -> str:
        """Calculate SHA-256 hash of file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hex digest of SHA-256 hash
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    async def invalidate_cache(self, file_hash: str) -> bool:
        """Invalidate cached result for a file.
        
        Args:
            file_hash: File hash
            
        Returns:
            True if invalidated, False otherwise
        """
        return await self.cache.delete(file_hash)
    
    async def get_cache_stats(self) -> dict:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        # This is a placeholder - in production, you'd track hit/miss rates
        return {
            "cache_type": type(self.cache).__name__,
            "is_redis": isinstance(self.cache, DetectionCache)
        }


# Singleton instance for reuse
_detection_service: Optional[ContentDetectionService] = None


def get_detection_service(
    cache: Optional[DetectionCache] = None
) -> ContentDetectionService:
    """Get or create detection service singleton.
    
    Args:
        cache: Optional cache instance
        
    Returns:
        Content detection service
    """
    global _detection_service
    
    if _detection_service is None:
        _detection_service = ContentDetectionService(cache=cache)
    
    return _detection_service
