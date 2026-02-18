"""Integration tests for content detection storage and caching."""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from src.core.content_detection.cache import DetectionCache, NullDetectionCache
from src.core.content_detection.models import (
    ContentAnalysisResult,
    ContentDetectionRecord,
    ContentType,
    ImageStatistics,
    PageAnalysis,
    TextStatistics,
)


class TestNullDetectionCache:
    """Test NullDetectionCache (fallback when Redis unavailable)."""
    
    @pytest.fixture
    def null_cache(self):
        """Create null cache instance."""
        return NullDetectionCache()
    
    @pytest.fixture
    def sample_record(self):
        """Create sample detection record."""
        return ContentDetectionRecord(
            id=uuid4(),
            file_hash="abc123" * 8,  # 64 chars
            file_size=1024,
            content_type=ContentType.TEXT_BASED,
            confidence=0.95,
            recommended_parser="docling",
            alternative_parsers=["azure_ocr"],
            text_statistics=TextStatistics(
                total_pages=1,
                total_characters=1000,
                has_text_layer=True,
                font_count=2,
                encoding="UTF-8",
                average_chars_per_page=1000,
            ),
            image_statistics=ImageStatistics(
                total_images=0,
                image_area_ratio=0.0,
                average_dpi=None,
                color_pages=0,
                total_page_area=100000,
                total_image_area=0,
            ),
            page_results=[
                PageAnalysis(
                    page_number=1,
                    content_type=ContentType.TEXT_BASED,
                    confidence=0.95,
                    text_ratio=1.0,
                    image_ratio=0.0,
                    character_count=1000,
                    image_count=0,
                )
            ],
            processing_time_ms=100,
        )
    
    @pytest.mark.asyncio
    async def test_null_cache_get_returns_none(self, null_cache):
        """Test that null cache always returns None on get."""
        result = await null_cache.get("any_hash")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_null_cache_set_returns_false(self, null_cache, sample_record):
        """Test that null cache always returns False on set."""
        result = await null_cache.set("hash", sample_record)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_null_cache_exists_returns_false(self, null_cache):
        """Test that null cache always returns False on exists."""
        result = await null_cache.exists("any_hash")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_null_cache_delete_returns_false(self, null_cache):
        """Test that null cache always returns False on delete."""
        result = await null_cache.delete("any_hash")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_null_cache_get_ttl_returns_not_found(self, null_cache):
        """Test that null cache returns -2 (not found) for TTL."""
        result = await null_cache.get_ttl("any_hash")
        assert result == -2
    
    @pytest.mark.asyncio
    async def test_null_cache_clear_all_returns_zero(self, null_cache):
        """Test that null cache returns 0 on clear_all."""
        result = await null_cache.clear_all()
        assert result == 0


class TestDetectionCacheSerialization:
    """Test serialization/deserialization of detection records."""
    
    @pytest.fixture
    def sample_record(self):
        """Create sample detection record with all fields."""
        return ContentDetectionRecord(
            id=uuid4(),
            file_hash="abc123" * 8,
            file_size=1024,
            content_type=ContentType.MIXED,
            confidence=0.75,
            recommended_parser="docling",
            alternative_parsers=["azure_ocr"],
            text_statistics=TextStatistics(
                total_pages=2,
                total_characters=5000,
                has_text_layer=True,
                font_count=3,
                encoding="UTF-8",
                average_chars_per_page=2500,
            ),
            image_statistics=ImageStatistics(
                total_images=2,
                image_area_ratio=0.3,
                average_dpi=150,
                color_pages=1,
                total_page_area=200000,
                total_image_area=60000,
            ),
            page_results=[
                PageAnalysis(
                    page_number=1,
                    content_type=ContentType.TEXT_BASED,
                    confidence=0.9,
                    text_ratio=0.8,
                    image_ratio=0.1,
                    character_count=3000,
                    image_count=1,
                ),
                PageAnalysis(
                    page_number=2,
                    content_type=ContentType.MIXED,
                    confidence=0.6,
                    text_ratio=0.4,
                    image_ratio=0.5,
                    character_count=2000,
                    image_count=1,
                ),
            ],
            processing_time_ms=250,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=30),
            access_count=5,
            last_accessed_at=datetime.utcnow(),
        )
    
    def test_model_to_dict_conversion(self, sample_record):
        """Test conversion of model to dictionary."""
        # Create a mock cache to use the conversion method
        cache = DetectionCache.__new__(DetectionCache)
        
        data = cache._model_to_dict(sample_record)
        
        assert data["id"] == str(sample_record.id)
        assert data["file_hash"] == sample_record.file_hash
        assert data["content_type"] == sample_record.content_type
        assert data["confidence"] == sample_record.confidence
        assert data["text_statistics"]["total_pages"] == 2
        assert len(data["page_results"]) == 2
        assert data["access_count"] == 5
    
    def test_dict_to_model_conversion(self, sample_record):
        """Test conversion of dictionary back to model."""
        cache = DetectionCache.__new__(DetectionCache)
        
        # Convert to dict and back
        data = cache._model_to_dict(sample_record)
        restored = cache._dict_to_model(data)
        
        assert restored.id == sample_record.id
        assert restored.file_hash == sample_record.file_hash
        assert restored.content_type == sample_record.content_type
        assert restored.confidence == sample_record.confidence
        assert restored.text_statistics.total_pages == 2
        assert len(restored.page_results) == 2
        assert restored.access_count == 5


class TestContentDetectionServiceWithCache:
    """Test ContentDetectionService with caching."""
    
    @pytest.fixture
    def null_service(self):
        """Create service with null cache."""
        from src.core.content_detection.service import ContentDetectionService
        return ContentDetectionService()
    
    @pytest.mark.asyncio
    async def test_service_detect_with_null_cache(self, null_service, tmp_path):
        """Test that service works with null cache."""
        import fitz
        
        # Create a simple PDF
        pdf_path = tmp_path / "test.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Test content " * 100, fontsize=12)
        doc.save(str(pdf_path))
        doc.close()
        
        # Detect
        result = await null_service.detect(pdf_path)
        
        assert result is not None
        assert result.content_type == ContentType.TEXT_BASED
        assert result.confidence > 0.9
    
    @pytest.mark.asyncio
    async def test_service_detect_from_bytes(self, null_service, tmp_path):
        """Test detection from bytes."""
        import fitz
        
        # Create a PDF
        pdf_path = tmp_path / "test.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Test content " * 100, fontsize=12)
        doc.save(str(pdf_path))
        doc.close()
        
        # Read bytes
        content = pdf_path.read_bytes()
        
        # Detect from bytes
        result = await null_service.detect_from_bytes(content)
        
        assert result is not None
        assert result.content_type == ContentType.TEXT_BASED
        assert result.file_hash is not None
    
    @pytest.mark.asyncio
    async def test_service_skip_cache(self, null_service, tmp_path):
        """Test skip_cache flag forces re-analysis."""
        import fitz
        
        # Create a PDF
        pdf_path = tmp_path / "test.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Test content " * 100, fontsize=12)
        doc.save(str(pdf_path))
        doc.close()
        
        # First detection
        result1 = await null_service.detect(pdf_path)
        
        # Second detection with skip_cache
        result2 = await null_service.detect(pdf_path, skip_cache=True)
        
        # Results should be the same (same file)
        assert result1.content_type == result2.content_type
        assert result1.file_hash == result2.file_hash


class TestRepositoryIntegration:
    """Test DetectionResultRepository integration."""
    
    @pytest.fixture
    def sample_analysis_result(self):
        """Create sample analysis result."""
        return ContentAnalysisResult(
            id=uuid4(),
            file_hash="def456" * 8,
            file_size=2048,
            content_type=ContentType.SCANNED,
            confidence=0.92,
            recommended_parser="azure_ocr",
            alternative_parsers=["docling"],
            text_statistics=TextStatistics(
                total_pages=1,
                total_characters=100,
                has_text_layer=False,
                font_count=0,
                encoding=None,
                average_chars_per_page=100,
            ),
            image_statistics=ImageStatistics(
                total_images=1,
                image_area_ratio=0.95,
                average_dpi=300,
                color_pages=0,
                total_page_area=100000,
                total_image_area=95000,
            ),
            page_results=[
                PageAnalysis(
                    page_number=1,
                    content_type=ContentType.SCANNED,
                    confidence=0.92,
                    text_ratio=0.02,
                    image_ratio=0.95,
                    character_count=100,
                    image_count=1,
                )
            ],
            processing_time_ms=500,
        )
    
    @pytest.mark.asyncio
    async def test_repository_save_and_get(self, sample_analysis_result):
        """Test saving and retrieving detection result."""
        # This test would require a real database connection
        # For now, we just verify the repository structure
        from src.db.repositories.detection_result import DetectionResultRepository
        
        # Repository requires an async session - in real tests,
        # you'd use a test database or mock the session
        pass
    
    def test_record_model_creation(self, sample_analysis_result):
        """Test creation of ContentDetectionRecord from analysis result."""
        from datetime import datetime
        
        record = ContentDetectionRecord(
            id=sample_analysis_result.id,
            file_hash=sample_analysis_result.file_hash,
            file_size=sample_analysis_result.file_size,
            content_type=sample_analysis_result.content_type,
            confidence=sample_analysis_result.confidence,
            recommended_parser=sample_analysis_result.recommended_parser,
            alternative_parsers=sample_analysis_result.alternative_parsers,
            text_statistics=sample_analysis_result.text_statistics,
            image_statistics=sample_analysis_result.image_statistics,
            page_results=sample_analysis_result.page_results,
            processing_time_ms=sample_analysis_result.processing_time_ms,
            created_at=datetime.utcnow(),
        )
        
        assert record.file_hash == sample_analysis_result.file_hash
        assert record.content_type == ContentType.SCANNED
        assert record.confidence == 0.92
