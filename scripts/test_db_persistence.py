#!/usr/bin/env python3
"""Test database persistence directly using repositories."""

import asyncio
import os
import sys
from uuid import uuid4

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from src.core.content_detection.models import (
    ContentAnalysisResult,
    ImageStatistics,
    PageAnalysis,
    TextStatistics,
)
from src.db.repositories.detection_result import DetectionResultRepository


async def test_detection_result_repository():
    """Test that detection results can be saved to database."""
    print("Testing DetectionResultRepository.save()...")
    
    # Create engine
    engine = create_async_engine(
        "postgresql+asyncpg://postgres:postgres@localhost:5432/pipeline",
        echo=False
    )
    
    # Create session
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        # Create repository
        repo = DetectionResultRepository(session)
        
        # Create a sample detection result
        result = ContentAnalysisResult(
            file_hash="abc123" + str(uuid4())[:8],  # Unique hash
            file_size=1024,
            content_type="text_based",
            confidence=0.95,
            recommended_parser="docling",
            alternative_parsers=["azure_ocr", "tesseract"],
            text_statistics=TextStatistics(
                total_pages=3,
                total_characters=1500,
                has_text_layer=True,
                font_count=5,
                encoding="utf-8",
                average_chars_per_page=500.0,
            ),
            image_statistics=ImageStatistics(
                total_images=2,
                image_area_ratio=0.15,
                average_dpi=150,
                color_pages=1,
                total_page_area=168300.0,  # letter size in points
                total_image_area=25245.0,
            ),
            page_results=[
                PageAnalysis(
                    page_number=1,
                    content_type="text_based",
                    confidence=0.95,
                    text_ratio=0.8,
                    image_ratio=0.0,
                    character_count=500,
                    image_count=0,
                ),
                PageAnalysis(
                    page_number=2,
                    content_type="mixed",
                    confidence=0.90,
                    text_ratio=0.6,
                    image_ratio=0.2,
                    character_count=400,
                    image_count=2,
                ),
                PageAnalysis(
                    page_number=3,
                    content_type="text_based",
                    confidence=0.92,
                    text_ratio=0.85,
                    image_ratio=0.0,
                    character_count=600,
                    image_count=0,
                ),
            ],
            processing_time_ms=150,
        )
        
        # Save to database
        saved_record = await repo.save(result)
        
        print(f"  ✓ Saved detection result with ID: {saved_record.id}")
        print(f"  ✓ File hash: {saved_record.file_hash}")
        print(f"  ✓ Content type: {saved_record.content_type}")
        print(f"  ✓ Confidence: {saved_record.confidence}")
        print(f"  ✓ Recommended parser: {saved_record.recommended_parser}")
        
        # Verify it was saved by retrieving it
        retrieved = await repo.get_by_hash(result.file_hash)
        if retrieved:
            print(f"  ✓ Successfully retrieved from database")
            return saved_record.id
        else:
            print(f"  ✗ Failed to retrieve from database")
            return None
    
    await engine.dispose()


async def main():
    """Run all persistence tests."""
    print("═══════════════════════════════════════════════════════════════════")
    print("       DATABASE PERSISTENCE TEST - Direct Repository Access")
    print("═══════════════════════════════════════════════════════════════════\n")
    
    try:
        record_id = await test_detection_result_repository()
        
        if record_id:
            print("\n✅ DATABASE PERSISTENCE TEST PASSED")
            print(f"   Record ID: {record_id}")
            print("   Data was successfully saved to PostgreSQL")
        else:
            print("\n❌ TEST FAILED")
            
    except Exception as e:
        print(f"\n❌ TEST FAILED with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
