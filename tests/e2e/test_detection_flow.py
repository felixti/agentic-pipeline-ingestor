"""End-to-end tests for complete detection flow."""

from pathlib import Path
from uuid import uuid4

import fitz
import pytest

from src.core.content_detection.models import ContentType
from src.core.pipeline import run_pipeline


@pytest.fixture
def sample_corpus(tmp_path: Path) -> dict:
    """Create sample PDF corpus for E2E testing."""
    corpus = {}
    
    # 1. Pure text document (research paper style)
    research_path = tmp_path / "research_paper.pdf"
    doc = fitz.open()
    for i in range(5):
        page = doc.new_page()
        # Title on first page
        if i == 0:
            page.insert_text((72, 72), "Research Paper: PDF Content Detection", fontsize=18)
            page.insert_text((72, 120), "Abstract: This paper discusses...", fontsize=12)
        # Body text
        text = f"Section {i+1}: " + "Lorem ipsum dolor sit amet. " * 200
        page.insert_text((72, 200), text, fontsize=12)
    doc.save(str(research_path))
    doc.close()
    corpus["research_paper"] = research_path
    
    # 2. Scanned invoice
    invoice_path = tmp_path / "invoice.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # Create invoice-like image
    pix = fitz.Pixmap(fitz.csRGB, 500, 700, b"\xff\xff\xff" * 350000)  # White background
    page.insert_image(fitz.Rect(50, 50, 550, 750), pixmap=pix)
    # Add minimal text to simulate OCR-like content
    page.insert_text((100, 100), "INVOICE #001", fontsize=24)
    page.insert_text((100, 150), "Total: $100.00", fontsize=14)
    pix = None
    doc.save(str(invoice_path))
    doc.close()
    corpus["invoice"] = invoice_path
    
    # 3. Mixed brochure
    brochure_path = tmp_path / "brochure.pdf"
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page()
        # Text header
        page.insert_text((72, 72), f"Product Page {i+1}", fontsize=20)
        # Body text
        page.insert_text((72, 150), "Product description and details. " * 50, fontsize=11)
        # Product image
        pix = fitz.Pixmap(fitz.csRGB, 200, 200, b"\x00\x80\xff" * 40000)
        page.insert_image(fitz.Rect(300, 300, 500, 500), pixmap=pix)
        pix = None
    doc.save(str(brochure_path))
    doc.close()
    corpus["brochure"] = brochure_path
    
    # 4. Textbook with diagrams
    textbook_path = tmp_path / "textbook.pdf"
    doc = fitz.open()
    for i in range(4):
        page = doc.new_page()
        # Chapter heading
        page.insert_text((72, 72), f"Chapter {i+1}: Introduction", fontsize=16)
        # Main content
        content = "Educational content with explanations. " * 150
        page.insert_text((72, 120), content, fontsize=11)
        # Diagram
        if i % 2 == 0:
            pix = fitz.Pixmap(fitz.csRGB, 300, 200, b"\x80\xff\x80" * 60000)
            page.insert_image(fitz.Rect(150, 400, 450, 600), pixmap=pix)
            pix = None
    doc.save(str(textbook_path))
    doc.close()
    corpus["textbook"] = textbook_path
    
    return corpus


class TestCompleteUserFlow:
    """Test complete user journeys."""
    
    @pytest.mark.asyncio
    async def test_upload_text_document_flow(self, sample_corpus: dict):
        """E2E: Upload text document → Detection → Parser selection → Verification."""
        # Step 1: Simulate file upload (file already exists in corpus)
        pdf_path = sample_corpus["research_paper"]
        
        # Step 2: Run pipeline (includes detection and parser selection)
        job_id = uuid4()
        context = await run_pipeline(
            file_path=str(pdf_path),
            file_type="application/pdf",
            job_id=job_id
        )
        
        # Step 3: Verify detection result
        assert context.content_detection_result is not None
        detection = context.content_detection_result
        
        # Research paper should be text-based
        assert detection.content_type == ContentType.TEXT_BASED
        assert detection.confidence > 0.9
        assert detection.text_statistics.has_text_layer is True
        assert detection.text_statistics.total_pages == 5
        assert detection.text_statistics.total_characters > 1000
        
        # Step 4: Verify parser selection
        assert context.selected_parser == "docling"
        assert context.fallback_parser == "azure_ocr"
        
        # Step 5: Verify stage completion
        assert "ingest" in context.stage_results
        assert "detect" in context.stage_results
        assert "select_parser" in context.stage_results
        assert "parse" in context.stage_results
        
        # Step 6: Verify metadata
        assert str(context.job_id) == str(job_id)
        assert context.file_path == str(pdf_path)
    
    @pytest.mark.asyncio
    async def test_upload_scanned_document_flow(self, sample_corpus: dict):
        """E2E: Upload scanned document → Detection → OCR parser selection."""
        pdf_path = sample_corpus["invoice"]
        
        context = await run_pipeline(
            file_path=str(pdf_path),
            file_type="application/pdf"
        )
        
        # Verify detection
        assert context.content_detection_result is not None
        detection = context.content_detection_result
        
        # Invoice should be detected as scanned or mixed
        assert detection.content_type in [ContentType.SCANNED, ContentType.MIXED]
        assert detection.image_statistics.total_images > 0
        
        # Should use OCR parser
        assert context.selected_parser in ["azure_ocr", "docling"]
    
    @pytest.mark.asyncio
    async def test_mixed_content_flow(self, sample_corpus: dict):
        """E2E: Mixed content document with fallback parser."""
        pdf_path = sample_corpus["brochure"]
        
        context = await run_pipeline(
            file_path=str(pdf_path),
            file_type="application/pdf"
        )
        
        # Verify detection
        assert context.content_detection_result is not None
        detection = context.content_detection_result
        
        # Brochure should have mixed content
        assert detection.content_type in [ContentType.MIXED, ContentType.TEXT_BASED]
        assert detection.text_statistics.total_characters > 0
        assert detection.image_statistics.total_images > 0
        
        # Should have fallback parser configured
        assert context.fallback_parser is not None
    
    @pytest.mark.asyncio
    async def test_explicit_parser_override_flow(self, sample_corpus: dict):
        """E2E: User overrides automatic parser selection."""
        pdf_path = sample_corpus["research_paper"]
        
        # Configure explicit OCR
        config = {
            "parser": {
                "primary": "azure_ocr",
                "fallback": "docling",
                "force_ocr": True
            }
        }
        
        context = await run_pipeline(
            file_path=str(pdf_path),
            file_type="application/pdf",
            config=config
        )
        
        # Should use explicit config despite text-based detection
        assert context.selected_parser == "azure_ocr"
        
        # Verify override flag
        select_result = context.stage_results.get("select_parser", {})
        assert select_result.get("overridden") is True
    
    @pytest.mark.asyncio
    async def test_batch_processing_flow(self, sample_corpus: dict):
        """E2E: Process multiple documents in batch."""
        documents = [
            ("research", sample_corpus["research_paper"]),
            ("invoice", sample_corpus["invoice"]),
            ("brochure", sample_corpus["brochure"]),
        ]
        
        results = []
        for name, pdf_path in documents:
            context = await run_pipeline(
                file_path=str(pdf_path),
                file_type="application/pdf"
            )
            results.append((name, context))
        
        # Verify all processed
        assert len(results) == 3
        
        # Verify research paper used Docling
        research_ctx = next(ctx for name, ctx in results if name == "research")
        assert research_ctx.selected_parser == "docling"
        
        # Verify invoice used OCR
        invoice_ctx = next(ctx for name, ctx in results if name == "invoice")
        assert invoice_ctx.content_detection_result.content_type in [
            ContentType.SCANNED, ContentType.MIXED
        ]
    
    @pytest.mark.asyncio
    async def test_caching_flow(self, sample_corpus: dict):
        """E2E: Second analysis should use cache."""
        from src.core.content_detection.service import (
            ContentDetectionService,
            get_detection_service,
        )
        
        pdf_path = sample_corpus["textbook"]
        
        # First analysis
        service = ContentDetectionService()
        result1 = await service.detect(pdf_path)
        
        # Second analysis (should hit cache)
        result2 = await service.detect(pdf_path)
        
        # Results should be identical
        assert result1.file_hash == result2.file_hash
        assert result1.content_type == result2.content_type
        assert result1.confidence == result2.confidence
    
    @pytest.mark.asyncio
    async def test_error_handling_flow(self, tmp_path: Path):
        """E2E: Handle corrupted PDF gracefully."""
        # Create invalid PDF
        invalid_pdf = tmp_path / "invalid.pdf"
        invalid_pdf.write_bytes(b"This is not a PDF file")
        
        # Should raise error
        with pytest.raises(Exception):  # noqa: B017  # Any exception is expected for invalid PDF
            await run_pipeline(
                file_path=str(invalid_pdf),
                file_type="application/pdf"
            )


class TestProductionScenarios:
    """Test production-like scenarios."""
    
    @pytest.mark.asyncio
    async def test_high_volume_upload_simulation(self, tmp_path: Path):
        """E2E: Simulate high volume document uploads."""
        # Create 20 test documents
        documents = []
        for i in range(20):
            pdf_path = tmp_path / f"doc_{i:03d}.pdf"
            doc = fitz.open()
            page = doc.new_page()
            
            # Alternate between text-heavy and image-heavy
            if i % 3 == 0:
                # Text document
                page.insert_text((72, 72), f"Document {i}" * 200, fontsize=12)
            else:
                # Image document
                pix = fitz.Pixmap(fitz.csRGB, 400, 500, b"\xff\x00\x00" * 200000)
                page.insert_image(fitz.Rect(72, 72, 472, 572), pixmap=pix)
                pix = None
            
            doc.save(str(pdf_path))
            doc.close()
            documents.append(pdf_path)
        
        # Process all
        results = []
        for pdf_path in documents:
            context = await run_pipeline(
                file_path=str(pdf_path),
                file_type="application/pdf"
            )
            results.append(context.content_detection_result.content_type)
        
        # Verify all processed
        assert len(results) == 20
        
        # Verify distribution
        text_count = sum(1 for r in results if r == ContentType.TEXT_BASED)
        scanned_count = sum(1 for r in results if r == ContentType.SCANNED)
        mixed_count = sum(1 for r in results if r == ContentType.MIXED)
        
        print("\nProcessing distribution:")
        print(f"  Text-based: {text_count}")
        print(f"  Scanned: {scanned_count}")
        print(f"  Mixed: {mixed_count}")
        
        # All types should be detected
        assert text_count > 0
    
    @pytest.mark.asyncio
    async def test_integration_with_existing_pipeline(self, sample_corpus: dict):
        """E2E: Content detection integrates with existing pipeline stages."""
        pdf_path = sample_corpus["textbook"]
        
        context = await run_pipeline(
            file_path=str(pdf_path),
            file_type="application/pdf"
        )
        
        # Verify detection result is available to downstream stages
        assert context.content_detection_result is not None
        
        # Verify parser selection used detection
        select_result = context.stage_results.get("select_parser", {})
        assert "rationale" in select_result
        
        # Parse stage should have access to selected parser
        parse_result = context.stage_results.get("parse", {})
        assert "parser_used" in parse_result
        assert parse_result["parser_used"] == context.selected_parser
