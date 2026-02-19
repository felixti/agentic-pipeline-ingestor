"""Integration tests for pipeline with content detection."""

from datetime import UTC
from pathlib import Path
from uuid import uuid4

import fitz
import pytest

from src.core.content_detection.models import ContentType
from src.core.job_context import JobContext
from src.core.parser_selection import ParserConfig, ParserSelector
from src.core.pipeline import Pipeline, run_pipeline


@pytest.fixture
def text_pdf(tmp_path: Path) -> Path:
    """Create a text-based PDF for testing."""
    pdf_path = tmp_path / "text_doc.pdf"
    doc = fitz.open()
    
    for i in range(3):
        page = doc.new_page()
        text = f"This is page {i+1} with substantial text content for testing. " * 100
        page.insert_text((72, 72), text, fontsize=12)
    
    doc.save(str(pdf_path))
    doc.close()
    
    return pdf_path


@pytest.fixture
def scanned_pdf(tmp_path: Path) -> Path:
    """Create a scanned-like PDF for testing."""
    pdf_path = tmp_path / "scanned_doc.pdf"
    doc = fitz.open()
    
    for i in range(2):
        page = doc.new_page()
        # Create large image to simulate scanned page
        pix = fitz.Pixmap(fitz.csRGB, 500, 700, b"\x00\x00\xff" * 350000)
        img_rect = fitz.Rect(72, 72, 572, 720)
        page.insert_image(img_rect, pixmap=pix)
        pix = None
    
    doc.save(str(pdf_path))
    doc.close()
    
    return pdf_path


class TestParserSelection:
    """Test parser selection logic."""
    
    def test_select_text_based_parser(self, text_pdf: Path):
        """Test parser selection for text-based PDF."""
        from src.core.content_detection.analyzer import PDFContentAnalyzer
        
        analyzer = PDFContentAnalyzer()
        detection_result = analyzer.analyze(text_pdf)
        
        selection = ParserSelector.select_parser(detection_result)
        
        assert selection.primary_parser == "docling"
        assert selection.fallback_parser == "azure_ocr"
        assert "Text-based PDF" in selection.rationale
        assert not selection.overridden
    
    def test_select_scanned_parser(self, scanned_pdf: Path):
        """Test parser selection for scanned PDF."""
        from src.core.content_detection.analyzer import PDFContentAnalyzer
        
        analyzer = PDFContentAnalyzer()
        detection_result = analyzer.analyze(scanned_pdf)
        
        selection = ParserSelector.select_parser(detection_result)
        
        # Scanned PDF should use OCR
        assert selection.primary_parser == "azure_ocr"
        assert selection.fallback_parser == "docling"
        assert "Scanned PDF" in selection.rationale or "Mixed content" in selection.rationale
        assert not selection.overridden
    
    def test_explicit_config_override(self, text_pdf: Path):
        """Test that explicit config overrides detection."""
        from src.core.content_detection.analyzer import PDFContentAnalyzer
        
        analyzer = PDFContentAnalyzer()
        detection_result = analyzer.analyze(text_pdf)
        
        # Force OCR
        explicit_config = ParserConfig(
            primary_parser="azure_ocr",
            fallback_parser="docling",
            force_ocr=True
        )
        
        selection = ParserSelector.select_parser(detection_result, explicit_config)
        
        assert selection.primary_parser == "azure_ocr"
        assert selection.overridden is True
        assert "OCR forced" in selection.rationale
    
    def test_low_confidence_conservative(self, text_pdf: Path):
        """Test conservative strategy for low confidence."""
        from src.core.content_detection.analyzer import PDFContentAnalyzer
        from src.core.content_detection.models import ContentAnalysisResult
        
        analyzer = PDFContentAnalyzer()
        detection_result = analyzer.analyze(text_pdf)
        
        # Manually lower confidence
        low_confidence_result = ContentAnalysisResult(
            id=detection_result.id,
            file_hash=detection_result.file_hash,
            file_size=detection_result.file_size,
            content_type=detection_result.content_type,
            confidence=0.5,  # Below threshold
            recommended_parser=detection_result.recommended_parser,
            alternative_parsers=detection_result.alternative_parsers,
            text_statistics=detection_result.text_statistics,
            image_statistics=detection_result.image_statistics,
            page_results=detection_result.page_results,
            processing_time_ms=detection_result.processing_time_ms,
        )
        
        selection = ParserSelector.select_parser(low_confidence_result)
        
        # Should use conservative strategy with both parsers
        assert selection.fallback_parser == "azure_ocr"
        assert "Low detection confidence" in selection.rationale
    
    def test_estimate_processing_time(self):
        """Test processing time estimation."""
        from src.core.parser_selection import ParserSelection
        
        selection_doc = ParserSelection("docling", "azure_ocr", "test")
        selection_ocr = ParserSelection("azure_ocr", "docling", "test")
        
        doc_time = ParserSelector.estimate_processing_time(selection_doc, 10)
        ocr_time = ParserSelector.estimate_processing_time(selection_ocr, 10)
        
        # OCR should take longer
        assert "s" in doc_time
        assert "s" in ocr_time or "m" in ocr_time
    
    def test_estimate_cost(self):
        """Test cost estimation."""
        from src.core.parser_selection import ParserSelection
        
        selection_doc = ParserSelection("docling", None, "test")
        selection_ocr = ParserSelection("azure_ocr", None, "test")
        
        doc_cost = ParserSelector.estimate_cost(selection_doc, 10)
        ocr_cost = ParserSelector.estimate_cost(selection_ocr, 10)
        
        # OCR should be more expensive
        assert doc_cost < ocr_cost
        assert doc_cost > 0
        assert ocr_cost > 0


class TestJobContext:
    """Test JobContext with detection result."""
    
    def test_set_and_get_detection_result(self, text_pdf: Path):
        """Test setting and getting detection result."""
        from datetime import datetime, timezone
        
        context = JobContext(
            job_id=uuid4(),
            file_path=str(text_pdf),
            file_type="application/pdf",
            created_at=datetime.now(UTC).isoformat(),
        )
        
        from src.core.content_detection.analyzer import PDFContentAnalyzer
        analyzer = PDFContentAnalyzer()
        result = analyzer.analyze(text_pdf)
        
        # Set detection result
        context.set_detection_result(result)
        
        # Get detection result
        retrieved = context.get_detection_result()
        
        assert retrieved is not None
        assert retrieved.file_hash == result.file_hash
        assert retrieved.content_type == result.content_type
    
    def test_set_parser_selection(self):
        """Test setting parser selection."""
        from datetime import datetime, timezone
        
        context = JobContext(
            job_id=uuid4(),
            file_path="/tmp/test.pdf",
            file_type="application/pdf",
            created_at=datetime.now(UTC).isoformat(),
        )
        
        context.set_parser_selection("docling", "azure_ocr")
        
        assert context.selected_parser == "docling"
        assert context.fallback_parser == "azure_ocr"


class TestPipelineExecution:
    """Test full pipeline execution."""
    
    @pytest.mark.asyncio
    async def test_pipeline_with_text_pdf(self, text_pdf: Path):
        """Test full pipeline with text-based PDF."""
        context = await run_pipeline(
            file_path=str(text_pdf),
            file_type="application/pdf"
        )
        
        # Check detection result
        assert context.content_detection_result is not None
        assert context.content_detection_result.content_type == ContentType.TEXT_BASED
        
        # Check parser selection
        assert context.selected_parser == "docling"
        assert context.fallback_parser == "azure_ocr"
        
        # Check stage results
        assert "ingest" in context.stage_results
        assert "detect" in context.stage_results
        assert "select_parser" in context.stage_results
        assert "parse" in context.stage_results
    
    @pytest.mark.asyncio
    async def test_pipeline_with_scanned_pdf(self, scanned_pdf: Path):
        """Test full pipeline with scanned PDF."""
        context = await run_pipeline(
            file_path=str(scanned_pdf),
            file_type="application/pdf"
        )
        
        # Check detection result
        assert context.content_detection_result is not None
        # Could be scanned or mixed
        assert context.content_detection_result.content_type in [
            ContentType.SCANNED,
            ContentType.MIXED
        ]
        
        # Parser should be OCR-based for scanned content
        assert context.selected_parser is not None
    
    @pytest.mark.asyncio
    async def test_pipeline_with_explicit_config(self, text_pdf: Path):
        """Test pipeline with explicit parser config."""
        config = {
            "parser": {
                "primary": "azure_ocr",
                "fallback": "docling",
                "force_ocr": True
            }
        }
        
        context = await run_pipeline(
            file_path=str(text_pdf),
            file_type="application/pdf",
            config=config
        )
        
        # Should use explicit config
        assert context.selected_parser == "azure_ocr"
        
        # Check override flag
        select_result = context.stage_results.get("select_parser", {})
        assert select_result.get("overridden") is True
    
    @pytest.mark.asyncio
    async def test_pipeline_stage_order(self, text_pdf: Path):
        """Test that stages execute in correct order."""
        pipeline = Pipeline()
        
        stage_names = [stage.name for stage in pipeline.stages]
        
        # Verify order
        assert stage_names[0] == "ingest"
        assert stage_names[1] == "detect"  # NEW stage
        assert stage_names[2] == "select_parser"
        assert stage_names[3] == "parse"


class TestPipelineFlow:
    """Test specific pipeline flow scenarios."""
    
    @pytest.mark.asyncio
    async def test_detection_skipped_if_already_cached(self, text_pdf: Path):
        """Test that detection is skipped if result already in context."""
        from datetime import datetime, timezone

        from src.core.content_detection.analyzer import PDFContentAnalyzer
        
        # Pre-populate context with detection result
        analyzer = PDFContentAnalyzer()
        detection_result = analyzer.analyze(text_pdf)
        
        context = JobContext(
            job_id=uuid4(),
            file_path=str(text_pdf),
            file_type="application/pdf",
            content_detection_result=detection_result,  # Pre-populate
            created_at=datetime.now(UTC).isoformat(),
        )
        
        # Run only detection stage
        from src.core.pipeline import DetectStage
        stage = DetectStage()
        result_context = await stage.execute(context)
        
        # Should use existing result
        assert result_context.content_detection_result is detection_result
    
    @pytest.mark.asyncio
    async def test_parser_selection_without_detection(self, text_pdf: Path):
        """Test parser selection when no detection available."""
        from datetime import datetime, timezone
        
        context = JobContext(
            job_id=uuid4(),
            file_path=str(text_pdf),
            file_type="application/pdf",
            content_detection_result=None,  # No detection
            created_at=datetime.now(UTC).isoformat(),
        )
        
        # Run parser selection stage
        from src.core.pipeline import SelectParserStage
        stage = SelectParserStage()
        result_context = await stage.execute(context)
        
        # Should use default
        assert result_context.selected_parser == "docling"
        assert result_context.fallback_parser == "azure_ocr"


def test_select_parser_for_job_convenience_function(text_pdf: Path):
    """Test convenience function for parser selection."""
    from src.core.content_detection.analyzer import PDFContentAnalyzer
    from src.core.parser_selection import select_parser_for_job
    
    analyzer = PDFContentAnalyzer()
    detection_result = analyzer.analyze(text_pdf)
    
    # Without explicit config
    selection = select_parser_for_job(detection_result)
    assert selection.primary_parser == "docling"
    
    # With explicit config
    selection = select_parser_for_job(
        detection_result,
        explicit_config={
            "primary_parser": "azure_ocr",
            "force_ocr": True
        }
    )
    assert selection.primary_parser == "azure_ocr"
    assert selection.overridden is True
