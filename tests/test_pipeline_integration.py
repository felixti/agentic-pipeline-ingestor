"""Integration tests for the 7-stage pipeline.

These tests verify the complete pipeline flow from ingestion to output.
"""

import tempfile
from datetime import UTC, datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from src.api.models import (
    Job,
    JobStatus,
    PipelineConfig,
    ProcessingMode,
    SourceType,
)
from src.core.engine import OrchestrationEngine
from src.core.pipeline import PipelineExecutor
from src.plugins.destinations.cognee import CogneeMockDestination
from src.plugins.parsers.docling_parser import DoclingParser
from src.plugins.registry import PluginRegistry


@pytest.fixture
def plugin_registry():
    """Create a plugin registry with test plugins."""
    registry = PluginRegistry()

    # Register mock parsers
    docling = DoclingParser()
    registry.register_parser(docling)

    # Register mock destination
    cognee = CogneeMockDestination()
    registry.register_destination(cognee)

    return registry


@pytest.fixture
def sample_text_file():
    """Create a sample text file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("This is a test document.\n\n")
        f.write("It has multiple paragraphs.\n\n")
        f.write("This is the final paragraph.")
        path = f.name

    yield path

    # Cleanup
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def sample_pdf_file():
    """Create a sample PDF file for testing (requires PyMuPDF)."""
    try:
        import fitz  # PyMuPDF

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((100, 100), "This is a test PDF document.")
            page.insert_text((100, 130), "It contains multiple lines of text.")
            doc.save(f.name)
            doc.close()
            path = f.name

        yield path

        # Cleanup
        Path(path).unlink(missing_ok=True)

    except ImportError:
        pytest.skip("PyMuPDF not installed")


class TestPipelineStages:
    """Test individual pipeline stages."""

    @pytest.mark.asyncio
    async def test_ingest_stage(self, sample_text_file):
        """Test the ingest stage."""
        from src.core.pipeline import IngestStage, PipelineContext

        job = Job(
            id=uuid4(),
            source_type=SourceType.UPLOAD,
            source_uri=sample_text_file,
            file_name="test.txt",
            status=JobStatus.CREATED,
            created_at=datetime.now(UTC),
        )

        config = PipelineConfig(name="test")
        context = PipelineContext(job, config)

        stage = IngestStage()
        result = await stage.execute(context)

        assert result["validated"] is True
        assert "file_hash" in result
        assert "file_size" in result

    @pytest.mark.asyncio
    async def test_detect_stage_text_file(self, sample_text_file):
        """Test content detection for text files."""
        from src.core.pipeline import DetectStage, IngestStage, PipelineContext

        job = Job(
            id=uuid4(),
            source_type=SourceType.UPLOAD,
            source_uri=sample_text_file,
            file_name="test.txt",
            mime_type="text/plain",
            status=JobStatus.CREATED,
            created_at=datetime.now(UTC),
        )

        config = PipelineConfig(name="test")
        context = PipelineContext(job, config)

        # First run ingest
        ingest_stage = IngestStage()
        ingest_result = await ingest_stage.execute(context)
        context.set_stage_result("ingest", ingest_result)

        # Then run detect
        detect_stage = DetectStage()
        result = await detect_stage.execute(context)

        assert "detection" in result
        assert "detected_type" in result
        assert "recommended_parser" in result
        assert result["confidence"] > 0

    @pytest.mark.asyncio
    async def test_transform_stage_chunking(self, sample_text_file):
        """Test the transform stage with chunking."""
        from src.core.pipeline import (
            PipelineContext,
            TransformStage,
        )

        job = Job(
            id=uuid4(),
            source_type=SourceType.UPLOAD,
            source_uri=sample_text_file,
            file_name="test.txt",
            status=JobStatus.CREATED,
            created_at=datetime.now(UTC),
        )

        config = PipelineConfig(name="test")
        context = PipelineContext(job, config)

        # Mock parse result
        context.set_stage_result("parse", {
            "success": True,
            "text": "This is a long text. " * 100,  # ~2300 chars
            "pages": ["Page 1 content. " * 50],
            "confidence": 0.95,
        })

        # Mock quality result
        context.set_stage_result("quality", {
            "passed": True,
            "overall_score": 0.95,
        })

        stage = TransformStage()
        result = await stage.execute(context)

        assert "chunks" in result
        assert result["chunk_count"] > 0
        assert all("content" in chunk for chunk in result["chunks"])


class TestPipelineIntegration:
    """Test complete pipeline flow."""

    @pytest.mark.skip(reason="Requires OpenTelemetry tracer setup")
    @pytest.mark.asyncio
    async def test_full_pipeline_text_file(self, sample_text_file, plugin_registry):
        """Test complete pipeline for a text file."""
        job = Job(
            id=uuid4(),
            source_type=SourceType.UPLOAD,
            source_uri=sample_text_file,
            file_name="test.txt",
            mime_type="text/plain",
            mode=ProcessingMode.SYNC,
            status=JobStatus.CREATED,
            created_at=datetime.now(UTC),
        )

        config = PipelineConfig(
            name="test",
            parser={"primary_parser": "docling", "fallback_parser": None},
        )
        job.pipeline_config = config

        executor = PipelineExecutor(
            config=config,
            plugin_registry=plugin_registry,
        )

        context = await executor.execute(job)

        # Verify all stages executed
        assert "ingest" in context.stage_results
        assert "detect" in context.stage_results
        assert "parse" in context.stage_results
        assert "enrich" in context.stage_results
        assert "quality" in context.stage_results
        assert "transform" in context.stage_results

        # Verify job completed
        assert job.status == JobStatus.COMPLETED
        assert job.result is not None
        assert job.result.success is True

    @pytest.mark.skip(reason="Requires OpenTelemetry tracer setup")
    @pytest.mark.asyncio
    async def test_pipeline_with_quality_failure(self, plugin_registry):
        """Test pipeline behavior when quality check fails."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("")  # Empty file - should fail quality
            empty_file = f.name

        try:
            job = Job(
                id=uuid4(),
                source_type=SourceType.UPLOAD,
                source_uri=empty_file,
                file_name="empty.txt",
                mode=ProcessingMode.SYNC,
                status=JobStatus.CREATED,
                created_at=datetime.now(UTC),
            )

            config = PipelineConfig(name="test")
            job.pipeline_config = config

            executor = PipelineExecutor(
                config=config,
                plugin_registry=plugin_registry,
            )

            context = await executor.execute(job)

            # Check quality result
            quality_result = context.get_stage_result("quality")
            if quality_result:
                # Empty file should have low quality score
                assert quality_result.get("overall_score", 1.0) < 0.7

        finally:
            Path(empty_file).unlink(missing_ok=True)


class TestOrchestrationEngine:
    """Test the orchestration engine."""

    @pytest.mark.asyncio
    async def test_create_job(self):
        """Test job creation."""
        engine = OrchestrationEngine()

        job_data = {
            "source_type": "upload",
            "source_uri": "/tmp/test.txt",
            "file_name": "test.txt",
            "mode": "async",
        }

        job = await engine.create_job(job_data)

        assert job.id is not None
        assert job.source_type.value == "upload"
        assert job.file_name == "test.txt"
        assert job.status == JobStatus.CREATED

    @pytest.mark.asyncio
    async def test_cancel_job(self):
        """Test job cancellation."""
        engine = OrchestrationEngine()

        job_data = {
            "source_type": "upload",
            "source_uri": "/tmp/test.txt",
            "file_name": "test.txt",
            "mode": "async",
        }

        job = await engine.create_job(job_data)

        # Cancel the job
        cancelled = await engine.cancel_job(job.id)
        assert cancelled is True

        # Verify status
        updated_job = await engine.get_job(job.id)
        assert updated_job.status == JobStatus.CANCELLED


class TestParserPlugins:
    """Test parser plugins."""

    @pytest.mark.asyncio
    async def test_docling_parser_supports(self):
        """Test Docling parser file support detection."""
        parser = DoclingParser()

        # Should support PDF
        result = await parser.supports("/path/to/file.pdf")
        assert result.supported is True

        # Should support DOCX
        result = await parser.supports("/path/to/file.docx")
        assert result.supported is True

        # Should not support unknown
        result = await parser.supports("/path/to/file.xyz")
        assert result.supported is False


class TestDestinationPlugins:
    """Test destination plugins."""

    @pytest.mark.asyncio
    async def test_cognee_mock_destination(self):
        """Test mock Cognee destination."""
        from uuid import uuid4

        from src.plugins.base import TransformedData

        destination = CogneeMockDestination()
        await destination.initialize({})

        conn = await destination.connect({"dataset_id": "test-dataset"})

        data = TransformedData(
            job_id=uuid4(),
            chunks=[{"content": "Test chunk", "metadata": {}}],
        )

        result = await destination.write(conn, data)

        assert result.success is True
        assert result.records_written == 1

        # Verify stored data
        stored = destination.get_stored_data("test-dataset")
        assert len(stored) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
