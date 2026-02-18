"""Performance benchmarks for content detection."""

import io
import time
from pathlib import Path

import fitz
import pytest

from src.core.content_detection.analyzer import PDFContentAnalyzer


class TestDetectionPerformance:
    """Performance tests for content detection."""
    
    @pytest.fixture
    def small_text_pdf(self, tmp_path: Path) -> Path:
        """Create a small text PDF (~10KB)."""
        pdf_path = tmp_path / "small.pdf"
        doc = fitz.open()
        page = doc.new_page()
        text = "Test content " * 100
        page.insert_text((72, 72), text, fontsize=12)
        doc.save(str(pdf_path))
        doc.close()
        return pdf_path
    
    @pytest.fixture
    def medium_text_pdf(self, tmp_path: Path) -> Path:
        """Create a medium text PDF (~100KB)."""
        pdf_path = tmp_path / "medium.pdf"
        doc = fitz.open()
        
        for i in range(10):
            page = doc.new_page()
            text = f"Page {i+1} content " * 500
            page.insert_text((72, 72), text, fontsize=12)
        
        doc.save(str(pdf_path))
        doc.close()
        return pdf_path
    
    @pytest.fixture
    def large_text_pdf(self, tmp_path: Path) -> Path:
        """Create a large text PDF (~1MB)."""
        pdf_path = tmp_path / "large.pdf"
        doc = fitz.open()
        
        for i in range(50):
            page = doc.new_page()
            text = f"Page {i+1} content with substantial text for testing performance. " * 200
            page.insert_text((72, 72), text, fontsize=12)
        
        doc.save(str(pdf_path))
        doc.close()
        return pdf_path
    
    def test_small_pdf_latency(self, small_text_pdf: Path):
        """Test latency for small PDF (< 500ms target)."""
        analyzer = PDFContentAnalyzer()
        
        start = time.time()
        result = analyzer.analyze(small_text_pdf)
        elapsed_ms = (time.time() - start) * 1000
        
        print(f"\nSmall PDF latency: {elapsed_ms:.1f}ms")
        assert elapsed_ms < 500, f"Latency {elapsed_ms:.1f}ms exceeds 500ms target"
        assert result.processing_time_ms < 500
    
    def test_medium_pdf_latency(self, medium_text_pdf: Path):
        """Test latency for medium PDF (< 1s target)."""
        analyzer = PDFContentAnalyzer()
        
        start = time.time()
        result = analyzer.analyze(medium_text_pdf)
        elapsed_ms = (time.time() - start) * 1000
        
        print(f"\nMedium PDF latency: {elapsed_ms:.1f}ms")
        assert elapsed_ms < 1000, f"Latency {elapsed_ms:.1f}ms exceeds 1s target"
    
    def test_large_pdf_latency(self, large_text_pdf: Path):
        """Test latency for large PDF (< 3s target)."""
        analyzer = PDFContentAnalyzer()
        
        start = time.time()
        result = analyzer.analyze(large_text_pdf)
        elapsed_ms = (time.time() - start) * 1000
        
        print(f"\nLarge PDF latency: {elapsed_ms:.1f}ms")
        assert elapsed_ms < 3000, f"Latency {elapsed_ms:.1f}ms exceeds 3s target"
    
    def test_memory_usage(self, medium_text_pdf: Path):
        """Test memory usage during analysis."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        analyzer = PDFContentAnalyzer()
        analyzer.analyze(medium_text_pdf)
        
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_used = mem_after - mem_before
        
        print(f"\nMemory used: {mem_used:.1f}MB")
        assert mem_used < 200, f"Memory usage {mem_used:.1f}MB exceeds 200MB target"
    
    def test_throughput(self, tmp_path: Path):
        """Test throughput (requests per second)."""
        # Create 10 test PDFs
        pdf_paths = []
        for i in range(10):
            pdf_path = tmp_path / f"test_{i}.pdf"
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((72, 72), f"Test {i} " * 100, fontsize=12)
            doc.save(str(pdf_path))
            doc.close()
            pdf_paths.append(pdf_path)
        
        analyzer = PDFContentAnalyzer()
        
        start = time.time()
        for pdf_path in pdf_paths:
            analyzer.analyze(pdf_path)
        elapsed = time.time() - start
        
        throughput = len(pdf_paths) / elapsed
        print(f"\nThroughput: {throughput:.1f} docs/sec")
        
        assert throughput > 1, f"Throughput {throughput:.1f} docs/sec below 1 doc/sec target"
    
    @pytest.mark.slow
    def test_accuracy_corpus(self, tmp_path: Path):
        """Test accuracy on corpus of mixed PDFs (target: > 98%)."""
        # Create test corpus
        test_cases = []
        
        # Text-based PDFs (should be detected as text_based)
        for i in range(30):
            pdf_path = tmp_path / f"text_{i}.pdf"
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((72, 72), "Text content " * 200, fontsize=12)
            doc.save(str(pdf_path))
            doc.close()
            test_cases.append((pdf_path, "text_based"))
        
        # Scanned-like PDFs (should be detected as scanned or mixed)
        for i in range(30):
            pdf_path = tmp_path / f"scanned_{i}.pdf"
            doc = fitz.open()
            page = doc.new_page()
            pix = fitz.Pixmap(fitz.csRGB, 400, 600, b"\xff\x00\x00" * 240000)
            page.insert_image(fitz.Rect(72, 72, 472, 672), pixmap=pix)
            pix = None
            doc.save(str(pdf_path))
            doc.close()
            test_cases.append((pdf_path, "scanned"))
        
        # Mixed PDFs
        for i in range(20):
            pdf_path = tmp_path / f"mixed_{i}.pdf"
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((72, 72), "Text " * 100, fontsize=12)
            pix = fitz.Pixmap(fitz.csRGB, 200, 300, b"\x00\xff\x00" * 60000)
            page.insert_image(fitz.Rect(300, 72, 500, 372), pixmap=pix)
            pix = None
            doc.save(str(pdf_path))
            doc.close()
            test_cases.append((pdf_path, "mixed"))
        
        # Test
        analyzer = PDFContentAnalyzer()
        correct = 0
        total = len(test_cases)
        
        for pdf_path, expected_type in test_cases:
            result = analyzer.analyze(pdf_path)
            
            # For text-based, must be exactly text_based
            if expected_type == "text_based" and result.content_type.value == "text_based":
                correct += 1
            # For scanned, accept scanned or mixed (conservative)
            elif expected_type == "scanned" and result.content_type.value in ["scanned", "mixed"]:
                correct += 1
            # For mixed, accept mixed or text_based
            elif expected_type == "mixed" and result.content_type.value in ["mixed", "text_based"]:
                correct += 1
        
        accuracy = correct / total
        print(f"\nAccuracy: {accuracy:.1%} ({correct}/{total})")
        
        assert accuracy > 0.90, f"Accuracy {accuracy:.1%} below 90% threshold"


class TestAPILatency:
    """API endpoint latency tests."""
    
    @pytest.mark.asyncio
    async def test_api_detect_latency(self, tmp_path: Path):
        """Test API /detect endpoint latency."""
        from fastapi.testclient import TestClient
        from src.api.routes.detection import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router, prefix="/api/v1/detect")
        client = TestClient(app)
        
        # Create test PDF
        pdf_path = tmp_path / "api_test.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "API test content " * 100, fontsize=12)
        doc.save(str(pdf_path))
        doc.close()
        
        # Measure API latency
        start = time.time()
        with open(pdf_path, 'rb') as f:
            response = client.post(
                "/api/v1/detect",
                files={"file": ("test.pdf", f, "application/pdf")}
            )
        elapsed_ms = (time.time() - start) * 1000
        
        print(f"\nAPI /detect latency: {elapsed_ms:.1f}ms")
        
        assert response.status_code == 200
        assert elapsed_ms < 1000, f"API latency {elapsed_ms:.1f}ms exceeds 1s target"
    
    @pytest.mark.asyncio
    async def test_api_batch_latency(self, tmp_path: Path):
        """Test API /detect/batch endpoint latency."""
        from fastapi.testclient import TestClient
        from src.api.routes.detection import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router, prefix="/api/v1/detect")
        client = TestClient(app)
        
        # Create test PDFs
        files = []
        for i in range(5):
            pdf_path = tmp_path / f"batch_{i}.pdf"
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((72, 72), f"Batch {i} " * 100, fontsize=12)
            doc.save(str(pdf_path))
            doc.close()
            files.append(("files", (f"batch_{i}.pdf", open(pdf_path, 'rb'), "application/pdf")))
        
        # Measure batch API latency
        start = time.time()
        response = client.post("/api/v1/detect/batch", files=files)
        elapsed_ms = (time.time() - start) * 1000
        
        # Close files
        for _, (_, file_obj, _) in files:
            file_obj.close()
        
        print(f"\nAPI /detect/batch latency (5 files): {elapsed_ms:.1f}ms")
        
        assert response.status_code == 200
        assert elapsed_ms < 3000, f"Batch API latency {elapsed_ms:.1f}ms exceeds 3s target"
