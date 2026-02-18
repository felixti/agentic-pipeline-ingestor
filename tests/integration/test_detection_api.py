"""Integration tests for content detection API endpoints."""

import io
from pathlib import Path

import fitz
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.detection import router


@pytest.fixture
def app() -> FastAPI:
    """Create test FastAPI app."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/detect")
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def text_pdf() -> io.BytesIO:
    """Create a text-based PDF for testing."""
    doc = fitz.open()
    page = doc.new_page()
    text = "This is a test document with text content. " * 100
    page.insert_text((72, 72), text, fontsize=12)
    doc.new_page()
    page2 = doc[-1]
    page2.insert_text((72, 72), "More text on page 2. " * 100, fontsize=12)
    
    pdf_bytes = io.BytesIO()
    doc.save(pdf_bytes)
    doc.close()
    pdf_bytes.seek(0)
    return pdf_bytes


@pytest.fixture
def scanned_pdf() -> io.BytesIO:
    """Create a scanned-like PDF for testing."""
    doc = fitz.open()
    page = doc.new_page()
    # Create a large image
    pix = fitz.Pixmap(fitz.csRGB, 500, 700, b"\x00\x00\xff" * 350000)
    img_rect = fitz.Rect(72, 72, 572, 720)
    page.insert_image(img_rect, pixmap=pix)
    pix = None
    
    pdf_bytes = io.BytesIO()
    doc.save(pdf_bytes)
    doc.close()
    pdf_bytes.seek(0)
    return pdf_bytes


@pytest.fixture
def invalid_file() -> io.BytesIO:
    """Create an invalid file for testing."""
    return io.BytesIO(b"This is not a PDF file")


class TestDetectEndpoint:
    """Test POST /api/v1/detect endpoint."""
    
    def test_detect_text_based_pdf(self, client: TestClient, text_pdf: io.BytesIO):
        """Test detecting a text-based PDF."""
        response = client.post(
            "/api/v1/detect",
            files={"file": ("test.pdf", text_pdf, "application/pdf")},
            data={"detailed": "false"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "meta" in data
        
        result = data["data"]
        assert result["content_type"] == "text_based"
        assert result["confidence"] > 0.9
        assert result["recommended_parser"] == "docling"
        assert result["text_statistics"]["has_text_layer"] is True
        assert result["processing_time_ms"] > 0
        assert len(result["file_hash"]) == 64
    
    def test_detect_scanned_pdf(self, client: TestClient, scanned_pdf: io.BytesIO):
        """Test detecting a scanned PDF."""
        response = client.post(
            "/api/v1/detect",
            files={"file": ("scanned.pdf", scanned_pdf, "application/pdf")},
            data={"detailed": "false"}
        )
        
        assert response.status_code == 200
        data = response.json()
        result = data["data"]
        
        # Should be scanned or mixed
        assert result["content_type"] in ["scanned", "mixed"]
        assert result["image_statistics"]["total_images"] > 0
        assert result["processing_time_ms"] > 0
    
    def test_detect_with_detailed_flag(self, client: TestClient, text_pdf: io.BytesIO):
        """Test detection with detailed=true."""
        response = client.post(
            "/api/v1/detect",
            files={"file": ("test.pdf", text_pdf, "application/pdf")},
            data={"detailed": "true"}
        )
        
        assert response.status_code == 200
        data = response.json()
        result = data["data"]
        
        assert result["page_results"] is not None
        assert len(result["page_results"]) == 2
        assert result["page_results"][0]["page_number"] == 1
    
    def test_detect_without_detailed_flag(self, client: TestClient, text_pdf: io.BytesIO):
        """Test detection without detailed flag."""
        response = client.post(
            "/api/v1/detect",
            files={"file": ("test.pdf", text_pdf, "application/pdf")}
        )
        
        assert response.status_code == 200
        data = response.json()
        result = data["data"]
        
        # page_results should be None when detailed=false
        assert result.get("page_results") is None
    
    def test_detect_invalid_file_type(self, client: TestClient):
        """Test detecting a non-PDF file."""
        response = client.post(
            "/api/v1/detect",
            files={"file": ("test.txt", io.BytesIO(b"Not a PDF"), "text/plain")}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data["detail"]
        assert data["detail"]["error"]["code"] == "INVALID_FILE_FORMAT"
    
    def test_detect_corrupted_pdf(self, client: TestClient, invalid_file: io.BytesIO):
        """Test detecting an invalid/corrupted PDF."""
        response = client.post(
            "/api/v1/detect",
            files={"file": ("corrupted.pdf", invalid_file, "application/pdf")}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data["detail"]
        assert data["detail"]["error"]["code"] == "INVALID_FILE_FORMAT"
    
    def test_detect_no_file(self, client: TestClient):
        """Test detection without providing a file."""
        response = client.post("/api/v1/detect")
        
        assert response.status_code == 422  # Validation error


class TestDetectUrlEndpoint:
    """Test POST /api/v1/detect/url endpoint."""
    
    def test_detect_from_url(self, client: TestClient, text_pdf: io.BytesIO, monkeypatch):
        """Test detecting PDF from URL."""
        import httpx
        
        # Mock the HTTP client
        class MockResponse:
            status_code = 200
            content = text_pdf.getvalue()
            
            def raise_for_status(self):
                pass
        
        class MockClient:
            async def get(self, *args, **kwargs):
                return MockResponse()
            
            async def __aenter__(self):
                return self
            
            async def __aexit__(self, *args):
                pass
        
        monkeypatch.setattr(httpx, "AsyncClient", MockClient)
        
        response = client.post(
            "/api/v1/detect/url",
            json={
                "url": "https://example.com/test.pdf",
                "detailed": False
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert data["data"]["content_type"] == "text_based"
    
    def test_detect_from_url_invalid_url(self, client: TestClient):
        """Test detection with invalid URL."""
        response = client.post(
            "/api/v1/detect/url",
            json={
                "url": "not-a-valid-url",
                "detailed": False
            }
        )
        
        assert response.status_code == 422  # Validation error


class TestDetectBatchEndpoint:
    """Test POST /api/v1/detect/batch endpoint."""
    
    def test_detect_batch_success(self, client: TestClient, text_pdf: io.BytesIO):
        """Test batch detection with valid files."""
        text_pdf.seek(0)
        
        response = client.post(
            "/api/v1/detect/batch",
            files=[
                ("files", ("doc1.pdf", text_pdf, "application/pdf")),
                ("files", ("doc2.pdf", io.BytesIO(text_pdf.getvalue()), "application/pdf")),
            ]
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "results" in data["data"]
        assert "summary" in data["data"]
        
        results = data["data"]["results"]
        summary = data["data"]["summary"]
        
        assert len(results) == 2
        assert summary["total"] == 2
        assert summary["successful"] == 2
        assert all(r["error"] is None for r in results)
    
    def test_detect_batch_with_error(self, client: TestClient, text_pdf: io.BytesIO, invalid_file: io.BytesIO):
        """Test batch detection with one valid and one invalid file."""
        text_pdf.seek(0)
        
        response = client.post(
            "/api/v1/detect/batch",
            files=[
                ("files", ("valid.pdf", text_pdf, "application/pdf")),
                ("files", ("invalid.pdf", invalid_file, "application/pdf")),
            ]
        )
        
        assert response.status_code == 200
        data = response.json()
        summary = data["data"]["summary"]
        
        assert summary["total"] == 2
        assert summary["successful"] == 1
        assert summary["errors"] == 1
    
    def test_detect_batch_too_many_files(self, client: TestClient, text_pdf: io.BytesIO):
        """Test batch detection with too many files."""
        files = [("files", (f"doc{i}.pdf", io.BytesIO(text_pdf.getvalue()), "application/pdf")) 
                 for i in range(11)]
        
        response = client.post("/api/v1/detect/batch", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["error"]["code"] == "TOO_MANY_FILES"
    
    def test_detect_batch_empty(self, client: TestClient):
        """Test batch detection with no files."""
        response = client.post("/api/v1/detect/batch")
        
        assert response.status_code == 422  # Validation error


class TestRateLimiting:
    """Test rate limiting functionality."""
    
    def test_rate_limit_enforced(self, client: TestClient, text_pdf: io.BytesIO, monkeypatch):
        """Test that rate limiting is enforced."""
        from src.api.routes import detection
        
        # Set very low rate limit for testing
        original_limit = detection.RATE_LIMIT_REQUESTS
        detection.RATE_LIMIT_REQUESTS = 2
        detection.RATE_LIMIT_WINDOW = 60
        
        try:
            # Make requests up to the limit
            for i in range(2):
                text_pdf.seek(0)
                response = client.post(
                    "/api/v1/detect",
                    files={"file": (f"test{i}.pdf", text_pdf, "application/pdf")},
                    headers={"X-API-Key": "test-key"}
                )
                assert response.status_code == 200
            
            # Next request should be rate limited
            text_pdf.seek(0)
            response = client.post(
                "/api/v1/detect",
                files={"file": ("test_blocked.pdf", text_pdf, "application/pdf")},
                headers={"X-API-Key": "test-key"}
            )
            
            assert response.status_code == 429
            assert "Retry-After" in response.headers
            
        finally:
            # Restore original limit
            detection.RATE_LIMIT_REQUESTS = original_limit
    
    def test_rate_limit_per_client(self, client: TestClient, text_pdf: io.BytesIO, monkeypatch):
        """Test that rate limits are per-client."""
        from src.api.routes import detection
        
        # Set low rate limit
        original_limit = detection.RATE_LIMIT_REQUESTS
        detection.RATE_LIMIT_REQUESTS = 1
        detection.RATE_LIMIT_WINDOW = 60
        
        try:
            # Use different API keys
            for i in range(2):
                text_pdf.seek(0)
                response = client.post(
                    "/api/v1/detect",
                    files={"file": (f"test{i}.pdf", text_pdf, "application/pdf")},
                    headers={"X-API-Key": f"key-{i}"}
                )
                assert response.status_code == 200
            
        finally:
            detection.RATE_LIMIT_REQUESTS = original_limit


class TestResponseStructure:
    """Test API response structure."""
    
    def test_response_has_required_fields(self, client: TestClient, text_pdf: io.BytesIO):
        """Test that response has all required fields."""
        response = client.post(
            "/api/v1/detect",
            files={"file": ("test.pdf", text_pdf, "application/pdf")}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check top-level structure
        assert "data" in data
        assert "meta" in data
        
        # Check meta fields
        meta = data["meta"]
        assert "request_id" in meta
        assert "timestamp" in meta
        assert "api_version" in meta
        
        # Check data fields
        result = data["data"]
        required_fields = [
            "content_type", "confidence", "recommended_parser",
            "alternative_parsers", "text_statistics", "image_statistics",
            "processing_time_ms", "file_hash", "file_size"
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"
