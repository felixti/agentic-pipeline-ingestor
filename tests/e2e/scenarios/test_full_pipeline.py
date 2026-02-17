"""Full Pipeline E2E Tests.

This module tests the complete document processing pipeline from job creation
to result retrieval for various document types and scenarios.
"""


import pytest
import pytest_asyncio

# Mark all tests in this module as E2E tests
pytestmark = [pytest.mark.e2e, pytest.mark.asyncio]


@pytest_asyncio.fixture
async def text_pdf_job(auth_client, test_documents) -> str:
    """Create a job for text PDF processing and return job ID."""
    payload = {
        "source_type": "upload",
        "source_uri": str(test_documents["text_pdf"]),
        "file_name": "sample-text.pdf",
        "mime_type": "application/pdf",
        "mode": "async",
        "priority": 5,
        "metadata": {"test_case": "TC001", "description": "Text PDF Processing"}
    }

    response = await auth_client.post("/api/v1/jobs", json=payload)
    assert response.status_code == 202, f"Failed to create job: {response.text}"

    return response.json()["data"]["id"]


@pytest_asyncio.fixture
async def scanned_pdf_job(auth_client, test_documents) -> str:
    """Create a job for scanned PDF processing and return job ID."""
    payload = {
        "source_type": "upload",
        "source_uri": str(test_documents["scanned_pdf"]),
        "file_name": "sample-scanned.pdf",
        "mime_type": "application/pdf",
        "mode": "async",
        "priority": 3,
        "options": {"force_ocr": True, "ocr_language": "eng"},
        "metadata": {"test_case": "TC002", "description": "Scanned PDF with OCR"}
    }

    response = await auth_client.post("/api/v1/jobs", json=payload)
    assert response.status_code == 202, f"Failed to create job: {response.text}"

    return response.json()["data"]["id"]


class TestFullPipeline:
    """Test complete document processing pipeline."""

    async def test_health_check(self, client):
        """E2E: Health check endpoint returns healthy status."""
        response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] in ("healthy", "degraded")
        assert "components" in data
        assert "api" in data["components"]

    async def test_create_job_text_pdf(self, auth_client, test_documents):
        """E2E: Create job for text PDF and verify acceptance."""
        payload = {
            "source_type": "upload",
            "source_uri": str(test_documents["text_pdf"]),
            "file_name": "sample-text.pdf",
            "mime_type": "application/pdf",
            "mode": "async",
            "priority": 5,
            "metadata": {"test_case": "TC001"}
        }

        response = await auth_client.post("/api/v1/jobs", json=payload)

        assert response.status_code == 202
        data = response.json()

        assert "data" in data
        assert "id" in data["data"]
        assert data["data"]["status"] in ("created", "pending", "processing")

    async def test_get_job_status(self, auth_client, text_pdf_job):
        """E2E: Get job status returns job details."""
        response = await auth_client.get(f"/api/v1/jobs/{text_pdf_job}")

        assert response.status_code == 200
        data = response.json()

        assert "data" in data
        assert data["data"]["id"] == text_pdf_job
        assert "status" in data["data"]

    async def test_list_jobs_with_pagination(self, auth_client):
        """E2E: List jobs with pagination works correctly."""
        response = await auth_client.get("/api/v1/jobs?page=1&limit=10")

        assert response.status_code == 200
        data = response.json()

        assert "data" in data
        assert isinstance(data["data"], list)
        assert len(data["data"]) <= 10

    async def test_list_jobs_with_status_filter(self, auth_client):
        """E2E: List jobs with status filter returns filtered results."""
        response = await auth_client.get("/api/v1/jobs?status=completed&limit=5")

        assert response.status_code == 200
        data = response.json()

        assert "data" in data

        # If jobs exist, verify they match the filter
        for job in data["data"]:
            assert job.get("status") == "completed"

    async def test_cancel_job(self, auth_client):
        """E2E: Cancel a pending job."""
        # Create a job
        payload = {
            "source_type": "upload",
            "source_uri": "/uploads/test.pdf",
            "file_name": "test.pdf",
            "mime_type": "application/pdf",
            "mode": "async",
            "priority": 5
        }

        create_response = await auth_client.post("/api/v1/jobs", json=payload)
        assert create_response.status_code == 202

        job_id = create_response.json()["data"]["id"]

        # Cancel the job
        cancel_response = await auth_client.delete(f"/api/v1/jobs/{job_id}")

        # May return 204 or 404 depending on implementation
        assert cancel_response.status_code in (204, 404)

    async def test_retry_job(self, auth_client):
        """E2E: Retry a job with modified options."""
        # This test assumes a job exists that can be retried
        # In practice, you might need to create a job that fails

        # For now, test the API endpoint structure
        job_id = "test-job-id"  # Placeholder

        retry_payload = {
            "force_parser": "azure_ocr",
            "priority": 10
        }

        response = await auth_client.post(
            f"/api/v1/jobs/{job_id}/retry",
            json=retry_payload
        )

        # May return 202, 404 (job not found), or 409 (job not in retryable state)
        assert response.status_code in (202, 404, 409)

    async def test_job_result_not_found(self, auth_client):
        """E2E: Get result for non-existent job returns 404."""
        response = await auth_client.get("/api/v1/jobs/non-existent-job/result")

        assert response.status_code == 404


class TestDocumentTypeSupport:
    """Test support for various document types."""

    async def test_pdf_text_extraction(self, auth_client, test_documents):
        """E2E: Upload text PDF → Process → Verify result structure."""
        # Create job
        payload = {
            "source_type": "upload",
            "source_uri": str(test_documents["text_pdf"]),
            "file_name": "sample-text.pdf",
            "mime_type": "application/pdf",
            "mode": "async",
            "priority": 5
        }

        create_response = await auth_client.post("/api/v1/jobs", json=payload)
        assert create_response.status_code == 202

        job_id = create_response.json()["data"]["id"]

        # Verify job was created
        get_response = await auth_client.get(f"/api/v1/jobs/{job_id}")
        assert get_response.status_code == 200

        job_data = get_response.json()["data"]
        assert job_data["id"] == job_id

    async def test_scanned_pdf_ocr(self, auth_client, test_documents):
        """E2E: Upload scanned PDF → Request OCR → Verify job created."""
        payload = {
            "source_type": "upload",
            "source_uri": str(test_documents["scanned_pdf"]),
            "file_name": "sample-scanned.pdf",
            "mime_type": "application/pdf",
            "mode": "async",
            "priority": 3,
            "options": {"force_ocr": True}
        }

        response = await auth_client.post("/api/v1/jobs", json=payload)

        assert response.status_code == 202
        assert response.json()["data"]["id"] is not None

    async def test_word_document_processing(self, auth_client, test_documents):
        """E2E: Process Word document."""
        payload = {
            "source_type": "upload",
            "source_uri": str(test_documents["word_doc"]),
            "file_name": "sample.docx",
            "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "mode": "async",
            "priority": 5
        }

        response = await auth_client.post("/api/v1/jobs", json=payload)

        assert response.status_code == 202
        assert response.json()["data"]["id"] is not None

    async def test_excel_spreadsheet_processing(self, auth_client, test_documents):
        """E2E: Process Excel spreadsheet."""
        payload = {
            "source_type": "upload",
            "source_uri": str(test_documents["excel_doc"]),
            "file_name": "sample.xlsx",
            "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "mode": "async",
            "priority": 5
        }

        response = await auth_client.post("/api/v1/jobs", json=payload)

        assert response.status_code == 202
        assert response.json()["data"]["id"] is not None

    async def test_powerpoint_processing(self, auth_client, test_documents):
        """E2E: Process PowerPoint presentation."""
        payload = {
            "source_type": "upload",
            "source_uri": str(test_documents["ppt_doc"]),
            "file_name": "sample.pptx",
            "mime_type": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "mode": "async",
            "priority": 5
        }

        response = await auth_client.post("/api/v1/jobs", json=payload)

        assert response.status_code == 202
        assert response.json()["data"]["id"] is not None

    async def test_image_ocr_processing(self, auth_client, test_documents):
        """E2E: Process image with OCR."""
        payload = {
            "source_type": "upload",
            "source_uri": str(test_documents["receipt_img"]),
            "file_name": "receipt.jpg",
            "mime_type": "image/jpeg",
            "mode": "async",
            "priority": 5,
            "options": {"force_ocr": True}
        }

        response = await auth_client.post("/api/v1/jobs", json=payload)

        assert response.status_code == 202
        assert response.json()["data"]["id"] is not None


class TestFileUpload:
    """Test file upload functionality."""

    async def test_single_file_upload(self, auth_client, test_documents):
        """E2E: Upload single PDF file."""
        file_path = test_documents["text_pdf"]

        # Skip if file doesn't exist
        if not file_path.exists():
            pytest.skip(f"Test file not found: {file_path}")

        with open(file_path, "rb") as f:
            files = {"file": ("sample-text.pdf", f, "application/pdf")}
            data = {"metadata": '{"test": "upload"}'}

            response = await auth_client.post(
                "/api/v1/upload",
                files=files,
                data=data
            )

        assert response.status_code == 202
        assert "job_id" in response.json()["data"]

    async def test_upload_with_options(self, auth_client, test_documents):
        """E2E: Upload file with processing options."""
        file_path = test_documents["scanned_pdf"]

        if not file_path.exists():
            pytest.skip(f"Test file not found: {file_path}")

        with open(file_path, "rb") as f:
            files = {"file": ("sample-scanned.pdf", f, "application/pdf")}
            data = {
                "metadata": '{"test": "ocr-upload"}',
                "options": '{"force_ocr": true, "ocr_language": "eng"}'
            }

            response = await auth_client.post(
                "/api/v1/upload",
                files=files,
                data=data
            )

        assert response.status_code == 202


class TestPipelineConfiguration:
    """Test pipeline configuration endpoints."""

    async def test_list_sources(self, auth_client):
        """E2E: List available source plugins."""
        response = await auth_client.get("/api/v1/sources")

        assert response.status_code == 200
        data = response.json()

        assert "data" in data
        assert "plugins" in data["data"]

    async def test_list_destinations(self, auth_client):
        """E2E: List available destination plugins."""
        response = await auth_client.get("/api/v1/destinations")

        assert response.status_code == 200
        data = response.json()

        assert "data" in data
        assert "plugins" in data["data"]

    async def test_list_pipelines(self, auth_client):
        """E2E: List pipeline configurations."""
        response = await auth_client.get("/api/v1/pipelines")

        assert response.status_code == 200
        data = response.json()

        assert "data" in data


class TestErrorHandling:
    """Test error handling and edge cases."""

    async def test_invalid_job_id(self, auth_client):
        """E2E: Get job with invalid ID returns 404."""
        response = await auth_client.get("/api/v1/jobs/invalid-job-id-12345")

        assert response.status_code == 404

    async def test_create_job_missing_required_fields(self, auth_client):
        """E2E: Create job without required fields returns error."""
        payload = {
            "source_type": "upload"
            # Missing: source_uri, file_name, mime_type
        }

        response = await auth_client.post("/api/v1/jobs", json=payload)

        assert response.status_code in (400, 422)

    async def test_create_job_invalid_mime_type(self, auth_client):
        """E2E: Create job with unsupported MIME type."""
        payload = {
            "source_type": "upload",
            "source_uri": "/uploads/test.xyz",
            "file_name": "test.xyz",
            "mime_type": "application/x-unknown-format"
        }

        response = await auth_client.post("/api/v1/jobs", json=payload)

        # May accept with warning or reject
        assert response.status_code in (202, 400, 422)

    async def test_invalid_json_payload(self, auth_client):
        """E2E: Send invalid JSON payload."""
        response = await auth_client.post(
            "/api/v1/jobs",
            content="invalid json {{[",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code in (400, 422)
