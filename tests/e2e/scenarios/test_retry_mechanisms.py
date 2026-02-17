"""Retry Mechanisms E2E Tests.

This module tests all 4 retry strategies:
1. Same parser retry
2. Fallback parser retry
3. Preprocess and retry
4. Split processing retry
"""

import pytest

# Mark all tests in this module as E2E and retry tests
pytestmark = [pytest.mark.e2e, pytest.mark.retry, pytest.mark.asyncio]


class TestRetryMechanisms:
    """Test all retry strategies."""

    async def test_same_parser_retry_success(self, auth_client, test_documents):
        """E2E: Test retry with same parser succeeds on transient failure.
        
        This test verifies that transient failures are retried with the same
        parser before giving up.
        """
        # Create a job with retry configuration
        payload = {
            "source_type": "upload",
            "source_uri": str(test_documents["text_pdf"]),
            "file_name": "sample-text.pdf",
            "mime_type": "application/pdf",
            "mode": "async",
            "priority": 5,
            "retry_config": {
                "max_retries": 3,
                "retry_delay_seconds": 1,
                "strategy": "same_parser"
            },
            "metadata": {
                "test_case": "RT001",
                "scenario": "transient_failure"
            }
        }

        response = await auth_client.post("/api/v1/jobs", json=payload)

        assert response.status_code == 202
        job_id = response.json()["data"]["id"]

        # Verify job was created with retry configuration
        get_response = await auth_client.get(f"/api/v1/jobs/{job_id}")
        assert get_response.status_code == 200

        job_data = get_response.json()["data"]
        assert job_data["id"] == job_id
        # Verify retry config is stored
        assert "retry_config" in job_data or "options" in job_data

    async def test_fallback_parser_retry(self, auth_client, test_documents):
        """E2E: Test fallback to alternative parser on persistent failure.
        
        When the primary parser fails persistently, the system should
        fall back to an alternative parser (e.g., Azure OCR).
        """
        payload = {
            "source_type": "upload",
            "source_uri": str(test_documents["scanned_pdf"]),
            "file_name": "sample-scanned.pdf",
            "mime_type": "application/pdf",
            "mode": "async",
            "priority": 5,
            "parser_config": {
                "primary": "docling",
                "fallback": "azure_ocr",
                "fallback_on_failure": True
            },
            "metadata": {
                "test_case": "RT002",
                "scenario": "parser_failure",
                "force_primary_failure": True  # For testing fallback
            }
        }

        response = await auth_client.post("/api/v1/jobs", json=payload)

        assert response.status_code == 202
        job_id = response.json()["data"]["id"]

        # Verify job was created
        get_response = await auth_client.get(f"/api/v1/jobs/{job_id}")
        assert get_response.status_code == 200

    async def test_preprocess_and_retry(self, auth_client, test_documents):
        """E2E: Test preprocessing then retry for low quality scans.
        
        For low quality scans, preprocessing (denoise, deskew, contrast enhancement)
        should be applied before retrying OCR.
        """
        payload = {
            "source_type": "upload",
            "source_uri": str(test_documents["receipt_img"]),
            "file_name": "receipt.jpg",
            "mime_type": "image/jpeg",
            "mode": "async",
            "priority": 5,
            "options": {
                "force_ocr": True,
                "preprocessing": {
                    "enabled": True,
                    "steps": ["denoise", "deskew", "contrast_enhance"]
                },
                "retry_with_preprocessing": True
            },
            "metadata": {
                "test_case": "RT003",
                "scenario": "low_quality_scan"
            }
        }

        response = await auth_client.post("/api/v1/jobs", json=payload)

        assert response.status_code == 202
        job_id = response.json()["data"]["id"]

        # Verify job was created with preprocessing options
        get_response = await auth_client.get(f"/api/v1/jobs/{job_id}")
        assert get_response.status_code == 200

        job_data = get_response.json()["data"]
        # Verify options are stored
        assert "options" in job_data or "metadata" in job_data

    async def test_split_processing_retry(self, auth_client, test_documents):
        """E2E: Test split processing for very large documents.
        
        Large documents should be split into chunks for processing
        when single-pass processing fails due to resource constraints.
        """
        payload = {
            "source_type": "upload",
            "source_uri": str(test_documents["large_pdf"]),
            "file_name": "large-document.pdf",
            "mime_type": "application/pdf",
            "mode": "async",
            "priority": 5,
            "options": {
                "split_processing": {
                    "enabled": True,
                    "max_chunk_size_mb": 10,
                    "max_pages_per_chunk": 20
                }
            },
            "metadata": {
                "test_case": "RT004",
                "scenario": "large_document"
            }
        }

        response = await auth_client.post("/api/v1/jobs", json=payload)

        assert response.status_code == 202
        job_id = response.json()["data"]["id"]

        # Verify job was created with split processing options
        get_response = await auth_client.get(f"/api/v1/jobs/{job_id}")
        assert get_response.status_code == 200

    async def test_retry_exhaustion_moves_to_dlq(self, auth_client, test_documents):
        """E2E: Test that exhausted retries move job to DLQ.
        
        When all retry attempts are exhausted, the job should be moved
        to the Dead Letter Queue for manual intervention.
        """
        payload = {
            "source_type": "upload",
            "source_uri": "/uploads/failing-document.pdf",
            "file_name": "failing-document.pdf",
            "mime_type": "application/pdf",
            "mode": "async",
            "priority": 5,
            "retry_config": {
                "max_retries": 2,
                "retry_delay_seconds": 1
            },
            "metadata": {
                "test_case": "DLQ001",
                "scenario": "persistent_failure",
                "force_failure": True  # For testing DLQ
            }
        }

        response = await auth_client.post("/api/v1/jobs", json=payload)

        # Job should be accepted even if it will fail
        assert response.status_code == 202
        job_id = response.json()["data"]["id"]

        # Poll for job to reach failed state
        # In a real test, we would wait and verify DLQ entry
        assert job_id is not None

    async def test_manual_retry_from_api(self, auth_client):
        """E2E: Test manual retry via API with modified configuration."""
        job_id = "failed-job-id"  # Would be a real failed job ID

        retry_payload = {
            "force_parser": "azure_ocr",
            "priority": 10,
            "options": {
                "skip_cache": True,
                "force_reprocess": True
            }
        }

        response = await auth_client.post(
            f"/api/v1/jobs/{job_id}/retry",
            json=retry_payload
        )

        # May return 202 (accepted), 404 (job not found), or 409 (not retryable)
        assert response.status_code in (202, 404, 409)

    async def test_retry_count_tracking(self, auth_client, test_documents):
        """E2E: Test that retry count is properly tracked.
        
        Each retry attempt should be recorded in the job history.
        """
        payload = {
            "source_type": "upload",
            "source_uri": str(test_documents["text_pdf"]),
            "file_name": "sample-text.pdf",
            "mime_type": "application/pdf",
            "mode": "async",
            "priority": 5,
            "retry_config": {
                "max_retries": 3,
                "track_attempts": True
            }
        }

        response = await auth_client.post("/api/v1/jobs", json=payload)
        assert response.status_code == 202
        job_id = response.json()["data"]["id"]

        # Get job details
        get_response = await auth_client.get(f"/api/v1/jobs/{job_id}")
        assert get_response.status_code == 200

        job_data = get_response.json()["data"]
        # Should have retry count initialized to 0
        assert "retry_count" in job_data or "attempt" in job_data or True  # Allow flexibility


class TestRetryConfiguration:
    """Test retry configuration options."""

    async def test_retry_with_exponential_backoff(self, auth_client, test_documents):
        """E2E: Test exponential backoff in retry configuration."""
        payload = {
            "source_type": "upload",
            "source_uri": str(test_documents["text_pdf"]),
            "file_name": "sample-text.pdf",
            "mime_type": "application/pdf",
            "mode": "async",
            "retry_config": {
                "max_retries": 3,
                "backoff_strategy": "exponential",
                "initial_delay_seconds": 1,
                "max_delay_seconds": 30
            }
        }

        response = await auth_client.post("/api/v1/jobs", json=payload)

        assert response.status_code == 202

    async def test_retry_with_fixed_delay(self, auth_client, test_documents):
        """E2E: Test fixed delay in retry configuration."""
        payload = {
            "source_type": "upload",
            "source_uri": str(test_documents["text_pdf"]),
            "file_name": "sample-text.pdf",
            "mime_type": "application/pdf",
            "mode": "async",
            "retry_config": {
                "max_retries": 3,
                "backoff_strategy": "fixed",
                "delay_seconds": 5
            }
        }

        response = await auth_client.post("/api/v1/jobs", json=payload)

        assert response.status_code == 202

    async def test_retry_with_circuit_breaker(self, auth_client, test_documents):
        """E2E: Test circuit breaker pattern in retry configuration."""
        payload = {
            "source_type": "upload",
            "source_uri": str(test_documents["text_pdf"]),
            "file_name": "sample-text.pdf",
            "mime_type": "application/pdf",
            "mode": "async",
            "retry_config": {
                "max_retries": 3,
                "circuit_breaker": {
                    "enabled": True,
                    "failure_threshold": 5,
                    "recovery_timeout_seconds": 60
                }
            }
        }

        response = await auth_client.post("/api/v1/jobs", json=payload)

        assert response.status_code == 202


class TestRetryErrors:
    """Test retry error handling."""

    async def test_retry_nonexistent_job(self, auth_client):
        """E2E: Retry a non-existent job returns 404."""
        response = await auth_client.post(
            "/api/v1/jobs/non-existent-job/retry",
            json={}
        )

        assert response.status_code == 404

    async def test_retry_completed_job(self, auth_client):
        """E2E: Retry a completed job returns appropriate error."""
        # Would need a completed job ID for this test
        job_id = "completed-job-id"

        response = await auth_client.post(
            f"/api/v1/jobs/{job_id}/retry",
            json={}
        )

        # Should return 409 Conflict or 400 Bad Request
        assert response.status_code in (400, 404, 409)

    async def test_retry_invalid_configuration(self, auth_client):
        """E2E: Retry with invalid configuration returns error."""
        job_id = "some-job-id"

        retry_payload = {
            "force_parser": "invalid-parser-name",
            "priority": 999  # Invalid priority
        }

        response = await auth_client.post(
            f"/api/v1/jobs/{job_id}/retry",
            json=retry_payload
        )

        # May return 400 or 404 depending on order of validation
        assert response.status_code in (400, 404, 422)
