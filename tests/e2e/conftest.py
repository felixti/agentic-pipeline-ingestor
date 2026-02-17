"""Pytest configuration for E2E tests.

This module provides fixtures and configuration for E2E testing of the
Agentic Data Pipeline Ingestor following shift-left engineering principles.
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional
from urllib.parse import urljoin

import pytest
import pytest_asyncio

# Import httpx for HTTP client
pytest.importorskip("httpx")
import httpx


# =============================================================================
# Configuration
# =============================================================================

E2E_BASE_URL = os.getenv("E2E_BASE_URL", "http://localhost:8000")
E2E_API_KEY = os.getenv("E2E_API_KEY", "test-api-key")
E2E_ADMIN_API_KEY = os.getenv("E2E_ADMIN_API_KEY", "admin-api-key")
E2E_VIEWER_API_KEY = os.getenv("E2E_VIEWER_API_KEY", "viewer-api-key")
E2E_REQUEST_TIMEOUT = int(os.getenv("E2E_REQUEST_TIMEOUT", "30"))
E2E_POLL_INTERVAL = float(os.getenv("E2E_POLL_INTERVAL", "2.0"))
E2E_MAX_POLL_ATTEMPTS = int(os.getenv("E2E_MAX_POLL_ATTEMPTS", "60"))

FIXTURES_DIR = Path(__file__).parent / "fixtures"
DATASETS_DIR = FIXTURES_DIR / "datasets"
DOCUMENTS_DIR = FIXTURES_DIR / "documents"


# =============================================================================
# Event Loop Configuration
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# HTTP Client Fixtures
# =============================================================================

@pytest_asyncio.fixture(scope="session")
async def http_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create a shared HTTP client for the test session."""
    async with httpx.AsyncClient(
        base_url=E2E_BASE_URL,
        timeout=E2E_REQUEST_TIMEOUT,
        headers={"Accept": "application/json"},
    ) as client:
        yield client


@pytest_asyncio.fixture
async def client(http_client: httpx.AsyncClient) -> httpx.AsyncClient:
    """Get the HTTP client (alias for http_client)."""
    return http_client


@pytest_asyncio.fixture
async def auth_client(http_client: httpx.AsyncClient) -> httpx.AsyncClient:
    """Get an authenticated HTTP client with standard API key."""
    http_client.headers["X-API-Key"] = E2E_API_KEY
    return http_client


@pytest_asyncio.fixture
async def admin_client(http_client: httpx.AsyncClient) -> httpx.AsyncClient:
    """Get an authenticated HTTP client with admin API key."""
    http_client.headers["X-API-Key"] = E2E_ADMIN_API_KEY
    return http_client


@pytest_asyncio.fixture
async def viewer_client(http_client: httpx.AsyncClient) -> httpx.AsyncClient:
    """Get an authenticated HTTP client with viewer API key."""
    http_client.headers["X-API-Key"] = E2E_VIEWER_API_KEY
    return http_client


@pytest_asyncio.fixture
async def unauth_client(http_client: httpx.AsyncClient) -> httpx.AsyncClient:
    """Get an unauthenticated HTTP client."""
    # Remove any existing API key header
    http_client.headers.pop("X-API-Key", None)
    http_client.headers.pop("Authorization", None)
    return http_client


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def test_dataset() -> Dict[str, Any]:
    """Load the test dataset configuration."""
    dataset_path = DATASETS_DIR / "test-dataset.json"
    if dataset_path.exists():
        with open(dataset_path) as f:
            return json.load(f)
    return {"test_cases": [], "retry_test_cases": [], "dlq_test_cases": []}


@pytest.fixture
def test_documents() -> Dict[str, Path]:
    """Get paths to test documents."""
    return {
        "text_pdf": DOCUMENTS_DIR / "pdf" / "sample-text.pdf",
        "scanned_pdf": DOCUMENTS_DIR / "pdf" / "sample-scanned.pdf",
        "mixed_pdf": DOCUMENTS_DIR / "pdf" / "sample-mixed.pdf",
        "large_pdf": DOCUMENTS_DIR / "pdf" / "large-document.pdf",
        "word_doc": DOCUMENTS_DIR / "office" / "sample.docx",
        "excel_doc": DOCUMENTS_DIR / "office" / "sample.xlsx",
        "ppt_doc": DOCUMENTS_DIR / "office" / "sample.pptx",
        "receipt_img": DOCUMENTS_DIR / "images" / "receipt.jpg",
        "document_img": DOCUMENTS_DIR / "images" / "document.png",
        "archive": DOCUMENTS_DIR / "archives" / "documents.zip",
    }


@pytest.fixture
def sample_job_payload() -> Dict[str, Any]:
    """Get a sample job creation payload."""
    return {
        "source_type": "upload",
        "source_uri": "/uploads/sample-text.pdf",
        "file_name": "sample-text.pdf",
        "mime_type": "application/pdf",
        "mode": "async",
        "priority": 5,
        "metadata": {
            "source": "e2e-test",
            "test_case": "sample-job"
        }
    }


# =============================================================================
# Helper Functions
# =============================================================================

async def wait_for_job_completion(
    client: httpx.AsyncClient,
    job_id: str,
    max_attempts: int = E2E_MAX_POLL_ATTEMPTS,
    interval: float = E2E_POLL_INTERVAL,
) -> Dict[str, Any]:
    """Wait for a job to complete.
    
    Args:
        client: HTTP client
        job_id: ID of the job to wait for
        max_attempts: Maximum number of polling attempts
        interval: Seconds between polling attempts
        
    Returns:
        Final job status
        
    Raises:
        TimeoutError: If job doesn't complete within max_attempts
    """
    for attempt in range(max_attempts):
        response = await client.get(f"/api/v1/jobs/{job_id}")
        
        if response.status_code == 404:
            raise ValueError(f"Job {job_id} not found")
        
        response.raise_for_status()
        data = response.json()
        
        job_data = data.get("data", {})
        status = job_data.get("status", "unknown")
        
        if status in ("completed", "failed", "cancelled"):
            return job_data
        
        await asyncio.sleep(interval)
    
    raise TimeoutError(f"Job {job_id} did not complete within {max_attempts} attempts")


async def create_job(
    client: httpx.AsyncClient,
    payload: Dict[str, Any],
) -> str:
    """Create a job and return the job ID.
    
    Args:
        client: HTTP client
        payload: Job creation payload
        
    Returns:
        Job ID
    """
    response = await client.post("/api/v1/jobs", json=payload)
    response.raise_for_status()
    
    data = response.json()
    return data["data"]["id"]


async def upload_file(
    client: httpx.AsyncClient,
    file_path: Path,
    metadata: Optional[Dict[str, Any]] = None,
    options: Optional[Dict[str, Any]] = None,
) -> str:
    """Upload a file and return the job ID.
    
    Args:
        client: HTTP client
        file_path: Path to file to upload
        metadata: Optional metadata
        options: Optional processing options
        
    Returns:
        Job ID
    """
    files = {"file": (file_path.name, open(file_path, "rb"), "application/octet-stream")}
    data = {}
    
    if metadata:
        data["metadata"] = json.dumps(metadata)
    if options:
        data["options"] = json.dumps(options)
    
    response = await client.post("/api/v1/upload", files=files, data=data)
    response.raise_for_status()
    
    return response.json()["data"]["job_id"]


# =============================================================================
# Custom Markers
# =============================================================================

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow tests that may take a while")
    config.addinivalue_line("markers", "auth: Authentication tests")
    config.addinivalue_line("markers", "retry: Retry mechanism tests")
    config.addinivalue_line("markers", "dlq: Dead Letter Queue tests")


# =============================================================================
# Test Helpers
# =============================================================================

class JobHelper:
    """Helper class for job-related operations."""
    
    def __init__(self, client: httpx.AsyncClient):
        self.client = client
    
    async def create(self, payload: Dict[str, Any]) -> str:
        """Create a job."""
        return await create_job(self.client, payload)
    
    async def get(self, job_id: str) -> Dict[str, Any]:
        """Get job details."""
        response = await self.client.get(f"/api/v1/jobs/{job_id}")
        response.raise_for_status()
        return response.json()["data"]
    
    async def wait_for_completion(
        self,
        job_id: str,
        timeout: int = 120,
    ) -> Dict[str, Any]:
        """Wait for job completion."""
        return await wait_for_job_completion(
            self.client,
            job_id,
            max_attempts=int(timeout / E2E_POLL_INTERVAL),
        )
    
    async def cancel(self, job_id: str) -> None:
        """Cancel a job."""
        response = await self.client.delete(f"/api/v1/jobs/{job_id}")
        response.raise_for_status()
    
    async def retry(self, job_id: str, options: Optional[Dict] = None) -> str:
        """Retry a failed job."""
        response = await self.client.post(
            f"/api/v1/jobs/{job_id}/retry",
            json=options or {}
        )
        response.raise_for_status()
        return response.json()["data"]["id"]
    
    async def get_result(self, job_id: str) -> Dict[str, Any]:
        """Get job result."""
        response = await self.client.get(f"/api/v1/jobs/{job_id}/result")
        response.raise_for_status()
        return response.json()["data"]


@pytest.fixture
def job_helper(auth_client: httpx.AsyncClient) -> JobHelper:
    """Get a JobHelper instance."""
    return JobHelper(auth_client)


# =============================================================================
# Assertions
# =============================================================================

class E2EAssertions:
    """Custom assertion helpers for E2E tests."""
    
    @staticmethod
    def assert_job_completed(job_data: Dict[str, Any]) -> None:
        """Assert that a job completed successfully."""
        assert job_data.get("status") == "completed", \
            f"Expected job status 'completed', got '{job_data.get('status')}'"
    
    @staticmethod
    def assert_job_failed(job_data: Dict[str, Any]) -> None:
        """Assert that a job failed."""
        assert job_data.get("status") == "failed", \
            f"Expected job status 'failed', got '{job_data.get('status')}'"
    
    @staticmethod
    def assert_has_text(result: Dict[str, Any]) -> None:
        """Assert that result contains text."""
        text = result.get("text", "")
        assert text and len(text) > 0, "Expected result to have text content"
    
    @staticmethod
    def assert_quality_score(result: Dict[str, Any], min_score: float = 0.8) -> None:
        """Assert that quality score meets minimum."""
        score = result.get("quality_score", 0)
        assert score >= min_score, \
            f"Expected quality score >= {min_score}, got {score}"
    
    @staticmethod
    def assert_response_time(response_time_ms: float, max_ms: float = 2000) -> None:
        """Assert that response time is within limit."""
        assert response_time_ms < max_ms, \
            f"Expected response time < {max_ms}ms, got {response_time_ms}ms"


@pytest.fixture
def assert_helper() -> E2EAssertions:
    """Get E2EAssertions helper."""
    return E2EAssertions()
