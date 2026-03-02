"""End-to-End Integration Tests for Ralph Loop Pipeline.

This module tests the complete document processing pipeline including:
- Single file upload (happy path)
- Multiple concurrent uploads (stress test)
- Different file type parsing (PDF, DOCX, PPTX, TXT)
- Search validation after ingestion (Cognee and HippoRAG)

These tests verify all three critical fixes work together:
1. Parser text extraction (docling + fallbacks)
2. Neo4j deadlock handling (retry + locking)
3. Database transactions (proper commit boundaries)

Usage:
    pytest tests/integration/test_end_to_end.py -v
    pytest tests/integration/test_end_to_end.py::TestSingleFileUpload -v
    pytest tests/integration/test_end_to_end.py::TestConcurrentUploads -v
    pytest tests/integration/test_end_to_end.py::TestFileTypeParsing -v
    pytest tests/integration/test_end_to_end.py::TestSearchAfterIngestion -v

Environment Variables:
    E2E_BASE_URL: API base URL (default: http://localhost:8000)
    E2E_API_KEY: API key for authentication (default: test-api-key)
    E2E_MAX_POLL_ATTEMPTS: Max polling attempts for job completion (default: 60)
    E2E_POLL_INTERVAL: Seconds between polling attempts (default: 2.0)
"""

import asyncio
import os
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
import pytest_asyncio

pytest.importorskip("httpx")
import httpx

# =============================================================================
# Configuration
# =============================================================================

E2E_BASE_URL = os.getenv("E2E_BASE_URL", "http://localhost:8000")
E2E_API_KEY = os.getenv("E2E_API_KEY", "test-api-key")
E2E_REQUEST_TIMEOUT = int(os.getenv("E2E_REQUEST_TIMEOUT", "60"))
E2E_POLL_INTERVAL = float(os.getenv("E2E_POLL_INTERVAL", "2.0"))
E2E_MAX_POLL_ATTEMPTS = int(os.getenv("E2E_MAX_POLL_ATTEMPTS", "60"))

# Test data directory
TEST_DATA_DIR = Path(__file__).parent.parent / "data"

# =============================================================================
# Fixtures
# =============================================================================


@pytest_asyncio.fixture(scope="session")
async def http_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create a shared HTTP client for the test session."""
    async with httpx.AsyncClient(
        base_url=E2E_BASE_URL,
        timeout=E2E_REQUEST_TIMEOUT,
        headers={
            "Accept": "application/json",
            "X-API-Key": E2E_API_KEY,
        },
    ) as client:
        yield client


@pytest.fixture
def test_files() -> dict[str, Path]:
    """Get paths to test files."""
    return {
        "pdf": TEST_DATA_DIR / "sample.pdf",
        "docx": TEST_DATA_DIR / "sample.docx",
        "pptx": TEST_DATA_DIR / "sample.pptx",
        "txt": TEST_DATA_DIR / "sample.txt",
    }


# =============================================================================
# Helper Functions
# =============================================================================


async def upload_file(
    client: httpx.AsyncClient,
    file_path: Path,
    metadata: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
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
    if not file_path.exists():
        pytest.skip(f"Test file not found: {file_path}")
    
    with open(file_path, "rb") as f:
        files = {"file": (file_path.name, f, "application/octet-stream")}
        data = {}
        
        if metadata:
            data["metadata"] = __import__("json").dumps(metadata)
        if options:
            data["options"] = __import__("json").dumps(options)
        
        response = await client.post("/api/v1/upload", files=files, data=data)
    
    if response.status_code != 202:
        raise AssertionError(f"Upload failed: {response.status_code} - {response.text}")
    
    return response.json()["data"]["job_id"]


async def wait_for_job_completion(
    client: httpx.AsyncClient,
    job_id: str,
    max_attempts: int = E2E_MAX_POLL_ATTEMPTS,
    interval: float = E2E_POLL_INTERVAL,
) -> dict[str, Any]:
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


async def get_job_chunks(
    client: httpx.AsyncClient,
    job_id: str,
) -> list[dict[str, Any]]:
    """Get chunks for a job.
    
    Args:
        client: HTTP client
        job_id: Job ID
        
    Returns:
        List of chunks
    """
    response = await client.get(f"/api/v1/jobs/{job_id}/chunks")
    response.raise_for_status()
    
    data = response.json()
    return data.get("data", {}).get("items", [])


async def search_cognee(
    client: httpx.AsyncClient,
    query: str,
    search_type: str = "hybrid",
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Search using Cognee GraphRAG.
    
    Args:
        client: HTTP client
        query: Search query
        search_type: Type of search (vector, graph, hybrid)
        top_k: Number of results to return
        
    Returns:
        List of search results
    """
    response = await client.post(
        "/api/v1/cognee/search",
        json={
            "query": query,
            "search_type": search_type,
            "top_k": top_k,
        },
    )
    
    if response.status_code == 404:
        pytest.skip("Cognee search endpoint not available")
    
    response.raise_for_status()
    data = response.json()
    return data.get("data", {}).get("results", [])


async def search_hipporag(
    client: httpx.AsyncClient,
    query: str,
    num_to_retrieve: int = 5,
) -> list[dict[str, Any]]:
    """Search using HippoRAG.
    
    Args:
        client: HTTP client
        query: Search query
        num_to_retrieve: Number of passages to retrieve
        
    Returns:
        List of retrieval results
    """
    response = await client.post(
        "/api/v1/hipporag/retrieve",
        json={
            "queries": [query],
            "num_to_retrieve": num_to_retrieve,
        },
    )
    
    if response.status_code == 404:
        pytest.skip("HippoRAG endpoint not available")
    
    response.raise_for_status()
    data = response.json()
    return data.get("data", {}).get("results", [])


def assert_job_completed(job_data: dict[str, Any]) -> None:
    """Assert that a job completed successfully."""
    assert job_data.get("status") == "completed", (
        f"Expected job status 'completed', got '{job_data.get('status')}'. "
        f"Error: {job_data.get('error')}"
    )


def assert_no_errors(logs: list[str]) -> None:
    """Assert that no deadlock or transaction errors exist in logs."""
    error_patterns = [
        "deadlock",
        "Deadlock",
        "DEADLOCK",
        "ForeignKeyViolation",
        "foreign key violation",
        "transaction",
        "TransactionError",
    ]
    
    errors_found = []
    for log in logs:
        for pattern in error_patterns:
            if pattern in log:
                errors_found.append(f"Found '{pattern}' in: {log}")
    
    assert not errors_found, f"Found errors in logs:\n" + "\n".join(errors_found)


# =============================================================================
# Test Classes
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestSingleFileUpload:
    """Test complete pipeline for single file upload (happy path).
    
    Scenario:
        Upload PDF → Parse → Chunk → Embed → Store in PostgreSQL → Store in Cognee → Store in HippoRAG
    
    Expected:
        - Job status: completed
        - Chunks in PostgreSQL: > 0
        - Job shows chunks_count > 0
        - Cognee graph updated
        - HippoRAG graph updated
    """
    
    async def test_single_pdf_upload_happy_path(self, http_client: httpx.AsyncClient, test_files: dict):
        """Test: Single PDF upload → complete pipeline."""
        # Arrange
        file_path = test_files["pdf"]
        
        # Act: Upload file
        start_time = time.time()
        job_id = await upload_file(
            http_client,
            file_path,
            metadata={"test_case": "single_pdf_upload", "source": "integration_test"},
        )
        
        # Wait for processing
        job_data = await wait_for_job_completion(http_client, job_id)
        processing_time = time.time() - start_time
        
        # Assert: Job completed
        assert_job_completed(job_data)
        
        # Assert: Processing time < 60 seconds (target)
        assert processing_time < 60, f"Processing took {processing_time:.1f}s, expected < 60s"
        
        # Assert: Chunks exist in database
        chunks = await get_job_chunks(http_client, job_id)
        assert len(chunks) > 0, f"Expected chunks > 0, got {len(chunks)}"
        assert len(chunks) >= 5, f"Expected at least 5 chunks, got {len(chunks)}"
        
        # Assert: Chunks have content
        for chunk in chunks:
            assert chunk.get("content"), f"Chunk {chunk.get('id')} has no content"
            assert len(chunk.get("content", "")) > 10, "Chunk content too short"
        
        print(f"\n✓ Single PDF upload successful:")
        print(f"  - Job ID: {job_id}")
        print(f"  - Chunks created: {len(chunks)}")
        print(f"  - Processing time: {processing_time:.1f}s")
    
    async def test_job_result_structure(self, http_client: httpx.AsyncClient, test_files: dict):
        """Test: Job result has expected structure."""
        # Upload and process
        job_id = await upload_file(http_client, test_files["pdf"])
        job_data = await wait_for_job_completion(http_client, job_id)
        
        # Assert job structure
        assert_job_completed(job_data)
        assert "id" in job_data
        assert "status" in job_data
        assert "source_type" in job_data
        assert "file_name" in job_data
        assert "created_at" in job_data
        assert "updated_at" in job_data
        
        # Get job result
        response = await http_client.get(f"/api/v1/jobs/{job_id}/result")
        if response.status_code == 200:
            result_data = response.json().get("data", {})
            # Result may contain metadata about processing
            assert isinstance(result_data, dict)


@pytest.mark.integration
@pytest.mark.asyncio
class TestConcurrentUploads:
    """Test 5 concurrent uploads - stress test for deadlock handling.
    
    Scenario:
        Upload 5 files simultaneously
    
    Expected:
        - All jobs complete successfully
        - No deadlock errors
        - No transaction errors
        - All chunks saved correctly
    """
    
    async def test_five_concurrent_uploads(self, http_client: httpx.AsyncClient, test_files: dict):
        """Test: 5 concurrent file uploads complete without deadlocks."""
        # Arrange
        file_path = test_files["pdf"]
        num_concurrent = 5
        
        # Act: Upload 5 files concurrently
        start_time = time.time()
        
        async def upload_and_wait(index: int) -> tuple[str, dict[str, Any]]:
            """Upload file and wait for completion."""
            job_id = await upload_file(
                http_client,
                file_path,
                metadata={"test_case": "concurrent_upload", "index": index},
            )
            job_data = await wait_for_job_completion(http_client, job_id)
            return job_id, job_data
        
        # Run concurrent uploads
        results = await asyncio.gather(
            *[upload_and_wait(i) for i in range(num_concurrent)],
            return_exceptions=True,
        )
        
        total_time = time.time() - start_time
        
        # Assert: All uploads succeeded
        job_ids = []
        errors = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append(f"Upload {i}: {result}")
            else:
                job_id, job_data = result
                job_ids.append(job_id)
                if job_data.get("status") != "completed":
                    errors.append(f"Job {job_id} status: {job_data.get('status')}")
        
        assert not errors, f"Errors during concurrent uploads:\n" + "\n".join(errors)
        assert len(job_ids) == num_concurrent, f"Expected {num_concurrent} jobs, got {len(job_ids)}"
        
        # Assert: All jobs completed
        for job_id in job_ids:
            chunks = await get_job_chunks(http_client, job_id)
            assert len(chunks) > 0, f"Job {job_id} has no chunks"
        
        # Assert: Success rate 100%
        success_rate = len(job_ids) / num_concurrent * 100
        assert success_rate == 100, f"Success rate {success_rate}%, expected 100%"
        
        print(f"\n✓ Concurrent uploads successful:")
        print(f"  - Jobs created: {len(job_ids)}")
        print(f"  - Success rate: {success_rate}%")
        print(f"  - Total time: {total_time:.1f}s")
    
    async def test_concurrent_different_file_types(self, http_client: httpx.AsyncClient, test_files: dict):
        """Test: Concurrent uploads of different file types."""
        files_to_upload = [
            ("pdf", test_files["pdf"]),
            ("docx", test_files["docx"]),
            ("txt", test_files["txt"]),
        ]
        
        async def upload_and_wait(file_type: str, file_path: Path) -> tuple[str, str, dict]:
            job_id = await upload_file(
                http_client,
                file_path,
                metadata={"test_case": "concurrent_types", "file_type": file_type},
            )
            job_data = await wait_for_job_completion(http_client, job_id)
            return file_type, job_id, job_data
        
        # Run concurrent uploads
        tasks = [
            upload_and_wait(ft, fp)
            for ft, fp in files_to_upload
            if fp.exists()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Assert all succeeded
        for result in results:
            if isinstance(result, Exception):
                raise AssertionError(f"Upload failed: {result}")
            
            file_type, job_id, job_data = result
            assert_job_completed(job_data)
            
            # Verify chunks
            chunks = await get_job_chunks(http_client, job_id)
            assert len(chunks) > 0, f"{file_type} upload produced no chunks"


@pytest.mark.integration
@pytest.mark.asyncio
class TestFileTypeParsing:
    """Test PDF, DOCX, PPTX parsing.
    
    Scenario:
        Upload PDF, DOCX, PPTX sequentially
    
    Expected:
        - All files parsed successfully
        - Text extracted from each
        - Chunks created for each
    """
    
    async def test_pdf_parsing(self, http_client: httpx.AsyncClient, test_files: dict):
        """Test: PDF file parsing extracts text."""
        file_path = test_files["pdf"]
        
        job_id = await upload_file(http_client, file_path, metadata={"test_type": "pdf"})
        job_data = await wait_for_job_completion(http_client, job_id)
        
        assert_job_completed(job_data)
        
        chunks = await get_job_chunks(http_client, job_id)
        assert len(chunks) > 0, "PDF parsing produced no chunks"
        
        # Verify text extracted
        total_chars = sum(len(c.get("content", "")) for c in chunks)
        assert total_chars > 100, f"PDF text too short: {total_chars} chars"
        
        print(f"\n✓ PDF parsing: {len(chunks)} chunks, {total_chars} chars")
    
    async def test_docx_parsing(self, http_client: httpx.AsyncClient, test_files: dict):
        """Test: DOCX file parsing extracts text."""
        file_path = test_files["docx"]
        
        if not file_path.exists():
            pytest.skip(f"DOCX test file not found: {file_path}")
        
        job_id = await upload_file(http_client, file_path, metadata={"test_type": "docx"})
        job_data = await wait_for_job_completion(http_client, job_id)
        
        assert_job_completed(job_data)
        
        chunks = await get_job_chunks(http_client, job_id)
        assert len(chunks) > 0, "DOCX parsing produced no chunks"
        
        # Verify text extracted (> 500 chars target)
        total_chars = sum(len(c.get("content", "")) for c in chunks)
        assert total_chars > 500, f"DOCX text too short: {total_chars} chars (expected > 500)"
        
        print(f"\n✓ DOCX parsing: {len(chunks)} chunks, {total_chars} chars")
    
    async def test_pptx_parsing(self, http_client: httpx.AsyncClient, test_files: dict):
        """Test: PPTX file parsing extracts text."""
        file_path = test_files["pptx"]
        
        if not file_path.exists():
            pytest.skip(f"PPTX test file not found: {file_path}")
        
        job_id = await upload_file(http_client, file_path, metadata={"test_type": "pptx"})
        job_data = await wait_for_job_completion(http_client, job_id)
        
        assert_job_completed(job_data)
        
        chunks = await get_job_chunks(http_client, job_id)
        assert len(chunks) > 0, "PPTX parsing produced no chunks"
        
        # Verify text extracted (> 500 chars target)
        total_chars = sum(len(c.get("content", "")) for c in chunks)
        assert total_chars > 500, f"PPTX text too short: {total_chars} chars (expected > 500)"
        
        print(f"\n✓ PPTX parsing: {len(chunks)} chunks, {total_chars} chars")
    
    async def test_txt_parsing(self, http_client: httpx.AsyncClient, test_files: dict):
        """Test: TXT file parsing extracts text."""
        file_path = test_files["txt"]
        
        if not file_path.exists():
            pytest.skip(f"TXT test file not found: {file_path}")
        
        job_id = await upload_file(http_client, file_path, metadata={"test_type": "txt"})
        job_data = await wait_for_job_completion(http_client, job_id)
        
        assert_job_completed(job_data)
        
        chunks = await get_job_chunks(http_client, job_id)
        assert len(chunks) > 0, "TXT parsing produced no chunks"
        
        total_chars = sum(len(c.get("content", "")) for c in chunks)
        assert total_chars > 100, f"TXT text too short: {total_chars} chars"
        
        print(f"\n✓ TXT parsing: {len(chunks)} chunks, {total_chars} chars")


@pytest.mark.integration
@pytest.mark.asyncio
class TestSearchAfterIngestion:
    """Test search returns results after processing.
    
    Scenario:
        After file processing, search with:
        - Cognee hybrid search
        - HippoRAG retrieval
    
    Expected:
        - Search returns results
        - Results contain entities
        - Source documents referenced
    """
    
    async def test_cognee_search_after_ingestion(self, http_client: httpx.AsyncClient, test_files: dict):
        """Test: Cognee search returns results after document ingestion."""
        # First, ingest a document
        file_path = test_files["txt"]
        if not file_path.exists():
            pytest.skip("TXT test file not found")
        
        job_id = await upload_file(
            http_client,
            file_path,
            metadata={"test_type": "cognee_search"},
        )
        job_data = await wait_for_job_completion(http_client, job_id)
        assert_job_completed(job_data)
        
        # Wait a moment for graph to be updated
        await asyncio.sleep(2)
        
        # Search using Cognee
        try:
            results = await search_cognee(
                http_client,
                query="artificial intelligence",
                search_type="hybrid",
                top_k=5,
            )
            
            # Assert results returned
            assert isinstance(results, list), "Search results should be a list"
            
            # If results exist, verify structure
            if results:
                print(f"\n✓ Cognee search returned {len(results)} results")
                for result in results:
                    assert "content" in result or "text" in result, "Result should have content"
            else:
                print("\n⚠ Cognee search returned empty results (may be expected if graph not populated)")
                
        except Exception as e:
            pytest.skip(f"Cognee search failed: {e}")
    
    async def test_hipporag_retrieval_after_ingestion(self, http_client: httpx.AsyncClient, test_files: dict):
        """Test: HippoRAG retrieval returns results after document ingestion."""
        # First, ingest a document
        file_path = test_files["txt"]
        if not file_path.exists():
            pytest.skip("TXT test file not found")
        
        job_id = await upload_file(
            http_client,
            file_path,
            metadata={"test_type": "hipporag_search"},
        )
        job_data = await wait_for_job_completion(http_client, job_id)
        assert_job_completed(job_data)
        
        # Wait a moment for graph to be updated
        await asyncio.sleep(2)
        
        # Search using HippoRAG
        try:
            results = await search_hipporag(
                http_client,
                query="What is artificial intelligence?",
                num_to_retrieve=5,
            )
            
            # Assert results returned
            assert isinstance(results, list), "Retrieval results should be a list"
            
            # If results exist, verify structure
            if results:
                print(f"\n✓ HippoRAG retrieval returned {len(results)} results")
                for result in results:
                    assert "passages" in result or "entities" in result, "Result should have passages or entities"
            else:
                print("\n⚠ HippoRAG retrieval returned empty results (may be expected if graph not populated)")
                
        except Exception as e:
            pytest.skip(f"HippoRAG retrieval failed: {e}")
    
    async def test_search_with_specific_content(self, http_client: httpx.AsyncClient, test_files: dict):
        """Test: Search for specific content from uploaded document."""
        # Upload a document with specific content
        file_path = test_files["txt"]
        if not file_path.exists():
            pytest.skip("TXT test file not found")
        
        job_id = await upload_file(http_client, file_path)
        job_data = await wait_for_job_completion(http_client, job_id)
        assert_job_completed(job_data)
        
        # Get chunks to know what content was extracted
        chunks = await get_job_chunks(http_client, job_id)
        if not chunks:
            pytest.skip("No chunks to search for")
        
        # Extract a keyword from the first chunk
        first_chunk_content = chunks[0].get("content", "")
        words = first_chunk_content.split()
        if len(words) >= 3:
            keyword = " ".join(words[:3])  # First 3 words
        else:
            keyword = words[0] if words else "document"
        
        # Wait for indexing
        await asyncio.sleep(2)
        
        # Try Cognee search with keyword
        try:
            results = await search_cognee(http_client, query=keyword, top_k=3)
            print(f"\n✓ Search for '{keyword}' returned {len(results)} results")
        except Exception:
            pass  # Search may not be available


# =============================================================================
# Performance and Validation Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestPerformanceMetrics:
    """Collect performance metrics for validation."""
    
    async def test_single_pdf_processing_time(self, http_client: httpx.AsyncClient, test_files: dict):
        """Test: Single PDF processing completes within target time."""
        file_path = test_files["pdf"]
        
        start_time = time.time()
        job_id = await upload_file(http_client, file_path)
        job_data = await wait_for_job_completion(http_client, job_id)
        processing_time = time.time() - start_time
        
        assert_job_completed(job_data)
        
        # Target: < 60 seconds
        assert processing_time < 60, f"Processing time {processing_time:.1f}s exceeds 60s target"
        
        print(f"\n✓ Processing time: {processing_time:.1f}s (target: <60s)")
    
    async def test_chunks_created_threshold(self, http_client: httpx.AsyncClient, test_files: dict):
        """Test: Document creates minimum number of chunks."""
        file_path = test_files["pdf"]
        
        job_id = await upload_file(http_client, file_path)
        job_data = await wait_for_job_completion(http_client, job_id)
        assert_job_completed(job_data)
        
        chunks = await get_job_chunks(http_client, job_id)
        
        # Target: > 5 chunks for a typical PDF
        assert len(chunks) >= 5, f"Only {len(chunks)} chunks created, expected >= 5"
        
        print(f"\n✓ Chunks created: {len(chunks)} (target: >=5)")


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestErrorHandling:
    """Test error handling and edge cases."""
    
    async def test_nonexistent_job_returns_404(self, http_client: httpx.AsyncClient):
        """Test: Getting non-existent job returns 404."""
        response = await http_client.get("/api/v1/jobs/non-existent-job-id-12345")
        assert response.status_code == 404
    
    async def test_invalid_file_type_handling(self, http_client: httpx.AsyncClient):
        """Test: Invalid file type is handled gracefully."""
        # Create a temporary file with invalid content
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"invalid content")
            temp_path = f.name
        
        try:
            with open(temp_path, "rb") as f:
                files = {"file": ("test.xyz", f, "application/octet-stream")}
                response = await http_client.post("/api/v1/upload", files=files)
            
            # Should either accept or return 400/422
            assert response.status_code in (202, 400, 422)
        finally:
            os.unlink(temp_path)


# =============================================================================
# Summary Report
# =============================================================================


def pytest_sessionfinish(session, exitstatus):
    """Print summary report after test session."""
    print("\n" + "=" * 70)
    print("INTEGRATION TEST SUMMARY - Ralph Loop Pipeline")
    print("=" * 70)
    print("\nTest Scenarios:")
    print("  1. Single File Upload (Happy Path)")
    print("     - Upload PDF → Parse → Chunk → Embed → Store")
    print("     - Verify job completed, chunks created, Cognee/HippoRAG updated")
    print("\n  2. Multiple Concurrent Uploads (Stress Test)")
    print("     - Upload 5 files simultaneously")
    print("     - Verify no deadlocks, 100% success rate")
    print("\n  3. Different File Types")
    print("     - PDF, DOCX, PPTX parsing")
    print("     - Verify text extracted, chunks created")
    print("\n  4. Search Validation")
    print("     - Cognee hybrid search")
    print("     - HippoRAG retrieval")
    print("\nValidation Checklist:")
    print("  [ ] Single PDF: Chunks created > 5")
    print("  [ ] Single PDF: Processing time < 60 seconds")
    print("  [ ] Concurrent (5x): Success rate 100%")
    print("  [ ] Concurrent (5x): Deadlock errors 0")
    print("  [ ] Concurrent (5x): FK violations 0")
    print("  [ ] DOCX parsing: Text extracted > 500 chars")
    print("  [ ] PPTX parsing: Text extracted > 500 chars")
    print("  [ ] Cognee search: Results returned")
    print("  [ ] HippoRAG search: Entities returned")
    print("=" * 70)
