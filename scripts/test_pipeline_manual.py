#!/usr/bin/env python3
"""
Manual pipeline test script for end-to-end verification.

This script provides a command-line interface for testing the document
processing pipeline manually. It supports:
- Single file upload and processing verification
- Batch file processing
- Job status monitoring
- Results verification

Usage:
    python scripts/test_pipeline_manual.py --file tests/data/sample.pdf
    python scripts/test_pipeline_manual.py --file tests/data/sample.pdf --wait
    python scripts/test_pipeline_manual.py --dir tests/data/ --pattern "*.pdf"
    python scripts/test_pipeline_manual.py --job-id <uuid> --check

Environment Variables:
    API_URL: API base URL (default: https://pipeline-api.felixtek.cloud/api/v1)
    API_KEY: API key for authentication (default: test-api-key-123)
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_API_URL = os.getenv("API_URL", "https://pipeline-api.felixtek.cloud/api/v1")
DEFAULT_API_KEY = os.getenv("API_KEY", "test-api-key-123")
DEFAULT_TIMEOUT = 60
DEFAULT_POLL_INTERVAL = 2.0
DEFAULT_MAX_POLLS = 60

# Colors for terminal output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"


# =============================================================================
# Helper Functions
# =============================================================================

def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.END}\n")


def print_success(text: str) -> None:
    """Print a success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text: str) -> None:
    """Print an error message."""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_info(text: str) -> None:
    """Print an info message."""
    print(f"{Colors.CYAN}ℹ {text}{Colors.END}")


def print_warning(text: str) -> None:
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def format_bytes(size: int) -> str:
    """Format bytes to human readable."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"


# =============================================================================
# API Client
# =============================================================================

class PipelineApiClient:
    """Client for the pipeline API."""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=DEFAULT_TIMEOUT,
            headers={
                "Accept": "application/json",
                "X-API-Key": api_key,
            },
        )
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def health_check(self) -> dict[str, Any]:
        """Check API health."""
        response = await self.client.get("/health")
        response.raise_for_status()
        return response.json()
    
    async def upload_file(
        self,
        file_path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Upload a file and return job ID.
        
        Args:
            file_path: Path to file
            metadata: Optional metadata
            
        Returns:
            Job ID
        """
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "application/octet-stream")}
            data = {}
            if metadata:
                data["metadata"] = json.dumps(metadata)
            
            response = await self.client.post("/api/v1/upload", files=files, data=data)
        
        response.raise_for_status()
        result = response.json()
        return result["data"]["job_id"]
    
    async def get_job(self, job_id: str) -> dict[str, Any]:
        """Get job details.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job data
        """
        response = await self.client.get(f"/api/v1/jobs/{job_id}")
        response.raise_for_status()
        return response.json()["data"]
    
    async def get_job_chunks(self, job_id: str) -> list[dict[str, Any]]:
        """Get chunks for a job.
        
        Args:
            job_id: Job ID
            
        Returns:
            List of chunks
        """
        response = await self.client.get(f"/api/v1/jobs/{job_id}/chunks")
        if response.status_code == 404:
            return []
        response.raise_for_status()
        return response.json().get("data", {}).get("items", [])
    
    async def wait_for_job(
        self,
        job_id: str,
        max_attempts: int = DEFAULT_MAX_POLLS,
        interval: float = DEFAULT_POLL_INTERVAL,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Wait for job completion.
        
        Args:
            job_id: Job ID
            max_attempts: Max polling attempts
            interval: Seconds between polls
            verbose: Print progress
            
        Returns:
            Final job data
        """
        if verbose:
            print_info(f"Waiting for job {job_id}...")
        
        for attempt in range(max_attempts):
            job_data = await self.get_job(job_id)
            status = job_data.get("status", "unknown")
            
            if verbose and attempt % 5 == 0:
                print(f"  Status: {status} (attempt {attempt + 1}/{max_attempts})")
            
            if status in ("completed", "failed", "cancelled"):
                return job_data
            
            await asyncio.sleep(interval)
        
        raise TimeoutError(f"Job {job_id} did not complete within {max_attempts} attempts")
    
    async def search_cognee(
        self,
        query: str,
        search_type: str = "hybrid",
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search using Cognee.
        
        Args:
            query: Search query
            search_type: Search type
            top_k: Number of results
            
        Returns:
            Search results
        """
        response = await self.client.post(
            "/api/v1/cognee/search",
            json={
                "query": query,
                "search_type": search_type,
                "top_k": top_k,
            },
        )
        
        if response.status_code == 404:
            return []
        
        response.raise_for_status()
        return response.json().get("data", {}).get("results", [])
    
    async def search_hipporag(
        self,
        query: str,
        num_to_retrieve: int = 5,
    ) -> list[dict[str, Any]]:
        """Search using HippoRAG.
        
        Args:
            query: Search query
            num_to_retrieve: Number of passages
            
        Returns:
            Retrieval results
        """
        response = await self.client.post(
            "/api/v1/hipporag/retrieve",
            json={
                "queries": [query],
                "num_to_retrieve": num_to_retrieve,
            },
        )
        
        if response.status_code == 404:
            return []
        
        response.raise_for_status()
        return response.json().get("data", {}).get("results", [])


# =============================================================================
# Test Functions
# =============================================================================

async def test_upload_and_verify(
    client: PipelineApiClient,
    file_path: Path,
    wait: bool = True,
) -> dict[str, Any]:
    """Test file upload and verify processing.
    
    Args:
        client: API client
        file_path: Path to file
        wait: Whether to wait for completion
        
    Returns:
        Test results summary
    """
    print_header(f"Testing: {file_path.name}")
    
    results = {
        "file": file_path.name,
        "file_size": file_path.stat().st_size,
        "success": False,
        "job_id": None,
        "processing_time": 0,
        "chunks_count": 0,
        "errors": [],
    }
    
    # Check file exists
    if not file_path.exists():
        print_error(f"File not found: {file_path}")
        results["errors"].append("File not found")
        return results
    
    print_info(f"File size: {format_bytes(results['file_size'])}")
    
    # Upload file
    try:
        print_info("Uploading file...")
        start_time = time.time()
        job_id = await client.upload_file(
            file_path,
            metadata={
                "test_source": "manual_test",
                "file_name": file_path.name,
            },
        )
        results["job_id"] = job_id
        print_success(f"Uploaded successfully. Job ID: {job_id}")
    except Exception as e:
        print_error(f"Upload failed: {e}")
        results["errors"].append(f"Upload failed: {e}")
        return results
    
    if not wait:
        print_info("Skipping wait for completion (--wait not specified)")
        results["success"] = True
        return results
    
    # Wait for completion
    try:
        print_info("Waiting for processing to complete...")
        job_data = await client.wait_for_job(job_id, verbose=True)
        results["processing_time"] = time.time() - start_time
        
        status = job_data.get("status")
        
        if status == "completed":
            print_success(f"Processing completed in {format_duration(results['processing_time'])}")
            results["success"] = True
        elif status == "failed":
            print_error(f"Processing failed: {job_data.get('error')}")
            results["errors"].append(f"Processing failed: {job_data.get('error')}")
            return results
        else:
            print_warning(f"Unexpected status: {status}")
            results["errors"].append(f"Unexpected status: {status}")
            return results
        
    except TimeoutError as e:
        print_error(f"Timeout waiting for job: {e}")
        results["errors"].append(f"Timeout: {e}")
        return results
    except Exception as e:
        print_error(f"Error waiting for job: {e}")
        results["errors"].append(f"Wait error: {e}")
        return results
    
    # Verify chunks
    try:
        print_info("Verifying chunks...")
        chunks = await client.get_job_chunks(job_id)
        results["chunks_count"] = len(chunks)
        
        if len(chunks) > 0:
            print_success(f"Found {len(chunks)} chunks")
            
            # Show first chunk preview
            if chunks:
                first_chunk = chunks[0]
                content_preview = first_chunk.get("content", "")[:100]
                print_info(f"First chunk preview: {content_preview}...")
        else:
            print_error("No chunks found")
            results["errors"].append("No chunks created")
            results["success"] = False
        
    except Exception as e:
        print_error(f"Error getting chunks: {e}")
        results["errors"].append(f"Chunks error: {e}")
    
    return results


async def test_search_after_ingestion(
    client: PipelineApiClient,
    query: str = "artificial intelligence",
) -> dict[str, Any]:
    """Test search after document ingestion.
    
    Args:
        client: API client
        query: Search query
        
    Returns:
        Search results summary
    """
    print_header("Testing Search After Ingestion")
    
    results = {
        "query": query,
        "cognee_results": 0,
        "hipporag_results": 0,
        "errors": [],
    }
    
    # Test Cognee search
    try:
        print_info(f"Searching Cognee for: '{query}'")
        cognee_results = await client.search_cognee(query)
        results["cognee_results"] = len(cognee_results)
        
        if cognee_results:
            print_success(f"Cognee returned {len(cognee_results)} results")
        else:
            print_warning("Cognee returned no results (graph may not be populated)")
    except Exception as e:
        print_error(f"Cognee search failed: {e}")
        results["errors"].append(f"Cognee search: {e}")
    
    # Test HippoRAG search
    try:
        print_info(f"Searching HippoRAG for: '{query}'")
        hipporag_results = await client.search_hipporag(query)
        results["hipporag_results"] = len(hipporag_results)
        
        if hipporag_results:
            print_success(f"HippoRAG returned {len(hipporag_results)} results")
        else:
            print_warning("HippoRAG returned no results (graph may not be populated)")
    except Exception as e:
        print_error(f"HippoRAG search failed: {e}")
        results["errors"].append(f"HippoRAG search: {e}")
    
    return results


async def check_job_status(client: PipelineApiClient, job_id: str) -> None:
    """Check status of a specific job.
    
    Args:
        client: API client
        job_id: Job ID to check
    """
    print_header(f"Checking Job: {job_id}")
    
    try:
        job_data = await client.get_job(job_id)
        
        print_info(f"Status: {job_data.get('status')}")
        print_info(f"Source: {job_data.get('source_type')}")
        print_info(f"File: {job_data.get('file_name')}")
        print_info(f"Created: {job_data.get('created_at')}")
        
        if job_data.get("error"):
            print_error(f"Error: {job_data['error']}")
        
        # Get chunks
        chunks = await client.get_job_chunks(job_id)
        print_info(f"Chunks: {len(chunks)}")
        
        for i, chunk in enumerate(chunks[:3]):  # Show first 3
            content = chunk.get("content", "")[:80]
            print(f"  Chunk {i + 1}: {content}...")
        
        if len(chunks) > 3:
            print(f"  ... and {len(chunks) - 3} more")
            
    except Exception as e:
        print_error(f"Failed to get job: {e}")


# =============================================================================
# Main
# =============================================================================

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Manual pipeline test script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test single file
  python scripts/test_pipeline_manual.py --file tests/data/sample.pdf
  
  # Test and wait for completion
  python scripts/test_pipeline_manual.py --file tests/data/sample.pdf --wait
  
  # Test all files in directory
  python scripts/test_pipeline_manual.py --dir tests/data/ --pattern "*.pdf"
  
  # Check job status
  python scripts/test_pipeline_manual.py --job-id <uuid> --check
  
  # Test with custom API URL
  API_URL=http://localhost:8000/api/v1 python scripts/test_pipeline_manual.py --file sample.pdf
        """
    )
    
    parser.add_argument(
        "--file",
        type=Path,
        help="Single file to test",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        help="Directory of files to test",
    )
    parser.add_argument(
        "--pattern",
        default="*",
        help="File pattern when using --dir (default: *)",
    )
    parser.add_argument(
        "--job-id",
        help="Job ID to check",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check job status only",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        default=True,
        help="Wait for job completion (default: True)",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for job completion",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="Test search after ingestion",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"API base URL (default: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="API key",
    )
    
    args = parser.parse_args()
    
    # Determine wait behavior
    wait = args.wait and not args.no_wait
    
    # Create client
    client = PipelineApiClient(args.api_url, args.api_key)
    
    try:
        # Health check
        print_header("Pipeline API Health Check")
        try:
            health = await client.health_check()
            print_success(f"API is healthy: {health.get('status')}")
            print_info(f"Version: {health.get('version')}")
        except Exception as e:
            print_error(f"Health check failed: {e}")
            print_info(f"API URL: {args.api_url}")
            sys.exit(1)
        
        # Check job status
        if args.check and args.job_id:
            await check_job_status(client, args.job_id)
            return
        
        # Test single file
        if args.file:
            results = await test_upload_and_verify(client, args.file, wait=wait)
            
            if results["success"] and args.search:
                await test_search_after_ingestion(client)
            
            # Summary
            print_header("Test Summary")
            print_info(f"File: {results['file']}")
            print_info(f"Job ID: {results['job_id']}")
            print_info(f"Status: {'SUCCESS' if results['success'] else 'FAILED'}")
            print_info(f"Processing time: {format_duration(results['processing_time'])}")
            print_info(f"Chunks: {results['chunks_count']}")
            
            if results["errors"]:
                print_error(f"Errors: {len(results['errors'])}")
                for error in results["errors"]:
                    print(f"  - {error}")
            
            sys.exit(0 if results["success"] else 1)
        
        # Test directory
        if args.dir:
            if not args.dir.exists():
                print_error(f"Directory not found: {args.dir}")
                sys.exit(1)
            
            files = list(args.dir.glob(args.pattern))
            print_info(f"Found {len(files)} files matching '{args.pattern}'")
            
            all_results = []
            for file_path in files:
                if file_path.is_file():
                    results = await test_upload_and_verify(client, file_path, wait=wait)
                    all_results.append(results)
            
            # Summary
            print_header("Batch Test Summary")
            success_count = sum(1 for r in all_results if r["success"])
            total_chunks = sum(r["chunks_count"] for r in all_results)
            
            print_info(f"Files tested: {len(all_results)}")
            print_success(f"Successful: {success_count}")
            print_error(f"Failed: {len(all_results) - success_count}")
            print_info(f"Total chunks: {total_chunks}")
            
            sys.exit(0 if success_count == len(all_results) else 1)
        
        # No action specified
        print_error("No action specified. Use --file, --dir, or --job-id")
        parser.print_help()
        sys.exit(1)
        
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
