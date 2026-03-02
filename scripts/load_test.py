#!/usr/bin/env python3
"""
Load test for concurrent document processing.

This script tests the pipeline's ability to handle concurrent uploads
and verifies that the deadlock fixes and transaction handling work
correctly under load.

Usage:
    python scripts/load_test.py --count 10
    python scripts/load_test.py --count 20 --concurrent 5
    python scripts/load_test.py --count 50 --file tests/data/sample.pdf
    python scripts/load_test.py --count 10 --report json --output results.json

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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_API_URL = os.getenv("API_URL", "https://pipeline-api.felixtek.cloud/api/v1")
DEFAULT_API_KEY = os.getenv("API_KEY", "test-api-key-123")
DEFAULT_TIMEOUT = 120

# Colors for terminal output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    END = "\033[0m"


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


def print_metric(name: str, value: str, unit: str = "") -> None:
    """Print a metric."""
    print(f"{Colors.BOLD}{name}:{Colors.END} {Colors.MAGENTA}{value}{Colors.END} {unit}")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class UploadResult:
    """Result of a single upload."""
    job_id: str | None = None
    success: bool = False
    upload_time: float = 0.0
    processing_time: float = 0.0
    chunks_count: int = 0
    error: str | None = None
    status: str = "unknown"


@dataclass
class LoadTestResults:
    """Aggregate results from load test."""
    total_uploads: int = 0
    successful_uploads: int = 0
    failed_uploads: int = 0
    total_chunks: int = 0
    deadlock_errors: int = 0
    fk_violations: int = 0
    other_errors: int = 0
    
    upload_times: list[float] = field(default_factory=list)
    processing_times: list[float] = field(default_factory=list)
    
    start_time: float = 0.0
    end_time: float = 0.0
    
    individual_results: list[UploadResult] = field(default_factory=list)
    
    @property
    def total_time(self) -> float:
        """Total test duration."""
        return self.end_time - self.start_time if self.end_time > self.start_time else 0
    
    @property
    def success_rate(self) -> float:
        """Success rate percentage."""
        if self.total_uploads == 0:
            return 0.0
        return (self.successful_uploads / self.total_uploads) * 100
    
    @property
    def avg_upload_time(self) -> float:
        """Average upload time."""
        if not self.upload_times:
            return 0.0
        return sum(self.upload_times) / len(self.upload_times)
    
    @property
    def avg_processing_time(self) -> float:
        """Average processing time."""
        times = [t for t in self.processing_times if t > 0]
        if not times:
            return 0.0
        return sum(times) / len(times)
    
    @property
    def min_processing_time(self) -> float:
        """Minimum processing time."""
        times = [t for t in self.processing_times if t > 0]
        return min(times) if times else 0.0
    
    @property
    def max_processing_time(self) -> float:
        """Maximum processing time."""
        times = [t for t in self.processing_times if t > 0]
        return max(times) if times else 0.0
    
    @property
    def throughput(self) -> float:
        """Uploads per minute."""
        if self.total_time == 0:
            return 0.0
        return (self.successful_uploads / self.total_time) * 60


# =============================================================================
# API Client
# =============================================================================

class LoadTestClient:
    """Client for load testing."""
    
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
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
            ),
        )
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def upload_file(
        self,
        file_path: Path,
        index: int,
    ) -> UploadResult:
        """Upload a file and track timing.
        
        Args:
            file_path: Path to file
            index: Upload index
            
        Returns:
            Upload result
        """
        result = UploadResult()
        
        try:
            with open(file_path, "rb") as f:
                files = {"file": (f"test_{index}_{file_path.name}", f, "application/octet-stream")}
                data = {"metadata": json.dumps({"load_test_index": index, "test_type": "load"})}
                
                upload_start = time.time()
                response = await self.client.post("/api/v1/upload", files=files, data=data)
                result.upload_time = time.time() - upload_start
            
            if response.status_code != 202:
                result.error = f"Upload failed: HTTP {response.status_code}"
                return result
            
            result.job_id = response.json()["data"]["job_id"]
            result.success = True
            
        except Exception as e:
            result.error = f"Upload exception: {str(e)}"
        
        return result
    
    async def wait_for_job(self, job_id: str) -> dict[str, Any]:
        """Wait for job completion.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job data
        """
        max_attempts = 60
        interval = 2.0
        
        for attempt in range(max_attempts):
            try:
                response = await self.client.get(f"/api/v1/jobs/{job_id}")
                response.raise_for_status()
                job_data = response.json()["data"]
                
                status = job_data.get("status")
                if status in ("completed", "failed", "cancelled"):
                    return job_data
                
                await asyncio.sleep(interval)
            except Exception:
                await asyncio.sleep(interval)
        
        return {"status": "timeout", "error": "Max polling attempts reached"}
    
    async def get_job_chunks(self, job_id: str) -> int:
        """Get chunk count for a job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Number of chunks
        """
        try:
            response = await self.client.get(f"/api/v1/jobs/{job_id}/chunks")
            if response.status_code == 200:
                return response.json().get("data", {}).get("total", 0)
        except Exception:
            pass
        return 0


# =============================================================================
# Load Test Functions
# =============================================================================

async def run_sequential_uploads(
    client: LoadTestClient,
    file_path: Path,
    count: int,
    results: LoadTestResults,
) -> None:
    """Run uploads sequentially.
    
    Args:
        client: API client
        file_path: File to upload
        count: Number of uploads
        results: Results collector
    """
    print_info(f"Running {count} sequential uploads...")
    
    for i in range(count):
        result = await client.upload_file(file_path, i)
        results.individual_results.append(result)
        results.upload_times.append(result.upload_time)
        
        if result.success:
            # Wait for processing
            job_data = await client.wait_for_job(result.job_id)
            result.status = job_data.get("status", "unknown")
            
            if result.status == "completed":
                result.processing_time = job_data.get("processing_time_ms", 0) / 1000
                result.chunks_count = await client.get_job_chunks(result.job_id)
                results.processing_times.append(result.processing_time)
                results.total_chunks += result.chunks_count
                results.successful_uploads += 1
                print(f"  [{i + 1}/{count}] ✓ Job {result.job_id[:8]}... completed ({result.chunks_count} chunks)")
            else:
                result.error = job_data.get("error", {}).get("message", "Unknown error")
                results.failed_uploads += 1
                print(f"  [{i + 1}/{count}] ✗ Job {result.job_id[:8]}... failed: {result.error}")
        else:
            results.failed_uploads += 1
            print(f"  [{i + 1}/{count}] ✗ Upload failed: {result.error}")
        
        results.total_uploads += 1


async def run_concurrent_uploads(
    client: LoadTestClient,
    file_path: Path,
    count: int,
    concurrent: int,
    results: LoadTestResults,
) -> None:
    """Run uploads concurrently with semaphore control.
    
    Args:
        client: API client
        file_path: File to upload
        count: Number of uploads
        concurrent: Max concurrent uploads
        results: Results collector
    """
    print_info(f"Running {count} uploads with concurrency of {concurrent}...")
    
    semaphore = asyncio.Semaphore(concurrent)
    completed = 0
    
    async def upload_with_limit(index: int) -> UploadResult:
        """Upload with semaphore limit."""
        nonlocal completed
        
        async with semaphore:
            result = await client.upload_file(file_path, index)
            
            if result.success:
                job_data = await client.wait_for_job(result.job_id)
                result.status = job_data.get("status", "unknown")
                
                if result.status == "completed":
                    result.processing_time = job_data.get("processing_time_ms", 0) / 1000
                    result.chunks_count = await client.get_job_chunks(result.job_id)
                else:
                    result.error = job_data.get("error", {}).get("message", "Unknown error")
            
            completed += 1
            status_char = "✓" if result.status == "completed" else "✗"
            chunks_info = f"({result.chunks_count} chunks)" if result.chunks_count > 0 else ""
            print(f"  [{completed}/{count}] {status_char} Job {result.job_id[:8] if result.job_id else 'N/A'}... {result.status} {chunks_info}")
            
            return result
    
    # Run all uploads concurrently with limit
    tasks = [upload_with_limit(i) for i in range(count)]
    individual_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    for result in individual_results:
        if isinstance(result, Exception):
            error_result = UploadResult(success=False, error=str(result))
            results.individual_results.append(error_result)
            results.failed_uploads += 1
            results.upload_times.append(0)
        else:
            results.individual_results.append(result)
            results.upload_times.append(result.upload_time)
            
            if result.status == "completed":
                results.successful_uploads += 1
                results.processing_times.append(result.processing_time)
                results.total_chunks += result.chunks_count
            else:
                results.failed_uploads += 1
                
                # Categorize errors
                error_str = str(result.error).lower()
                if "deadlock" in error_str:
                    results.deadlock_errors += 1
                elif "foreign key" in error_str or "fk violation" in error_str:
                    results.fk_violations += 1
                else:
                    results.other_errors += 1
        
        results.total_uploads += 1


# =============================================================================
# Report Generation
# =============================================================================

def print_summary(results: LoadTestResults) -> None:
    """Print test summary.
    
    Args:
        results: Test results
    """
    print_header("LOAD TEST SUMMARY")
    
    print_metric("Total uploads", str(results.total_uploads))
    print_metric("Successful", str(results.successful_uploads))
    print_metric("Failed", str(results.failed_uploads))
    print_metric("Success rate", f"{results.success_rate:.1f}", "%")
    print()
    print_metric("Total chunks created", str(results.total_chunks))
    print_metric("Avg chunks per job", f"{results.total_chunks / max(results.successful_uploads, 1):.1f}")
    print()
    print_metric("Total time", f"{results.total_time:.1f}", "seconds")
    print_metric("Throughput", f"{results.throughput:.1f}", "uploads/minute")
    print()
    print_metric("Avg upload time", f"{results.avg_upload_time:.2f}", "seconds")
    print_metric("Avg processing time", f"{results.avg_processing_time:.2f}", "seconds")
    print_metric("Min processing time", f"{results.min_processing_time:.2f}", "seconds")
    print_metric("Max processing time", f"{results.max_processing_time:.2f}", "seconds")
    
    if results.deadlock_errors > 0 or results.fk_violations > 0:
        print()
        print_error("ERRORS DETECTED:")
        if results.deadlock_errors > 0:
            print(f"  - Deadlock errors: {results.deadlock_errors}")
        if results.fk_violations > 0:
            print(f"  - FK violations: {results.fk_violations}")
        if results.other_errors > 0:
            print(f"  - Other errors: {results.other_errors}")
    else:
        print()
        print_success("No deadlock or transaction errors detected!")


def generate_json_report(results: LoadTestResults, output_path: Path) -> None:
    """Generate JSON report.
    
    Args:
        results: Test results
        output_path: Output file path
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_uploads": results.total_uploads,
            "successful_uploads": results.successful_uploads,
            "failed_uploads": results.failed_uploads,
            "success_rate_percent": round(results.success_rate, 2),
            "total_chunks": results.total_chunks,
            "deadlock_errors": results.deadlock_errors,
            "fk_violations": results.fk_violations,
            "other_errors": results.other_errors,
        },
        "timing": {
            "total_time_seconds": round(results.total_time, 2),
            "throughput_uploads_per_minute": round(results.throughput, 2),
            "avg_upload_time_seconds": round(results.avg_upload_time, 2),
            "avg_processing_time_seconds": round(results.avg_processing_time, 2),
            "min_processing_time_seconds": round(results.min_processing_time, 2),
            "max_processing_time_seconds": round(results.max_processing_time, 2),
        },
        "individual_results": [
            {
                "job_id": r.job_id,
                "success": r.success,
                "status": r.status,
                "upload_time_seconds": round(r.upload_time, 2),
                "processing_time_seconds": round(r.processing_time, 2),
                "chunks_count": r.chunks_count,
                "error": r.error,
            }
            for r in results.individual_results
        ],
    }
    
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print_success(f"JSON report saved to: {output_path}")


def generate_csv_report(results: LoadTestResults, output_path: Path) -> None:
    """Generate CSV report.
    
    Args:
        results: Test results
        output_path: Output file path
    """
    import csv
    
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "job_id", "success", "status", "upload_time", "processing_time",
            "chunks_count", "error"
        ])
        
        for r in results.individual_results:
            writer.writerow([
                r.job_id,
                r.success,
                r.status,
                r.upload_time,
                r.processing_time,
                r.chunks_count,
                r.error,
            ])
    
    print_success(f"CSV report saved to: {output_path}")


# =============================================================================
# Validation
# =============================================================================

def validate_results(results: LoadTestResults) -> list[str]:
    """Validate results against targets.
    
    Args:
        results: Test results
        
    Returns:
        List of validation messages
    """
    messages = []
    
    # Target: 100% success rate for concurrent uploads
    if results.success_rate < 100:
        messages.append(f"❌ Success rate {results.success_rate:.1f}% < 100% target")
    else:
        messages.append(f"✅ Success rate: {results.success_rate:.1f}% (target: 100%)")
    
    # Target: 0 deadlock errors
    if results.deadlock_errors > 0:
        messages.append(f"❌ Deadlock errors: {results.deadlock_errors} (target: 0)")
    else:
        messages.append("✅ Deadlock errors: 0 (target: 0)")
    
    # Target: 0 FK violations
    if results.fk_violations > 0:
        messages.append(f"❌ FK violations: {results.fk_violations} (target: 0)")
    else:
        messages.append("✅ FK violations: 0 (target: 0)")
    
    # Target: Chunks created per job > 0
    avg_chunks = results.total_chunks / max(results.successful_uploads, 1)
    if avg_chunks == 0:
        messages.append("❌ No chunks created (target: > 0)")
    else:
        messages.append(f"✅ Avg chunks per job: {avg_chunks:.1f}")
    
    return messages


# =============================================================================
# Main
# =============================================================================

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Load test for concurrent document processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 10 concurrent uploads
  python scripts/load_test.py --count 10
  
  # Run 20 uploads with max 5 concurrent
  python scripts/load_test.py --count 20 --concurrent 5
  
  # Use specific file
  python scripts/load_test.py --count 50 --file tests/data/sample.pdf
  
  # Generate JSON report
  python scripts/load_test.py --count 10 --report json --output results.json
  
  # Generate CSV report
  python scripts/load_test.py --count 10 --report csv --output results.csv
  
  # Run sequentially (no concurrency)
  python scripts/load_test.py --count 5 --sequential
        """
    )
    
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of uploads to perform (default: 10)",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=5,
        help="Max concurrent uploads (default: 5)",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("tests/data/sample.pdf"),
        help="File to upload (default: tests/data/sample.pdf)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run uploads sequentially (no concurrency)",
    )
    parser.add_argument(
        "--report",
        choices=["json", "csv"],
        help="Generate report in specified format",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for report",
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
    
    # Validate file
    if not args.file.exists():
        print_error(f"File not found: {args.file}")
        sys.exit(1)
    
    print_header("LOAD TEST - Ralph Loop Pipeline")
    print_info(f"API URL: {args.api_url}")
    print_info(f"Test file: {args.file}")
    print_info(f"File size: {args.file.stat().st_size:,} bytes")
    print_info(f"Upload count: {args.count}")
    
    if args.sequential:
        print_info("Mode: Sequential")
    else:
        print_info(f"Mode: Concurrent (max {args.concurrent} parallel)")
    
    # Create client
    client = LoadTestClient(args.api_url, args.api_key)
    results = LoadTestResults()
    
    try:
        # Health check
        print_header("Health Check")
        try:
            health_response = await client.client.get("/health")
            health_response.raise_for_status()
            health = health_response.json()
            print_success(f"API is healthy: {health.get('status')}")
        except Exception as e:
            print_error(f"Health check failed: {e}")
            sys.exit(1)
        
        # Run load test
        print_header("Running Load Test")
        results.start_time = time.time()
        
        if args.sequential:
            await run_sequential_uploads(client, args.file, args.count, results)
        else:
            await run_concurrent_uploads(client, args.file, args.count, args.concurrent, results)
        
        results.end_time = time.time()
        
        # Print summary
        print_summary(results)
        
        # Validation
        print_header("Validation Against Targets")
        validation_messages = validate_results(results)
        for msg in validation_messages:
            if msg.startswith("❌"):
                print_error(msg)
            else:
                print_success(msg)
        
        # Generate report if requested
        if args.report:
            print_header("Generating Report")
            
            if args.output:
                output_path = args.output
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = Path(f"load_test_results_{timestamp}.{args.report}")
            
            if args.report == "json":
                generate_json_report(results, output_path)
            elif args.report == "csv":
                generate_csv_report(results, output_path)
        
        # Exit with error code if validation failed
        all_passed = all(msg.startswith("✅") for msg in validation_messages)
        sys.exit(0 if all_passed else 1)
        
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
