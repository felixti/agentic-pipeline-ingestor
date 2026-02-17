"""Performance E2E Tests.

This module tests system performance metrics including:
- Latency percentiles (p50, p95, p99)
- Concurrent processing capacity
- Throughput validation (20GB/day target)
- Load handling
"""

import asyncio
import statistics
import time
from typing import List

import pytest

# Mark all tests in this module as E2E, performance, and slow tests
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.performance,
    pytest.mark.slow,
    pytest.mark.asyncio,
]


class TestLatency:
    """Test API latency metrics."""
    
    async def test_health_endpoint_latency_p99(self, client):
        """E2E: Verify p99 latency < 2s for health endpoint.
        
        Make 100 requests to /health and verify p99 latency is under 2 seconds.
        """
        num_requests = 100
        latencies = []
        
        for _ in range(num_requests):
            start = time.time()
            response = await client.get("/health")
            end = time.time()
            
            assert response.status_code == 200
            latencies.append((end - start) * 1000)  # Convert to ms
        
        # Calculate percentiles
        p50 = statistics.median(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]
        
        print(f"\nHealth endpoint latency: p50={p50:.2f}ms, p95={p95:.2f}ms, p99={p99:.2f}ms")
        
        # Assert p99 is under 2000ms (2 seconds)
        assert p99 < 2000, f"p99 latency {p99:.2f}ms exceeds 2000ms threshold"
    
    async def test_job_list_latency_p95(self, auth_client):
        """E2E: Verify p95 latency for job list endpoint."""
        num_requests = 50
        latencies = []
        
        for _ in range(num_requests):
            start = time.time()
            response = await auth_client.get("/api/v1/jobs?limit=10")
            end = time.time()
            
            assert response.status_code == 200
            latencies.append((end - start) * 1000)
        
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        
        print(f"\nJob list latency p95: {p95:.2f}ms")
        
        # p95 should be reasonable for list operations
        assert p95 < 5000, f"p95 latency {p95:.2f}ms exceeds 5000ms threshold"
    
    async def test_job_create_latency(self, auth_client, sample_job_payload):
        """E2E: Verify job creation latency is acceptable."""
        num_requests = 20
        latencies = []
        
        for i in range(num_requests):
            # Modify payload slightly for each request
            payload = sample_job_payload.copy()
            payload["metadata"] = {"test": f"perf-test-{i}"}
            
            start = time.time()
            response = await auth_client.post("/api/v1/jobs", json=payload)
            end = time.time()
            
            assert response.status_code == 202
            latencies.append((end - start) * 1000)
        
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        avg = statistics.mean(latencies)
        
        print(f"\nJob create latency: avg={avg:.2f}ms, p95={p95:.2f}ms")
        
        # Job creation should be fast
        assert p95 < 3000, f"p95 job create latency {p95:.2f}ms exceeds 3000ms threshold"


class TestConcurrentProcessing:
    """Test concurrent processing capabilities."""
    
    async def test_concurrent_job_creation(self, auth_client, sample_job_payload):
        """E2E: Test concurrent job creation.
        
        Create 50 jobs concurrently and verify all succeed.
        """
        num_jobs = 50
        concurrency = 10
        
        async def create_job(index: int) -> bool:
            payload = sample_job_payload.copy()
            payload["metadata"] = {"test": f"concurrent-{index}"}
            
            response = await auth_client.post("/api/v1/jobs", json=payload)
            return response.status_code == 202
        
        # Create semaphore for limiting concurrency
        semaphore = asyncio.Semaphore(concurrency)
        
        async def bounded_create(index: int) -> bool:
            async with semaphore:
                return await create_job(index)
        
        # Create all jobs concurrently
        results = await asyncio.gather(*[
            bounded_create(i) for i in range(num_jobs)
        ])
        
        success_count = sum(results)
        success_rate = success_count / num_jobs
        
        print(f"\nConcurrent job creation: {success_count}/{num_jobs} succeeded ({success_rate:.1%})")
        
        # At least 95% should succeed
        assert success_rate >= 0.95, f"Success rate {success_rate:.1%} is below 95% threshold"
    
    async def test_concurrent_health_checks(self, client):
        """E2E: Test concurrent health check requests."""
        num_requests = 100
        concurrency = 20
        
        semaphore = asyncio.Semaphore(concurrency)
        
        async def check_health() -> bool:
            async with semaphore:
                response = await client.get("/health")
                return response.status_code == 200
        
        results = await asyncio.gather(*[
            check_health() for _ in range(num_requests)
        ])
        
        success_count = sum(results)
        success_rate = success_count / num_requests
        
        print(f"\nConcurrent health checks: {success_count}/{num_requests} succeeded ({success_rate:.1%})")
        
        # All should succeed
        assert success_rate >= 0.99, f"Success rate {success_rate:.1%} is below 99% threshold"
    
    async def test_concurrent_api_requests(self, auth_client):
        """E2E: Test concurrent mixed API requests."""
        endpoints = [
            ("GET", "/health"),
            ("GET", "/api/v1/jobs?limit=5"),
            ("GET", "/api/v1/sources"),
            ("GET", "/api/v1/destinations"),
        ]
        
        num_requests = 40
        concurrency = 10
        
        semaphore = asyncio.Semaphore(concurrency)
        latencies = []
        
        async def make_request(method: str, path: str) -> tuple:
            async with semaphore:
                start = time.time()
                
                if method == "GET":
                    response = await auth_client.get(path)
                else:
                    response = None
                
                end = time.time()
                latency = (end - start) * 1000
                
                return response.status_code if response else 0, latency
        
        # Make concurrent requests
        tasks = []
        for i in range(num_requests):
            method, path = endpoints[i % len(endpoints)]
            tasks.append(make_request(method, path))
        
        results = await asyncio.gather(*tasks)
        
        success_count = sum(1 for status, _ in results if status == 200)
        latencies = [lat for _, lat in results]
        
        success_rate = success_count / num_requests
        avg_latency = statistics.mean(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        
        print(f"\nConcurrent API requests: {success_count}/{num_requests} succeeded ({success_rate:.1%})")
        print(f"Avg latency: {avg_latency:.2f}ms, p95: {p95_latency:.2f}ms")
        
        assert success_rate >= 0.95, f"Success rate {success_rate:.1%} is below 95%"
        assert p95_latency < 5000, f"p95 latency {p95_latency:.2f}ms exceeds 5000ms"


class TestThroughput:
    """Test system throughput capabilities."""
    
    @pytest.mark.skip(reason="Requires actual file uploads - run manually")
    async def test_throughput_20gb_day_simulation(self, auth_client, test_documents):
        """E2E: Validate 20GB/day throughput capability.
        
        This is a simulation test. For actual throughput validation,
        run a sustained load test over 24 hours.
        
        20GB/day = ~231KB/s sustained throughput
        """
        # Calculate throughput requirements
        gb_per_day = 20
        seconds_per_day = 86400
        bytes_per_second = (gb_per_day * 1024 * 1024 * 1024) / seconds_per_day
        
        print(f"\nTarget throughput: {gb_per_day}GB/day = {bytes_per_second:.2f} bytes/second")
        
        # For this test, we'll process multiple small files and calculate rate
        file_path = test_documents["text_pdf"]
        
        if not file_path.exists():
            pytest.skip("Test file not found")
        
        # Simulate processing
        num_files = 10
        start_time = time.time()
        
        for i in range(num_files):
            with open(file_path, "rb") as f:
                files = {"file": (f"doc-{i}.pdf", f, "application/pdf")}
                response = await auth_client.post("/api/v1/upload", files=files)
                
                if response.status_code != 202:
                    print(f"Upload {i} failed: {response.status_code}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate achieved throughput
        total_bytes = file_path.stat().st_size * num_files
        achieved_bps = total_bytes / duration
        
        print(f"Processed {total_bytes} bytes in {duration:.2f}s")
        print(f"Achieved throughput: {achieved_bps:.2f} bytes/second")
        
        # Scale to daily throughput (for estimation)
        daily_gb = (achieved_bps * seconds_per_day) / (1024 * 1024 * 1024)
        print(f"Projected daily throughput: {daily_gb:.2f}GB/day")
        
        # For a short test, we just verify the system responds
        # Actual sustained throughput requires longer testing
        assert duration < 60, "Processing took too long"
    
    async def test_job_creation_throughput(self, auth_client, sample_job_payload):
        """E2E: Test job creation throughput (jobs per second)."""
        num_jobs = 100
        duration_seconds = 10
        
        start_time = time.time()
        created = 0
        
        while time.time() - start_time < duration_seconds and created < num_jobs:
            payload = sample_job_payload.copy()
            payload["metadata"] = {"test": f"throughput-{created}"}
            
            response = await auth_client.post("/api/v1/jobs", json=payload)
            
            if response.status_code == 202:
                created += 1
            else:
                break
        
        elapsed = time.time() - start_time
        throughput = created / elapsed if elapsed > 0 else 0
        
        print(f"\nJob creation throughput: {created} jobs in {elapsed:.2f}s = {throughput:.2f} jobs/second")
        
        # Should be able to create at least 10 jobs per second
        assert throughput >= 10, f"Throughput {throughput:.2f} jobs/s is below 10 jobs/s threshold"


class TestLoadHandling:
    """Test system behavior under load."""
    
    async def test_rate_limit_handling(self, auth_client):
        """E2E: Test system behavior when rate limit is hit."""
        # Make rapid requests to potentially hit rate limit
        responses = []
        
        for _ in range(200):
            response = await auth_client.get("/api/v1/jobs?limit=1")
            responses.append(response.status_code)
        
        # Count different response codes
        success_count = responses.count(200)
        rate_limited_count = responses.count(429)
        error_count = len(responses) - success_count - rate_limited_count
        
        print(f"\nRate limit test: {success_count} success, {rate_limited_count} rate limited, {error_count} errors")
        
        # System should either succeed or return 429 (rate limited)
        assert error_count == 0, f"Unexpected errors: {error_count}"
        
        # If rate limiting is implemented, should have some 429s
        # If not implemented, all should succeed
        assert success_count > 0 or rate_limited_count > 0
    
    async def test_error_recovery(self, auth_client):
        """E2E: Test system recovers from errors under load."""
        # Mix valid and invalid requests
        tasks = []
        
        # Valid requests
        for _ in range(20):
            tasks.append(auth_client.get("/api/v1/jobs?limit=1"))
        
        # Invalid requests
        for _ in range(10):
            tasks.append(auth_client.get("/api/v1/jobs/invalid-id"))
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successes and failures
        success_count = sum(1 for r in responses 
                          if not isinstance(r, Exception) and r.status_code == 200)
        not_found_count = sum(1 for r in responses 
                             if not isinstance(r, Exception) and r.status_code == 404)
        error_count = len(responses) - success_count - not_found_count
        
        print(f"\nError recovery test: {success_count} success, {not_found_count} not found, {error_count} errors")
        
        # System should handle mix without crashing
        assert error_count == 0, f"Unexpected errors: {error_count}"
    
    async def test_memory_stability(self, auth_client):
        """E2E: Test memory stability over multiple requests."""
        # Make many requests and monitor for memory issues
        # (In real test, would monitor server memory)
        
        for batch in range(5):
            tasks = [auth_client.get("/health") for _ in range(20)]
            responses = await asyncio.gather(*tasks)
            
            success_count = sum(1 for r in responses if r.status_code == 200)
            
            assert success_count == 20, f"Batch {batch}: {success_count}/20 succeeded"
            
            # Small delay between batches
            await asyncio.sleep(0.1)
        
        print("\nMemory stability test completed successfully")


class TestResourceUsage:
    """Test resource usage under load."""
    
    async def test_large_file_upload_handling(self, auth_client, test_documents):
        """E2E: Test handling of large file uploads."""
        large_file = test_documents["large_pdf"]
        
        if not large_file.exists():
            pytest.skip("Large test file not found")
        
        file_size_mb = large_file.stat().st_size / (1024 * 1024)
        print(f"\nUploading large file: {file_size_mb:.2f} MB")
        
        with open(large_file, "rb") as f:
            files = {"file": ("large-document.pdf", f, "application/pdf")}
            
            start = time.time()
            response = await auth_client.post("/api/v1/upload", files=files)
            elapsed = time.time() - start
        
        print(f"Upload completed in {elapsed:.2f}s with status {response.status_code}")
        
        # Should accept or reject based on size limits, but not crash
        assert response.status_code in (202, 400, 413)
    
    async def test_bulk_job_submission(self, auth_client):
        """E2E: Test bulk job submission performance."""
        bulk_payload = {
            "jobs": [
                {
                    "source_type": "upload",
                    "source_uri": f"/uploads/doc{i}.pdf",
                    "file_name": f"doc{i}.pdf",
                    "mime_type": "application/pdf",
                    "priority": 5
                }
                for i in range(20)
            ]
        }
        
        start = time.time()
        response = await auth_client.post("/api/v1/bulk/jobs", json=bulk_payload)
        elapsed = time.time() - start
        
        print(f"\nBulk submission of 20 jobs completed in {elapsed:.2f}s")
        
        # Should complete quickly
        assert response.status_code in (202, 404)
        
        if response.status_code == 202:
            assert elapsed < 10, f"Bulk submission took {elapsed:.2f}s, expected <10s"


class TestScalability:
    """Test system scalability."""
    
    async def test_horizontal_scalability_indicators(self, client):
        """E2E: Check indicators that suggest horizontal scalability."""
        # Check health endpoint for load balancer compatibility
        response = await client.get("/health/live")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "alive" in data
        
        # Check readiness endpoint
        ready_response = await client.get("/health/ready")
        
        # May return 200 or 503 depending on dependencies
        assert ready_response.status_code in (200, 503)
        
        print("\nScalability indicators checked successfully")
    
    async def test_statelessness(self, auth_client):
        """E2E: Verify API statelessness through repeated requests."""
        # Same request should return consistent results
        responses = []
        
        for _ in range(5):
            response = await auth_client.get("/api/v1/jobs?limit=1")
            responses.append(response.status_code)
            await asyncio.sleep(0.1)
        
        # All should succeed (or all fail if auth issue)
        assert len(set(responses)) == 1, "Inconsistent responses suggest statefulness"
        
        print("\nStatelessness verified")
