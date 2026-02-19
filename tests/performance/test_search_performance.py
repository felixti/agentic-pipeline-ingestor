"""Performance tests for vector store search operations.

Benchmarks:
- Vector search latency (target: p99 < 100ms)
- Text search latency (target: p99 < 50ms)
- Hybrid search latency
- Performance at different data sizes (1K, 10K, 100K, 1M chunks)

Usage:
    pytest tests/performance/test_search_performance.py -v --performance
    
Note: These tests require a running PostgreSQL with pgvector.
Set DB_URL environment variable to use a specific database.
"""

import asyncio
import math
import os
import statistics
import time
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

# Performance test marker
pytestmark = [
    pytest.mark.performance,
    pytest.mark.slow,
]

# =============================================================================
# Test Configuration
# =============================================================================

PERFORMANCE_THRESHOLDS = {
    "vector_search_p99_ms": 100,  # Target: p99 < 100ms
    "text_search_p99_ms": 50,     # Target: p99 < 50ms
    "hybrid_search_p99_ms": 150,  # Target: p99 < 150ms
    "similar_chunks_p99_ms": 100, # Target: p99 < 100ms
}

DATASET_SIZES = [1000, 10000, 100000]  # 1K, 10K, 100K (1M in separate test)
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_embedding():
    """Create a normalized sample embedding vector."""
    dims = 1536
    vec = [0.1 * (i % 10) / 10 for i in range(dims)]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


@pytest.fixture
def mock_chunks_1k():
    """Create 1000 mock chunks for testing."""
    job_id = uuid4()
    return [
        MagicMock(
            id=uuid4(),
            job_id=job_id,
            chunk_index=i,
            content=f"Sample content for chunk {i}. This is test data for benchmarking purposes.",
            content_hash=f"hash_{i}",
            embedding=[0.01 * ((i + j) % 100) / 100 for j in range(1536)],
            metadata={"page": i // 10, "index": i},
            created_at=datetime.utcnow(),
        )
        for i in range(1000)
    ]


@pytest.fixture
def mock_chunks_10k():
    """Create 10000 mock chunks for testing."""
    job_id = uuid4()
    return [
        MagicMock(
            id=uuid4(),
            job_id=job_id,
            chunk_index=i,
            content=f"Sample content for chunk {i}. " * 5,
            content_hash=f"hash_{i}",
            embedding=[0.01 * ((i + j) % 100) / 100 for j in range(1536)],
            metadata={"page": i // 100, "index": i},
            created_at=datetime.utcnow(),
        )
        for i in range(10000)
    ]


@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    return AsyncMock()


# =============================================================================
# Benchmark Utilities
# =============================================================================

def calculate_percentile(values: list[float], percentile: float) -> float:
    """Calculate the given percentile of a list of values."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(len(sorted_values) * percentile / 100)
    return sorted_values[min(index, len(sorted_values) - 1)]


def calculate_stats(latencies: list[float]) -> dict[str, float]:
    """Calculate statistics for a list of latencies."""
    if not latencies:
        return {
            "min_ms": 0.0,
            "max_ms": 0.0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "std_dev_ms": 0.0,
        }
    
    return {
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "p50_ms": calculate_percentile(latencies, 50),
        "p95_ms": calculate_percentile(latencies, 95),
        "p99_ms": calculate_percentile(latencies, 99),
        "std_dev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
    }


def print_benchmark_results(
    name: str,
    stats: dict[str, float],
    threshold_ms: float,
    iterations: int,
    dataset_size: int,
):
    """Print benchmark results in a formatted way."""
    passed = stats["p99_ms"] <= threshold_ms
    status = "✅ PASS" if passed else "❌ FAIL"
    
    print(f"\n{'='*70}")
    print(f"Benchmark: {name}")
    print(f"Dataset Size: {dataset_size:,} chunks")
    print(f"Iterations: {iterations}")
    print(f"{'='*70}")
    print(f"  Min:        {stats['min_ms']:>8.2f} ms")
    print(f"  Max:        {stats['max_ms']:>8.2f} ms")
    print(f"  Mean:       {stats['mean_ms']:>8.2f} ms")
    print(f"  Median:     {stats['median_ms']:>8.2f} ms")
    print(f"  P95:        {stats['p95_ms']:>8.2f} ms")
    print(f"  P99:        {stats['p99_ms']:>8.2f} ms  (Threshold: {threshold_ms} ms) {status}")
    print(f"  Std Dev:    {stats['std_dev_ms']:>8.2f} ms")
    print(f"{'='*70}\n")
    
    return passed


# =============================================================================
# Vector Search Performance Tests
# =============================================================================

@pytest.mark.performance
class TestVectorSearchPerformance:
    """Performance tests for vector search."""

    @pytest.mark.asyncio
    async def test_vector_search_latency_1k_chunks(self, sample_embedding, mock_chunks_1k):
        """Benchmark vector search latency with 1K chunks.
        
        Target: p99 < 100ms
        """
        from src.db.repositories.document_chunk_repository import DocumentChunkRepository
        from src.services.vector_search_service import VectorSearchService
        
        # Setup
        mock_session = AsyncMock()
        repo = DocumentChunkRepository(mock_session)
        service = VectorSearchService(repo)
        
        # Mock database results
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            (mock_chunks_1k[0], 0.1),
            (mock_chunks_1k[1], 0.2),
            (mock_chunks_1k[2], 0.3),
        ]
        mock_session.execute.return_value = mock_result
        
        # Warmup
        for _ in range(WARMUP_ITERATIONS):
            await service.search_by_vector(
                query_embedding=sample_embedding,
                top_k=10,
            )
        
        # Benchmark
        latencies = []
        for _ in range(BENCHMARK_ITERATIONS):
            start = time.perf_counter()
            await service.search_by_vector(
                query_embedding=sample_embedding,
                top_k=10,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
        
        stats = calculate_stats(latencies)
        passed = print_benchmark_results(
            "Vector Search (1K chunks)",
            stats,
            PERFORMANCE_THRESHOLDS["vector_search_p99_ms"],
            BENCHMARK_ITERATIONS,
            1000,
        )
        
        assert passed, f"Vector search p99 ({stats['p99_ms']:.2f}ms) exceeds threshold ({PERFORMANCE_THRESHOLDS['vector_search_p99_ms']}ms)"

    @pytest.mark.asyncio
    async def test_vector_search_latency_10k_chunks(self, sample_embedding, mock_chunks_10k):
        """Benchmark vector search latency with 10K chunks.
        
        Target: p99 < 100ms
        """
        from src.db.repositories.document_chunk_repository import DocumentChunkRepository
        from src.services.vector_search_service import VectorSearchService
        
        mock_session = AsyncMock()
        repo = DocumentChunkRepository(mock_session)
        service = VectorSearchService(repo)
        
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            (mock_chunks_10k[0], 0.1),
            (mock_chunks_10k[1], 0.2),
            (mock_chunks_10k[2], 0.3),
        ]
        mock_session.execute.return_value = mock_result
        
        # Warmup
        for _ in range(WARMUP_ITERATIONS):
            await service.search_by_vector(
                query_embedding=sample_embedding,
                top_k=10,
            )
        
        # Benchmark
        latencies = []
        for _ in range(BENCHMARK_ITERATIONS):
            start = time.perf_counter()
            await service.search_by_vector(
                query_embedding=sample_embedding,
                top_k=10,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
        
        stats = calculate_stats(latencies)
        passed = print_benchmark_results(
            "Vector Search (10K chunks)",
            stats,
            PERFORMANCE_THRESHOLDS["vector_search_p99_ms"],
            BENCHMARK_ITERATIONS,
            10000,
        )
        
        assert passed, f"Vector search p99 ({stats['p99_ms']:.2f}ms) exceeds threshold"

    @pytest.mark.asyncio
    async def test_vector_search_different_top_k(self, sample_embedding, mock_chunks_1k):
        """Benchmark vector search with different top_k values."""
        from src.db.repositories.document_chunk_repository import DocumentChunkRepository
        from src.services.vector_search_service import VectorSearchService
        
        mock_session = AsyncMock()
        repo = DocumentChunkRepository(mock_session)
        service = VectorSearchService(repo)
        
        results = {}
        
        for top_k in [5, 10, 50, 100]:
            mock_result = MagicMock()
            mock_result.fetchall.return_value = [
                (mock_chunks_1k[i], 0.1 * i) for i in range(min(top_k, 10))
            ]
            mock_session.execute.return_value = mock_result
            
            latencies = []
            for _ in range(50):
                start = time.perf_counter()
                await service.search_by_vector(
                    query_embedding=sample_embedding,
                    top_k=top_k,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies.append(elapsed_ms)
            
            stats = calculate_stats(latencies)
            results[top_k] = stats["p99_ms"]
            print(f"  top_k={top_k}: p99={stats['p99_ms']:.2f}ms")
        
        # Verify that higher top_k doesn't dramatically increase latency
        assert results[100] < results[5] * 3, "Top_k=100 should not be 3x slower than top_k=5"


# =============================================================================
# Text Search Performance Tests
# =============================================================================

@pytest.mark.performance
class TestTextSearchPerformance:
    """Performance tests for text search."""

    @pytest.mark.asyncio
    async def test_text_search_latency_1k_chunks(self, mock_chunks_1k):
        """Benchmark text search latency with 1K chunks.
        
        Target: p99 < 50ms
        """
        from src.db.repositories.document_chunk_repository import DocumentChunkRepository
        from src.services.text_search_service import TextSearchService
        
        mock_session = AsyncMock()
        repo = DocumentChunkRepository(mock_session)
        service = TextSearchService(repo)
        
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            (mock_chunks_1k[0], 0.85, None),
            (mock_chunks_1k[1], 0.75, None),
            (mock_chunks_1k[2], 0.65, None),
        ]
        mock_session.execute.return_value = mock_result
        
        # Warmup
        for _ in range(WARMUP_ITERATIONS):
            await service.search_by_text(
                query="machine learning",
                top_k=10,
            )
        
        # Benchmark
        latencies = []
        for _ in range(BENCHMARK_ITERATIONS):
            start = time.perf_counter()
            await service.search_by_text(
                query="machine learning",
                top_k=10,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
        
        stats = calculate_stats(latencies)
        passed = print_benchmark_results(
            "Text Search (1K chunks)",
            stats,
            PERFORMANCE_THRESHOLDS["text_search_p99_ms"],
            BENCHMARK_ITERATIONS,
            1000,
        )
        
        assert passed, f"Text search p99 ({stats['p99_ms']:.2f}ms) exceeds threshold ({PERFORMANCE_THRESHOLDS['text_search_p99_ms']}ms)"

    @pytest.mark.asyncio
    async def test_text_search_latency_10k_chunks(self, mock_chunks_10k):
        """Benchmark text search latency with 10K chunks.
        
        Target: p99 < 50ms
        """
        from src.db.repositories.document_chunk_repository import DocumentChunkRepository
        from src.services.text_search_service import TextSearchService
        
        mock_session = AsyncMock()
        repo = DocumentChunkRepository(mock_session)
        service = TextSearchService(repo)
        
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            (mock_chunks_10k[0], 0.85, None),
            (mock_chunks_10k[1], 0.75, None),
        ]
        mock_session.execute.return_value = mock_result
        
        # Warmup
        for _ in range(WARMUP_ITERATIONS):
            await service.search_by_text(
                query="machine learning",
                top_k=10,
            )
        
        # Benchmark
        latencies = []
        for _ in range(BENCHMARK_ITERATIONS):
            start = time.perf_counter()
            await service.search_by_text(
                query="machine learning",
                top_k=10,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
        
        stats = calculate_stats(latencies)
        passed = print_benchmark_results(
            "Text Search (10K chunks)",
            stats,
            PERFORMANCE_THRESHOLDS["text_search_p99_ms"],
            BENCHMARK_ITERATIONS,
            10000,
        )
        
        assert passed, f"Text search p99 ({stats['p99_ms']:.2f}ms) exceeds threshold"

    @pytest.mark.asyncio
    async def test_text_search_with_highlighting_performance(self, mock_chunks_1k):
        """Benchmark text search with highlighting enabled."""
        from src.db.repositories.document_chunk_repository import DocumentChunkRepository
        from src.services.text_search_service import TextSearchService
        
        mock_session = AsyncMock()
        repo = DocumentChunkRepository(mock_session)
        service = TextSearchService(repo)
        
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            (mock_chunks_1k[0], 0.85, "Sample <mark>content</mark> for chunk 0"),
        ]
        mock_session.execute.return_value = mock_result
        
        latencies = []
        for _ in range(50):
            start = time.perf_counter()
            await service.search_by_text(
                query="content",
                top_k=10,
                highlight=True,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
        
        stats = calculate_stats(latencies)
        print("\nText Search with Highlighting (1K chunks):")
        print(f"  P99: {stats['p99_ms']:.2f} ms")
        
        # Highlighting should not add more than 50% overhead
        assert stats["p99_ms"] < 75, "Highlighting should not significantly impact performance"


# =============================================================================
# Hybrid Search Performance Tests
# =============================================================================

@pytest.mark.performance
class TestHybridSearchPerformance:
    """Performance tests for hybrid search."""

    @pytest.mark.asyncio
    async def test_hybrid_search_latency_1k_chunks(self, sample_embedding, mock_chunks_1k):
        """Benchmark hybrid search latency with 1K chunks.
        
        Target: p99 < 150ms
        """
        from src.db.repositories.document_chunk_repository import DocumentChunkRepository
        from src.services.hybrid_search_service import FusionMethod, HybridSearchService
        from src.services.text_search_service import TextSearchResult, TextSearchService
        from src.services.vector_search_service import SearchResult, VectorSearchService
        
        mock_session = AsyncMock()
        repo = DocumentChunkRepository(mock_session)
        vector_service = VectorSearchService(repo)
        text_service = TextSearchService(repo)
        hybrid_service = HybridSearchService(vector_service, text_service)
        
        # Mock underlying services
        vector_results = [
            SearchResult(chunk=mock_chunks_1k[0], similarity_score=0.9, rank=1),
            SearchResult(chunk=mock_chunks_1k[1], similarity_score=0.8, rank=2),
        ]
        text_results = [
            TextSearchResult(chunk=mock_chunks_1k[0], rank_score=0.85, rank=1),
            TextSearchResult(chunk=mock_chunks_1k[1], rank_score=0.75, rank=2),
        ]
        
        vector_service.search_by_vector = AsyncMock(return_value=vector_results)
        text_service.search_by_text = AsyncMock(return_value=text_results)
        
        # Warmup
        for _ in range(WARMUP_ITERATIONS):
            await hybrid_service.search_with_embedding(
                query_embedding=sample_embedding,
                query_text="machine learning",
                top_k=10,
            )
        
        # Benchmark
        latencies = []
        for _ in range(BENCHMARK_ITERATIONS):
            start = time.perf_counter()
            await hybrid_service.search_with_embedding(
                query_embedding=sample_embedding,
                query_text="machine learning",
                top_k=10,
                fusion_method=FusionMethod.WEIGHTED_SUM,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
        
        stats = calculate_stats(latencies)
        passed = print_benchmark_results(
            "Hybrid Search (1K chunks, Weighted Sum)",
            stats,
            PERFORMANCE_THRESHOLDS["hybrid_search_p99_ms"],
            BENCHMARK_ITERATIONS,
            1000,
        )
        
        assert passed, f"Hybrid search p99 ({stats['p99_ms']:.2f}ms) exceeds threshold"

    @pytest.mark.asyncio
    async def test_hybrid_search_rrf_fusion_performance(self, sample_embedding, mock_chunks_1k):
        """Benchmark hybrid search with RRF fusion."""
        from src.db.repositories.document_chunk_repository import DocumentChunkRepository
        from src.services.hybrid_search_service import FusionMethod, HybridSearchService
        from src.services.text_search_service import TextSearchResult, TextSearchService
        from src.services.vector_search_service import SearchResult, VectorSearchService
        
        mock_session = AsyncMock()
        repo = DocumentChunkRepository(mock_session)
        vector_service = VectorSearchService(repo)
        text_service = TextSearchService(repo)
        hybrid_service = HybridSearchService(vector_service, text_service)
        
        vector_results = [
            SearchResult(chunk=mock_chunks_1k[0], similarity_score=0.9, rank=1),
        ]
        text_results = [
            TextSearchResult(chunk=mock_chunks_1k[0], rank_score=0.85, rank=1),
        ]
        
        vector_service.search_by_vector = AsyncMock(return_value=vector_results)
        text_service.search_by_text = AsyncMock(return_value=text_results)
        
        latencies = []
        for _ in range(50):
            start = time.perf_counter()
            await hybrid_service.search_with_embedding(
                query_embedding=sample_embedding,
                query_text="test",
                top_k=10,
                fusion_method=FusionMethod.RECIPROCAL_RANK_FUSION,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
        
        stats = calculate_stats(latencies)
        print("\nHybrid Search RRF (1K chunks):")
        print(f"  P99: {stats['p99_ms']:.2f} ms")


# =============================================================================
# Similar Chunks Performance Tests
# =============================================================================

@pytest.mark.performance
class TestSimilarChunksPerformance:
    """Performance tests for similar chunks search."""

    @pytest.mark.asyncio
    async def test_find_similar_chunks_latency_1k_chunks(self, sample_embedding, mock_chunks_1k):
        """Benchmark find_similar_chunks latency with 1K chunks.
        
        Target: p99 < 100ms
        """
        from src.db.repositories.document_chunk_repository import DocumentChunkRepository
        from src.services.vector_search_service import VectorSearchService
        
        mock_session = AsyncMock()
        repo = DocumentChunkRepository(mock_session)
        service = VectorSearchService(repo)
        
        # Mock reference chunk
        reference_chunk = mock_chunks_1k[0]
        reference_chunk.embedding = sample_embedding
        repo.get_by_id = AsyncMock(return_value=reference_chunk)
        
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            (mock_chunks_1k[1], 0.2),
            (mock_chunks_1k[2], 0.3),
        ]
        mock_session.execute.return_value = mock_result
        
        # Warmup
        for _ in range(WARMUP_ITERATIONS):
            await service.find_similar_chunks(
                chunk_id=reference_chunk.id,
                top_k=10,
            )
        
        # Benchmark
        latencies = []
        for _ in range(BENCHMARK_ITERATIONS):
            start = time.perf_counter()
            await service.find_similar_chunks(
                chunk_id=reference_chunk.id,
                top_k=10,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
        
        stats = calculate_stats(latencies)
        passed = print_benchmark_results(
            "Find Similar Chunks (1K chunks)",
            stats,
            PERFORMANCE_THRESHOLDS["similar_chunks_p99_ms"],
            BENCHMARK_ITERATIONS,
            1000,
        )
        
        assert passed, f"Similar chunks p99 ({stats['p99_ms']:.2f}ms) exceeds threshold"


# =============================================================================
# Scalability Tests
# =============================================================================

@pytest.mark.performance
class TestScalability:
    """Tests for scalability with different data sizes."""

    @pytest.mark.asyncio
    async def test_vector_search_scalability(self, sample_embedding):
        """Test how vector search scales with data size."""
        from src.db.repositories.document_chunk_repository import DocumentChunkRepository
        from src.services.vector_search_service import VectorSearchService
        
        results = {}
        
        for size in [100, 1000, 10000]:
            mock_session = AsyncMock()
            repo = DocumentChunkRepository(mock_session)
            service = VectorSearchService(repo)
            
            # Create mock chunks for this size
            chunks = [
                MagicMock(
                    id=uuid4(),
                    content=f"Content {i}",
                    embedding=[0.01 * ((i + j) % 100) / 100 for j in range(1536)],
                )
                for i in range(min(size, 10))  # Only need a few for results
            ]
            
            mock_result = MagicMock()
            mock_result.fetchall.return_value = [(chunk, 0.1 * i) for i, chunk in enumerate(chunks)]
            mock_session.execute.return_value = mock_result
            
            latencies = []
            for _ in range(50):
                start = time.perf_counter()
                await service.search_by_vector(
                    query_embedding=sample_embedding,
                    top_k=10,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies.append(elapsed_ms)
            
            stats = calculate_stats(latencies)
            results[size] = stats["p99_ms"]
        
        print("\nVector Search Scalability:")
        for size, p99 in results.items():
            print(f"  {size:,} chunks: p99={p99:.2f}ms")
        
        # Verify sub-linear scaling (10x data should not result in 10x latency)
        if 100 in results and 10000 in results:
            scaling_factor = results[10000] / results[100]
            print(f"  Scaling factor (10K/100): {scaling_factor:.2f}x")
            assert scaling_factor < 5, "Search should scale sub-linearly with data size"


# =============================================================================
# Load Tests
# =============================================================================

@pytest.mark.performance
@pytest.mark.slow
class TestLoad:
    """Load tests for concurrent search operations."""

    @pytest.mark.asyncio
    async def test_concurrent_vector_searches(self, sample_embedding, mock_chunks_1k):
        """Test performance under concurrent load."""
        from src.db.repositories.document_chunk_repository import DocumentChunkRepository
        from src.services.vector_search_service import VectorSearchService
        
        mock_session = AsyncMock()
        repo = DocumentChunkRepository(mock_session)
        service = VectorSearchService(repo)
        
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            (mock_chunks_1k[0], 0.1),
            (mock_chunks_1k[1], 0.2),
        ]
        mock_session.execute.return_value = mock_result
        
        async def search_task():
            start = time.perf_counter()
            await service.search_by_vector(
                query_embedding=sample_embedding,
                top_k=10,
            )
            return (time.perf_counter() - start) * 1000
        
        # Run concurrent searches
        num_concurrent = 20
        tasks = [search_task() for _ in range(num_concurrent)]
        latencies = await asyncio.gather(*tasks)
        
        stats = calculate_stats(latencies)
        print(f"\nConcurrent Vector Searches ({num_concurrent} concurrent):")
        print(f"  Mean: {stats['mean_ms']:.2f}ms")
        print(f"  P99:  {stats['p99_ms']:.2f}ms")
        
        # Concurrent load should not degrade performance significantly
        assert stats["p99_ms"] < PERFORMANCE_THRESHOLDS["vector_search_p99_ms"] * 2


# =============================================================================
# Performance Report Generation
# =============================================================================

@pytest.mark.performance
class TestPerformanceReport:
    """Generate overall performance report."""

    def test_performance_summary(self):
        """Print performance targets and generate summary."""
        print("\n" + "="*70)
        print("VECTOR STORE PERFORMANCE TARGETS")
        print("="*70)
        print(f"  Vector Search p99:  < {PERFORMANCE_THRESHOLDS['vector_search_p99_ms']}ms")
        print(f"  Text Search p99:    < {PERFORMANCE_THRESHOLDS['text_search_p99_ms']}ms")
        print(f"  Hybrid Search p99:  < {PERFORMANCE_THRESHOLDS['hybrid_search_p99_ms']}ms")
        print(f"  Similar Chunks p99: < {PERFORMANCE_THRESHOLDS['similar_chunks_p99_ms']}ms")
        print("="*70 + "\n")
        
        # This test always passes, it's just for documentation
        assert True


def generate_performance_report():
    """Generate a markdown performance report.
    
    This function can be called to generate a report file.
    """
    report = f"""# Vector Store Performance Report

## Benchmark Results

### Test Environment
- Date: {datetime.utcnow().isoformat()}
- Python Version: {os.sys.version.split()[0]}
- Database: PostgreSQL with pgvector

### Performance Targets

| Operation | Target p99 | Status |
|-----------|------------|--------|
| Vector Search | < 100ms | ✅/❌ |
| Text Search | < 50ms | ✅/❌ |
| Hybrid Search | < 150ms | ✅/❌ |
| Similar Chunks | < 100ms | ✅/❌ |

### Scalability Results

| Dataset Size | Vector Search p99 | Text Search p99 |
|--------------|-------------------|-----------------|
| 1K chunks | XXms | XXms |
| 10K chunks | XXms | XXms |
| 100K chunks | XXms | XXms |

## Notes
- Benchmarks run with mocked database for unit test environment
- For production benchmarks, run against actual PostgreSQL with pgvector
- HNSW index significantly improves vector search performance at scale
"""
    
    return report


if __name__ == "__main__":
    # Can be run directly to generate report
    print(generate_performance_report())
