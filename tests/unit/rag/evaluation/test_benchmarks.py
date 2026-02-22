"""Unit tests for benchmark framework.

Tests the BenchmarkRunner class including dataset loading, benchmark execution,
A/B testing, and strategy comparison.
"""

import json
import tempfile
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.rag.evaluation.benchmarks import BenchmarkRunner
from src.rag.evaluation.models import BenchmarkConfig, BenchmarkDataset


# Helper classes
class MockResult:
    """Mock retrieval result."""
    def __init__(self, id: str, content: str = ""):
        self.id = id
        self.content = content


class MockRAGResponse:
    """Mock RAG system response."""
    def __init__(self, answer: str, retrieved_chunks: list, latency_ms: float = 100.0):
        self.answer = answer
        self.retrieved_chunks = retrieved_chunks
        self.latency_ms = latency_ms


class TestBenchmarkRunnerInitialization:
    """Tests for BenchmarkRunner initialization."""
    
    def test_default_initialization(self):
        """Test initialization with default values."""
        runner = BenchmarkRunner()
        
        assert runner.random_seed == 42
    
    def test_custom_initialization(self):
        """Test initialization with custom seed."""
        runner = BenchmarkRunner(random_seed=123)
        
        assert runner.random_seed == 123


class TestBenchmarkRunnerDatasetLoading:
    """Tests for dataset loading."""
    
    @pytest.mark.asyncio
    async def test_load_ms_marco(self):
        """Test loading MS MARCO dataset."""
        runner = BenchmarkRunner()
        
        dataset = await runner.load_dataset("ms_marco", max_queries=3)
        
        assert len(dataset) <= 3
        assert all(isinstance(d, BenchmarkDataset) for d in dataset)
        assert all(d.id.startswith("ms_marco") for d in dataset)
    
    @pytest.mark.asyncio
    async def test_load_ms_marco_all(self):
        """Test loading all MS MARCO queries."""
        runner = BenchmarkRunner()
        
        dataset = await runner.load_dataset("ms_marco")
        
        assert len(dataset) > 0
    
    @pytest.mark.asyncio
    async def test_load_custom_qa_from_file(self):
        """Test loading custom QA dataset from file."""
        runner = BenchmarkRunner()
        
        # Create temporary JSON file
        test_data = [
            {
                "id": "q1",
                "query": "What is AI?",
                "ground_truth_relevant_ids": ["doc1"],
                "ground_truth_answer": "AI is artificial intelligence."
            },
            {
                "id": "q2",
                "query": "What is ML?",
                "ground_truth_relevant_ids": ["doc2"],
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            dataset = await runner.load_dataset(
                "custom_qa",
                dataset_path=temp_path
            )
            
            assert len(dataset) == 2
            assert dataset[0].id == "q1"
            assert dataset[0].query == "What is AI?"
            assert dataset[0].ground_truth_relevant_ids == ["doc1"]
        finally:
            import os
            os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_load_custom_qa_with_max_queries(self):
        """Test loading custom QA with max_queries limit."""
        runner = BenchmarkRunner()
        
        test_data = [
            {"id": f"q{i}", "query": f"Query {i}"}
            for i in range(10)
        ]
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            dataset = await runner.load_dataset(
                "custom_qa",
                dataset_path=temp_path,
                max_queries=5
            )
            
            assert len(dataset) == 5
        finally:
            import os
            os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_load_unknown_dataset(self):
        """Test loading unknown dataset raises error."""
        runner = BenchmarkRunner()
        
        with pytest.raises(ValueError, match="Unknown benchmark"):
            await runner.load_dataset("unknown_dataset")


class TestBenchmarkRunnerRunBenchmark:
    """Tests for benchmark execution."""
    
    @pytest.mark.asyncio
    async def test_run_benchmark_success(self):
        """Test successful benchmark run."""
        runner = BenchmarkRunner()
        
        # Mock RAG system
        mock_system = MagicMock()
        mock_system.query = AsyncMock(return_value=MockRAGResponse(
            answer="Test answer",
            retrieved_chunks=[MockResult("doc_1")],
            latency_ms=100.0
        ))
        
        config = BenchmarkConfig(
            name="test",
            dataset_path="ms_marco",
            max_queries=2,
            k_values=[5]
        )
        
        result = await runner.run_benchmark(
            benchmark_name="ms_marco",
            rag_system=mock_system,
            config=config
        )
        
        assert result.benchmark_name == "ms_marco"
        assert result.config is not None
        assert result.total_queries == 2
        assert result.successful_queries == 2
        assert result.failed_queries == 0
        assert result.started_at is not None
    
    @pytest.mark.asyncio
    async def test_run_benchmark_with_failures(self):
        """Test benchmark run with some failures."""
        runner = BenchmarkRunner()
        
        # Mock RAG system that fails on first call
        mock_system = MagicMock()
        mock_system.query = AsyncMock(side_effect=[
            Exception("Error"),
            MockRAGResponse("Answer", [MockResult("doc_1")])
        ])
        
        config = BenchmarkConfig(
            name="test",
            dataset_path="ms_marco",
            max_queries=2,
            k_values=[5]
        )
        
        result = await runner.run_benchmark(
            benchmark_name="ms_marco",
            rag_system=mock_system,
            config=config
        )
        
        assert result.total_queries == 2
        assert result.successful_queries == 1
        assert result.failed_queries == 1
    
    @pytest.mark.asyncio
    async def test_run_benchmark_aggregate_metrics(self):
        """Test that aggregate metrics are computed."""
        runner = BenchmarkRunner()
        
        mock_system = MagicMock()
        mock_system.query = AsyncMock(return_value=MockRAGResponse(
            answer="Answer",
            retrieved_chunks=[MockResult("doc_1")],
            latency_ms=100.0
        ))
        
        config = BenchmarkConfig(
            name="test",
            dataset_path="ms_marco",
            max_queries=3,
            k_values=[5]
        )
        
        result = await runner.run_benchmark(
            benchmark_name="ms_marco",
            rag_system=mock_system,
            config=config
        )
        
        # Should have aggregate metrics
        assert len(result.aggregate_metrics) > 0


class TestBenchmarkRunnerAggregateMetrics:
    """Tests for aggregate metrics computation."""
    
    def test_compute_aggregate_metrics_empty(self):
        """Test aggregate metrics with empty results."""
        runner = BenchmarkRunner()
        
        aggregate = runner._compute_aggregate_metrics([])
        
        assert aggregate == {}
    
    def test_compute_aggregate_metrics_basic(self):
        """Test basic aggregate metrics computation."""
        runner = BenchmarkRunner()
        
        per_query_results = [
            {
                "query_id": "q1",
                "retrieval": {"mrr": 0.8, "latency_ms": 10.0},
                "generation": {"bertscore_f1": 0.9},
                "latency_ms": 100.0
            },
            {
                "query_id": "q2",
                "retrieval": {"mrr": 0.6, "latency_ms": 15.0},
                "generation": {"bertscore_f1": 0.85},
                "latency_ms": 120.0
            }
        ]
        
        aggregate = runner._compute_aggregate_metrics(per_query_results)
        
        assert "mean_mrr" in aggregate
        assert "median_mrr" in aggregate
        assert "std_mrr" in aggregate
        assert aggregate["mean_mrr"] == pytest.approx(0.7, abs=0.01)


class TestBenchmarkRunnerABTest:
    """Tests for A/B testing."""
    
    @pytest.mark.asyncio
    async def test_run_ab_test(self):
        """Test running an A/B test."""
        runner = BenchmarkRunner()
        
        # Mock control system (lower performance)
        mock_control = MagicMock()
        mock_control.query = AsyncMock(return_value=MockRAGResponse(
            answer="Control",
            retrieved_chunks=[MockResult("doc_1")],
            latency_ms=100.0
        ))
        
        # Mock treatment system (higher performance)
        mock_treatment = MagicMock()
        mock_treatment.query = AsyncMock(return_value=MockRAGResponse(
            answer="Treatment",
            retrieved_chunks=[MockResult("doc_1"), MockResult("doc_2")],
            latency_ms=120.0
        ))
        
        result = await runner.run_ab_test(
            test_name="test_ab",
            control_system=mock_control,
            treatment_system=mock_treatment,
            sample_size=3
        )
        
        assert result.test_name == "test_ab"
        assert result.control_strategy == "control"
        assert result.treatment_strategy == "treatment"
        assert "mrr" in result.control_metrics or len(result.control_metrics) == 0
        assert len(result.recommendation) > 0
    
    def test_generate_recommendation_improvement(self):
        """Test recommendation generation for improvements."""
        runner = BenchmarkRunner()
        
        improvements = {"mrr": 10.0}
        is_significant = {"mrr": True}
        control_metrics = {"mrr": 0.7}
        treatment_metrics = {"mrr": 0.77}
        
        rec = runner._generate_recommendation(
            improvements, is_significant, control_metrics, treatment_metrics
        )
        
        assert "Deploy treatment" in rec
    
    def test_generate_recommendation_regression(self):
        """Test recommendation generation for regressions."""
        runner = BenchmarkRunner()
        
        improvements = {"mrr": -10.0}
        is_significant = {"mrr": True}
        control_metrics = {"mrr": 0.7}
        treatment_metrics = {"mrr": 0.63}
        
        rec = runner._generate_recommendation(
            improvements, is_significant, control_metrics, treatment_metrics
        )
        
        assert "Keep control" in rec
    
    def test_generate_recommendation_mixed(self):
        """Test recommendation generation for mixed results."""
        runner = BenchmarkRunner()
        
        improvements = {"mrr": 10.0, "latency": -20.0}
        is_significant = {"mrr": True, "latency": True}
        control_metrics = {"mrr": 0.7, "latency": 100.0}
        treatment_metrics = {"mrr": 0.77, "latency": 120.0}
        
        rec = runner._generate_recommendation(
            improvements, is_significant, control_metrics, treatment_metrics
        )
        
        assert "Mixed results" in rec


class TestBenchmarkRunnerStrategyComparison:
    """Tests for strategy comparison."""
    
    @pytest.mark.asyncio
    async def test_compare_strategies(self):
        """Test comparing two strategies."""
        runner = BenchmarkRunner()
        
        mock_a = MagicMock()
        mock_a.query = AsyncMock(return_value=MockRAGResponse(
            answer="A",
            retrieved_chunks=[MockResult("doc_1")],
            latency_ms=100.0
        ))
        
        mock_b = MagicMock()
        mock_b.query = AsyncMock(return_value=MockRAGResponse(
            answer="B",
            retrieved_chunks=[MockResult("doc_2")],
            latency_ms=150.0
        ))
        
        result = await runner.compare_strategies(
            strategy_a_name="fast",
            strategy_b_name="accurate",
            strategy_a_system=mock_a,
            strategy_b_system=mock_b,
            sample_size=3
        )
        
        assert result.strategy_a_name == "fast"
        assert result.strategy_b_name == "accurate"
        assert result.sample_size == 3
        assert result.winner is not None


class TestBenchmarkRunnerExtractMetrics:
    """Tests for metric extraction utilities."""
    
    def test_extract_key_metrics(self):
        """Test extracting key metrics from benchmark result."""
        from src.rag.evaluation.models import BenchmarkResult
        
        runner = BenchmarkRunner()
        
        result = BenchmarkResult(
            benchmark_name="test",
            aggregate_metrics={
                "mean_mrr": 0.75,
                "mean_ndcg_at_k_10": 0.80,
                "mean_bertscore_f1": 0.90,
                "p95_overall_latency_ms": 500.0
            }
        )
        
        metrics = runner._extract_key_metrics(result)
        
        assert "mrr" in metrics
        assert metrics["mrr"] == 0.75
        assert "latency_p95_ms" in metrics
        assert metrics["latency_p95_ms"] == 500.0
    
    def test_extract_metric_values(self):
        """Test extracting per-query metric values."""
        from src.rag.evaluation.models import BenchmarkResult
        
        runner = BenchmarkRunner()
        
        result = BenchmarkResult(
            benchmark_name="test",
            per_query_results=[
                {
                    "retrieval": {"mrr": 0.8, "ndcg_at_k": {"10": 0.85}},
                    "generation": {"bertscore_f1": 0.9}
                },
                {
                    "retrieval": {"mrr": 0.7, "ndcg_at_k": {"10": 0.75}},
                    "generation": {"bertscore_f1": 0.85}
                }
            ]
        )
        
        mrr_values = runner._extract_metric_values(result, "mrr")
        assert len(mrr_values) == 2
        assert mrr_values[0] == 0.8
        assert mrr_values[1] == 0.7


class TestBenchmarkRunnerPerformance:
    """Tests for performance requirements."""
    
    @pytest.mark.asyncio
    async def test_benchmark_performance(self):
        """Test that benchmark runs in reasonable time."""
        import time
        
        runner = BenchmarkRunner()
        
        mock_system = MagicMock()
        mock_system.query = AsyncMock(return_value=MockRAGResponse(
            answer="Answer",
            retrieved_chunks=[MockResult("doc_1")],
            latency_ms=10.0
        ))
        
        config = BenchmarkConfig(
            name="test",
            dataset_path="ms_marco",
            max_queries=5,
            k_values=[5]
        )
        
        start = time.perf_counter()
        result = await runner.run_benchmark(
            benchmark_name="ms_marco",
            rag_system=mock_system,
            config=config
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # Should complete in reasonable time (adjust threshold as needed)
        assert elapsed_ms < 5000  # 5 seconds for 5 queries
        assert result.total_latency_ms < 10000  # Total reported latency
