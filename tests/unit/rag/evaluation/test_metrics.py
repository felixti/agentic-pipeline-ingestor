"""Unit tests for retrieval metrics.

Tests all retrieval metric computations including MRR, NDCG, Recall@K,
Precision@K, and Hit Rate@K.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.rag.evaluation.metrics import (
    BatchRetrievalMetrics,
    RetrievalMetrics,
    compute_hit_rate_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
)


# Helper class to create mock results
class MockResult:
    """Mock retrieval result for testing."""
    
    def __init__(self, id: str):
        self.id = id


class TestRetrievalMetricsMRR:
    """Tests for MRR (Mean Reciprocal Rank) metric."""
    
    def test_mrr_first_position(self):
        """Test MRR when relevant doc is at position 1."""
        results = [MockResult("doc_1"), MockResult("doc_2")]
        ground_truth = ["doc_1"]
        
        mrr = RetrievalMetrics.mrr(results, ground_truth)
        assert mrr == 1.0  # 1/1 = 1.0
    
    def test_mrr_second_position(self):
        """Test MRR when relevant doc is at position 2."""
        results = [MockResult("doc_1"), MockResult("doc_2"), MockResult("doc_3")]
        ground_truth = ["doc_2"]
        
        mrr = RetrievalMetrics.mrr(results, ground_truth)
        assert mrr == 0.5  # 1/2 = 0.5
    
    def test_mrr_third_position(self):
        """Test MRR when relevant doc is at position 3."""
        results = [
            MockResult("doc_1"),
            MockResult("doc_2"),
            MockResult("doc_3"),
            MockResult("doc_4")
        ]
        ground_truth = ["doc_3"]
        
        mrr = RetrievalMetrics.mrr(results, ground_truth)
        assert mrr == pytest.approx(0.333, abs=0.001)  # 1/3 ≈ 0.333
    
    def test_mrr_no_relevant_docs(self):
        """Test MRR when no relevant docs are retrieved."""
        results = [MockResult("doc_1"), MockResult("doc_2")]
        ground_truth = ["doc_3"]
        
        mrr = RetrievalMetrics.mrr(results, ground_truth)
        assert mrr == 0.0
    
    def test_mrr_multiple_relevant_first_wins(self):
        """Test MRR with multiple relevant docs - first position wins."""
        results = [
            MockResult("doc_1"),
            MockResult("doc_2"),
            MockResult("doc_3")
        ]
        ground_truth = ["doc_2", "doc_3"]  # doc_2 is at position 2
        
        mrr = RetrievalMetrics.mrr(results, ground_truth)
        assert mrr == 0.5  # First relevant (doc_2) is at rank 2
    
    def test_mrr_empty_ground_truth(self):
        """Test MRR with empty ground truth."""
        results = [MockResult("doc_1")]
        ground_truth = []
        
        mrr = RetrievalMetrics.mrr(results, ground_truth)
        assert mrr == 0.0
    
    def test_mrr_empty_results(self):
        """Test MRR with empty results."""
        results = []
        ground_truth = ["doc_1"]
        
        mrr = RetrievalMetrics.mrr(results, ground_truth)
        assert mrr == 0.0


class TestRetrievalMetricsNDCG:
    """Tests for NDCG@K metric."""
    
    def test_ndcg_perfect_ranking(self):
        """Test NDCG with perfect ranking."""
        results = [
            MockResult("doc_1"),
            MockResult("doc_2"),
            MockResult("doc_3")
        ]
        ground_truth = ["doc_1", "doc_2", "doc_3"]
        
        ndcg = RetrievalMetrics.ndcg_at_k(results, ground_truth, k=3)
        assert ndcg == pytest.approx(1.0, abs=0.001)
    
    def test_ndcg_reverse_ranking(self):
        """Test NDCG with reverse ranking."""
        results = [
            MockResult("doc_3"),
            MockResult("doc_2"),
            MockResult("doc_1")
        ]
        ground_truth = ["doc_1", "doc_2", "doc_3"]
        
        ndcg = RetrievalMetrics.ndcg_at_k(results, ground_truth, k=3)
        # When all docs are relevant, NDCG is 1.0 regardless of order
        # This is because ideal DCG = actual DCG (all have gain=1)
        assert ndcg == 1.0
    
    def test_ndcg_partial_relevance(self):
        """Test NDCG with partial relevance."""
        results = [
            MockResult("doc_1"),
            MockResult("doc_4"),
            MockResult("doc_2")
        ]
        ground_truth = ["doc_1", "doc_2"]
        
        ndcg = RetrievalMetrics.ndcg_at_k(results, ground_truth, k=3)
        # doc_1 at pos 1 (gain = 1/1 = 1.0)
        # doc_2 at pos 3 (gain = 1/log2(4) = 0.5)
        # Ideal: doc_1 at 1, doc_2 at 2 (gain = 1 + 1/log2(3) ≈ 1.63)
        assert 0.0 < ndcg < 1.0
    
    def test_ndcg_no_relevant_docs(self):
        """Test NDCG with no relevant docs."""
        results = [MockResult("doc_1"), MockResult("doc_2")]
        ground_truth = ["doc_3"]
        
        ndcg = RetrievalMetrics.ndcg_at_k(results, ground_truth, k=3)
        assert ndcg == 0.0
    
    def test_ndcg_k_smaller_than_results(self):
        """Test NDCG when k is smaller than results."""
        results = [
            MockResult("doc_1"),
            MockResult("doc_2"),
            MockResult("doc_3"),
            MockResult("doc_4")
        ]
        ground_truth = ["doc_1", "doc_2", "doc_3", "doc_4"]
        
        ndcg = RetrievalMetrics.ndcg_at_k(results, ground_truth, k=2)
        # Only top 2 considered
        assert 0.0 < ndcg <= 1.0
    
    def test_ndcg_empty_ground_truth(self):
        """Test NDCG with empty ground truth."""
        results = [MockResult("doc_1")]
        ground_truth = []
        
        ndcg = RetrievalMetrics.ndcg_at_k(results, ground_truth, k=5)
        assert ndcg == 0.0
    
    def test_ndcg_k_zero(self):
        """Test NDCG with k=0."""
        results = [MockResult("doc_1")]
        ground_truth = ["doc_1"]
        
        ndcg = RetrievalMetrics.ndcg_at_k(results, ground_truth, k=0)
        assert ndcg == 0.0


class TestRetrievalMetricsRecall:
    """Tests for Recall@K metric."""
    
    def test_recall_at_k_perfect(self):
        """Test Recall@K when all relevant docs are retrieved."""
        results = [
            MockResult("doc_1"),
            MockResult("doc_2"),
            MockResult("doc_3")
        ]
        ground_truth = ["doc_1", "doc_2"]
        
        recall = RetrievalMetrics.recall_at_k(results, ground_truth, k=3)
        assert recall == 1.0  # All relevant docs found
    
    def test_recall_at_k_partial(self):
        """Test Recall@K when some relevant docs are retrieved."""
        results = [
            MockResult("doc_1"),
            MockResult("doc_4"),
            MockResult("doc_5")
        ]
        ground_truth = ["doc_1", "doc_2", "doc_3"]
        
        recall = RetrievalMetrics.recall_at_k(results, ground_truth, k=3)
        assert recall == pytest.approx(0.333, abs=0.001)  # 1/3
    
    def test_recall_at_k_none(self):
        """Test Recall@K when no relevant docs are retrieved."""
        results = [MockResult("doc_4"), MockResult("doc_5")]
        ground_truth = ["doc_1", "doc_2"]
        
        recall = RetrievalMetrics.recall_at_k(results, ground_truth, k=3)
        assert recall == 0.0
    
    def test_recall_at_k_k_limits_results(self):
        """Test that k limits the results considered."""
        results = [
            MockResult("doc_1"),
            MockResult("doc_2"),
            MockResult("doc_3")
        ]
        ground_truth = ["doc_1", "doc_2", "doc_3"]
        
        recall = RetrievalMetrics.recall_at_k(results, ground_truth, k=2)
        assert recall == pytest.approx(0.667, abs=0.001)  # 2/3
    
    def test_recall_at_k_empty_ground_truth(self):
        """Test Recall@K with empty ground truth."""
        results = [MockResult("doc_1")]
        ground_truth = []
        
        recall = RetrievalMetrics.recall_at_k(results, ground_truth, k=5)
        assert recall == 0.0


class TestRetrievalMetricsPrecision:
    """Tests for Precision@K metric."""
    
    def test_precision_at_k_perfect(self):
        """Test Precision@K when all retrieved docs are relevant."""
        results = [MockResult("doc_1"), MockResult("doc_2")]
        ground_truth = ["doc_1", "doc_2"]
        
        precision = RetrievalMetrics.precision_at_k(results, ground_truth, k=2)
        assert precision == 1.0  # 2/2
    
    def test_precision_at_k_partial(self):
        """Test Precision@K when some retrieved docs are relevant."""
        results = [
            MockResult("doc_1"),
            MockResult("doc_4"),
            MockResult("doc_5")
        ]
        ground_truth = ["doc_1", "doc_2"]
        
        precision = RetrievalMetrics.precision_at_k(results, ground_truth, k=3)
        assert precision == pytest.approx(0.333, abs=0.001)  # 1/3
    
    def test_precision_at_k_none(self):
        """Test Precision@K when no retrieved docs are relevant."""
        results = [MockResult("doc_4"), MockResult("doc_5")]
        ground_truth = ["doc_1", "doc_2"]
        
        precision = RetrievalMetrics.precision_at_k(results, ground_truth, k=2)
        assert precision == 0.0
    
    def test_precision_at_k_k_larger_than_results(self):
        """Test Precision@K when k is larger than results."""
        results = [MockResult("doc_1")]
        ground_truth = ["doc_1"]
        
        precision = RetrievalMetrics.precision_at_k(results, ground_truth, k=5)
        # With k=5 and 1 relevant result out of 1 retrieved, precision = 1/1 = 1.0
        # (we only have 1 result, so min(k, len(results)) = 1)
        assert precision == 1.0
    
    def test_precision_at_k_k_zero(self):
        """Test Precision@K with k=0."""
        results = [MockResult("doc_1")]
        ground_truth = ["doc_1"]
        
        precision = RetrievalMetrics.precision_at_k(results, ground_truth, k=0)
        assert precision == 0.0
    
    def test_precision_at_k_empty_results(self):
        """Test Precision@K with empty results."""
        results = []
        ground_truth = ["doc_1"]
        
        precision = RetrievalMetrics.precision_at_k(results, ground_truth, k=5)
        assert precision == 0.0


class TestRetrievalMetricsHitRate:
    """Tests for Hit Rate@K metric."""
    
    def test_hit_rate_at_k_hit(self):
        """Test Hit Rate@K when at least one relevant doc is in top k."""
        results = [
            MockResult("doc_4"),
            MockResult("doc_1"),
            MockResult("doc_5")
        ]
        ground_truth = ["doc_1", "doc_2"]
        
        hit_rate = RetrievalMetrics.hit_rate_at_k(results, ground_truth, k=3)
        assert hit_rate == 1.0
    
    def test_hit_rate_at_k_miss(self):
        """Test Hit Rate@K when no relevant docs are in top k."""
        results = [
            MockResult("doc_4"),
            MockResult("doc_5"),
            MockResult("doc_6")
        ]
        ground_truth = ["doc_1", "doc_2"]
        
        hit_rate = RetrievalMetrics.hit_rate_at_k(results, ground_truth, k=3)
        assert hit_rate == 0.0
    
    def test_hit_rate_at_k_first_position(self):
        """Test Hit Rate@K when relevant doc is at first position."""
        results = [MockResult("doc_1"), MockResult("doc_4")]
        ground_truth = ["doc_1", "doc_2"]
        
        hit_rate = RetrievalMetrics.hit_rate_at_k(results, ground_truth, k=2)
        assert hit_rate == 1.0


class TestRetrievalMetricsAveragePrecision:
    """Tests for Average Precision metric."""
    
    def test_ap_perfect(self):
        """Test AP with perfect ranking."""
        results = [MockResult("doc_1"), MockResult("doc_2")]
        ground_truth = ["doc_1", "doc_2"]
        
        ap = RetrievalMetrics.average_precision(results, ground_truth)
        assert ap == 1.0
    
    def test_ap_partial(self):
        """Test AP with partial relevance."""
        results = [
            MockResult("doc_1"),
            MockResult("doc_4"),
            MockResult("doc_2")
        ]
        ground_truth = ["doc_1", "doc_2"]
        
        ap = RetrievalMetrics.average_precision(results, ground_truth)
        # doc_1 at rank 1: precision = 1/1
        # doc_2 at rank 3: precision = 2/3
        # AP = (1 + 2/3) / 2 = 0.833
        assert 0.0 < ap < 1.0


class TestRetrievalMetricsComputeAll:
    """Tests for compute_all_metrics method."""
    
    def test_compute_all_returns_all_metrics(self):
        """Test that compute_all returns all expected metrics."""
        results = [
            MockResult("doc_1"),
            MockResult("doc_2"),
            MockResult("doc_3")
        ]
        ground_truth = ["doc_1", "doc_2"]
        
        metrics = RetrievalMetrics.compute_all_metrics(
            results, ground_truth, k_values=[5, 10]
        )
        
        assert "mrr" in metrics
        assert "ndcg_at_5" in metrics
        assert "ndcg_at_10" in metrics
        assert "recall_at_5" in metrics
        assert "recall_at_10" in metrics
        assert "precision_at_5" in metrics
        assert "precision_at_10" in metrics
        assert "hit_rate_at_5" in metrics
        assert "hit_rate_at_10" in metrics
        assert "latency_ms" in metrics
    
    def test_compute_all_latency_tracking(self):
        """Test that latency is tracked."""
        results = [MockResult("doc_1")]
        ground_truth = ["doc_1"]
        
        metrics = RetrievalMetrics.compute_all_metrics(results, ground_truth)
        
        assert metrics["latency_ms"] >= 0.0
        # Should be very fast (< 10ms target)
        assert metrics["latency_ms"] < 10.0


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_compute_mrr(self):
        """Test compute_mrr convenience function."""
        results = [MockResult("doc_1"), MockResult("doc_2")]
        ground_truth = ["doc_2"]
        
        mrr = compute_mrr(results, ground_truth)
        assert mrr == 0.5
    
    def test_compute_ndcg_at_k(self):
        """Test compute_ndcg_at_k convenience function."""
        results = [MockResult("doc_1")]
        ground_truth = ["doc_1"]
        
        ndcg = compute_ndcg_at_k(results, ground_truth, k=5)
        assert ndcg == 1.0
    
    def test_compute_recall_at_k(self):
        """Test compute_recall_at_k convenience function."""
        results = [MockResult("doc_1")]
        ground_truth = ["doc_1", "doc_2"]
        
        recall = compute_recall_at_k(results, ground_truth, k=5)
        assert recall == 0.5
    
    def test_compute_precision_at_k(self):
        """Test compute_precision_at_k convenience function."""
        results = [MockResult("doc_1")]
        ground_truth = ["doc_1"]
        
        precision = compute_precision_at_k(results, ground_truth, k=5)
        # With k=5 and 1 relevant result out of 1 retrieved, precision = 1/1 = 1.0
        assert precision == 1.0
    
    def test_compute_hit_rate_at_k(self):
        """Test compute_hit_rate_at_k convenience function."""
        results = [MockResult("doc_1")]
        ground_truth = ["doc_1"]
        
        hit_rate = compute_hit_rate_at_k(results, ground_truth, k=5)
        assert hit_rate == 1.0


class TestBatchRetrievalMetrics:
    """Tests for BatchRetrievalMetrics class."""
    
    def test_add_query_results(self):
        """Test adding query results."""
        batch = BatchRetrievalMetrics(k_values=[5])
        
        results = [MockResult("doc_1")]
        ground_truth = ["doc_1"]
        
        metrics = batch.add_query_results("q1", results, ground_truth)
        
        assert "mrr" in metrics
        assert metrics["mrr"] == 1.0
    
    def test_get_aggregate_metrics(self):
        """Test getting aggregate metrics."""
        batch = BatchRetrievalMetrics(k_values=[5])
        
        # Add multiple queries
        batch.add_query_results("q1", [MockResult("doc_1")], ["doc_1"])
        batch.add_query_results("q2", [MockResult("doc_2")], ["doc_2"])
        batch.add_query_results("q3", [MockResult("doc_x")], ["doc_3"])
        
        aggregate = batch.get_aggregate_metrics()
        
        assert "mean_mrr" in aggregate
        assert "median_mrr" in aggregate
        assert "std_mrr" in aggregate
        assert "num_queries" in aggregate
        assert aggregate["num_queries"] == 3
    
    def test_get_aggregate_metrics_empty(self):
        """Test getting aggregate metrics with no queries."""
        batch = BatchRetrievalMetrics()
        
        aggregate = batch.get_aggregate_metrics()
        
        assert aggregate == {}
    
    def test_reset(self):
        """Test resetting batch metrics."""
        batch = BatchRetrievalMetrics()
        batch.add_query_results("q1", [MockResult("doc_1")], ["doc_1"])
        
        assert len(batch.query_results) == 1
        
        batch.reset()
        
        assert len(batch.query_results) == 0


class TestPerformanceRequirements:
    """Tests for performance requirements."""
    
    def test_retrieval_metrics_performance(self):
        """Test that metric computation is fast (< 10ms)."""
        import time
        
        # Create larger dataset
        results = [MockResult(f"doc_{i}") for i in range(100)]
        ground_truth = [f"doc_{i}" for i in range(20)]
        
        start = time.perf_counter()
        metrics = RetrievalMetrics.compute_all_metrics(
            results, ground_truth, k_values=[5, 10, 20]
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # Should complete in < 10ms
        assert elapsed_ms < 10.0, f"Metrics computation took {elapsed_ms:.2f}ms"
        assert metrics["latency_ms"] < 10.0
