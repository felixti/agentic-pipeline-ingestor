"""Unit tests for the main RAG Evaluator.

Tests the RAGEvaluator class including retrieval/generation evaluation,
benchmark runs, A/B testing, and alerting.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rag.evaluation import RAGEvaluator
from src.rag.evaluation.models import (
    AlertSeverity,
    BenchmarkConfig,
    EvaluationAlert,
    EvaluationConfig,
    MetricThresholds,
)


# Helper class for mock results
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


class TestRAGEvaluatorInitialization:
    """Tests for RAGEvaluator initialization."""
    
    def test_default_initialization(self):
        """Test initialization with default values."""
        evaluator = RAGEvaluator()
        
        assert evaluator.config is not None
        assert evaluator.retrieval_metrics is not None
        assert evaluator.generation_metrics is not None
        assert evaluator.benchmark_runner is not None
        assert evaluator.alert_history == []
    
    def test_custom_initialization(self):
        """Test initialization with custom values."""
        config = EvaluationConfig(enabled=False, sample_rate=0.5)
        
        evaluator = RAGEvaluator(config=config)
        
        assert evaluator.config.enabled is False
        assert evaluator.config.sample_rate == 0.5
    
    def test_load_config_from_settings(self):
        """Test loading config from application settings."""
        evaluator = RAGEvaluator()
        
        # Should have loaded defaults
        assert evaluator.config.enabled is True
        assert evaluator.config.sample_rate == 0.1


class TestRAGEvaluatorRetrievalEvaluation:
    """Tests for retrieval evaluation."""
    
    @pytest.mark.asyncio
    async def test_evaluate_retrieval_basic(self):
        """Test basic retrieval evaluation."""
        evaluator = RAGEvaluator()
        
        results = [
            MockResult("doc_1"),
            MockResult("doc_2"),
            MockResult("doc_3")
        ]
        ground_truth = ["doc_1", "doc_2"]
        
        eval_result = await evaluator.evaluate_retrieval(
            query="What is AI?",
            results=results,
            ground_truth=ground_truth,
            k_values=[5, 10]
        )
        
        assert eval_result.query == "What is AI?"
        assert 0.0 <= eval_result.mrr <= 1.0
        assert "5" in eval_result.ndcg_at_k
        assert "10" in eval_result.ndcg_at_k
        assert eval_result.latency_ms >= 0
        assert isinstance(eval_result.timestamp, datetime)
    
    @pytest.mark.asyncio
    async def test_evaluate_retrieval_perfect_match(self):
        """Test retrieval evaluation with perfect results."""
        evaluator = RAGEvaluator()
        
        results = [MockResult("doc_1"), MockResult("doc_2")]
        ground_truth = ["doc_1", "doc_2"]
        
        eval_result = await evaluator.evaluate_retrieval(
            query="test",
            results=results,
            ground_truth=ground_truth
        )
        
        assert eval_result.mrr == 1.0
        assert eval_result.hit_rate_at_k["5"] == 1.0
    
    @pytest.mark.asyncio
    async def test_evaluate_retrieval_performance(self):
        """Test that retrieval evaluation is fast (< 10ms)."""
        import time
        
        evaluator = RAGEvaluator()
        
        results = [MockResult(f"doc_{i}") for i in range(50)]
        ground_truth = [f"doc_{i}" for i in range(10)]
        
        start = time.perf_counter()
        eval_result = await evaluator.evaluate_retrieval(
            query="test",
            results=results,
            ground_truth=ground_truth,
            k_values=[5, 10, 20]
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        assert eval_result.latency_ms < 10.0
        assert elapsed_ms < 10.0


class TestRAGEvaluatorGenerationEvaluation:
    """Tests for generation evaluation."""
    
    @pytest.mark.asyncio
    async def test_evaluate_generation_basic(self):
        """Test basic generation evaluation."""
        evaluator = RAGEvaluator()
        
        eval_result = await evaluator.evaluate_generation(
            query="What is AI?",
            answer="AI is artificial intelligence.",
            ground_truth_answer="AI stands for artificial intelligence.",
            contexts=["AI is a field of computer science."]
        )
        
        assert eval_result.query == "What is AI?"
        assert eval_result.answer == "AI is artificial intelligence."
        assert 0.0 <= eval_result.bertscore_f1 <= 1.0
        assert 0.0 <= eval_result.faithfulness <= 1.0
        assert 0.0 <= eval_result.answer_relevance <= 1.0
        assert eval_result.latency_ms >= 0
    
    @pytest.mark.asyncio
    async def test_evaluate_generation_with_bleu_rouge(self):
        """Test generation evaluation with BLEU and ROUGE."""
        evaluator = RAGEvaluator()
        
        eval_result = await evaluator.evaluate_generation(
            query="test",
            answer="Machine learning is a subset of AI.",
            ground_truth_answer="ML is part of AI.",
            contexts=["Context."],
            include_bleu_rouge=True
        )
        
        assert eval_result.bleu_score is not None
        assert eval_result.rouge_l_f1 is not None


class TestRAGEvaluatorThresholds:
    """Tests for threshold checking."""
    
    def test_check_thresholds_no_violations(self):
        """Test threshold checking with no violations."""
        config = EvaluationConfig(alerting_enabled=True)
        evaluator = RAGEvaluator(config=config)
        
        metrics = {
            "mrr": 0.75,  # Above threshold of 0.70
            "ndcg_at_10": 0.80,  # Above threshold of 0.75
            "bertscore_f1": 0.90,  # Above threshold of 0.85
        }
        
        alerts = evaluator.check_thresholds(metrics)
        
        assert alerts == []
    
    def test_check_thresholds_mrr_violation(self):
        """Test threshold checking with MRR violation."""
        config = EvaluationConfig(
            alerting_enabled=True,
            thresholds=MetricThresholds(mrr_min=0.70)
        )
        evaluator = RAGEvaluator(config=config)
        
        metrics = {"mrr": 0.65}  # Below threshold
        
        alerts = evaluator.check_thresholds(metrics)
        
        assert len(alerts) == 1
        assert alerts[0].metric_name == "mrr"
        assert alerts[0].severity == AlertSeverity.WARNING
    
    def test_check_thresholds_faithfulness_violation(self):
        """Test threshold checking with faithfulness violation."""
        config = EvaluationConfig(alerting_enabled=True)
        evaluator = RAGEvaluator(config=config)
        
        metrics = {"faithfulness": 0.70}  # Below threshold of 0.80
        
        alerts = evaluator.check_thresholds(metrics)
        
        faithfulness_alerts = [a for a in alerts if a.metric_name == "faithfulness"]
        assert len(faithfulness_alerts) == 1
        assert faithfulness_alerts[0].severity == AlertSeverity.ERROR
    
    def test_check_thresholds_latency_violation(self):
        """Test threshold checking with latency violation."""
        config = EvaluationConfig(alerting_enabled=True)
        evaluator = RAGEvaluator(config=config)
        
        metrics = {"latency_p95_ms": 1200.0}  # Above threshold of 1000
        
        alerts = evaluator.check_thresholds(metrics)
        
        latency_alerts = [a for a in alerts if a.metric_name == "latency_p95_ms"]
        assert len(latency_alerts) == 1
    
    def test_check_thresholds_disabled(self):
        """Test that no alerts when alerting is disabled."""
        config = EvaluationConfig(alerting_enabled=False)
        evaluator = RAGEvaluator(config=config)
        
        metrics = {"mrr": 0.50}  # Would trigger alert
        
        alerts = evaluator.check_thresholds(metrics)
        
        assert alerts == []
    
    def test_check_thresholds_with_context(self):
        """Test threshold checking with context."""
        evaluator = RAGEvaluator()
        
        metrics = {"mrr": 0.65}
        context = {"benchmark": "ms_marco", "strategy": "fast"}
        
        alerts = evaluator.check_thresholds(metrics, context)
        
        assert len(alerts) > 0
        assert alerts[0].context["benchmark"] == "ms_marco"


class TestRAGEvaluatorAlertHistory:
    """Tests for alert history management."""
    
    def test_get_alert_history_empty(self):
        """Test getting empty alert history."""
        evaluator = RAGEvaluator()
        
        alerts = evaluator.get_alert_history()
        
        assert alerts == []
    
    def test_get_alert_history_with_alerts(self):
        """Test getting alert history with alerts."""
        evaluator = RAGEvaluator()
        
        # Manually add some alerts
        alert1 = EvaluationAlert(
            alert_type="test",
            severity=AlertSeverity.WARNING,
            metric_name="mrr",
            current_value=0.5,
            threshold_value=0.7,
            message="Test alert 1"
        )
        alert2 = EvaluationAlert(
            alert_type="test",
            severity=AlertSeverity.ERROR,
            metric_name="faithfulness",
            current_value=0.6,
            threshold_value=0.8,
            message="Test alert 2"
        )
        
        evaluator.alert_history = [alert1, alert2]
        
        # Get all alerts
        all_alerts = evaluator.get_alert_history()
        assert len(all_alerts) == 2
        
        # Filter by severity
        warning_alerts = evaluator.get_alert_history(severity=AlertSeverity.WARNING)
        assert len(warning_alerts) == 1
        
        # Filter by metric
        mrr_alerts = evaluator.get_alert_history(metric_name="mrr")
        assert len(mrr_alerts) == 1
    
    def test_clear_alert_history(self):
        """Test clearing alert history."""
        evaluator = RAGEvaluator()
        
        evaluator.alert_history = [MagicMock()]
        assert len(evaluator.alert_history) == 1
        
        evaluator.clear_alert_history()
        
        assert len(evaluator.alert_history) == 0


class TestRAGEvaluatorSampling:
    """Tests for query sampling."""
    
    def test_should_sample_disabled(self):
        """Test sampling when disabled."""
        config = EvaluationConfig(enabled=False, sample_rate=1.0)
        evaluator = RAGEvaluator(config=config)
        
        # Should never sample when disabled
        assert evaluator.should_sample_for_evaluation() is False
    
    def test_should_sample_zero_rate(self):
        """Test sampling with zero rate."""
        config = EvaluationConfig(sample_rate=0.0)
        evaluator = RAGEvaluator(config=config)
        
        # Should never sample with 0% rate
        assert evaluator.should_sample_for_evaluation() is False
    
    def test_should_sample_full_rate(self):
        """Test sampling with 100% rate."""
        config = EvaluationConfig(sample_rate=1.0)
        evaluator = RAGEvaluator(config=config)
        
        # Should always sample with 100% rate
        assert evaluator.should_sample_for_evaluation() is True


class TestRAGEvaluatorEndToEnd:
    """Tests for end-to-end evaluation."""
    
    @pytest.mark.asyncio
    async def test_evaluate_end_to_end(self):
        """Test end-to-end evaluation."""
        evaluator = RAGEvaluator()
        
        results = [
            MockResult("doc_1", "AI is artificial intelligence."),
            MockResult("doc_2", "ML is machine learning.")
        ]
        
        result = await evaluator.evaluate_end_to_end(
            query="What is AI?",
            answer="AI is artificial intelligence.",
            retrieved_results=results,
            ground_truth_relevant_ids=["doc_1"],
            ground_truth_answer="AI stands for artificial intelligence.",
            contexts=["AI is a field of computer science."]
        )
        
        assert "retrieval" in result
        assert "generation" in result
        assert "alerts" in result
        assert result["retrieval"] is not None
        assert result["generation"] is not None
    
    @pytest.mark.asyncio
    async def test_evaluate_end_to_end_without_generation(self):
        """Test end-to-end evaluation without ground truth answer."""
        evaluator = RAGEvaluator()
        
        results = [MockResult("doc_1")]
        
        result = await evaluator.evaluate_end_to_end(
            query="test",
            answer="",
            retrieved_results=results,
            ground_truth_relevant_ids=["doc_1"]
        )
        
        assert result["retrieval"] is not None
        assert result["generation"] is None


class TestRAGEvaluatorBenchmarking:
    """Tests for benchmark operations."""
    
    @pytest.mark.asyncio
    async def test_run_benchmark(self):
        """Test running a benchmark."""
        evaluator = RAGEvaluator()
        
        # Mock RAG system
        mock_system = MagicMock()
        mock_system.query = AsyncMock(return_value=MockRAGResponse(
            answer="Test answer",
            retrieved_chunks=[MockResult("doc_1")],
            latency_ms=100.0
        ))
        
        config = BenchmarkConfig(
            name="test_benchmark",
            dataset_path="test",
            max_queries=2,
            k_values=[5]
        )
        
        result = await evaluator.run_benchmark(
            benchmark_name="ms_marco",
            rag_system=mock_system,
            config=config
        )
        
        assert result.benchmark_name == "ms_marco"
        assert result.total_queries > 0
        assert "mean_mrr" in result.aggregate_metrics or len(result.per_query_results) > 0
    
    @pytest.mark.asyncio
    async def test_run_ab_test(self):
        """Test running an A/B test."""
        evaluator = RAGEvaluator()
        
        # Mock RAG systems
        mock_control = MagicMock()
        mock_control.query = AsyncMock(return_value=MockRAGResponse(
            answer="Control answer",
            retrieved_chunks=[MockResult("doc_1")],
            latency_ms=100.0
        ))
        
        mock_treatment = MagicMock()
        mock_treatment.query = AsyncMock(return_value=MockRAGResponse(
            answer="Treatment answer",
            retrieved_chunks=[MockResult("doc_1"), MockResult("doc_2")],
            latency_ms=120.0
        ))
        
        result = await evaluator.run_ab_test(
            test_name="test_ab",
            control_system=mock_control,
            treatment_system=mock_treatment,
            sample_size=2
        )
        
        assert result.test_name == "test_ab"
        assert result.control_strategy == "control"
        assert result.treatment_strategy == "treatment"
        assert len(result.recommendation) > 0
    
    @pytest.mark.asyncio
    async def test_compare_strategies(self):
        """Test comparing two strategies."""
        evaluator = RAGEvaluator()
        
        # Mock RAG systems
        mock_a = MagicMock()
        mock_a.query = AsyncMock(return_value=MockRAGResponse(
            answer="A answer",
            retrieved_chunks=[MockResult("doc_1")],
            latency_ms=100.0
        ))
        
        mock_b = MagicMock()
        mock_b.query = AsyncMock(return_value=MockRAGResponse(
            answer="B answer",
            retrieved_chunks=[MockResult("doc_2")],
            latency_ms=150.0
        ))
        
        result = await evaluator.compare_strategies(
            strategy_a_name="strategy_a",
            strategy_b_name="strategy_b",
            strategy_a_system=mock_a,
            strategy_b_system=mock_b,
            sample_size=2
        )
        
        assert result.strategy_a_name == "strategy_a"
        assert result.strategy_b_name == "strategy_b"
        assert result.winner is not None
