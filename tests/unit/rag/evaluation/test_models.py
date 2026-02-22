"""Unit tests for RAG evaluation models.

Tests all Pydantic models used in the evaluation framework.
"""

import json
from datetime import datetime

import pytest
from pydantic import ValidationError

from src.rag.evaluation.models import (
    ABTestResult,
    AlertSeverity,
    BenchmarkConfig,
    BenchmarkDataset,
    BenchmarkResult,
    EvaluationAlert,
    EvaluationConfig,
    GenerationEvaluation,
    MetricThresholds,
    RetrievalEvaluation,
    StrategyComparison,
)


class TestAlertSeverity:
    """Tests for AlertSeverity enum."""
    
    def test_enum_values(self):
        """Test that enum has all expected values."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.ERROR.value == "error"
        assert AlertSeverity.CRITICAL.value == "critical"
    
    def test_enum_from_string(self):
        """Test creating enum from string values."""
        assert AlertSeverity("info") == AlertSeverity.INFO
        assert AlertSeverity("warning") == AlertSeverity.WARNING
        assert AlertSeverity("error") == AlertSeverity.ERROR
        assert AlertSeverity("critical") == AlertSeverity.CRITICAL


class TestRetrievalEvaluation:
    """Tests for RetrievalEvaluation model."""
    
    def test_minimal_creation(self):
        """Test creating with minimal required fields."""
        eval_result = RetrievalEvaluation(
            query="What is AI?",
            mrr=0.75,
            ndcg_at_k={"5": 0.8, "10": 0.85},
            recall_at_k={"5": 0.6, "10": 0.7},
            precision_at_k={"5": 0.4, "10": 0.35},
            hit_rate_at_k={"5": 1.0, "10": 1.0},
        )
        
        assert eval_result.query == "What is AI?"
        assert eval_result.mrr == 0.75
        assert eval_result.k_values == [5, 10]
        assert eval_result.latency_ms >= 0
        assert isinstance(eval_result.timestamp, datetime)
    
    def test_full_creation(self):
        """Test creating with all fields."""
        eval_result = RetrievalEvaluation(
            query="What is ML?",
            mrr=0.8,
            ndcg_at_k={"5": 0.82},
            recall_at_k={"5": 0.65},
            precision_at_k={"5": 0.43},
            hit_rate_at_k={"5": 1.0},
            k_values=[5],
            latency_ms=5.5,
        )
        
        assert eval_result.latency_ms == 5.5
        assert eval_result.k_values == [5]
    
    def test_mrr_bounds(self):
        """Test that MRR must be between 0 and 1."""
        # Valid values
        RetrievalEvaluation(
            query="test",
            mrr=0.0,
            ndcg_at_k={},
            recall_at_k={},
            precision_at_k={},
            hit_rate_at_k={},
        )
        RetrievalEvaluation(
            query="test",
            mrr=1.0,
            ndcg_at_k={},
            recall_at_k={},
            precision_at_k={},
            hit_rate_at_k={},
        )
        
        # Invalid values
        with pytest.raises(ValidationError):
            RetrievalEvaluation(
                query="test",
                mrr=-0.1,
                ndcg_at_k={},
                recall_at_k={},
                precision_at_k={},
                hit_rate_at_k={},
            )
        
        with pytest.raises(ValidationError):
            RetrievalEvaluation(
                query="test",
                mrr=1.1,
                ndcg_at_k={},
                recall_at_k={},
                precision_at_k={},
                hit_rate_at_k={},
            )
    
    def test_serialization(self):
        """Test JSON serialization."""
        eval_result = RetrievalEvaluation(
            query="What is AI?",
            mrr=0.75,
            ndcg_at_k={"5": 0.8},
            recall_at_k={"5": 0.6},
            precision_at_k={"5": 0.4},
            hit_rate_at_k={"5": 1.0},
        )
        
        json_str = eval_result.model_dump_json()
        data = json.loads(json_str)
        
        assert data["query"] == "What is AI?"
        assert data["mrr"] == 0.75
        assert data["ndcg_at_k"]["5"] == 0.8


class TestGenerationEvaluation:
    """Tests for GenerationEvaluation model."""
    
    def test_minimal_creation(self):
        """Test creating with minimal required fields."""
        eval_result = GenerationEvaluation(
            query="What is AI?",
            answer="AI is artificial intelligence.",
            bertscore_precision=0.92,
            bertscore_recall=0.88,
            bertscore_f1=0.90,
            faithfulness=0.95,
            answer_relevance=0.93,
        )
        
        assert eval_result.query == "What is AI?"
        assert eval_result.answer == "AI is artificial intelligence."
        assert eval_result.bertscore_f1 == 0.90
        assert eval_result.faithfulness == 0.95
    
    def test_full_creation(self):
        """Test creating with optional fields."""
        eval_result = GenerationEvaluation(
            query="What is ML?",
            answer="ML is machine learning.",
            bertscore_precision=0.92,
            bertscore_recall=0.88,
            bertscore_f1=0.90,
            faithfulness=0.95,
            answer_relevance=0.93,
            bleu_score=0.45,
            rouge_l_f1=0.62,
            latency_ms=85.5,
        )
        
        assert eval_result.bleu_score == 0.45
        assert eval_result.rouge_l_f1 == 0.62
        assert eval_result.latency_ms == 85.5
    
    def test_score_bounds(self):
        """Test that scores must be between 0 and 1."""
        # Valid values
        GenerationEvaluation(
            query="test",
            answer="answer",
            bertscore_precision=0.5,
            bertscore_recall=0.5,
            bertscore_f1=0.5,
            faithfulness=1.0,
            answer_relevance=0.0,
        )
        
        # Invalid values
        with pytest.raises(ValidationError):
            GenerationEvaluation(
                query="test",
                answer="answer",
                bertscore_precision=1.5,
                bertscore_recall=0.5,
                bertscore_f1=0.5,
                faithfulness=0.5,
                answer_relevance=0.5,
            )


class TestBenchmarkDataset:
    """Tests for BenchmarkDataset model."""
    
    def test_minimal_creation(self):
        """Test creating with minimal fields."""
        dataset = BenchmarkDataset(
            id="q1",
            query="What is AI?",
        )
        
        assert dataset.id == "q1"
        assert dataset.query == "What is AI?"
        assert dataset.ground_truth_relevant_ids == []
        assert dataset.ground_truth_answer is None
    
    def test_full_creation(self):
        """Test creating with all fields."""
        dataset = BenchmarkDataset(
            id="q1",
            query="What is AI?",
            ground_truth_relevant_ids=["doc1", "doc2"],
            ground_truth_answer="AI is artificial intelligence.",
            metadata={"category": "definition"},
        )
        
        assert dataset.ground_truth_relevant_ids == ["doc1", "doc2"]
        assert dataset.ground_truth_answer == "AI is artificial intelligence."
        assert dataset.metadata["category"] == "definition"


class TestBenchmarkResult:
    """Tests for BenchmarkResult model."""
    
    def test_minimal_creation(self):
        """Test creating with minimal required fields."""
        result = BenchmarkResult(benchmark_name="ms_marco")
        
        assert result.benchmark_name == "ms_marco"
        assert result.total_queries == 0
        assert result.successful_queries == 0
        assert result.failed_queries == 0
    
    def test_full_creation(self):
        """Test creating with all fields."""
        result = BenchmarkResult(
            benchmark_name="ms_marco",
            aggregate_metrics={"mean_mrr": 0.72},
            per_query_results=[{"query_id": "q1"}],
            total_queries=100,
            successful_queries=95,
            failed_queries=5,
            total_latency_ms=50000.0,
        )
        
        assert result.aggregate_metrics["mean_mrr"] == 0.72
        assert result.total_queries == 100
        assert result.successful_queries == 95
        assert result.failed_queries == 5


class TestStrategyComparison:
    """Tests for StrategyComparison model."""
    
    def test_creation(self):
        """Test creating a strategy comparison."""
        comparison = StrategyComparison(
            strategy_a_name="balanced",
            strategy_b_name="fast",
            strategy_a_metrics={"mrr": 0.75},
            strategy_b_metrics={"mrr": 0.68},
            improvements={"mrr": 0.07},
            statistical_significance={"mrr": True},
            winner="A",
            sample_size=500,
        )
        
        assert comparison.strategy_a_name == "balanced"
        assert comparison.strategy_b_name == "fast"
        assert comparison.winner == "A"
        assert comparison.sample_size == 500


class TestABTestResult:
    """Tests for ABTestResult model."""
    
    def test_creation(self):
        """Test creating an A/B test result."""
        result = ABTestResult(
            test_name="reranking_test",
            control_strategy="baseline",
            treatment_strategy="with_reranking",
            control_metrics={"mrr": 0.68},
            treatment_metrics={"mrr": 0.75},
            relative_improvements={"mrr": 10.3},
            p_values={"mrr": 0.001},
            is_significant={"mrr": True},
            recommendation="Deploy treatment",
        )
        
        assert result.test_name == "reranking_test"
        assert result.recommendation == "Deploy treatment"
        assert result.confidence_level == 0.95  # default


class TestEvaluationAlert:
    """Tests for EvaluationAlert model."""
    
    def test_creation(self):
        """Test creating an evaluation alert."""
        alert = EvaluationAlert(
            alert_type="threshold_violation",
            severity=AlertSeverity.WARNING,
            metric_name="mrr",
            current_value=0.65,
            threshold_value=0.70,
            message="MRR dropped below threshold",
            context={"benchmark": "ms_marco"},
        )
        
        assert alert.alert_type == "threshold_violation"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.metric_name == "mrr"
        assert alert.current_value == 0.65


class TestMetricThresholds:
    """Tests for MetricThresholds model."""
    
    def test_defaults(self):
        """Test default threshold values."""
        thresholds = MetricThresholds()
        
        assert thresholds.mrr_min == 0.70
        assert thresholds.ndcg_at_10_min == 0.75
        assert thresholds.recall_at_5_min == 0.60
        assert thresholds.precision_at_5_min == 0.40
        assert thresholds.bertscore_f1_min == 0.85
        assert thresholds.faithfulness_min == 0.80
        assert thresholds.latency_p95_max == 1000.0
    
    def test_custom_values(self):
        """Test setting custom threshold values."""
        thresholds = MetricThresholds(
            mrr_min=0.80,
            latency_p95_max=500.0,
        )
        
        assert thresholds.mrr_min == 0.80
        assert thresholds.latency_p95_max == 500.0
    
    def test_threshold_bounds(self):
        """Test that threshold values must be valid."""
        # Invalid MRR (must be 0-1)
        with pytest.raises(ValidationError):
            MetricThresholds(mrr_min=1.5)
        
        with pytest.raises(ValidationError):
            MetricThresholds(mrr_min=-0.1)


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig model."""
    
    def test_creation(self):
        """Test creating a benchmark config."""
        config = BenchmarkConfig(
            name="ms_marco",
            dataset_path="microsoft/ms_marco",
        )
        
        assert config.name == "ms_marco"
        assert config.dataset_path == "microsoft/ms_marco"
        assert config.dataset_type == "json"
        assert config.max_queries == 1000
        assert config.k_values == [5, 10]
    
    def test_dataset_type_validation(self):
        """Test dataset type validation."""
        # Valid types
        BenchmarkConfig(name="test", dataset_path="test", dataset_type="ms_marco")
        BenchmarkConfig(name="test", dataset_path="test", dataset_type="json")
        BenchmarkConfig(name="test", dataset_path="test", dataset_type="s3")
        BenchmarkConfig(name="test", dataset_path="test", dataset_type="huggingface")
        
        # Invalid type
        with pytest.raises(ValidationError):
            BenchmarkConfig(
                name="test",
                dataset_path="test",
                dataset_type="invalid_type"
            )


class TestEvaluationConfig:
    """Tests for EvaluationConfig model."""
    
    def test_defaults(self):
        """Test default configuration values."""
        config = EvaluationConfig()
        
        assert config.enabled is True
        assert config.auto_evaluate_enabled is True
        assert config.sample_rate == 0.1
        assert config.alerting_enabled is True
        assert "log" in config.alert_channels
    
    def test_custom_values(self):
        """Test setting custom configuration values."""
        config = EvaluationConfig(
            enabled=False,
            sample_rate=0.25,
            alerting_enabled=False,
            alert_channels=["slack", "email"],
        )
        
        assert config.enabled is False
        assert config.sample_rate == 0.25
        assert config.alerting_enabled is False
        assert config.alert_channels == ["slack", "email"]
    
    def test_sample_rate_validation(self):
        """Test sample rate validation."""
        # Valid rates
        EvaluationConfig(sample_rate=0.0)
        EvaluationConfig(sample_rate=0.5)
        EvaluationConfig(sample_rate=1.0)
        
        # Invalid rates
        with pytest.raises(ValidationError):
            EvaluationConfig(sample_rate=-0.1)
        
        with pytest.raises(ValidationError):
            EvaluationConfig(sample_rate=1.5)
