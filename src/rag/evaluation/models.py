"""Pydantic models for RAG evaluation framework.

This module defines all data models used for evaluation results,
benchmark configurations, and alert structures.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AlertSeverity(str, Enum):
    """Alert severity levels for evaluation threshold violations."""
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RetrievalEvaluation(BaseModel):
    """Results of retrieval quality evaluation.
    
    Attributes:
        query: The original query string
        mrr: Mean Reciprocal Rank (0-1)
        ndcg_at_k: Normalized Discounted Cumulative Gain at K
        recall_at_k: Recall at K
        precision_at_k: Precision at K
        hit_rate_at_k: Hit Rate at K (at least one relevant doc in top K)
        k_values: List of K values used for metrics
        latency_ms: Time taken to compute metrics
        timestamp: When evaluation was performed
    """
    
    query: str = Field(..., description="Original query string")
    mrr: float = Field(..., ge=0.0, le=1.0, description="Mean Reciprocal Rank")
    ndcg_at_k: dict[str, float] = Field(
        default_factory=dict,
        description="NDCG scores at different K values"
    )
    recall_at_k: dict[str, float] = Field(
        default_factory=dict,
        description="Recall at different K values"
    )
    precision_at_k: dict[str, float] = Field(
        default_factory=dict,
        description="Precision at different K values"
    )
    hit_rate_at_k: dict[str, float] = Field(
        default_factory=dict,
        description="Hit Rate at different K values"
    )
    k_values: list[int] = Field(
        default_factory=lambda: [5, 10],
        description="K values used for metrics"
    )
    latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time taken to compute metrics in milliseconds"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When evaluation was performed"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What is machine learning?",
                "mrr": 0.75,
                "ndcg_at_k": {"5": 0.82, "10": 0.85},
                "recall_at_k": {"5": 0.60, "10": 0.80},
                "precision_at_k": {"5": 0.40, "10": 0.32},
                "hit_rate_at_k": {"5": 1.0, "10": 1.0},
                "k_values": [5, 10],
                "latency_ms": 2.5,
            }
        }
    )


class GenerationEvaluation(BaseModel):
    """Results of generation quality evaluation.
    
    Attributes:
        query: The original query string
        answer: Generated answer text
        bertscore_precision: BERTScore precision
        bertscore_recall: BERTScore recall
        bertscore_f1: BERTScore F1
        faithfulness: Faithfulness score (0-1)
        answer_relevance: Answer relevance score (0-1)
        bleu_score: BLEU score (optional)
        rouge_l_f1: ROUGE-L F1 score (optional)
        latency_ms: Time taken to compute metrics
        timestamp: When evaluation was performed
    """
    
    query: str = Field(..., description="Original query string")
    answer: str = Field(..., description="Generated answer text")
    bertscore_precision: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="BERTScore precision"
    )
    bertscore_recall: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="BERTScore recall"
    )
    bertscore_f1: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="BERTScore F1"
    )
    faithfulness: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Faithfulness score (grounded in contexts)"
    )
    answer_relevance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Answer relevance to query"
    )
    bleu_score: float | None = Field(
        default=None,
        ge=0.0,
        description="BLEU score (optional)"
    )
    rouge_l_f1: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="ROUGE-L F1 score (optional)"
    )
    latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time taken to compute metrics in milliseconds"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When evaluation was performed"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What is machine learning?",
                "answer": "Machine learning is...",
                "bertscore_precision": 0.92,
                "bertscore_recall": 0.88,
                "bertscore_f1": 0.90,
                "faithfulness": 0.95,
                "answer_relevance": 0.93,
                "bleu_score": 0.45,
                "rouge_l_f1": 0.62,
                "latency_ms": 85.3,
            }
        }
    )


class BenchmarkDataset(BaseModel):
    """A single item in a benchmark dataset.
    
    Attributes:
        id: Unique identifier for this dataset item
        query: Query string
        ground_truth_relevant_ids: List of relevant chunk/document IDs
        ground_truth_answer: Ground truth answer (optional)
        metadata: Additional metadata about this item
    """
    
    id: str = Field(..., description="Unique identifier")
    query: str = Field(..., description="Query string")
    ground_truth_relevant_ids: list[str] = Field(
        default_factory=list,
        description="List of relevant chunk/document IDs"
    )
    ground_truth_answer: str | None = Field(
        default=None,
        description="Ground truth answer (optional)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "ms_marco_001",
                "query": "What is machine learning?",
                "ground_truth_relevant_ids": ["doc_1", "doc_2"],
                "ground_truth_answer": "Machine learning is a subset of AI...",
                "metadata": {"source": "MS MARCO", "category": "definition"},
            }
        }
    )


class BenchmarkResult(BaseModel):
    """Results of a full benchmark run.
    
    Attributes:
        benchmark_name: Name of the benchmark
        config: Configuration used for this run
        aggregate_metrics: Aggregated metrics across all queries
        per_query_results: Individual results for each query
        strategy_config: RAG strategy configuration used
        total_queries: Total number of queries processed
        successful_queries: Number of successfully processed queries
        failed_queries: Number of failed queries
        total_latency_ms: Total time for benchmark run
        started_at: When benchmark started
        completed_at: When benchmark completed
    """
    
    benchmark_name: str = Field(..., description="Name of the benchmark")
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration used for this run"
    )
    aggregate_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Aggregated metrics across all queries"
    )
    per_query_results: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Individual results for each query"
    )
    strategy_config: dict[str, Any] = Field(
        default_factory=dict,
        description="RAG strategy configuration used"
    )
    total_queries: int = Field(default=0, ge=0, description="Total queries processed")
    successful_queries: int = Field(default=0, ge=0, description="Successful queries")
    failed_queries: int = Field(default=0, ge=0, description="Failed queries")
    total_latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Total time for benchmark run"
    )
    started_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When benchmark started"
    )
    completed_at: datetime | None = Field(
        default=None,
        description="When benchmark completed"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "benchmark_name": "ms_marco",
                "aggregate_metrics": {
                    "mean_mrr": 0.72,
                    "mean_ndcg_at_10": 0.78,
                    "mean_bertscore_f1": 0.88,
                },
                "total_queries": 1000,
                "successful_queries": 995,
                "failed_queries": 5,
                "total_latency_ms": 45000.0,
            }
        }
    )


class StrategyComparison(BaseModel):
    """Comparison between two RAG strategies.
    
    Attributes:
        strategy_a_name: Name of first strategy
        strategy_b_name: Name of second strategy
        strategy_a_metrics: Metrics for strategy A
        strategy_b_metrics: Metrics for strategy B
        improvements: Dict of metric improvements (positive = A better)
        statistical_significance: Whether differences are statistically significant
        winner: Which strategy performed better
        sample_size: Number of queries compared
    """
    
    strategy_a_name: str = Field(..., description="Name of first strategy")
    strategy_b_name: str = Field(..., description="Name of second strategy")
    strategy_a_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Metrics for strategy A"
    )
    strategy_b_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Metrics for strategy B"
    )
    improvements: dict[str, float] = Field(
        default_factory=dict,
        description="Metric improvements (positive = A better than B)"
    )
    statistical_significance: dict[str, bool] = Field(
        default_factory=dict,
        description="Whether differences are statistically significant per metric"
    )
    winner: str | None = Field(
        default=None,
        description="Which strategy performed better (A, B, or tie)"
    )
    sample_size: int = Field(default=0, ge=0, description="Number of queries compared")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "strategy_a_name": "balanced",
                "strategy_b_name": "fast",
                "strategy_a_metrics": {"mrr": 0.75, "latency_ms": 250.0},
                "strategy_b_metrics": {"mrr": 0.68, "latency_ms": 120.0},
                "improvements": {"mrr": 0.07, "latency_ms": -130.0},
                "winner": "A",
                "sample_size": 500,
            }
        }
    )


class ABTestResult(BaseModel):
    """Results of an A/B test between two strategies.
    
    Attributes:
        test_name: Name of the A/B test
        control_strategy: Name of control strategy
        treatment_strategy: Name of treatment strategy
        control_metrics: Metrics for control group
        treatment_metrics: Metrics for treatment group
        relative_improvements: Relative improvements (percentage)
        p_values: Statistical significance p-values
        is_significant: Whether results are statistically significant
        recommendation: Recommendation based on results
        confidence_level: Confidence level used for test
    """
    
    test_name: str = Field(..., description="Name of the A/B test")
    control_strategy: str = Field(..., description="Name of control strategy")
    treatment_strategy: str = Field(..., description="Name of treatment strategy")
    control_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Metrics for control group"
    )
    treatment_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Metrics for treatment group"
    )
    relative_improvements: dict[str, float] = Field(
        default_factory=dict,
        description="Relative improvements as percentages"
    )
    p_values: dict[str, float] = Field(
        default_factory=dict,
        description="Statistical significance p-values"
    )
    is_significant: dict[str, bool] = Field(
        default_factory=dict,
        description="Whether each metric is statistically significant"
    )
    recommendation: str = Field(
        default="",
        description="Recommendation based on results"
    )
    confidence_level: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Confidence level used for test"
    )
    sample_size_control: int = Field(default=0, ge=0, description="Control group size")
    sample_size_treatment: int = Field(default=0, ge=0, description="Treatment group size")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "test_name": "reranking_ab_test",
                "control_strategy": "standard",
                "treatment_strategy": "with_reranking",
                "control_metrics": {"mrr": 0.68, "ndcg_at_10": 0.72},
                "treatment_metrics": {"mrr": 0.75, "ndcg_at_10": 0.80},
                "relative_improvements": {"mrr": 10.3, "ndcg_at_10": 11.1},
                "p_values": {"mrr": 0.001, "ndcg_at_10": 0.0005},
                "is_significant": {"mrr": True, "ndcg_at_10": True},
                "recommendation": "Deploy treatment strategy (with_reranking)",
                "sample_size_control": 500,
                "sample_size_treatment": 500,
            }
        }
    )


class EvaluationAlert(BaseModel):
    """Alert triggered by evaluation threshold violations.
    
    Attributes:
        alert_type: Type of alert
        severity: Alert severity level
        metric_name: Name of the metric that violated threshold
        current_value: Current value of the metric
        threshold_value: Threshold that was violated
        message: Human-readable alert message
        timestamp: When alert was triggered
        context: Additional context about the alert
    """
    
    alert_type: str = Field(..., description="Type of alert")
    severity: AlertSeverity = Field(..., description="Alert severity level")
    metric_name: str = Field(..., description="Name of the metric")
    current_value: float = Field(..., description="Current value of the metric")
    threshold_value: float = Field(..., description="Threshold that was violated")
    message: str = Field(..., description="Human-readable alert message")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When alert was triggered"
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "alert_type": "threshold_violation",
                "severity": "warning",
                "metric_name": "mrr",
                "current_value": 0.65,
                "threshold_value": 0.70,
                "message": "MRR dropped below threshold: 0.65 < 0.70",
                "context": {"benchmark": "ms_marco", "strategy": "fast"},
            }
        }
    )


class MetricThresholds(BaseModel):
    """Threshold configuration for evaluation alerting.
    
    Attributes:
        mrr_min: Minimum acceptable MRR
        ndcg_at_10_min: Minimum acceptable NDCG@10
        recall_at_5_min: Minimum acceptable Recall@5
        precision_at_5_min: Minimum acceptable Precision@5
        bertscore_f1_min: Minimum acceptable BERTScore F1
        faithfulness_min: Minimum acceptable faithfulness
        answer_relevance_min: Minimum acceptable answer relevance
        latency_p95_max: Maximum acceptable P95 latency (ms)
        mrr_drop_threshold: Alert if MRR drops by this amount
    """
    
    mrr_min: float = Field(default=0.70, ge=0.0, le=1.0)
    ndcg_at_10_min: float = Field(default=0.75, ge=0.0, le=1.0)
    recall_at_5_min: float = Field(default=0.60, ge=0.0, le=1.0)
    precision_at_5_min: float = Field(default=0.40, ge=0.0, le=1.0)
    bertscore_f1_min: float = Field(default=0.85, ge=0.0, le=1.0)
    faithfulness_min: float = Field(default=0.80, ge=0.0, le=1.0)
    answer_relevance_min: float = Field(default=0.80, ge=0.0, le=1.0)
    latency_p95_max: float = Field(default=1000.0, ge=0.0)
    mrr_drop_threshold: float = Field(default=0.05, ge=0.0, le=1.0)


class BenchmarkConfig(BaseModel):
    """Configuration for a benchmark dataset.
    
    Attributes:
        name: Name of the benchmark
        dataset_path: Path or identifier for the dataset
        dataset_type: Type of dataset (ms_marco, json, s3, etc.)
        max_queries: Maximum number of queries to evaluate
        k_values: K values for retrieval metrics
        metrics: List of metrics to compute
    """
    
    name: str = Field(..., description="Name of the benchmark")
    dataset_path: str = Field(..., description="Path or identifier for the dataset")
    dataset_type: str = Field(
        default="json",
        description="Type of dataset (ms_marco, json, s3, etc.)"
    )
    max_queries: int = Field(
        default=1000,
        ge=1,
        description="Maximum number of queries to evaluate"
    )
    k_values: list[int] = Field(
        default_factory=lambda: [5, 10],
        description="K values for retrieval metrics"
    )
    metrics: list[str] = Field(
        default_factory=lambda: ["mrr", "ndcg", "recall", "precision", "hit_rate"],
        description="List of metrics to compute"
    )
    
    @field_validator("dataset_type")
    @classmethod
    def validate_dataset_type(cls, v: str) -> str:
        """Validate dataset type."""
        valid_types = {"ms_marco", "json", "s3", "huggingface", "csv"}
        if v.lower() not in valid_types:
            raise ValueError(f"Invalid dataset_type: {v}. Must be one of {valid_types}")
        return v.lower()


class EvaluationConfig(BaseModel):
    """Configuration for RAG evaluation framework.
    
    Attributes:
        enabled: Whether evaluation is enabled
        auto_evaluate_enabled: Whether to auto-evaluate queries
        sample_rate: Rate at which to sample queries for evaluation (0-1)
        metrics: List of metrics to compute
        benchmarks: List of benchmark configurations
        alerting_enabled: Whether alerting is enabled
        thresholds: Metric thresholds for alerting
        alert_channels: Channels for alerts (slack, email, etc.)
    """
    
    enabled: bool = Field(default=True, description="Whether evaluation is enabled")
    auto_evaluate_enabled: bool = Field(
        default=True,
        description="Whether to auto-evaluate queries"
    )
    sample_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Rate at which to sample queries for evaluation"
    )
    metrics: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "retrieval": ["mrr", "ndcg@10", "recall@5", "precision@5", "hit_rate@10"],
            "generation": ["bertscore", "faithfulness", "answer_relevance"],
        },
        description="Metrics to compute by category"
    )
    benchmarks: list[BenchmarkConfig] = Field(
        default_factory=list,
        description="Benchmark configurations"
    )
    alerting_enabled: bool = Field(default=True, description="Whether alerting is enabled")
    thresholds: MetricThresholds = Field(
        default_factory=MetricThresholds,
        description="Metric thresholds for alerting"
    )
    alert_channels: list[str] = Field(
        default_factory=lambda: ["log"],
        description="Channels for alerts (slack, email, log)"
    )
    
    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, v: float) -> float:
        """Validate sample rate."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("sample_rate must be between 0.0 and 1.0")
        return v
