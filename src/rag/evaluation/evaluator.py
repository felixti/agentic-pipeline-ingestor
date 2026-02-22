"""Main RAG Evaluator class.

This module provides the primary RAGEvaluator class that orchestrates
retrieval and generation evaluation, benchmark runs, and alerting.

The evaluator provides:
    - Unified interface for all evaluation metrics
    - Automated benchmark runs
    - A/B testing capabilities
    - Threshold-based alerting
    - Performance optimization (<10ms for retrieval, <100ms for BERTScore)

Example:
    >>> from src.rag.evaluation import RAGEvaluator
    >>>
    >>> evaluator = RAGEvaluator()
    >>>
    >>> # Evaluate retrieval
    >>> retrieval_eval = await evaluator.evaluate_retrieval(
    ...     query="What is RAG?",
    ...     results=retrieved_chunks,
    ...     ground_truth=["chunk_1", "chunk_2"]
    ... )
    >>>
    >>> # Evaluate generation
    >>> generation_eval = await evaluator.evaluate_generation(
    ...     query="What is RAG?",
    ...     answer="RAG is...",
    ...     ground_truth_answer="Retrieval-Augmented Generation is...",
    ...     contexts=["context1", "context2"]
    ... )
    >>>
    >>> # Run benchmark
    >>> benchmark_result = await evaluator.run_benchmark(
    ...     benchmark_name="ms_marco",
    ...     rag_system=my_rag_system
    ... )
"""

import random
import time
from typing import Any, Protocol

from src.config import get_settings
from src.rag.evaluation.benchmarks import BenchmarkRunner
from src.rag.evaluation.generation import GenerationMetrics
from src.rag.evaluation.metrics import RetrievalMetrics
from src.rag.evaluation.models import (
    ABTestResult,
    AlertSeverity,
    BenchmarkConfig,
    BenchmarkResult,
    EvaluationAlert,
    EvaluationConfig,
    GenerationEvaluation,
    RetrievalEvaluation,
    StrategyComparison,
)


class RetrievalResult(Protocol):
    """Protocol for retrieval results."""
    id: str


class RAGSystem(Protocol):
    """Protocol for RAG system."""
    
    async def query(self, query: str, **kwargs: Any) -> object:
        """Execute query."""
        ...


class RAGEvaluator:
    """Main evaluator for RAG systems.
    
    This class provides a unified interface for evaluating both retrieval
    and generation quality in RAG pipelines. It supports:
    
    - Individual query evaluation
    - Batch/benchmark evaluation
    - A/B testing between strategies
    - Automated alerting on threshold violations
    - Performance monitoring
    
    Attributes:
        retrieval_metrics: RetrievalMetrics instance
        generation_metrics: GenerationMetrics instance
        config: EvaluationConfig instance
        benchmark_runner: BenchmarkRunner instance
        alert_history: List of triggered alerts
    
    Example:
        >>> evaluator = RAGEvaluator()
        >>>
        >>> # Simple retrieval evaluation
        >>> eval_result = await evaluator.evaluate_retrieval(
        ...     query="What is AI?",
        ...     results=retrieved_docs,
        ...     ground_truth=["doc_1"]
        ... )
        >>> print(f"MRR: {eval_result.mrr}")
    """
    
    def __init__(
        self,
        config: EvaluationConfig | None = None,
        retrieval_metrics: RetrievalMetrics | None = None,
        generation_metrics: GenerationMetrics | None = None,
        benchmark_runner: BenchmarkRunner | None = None
    ):
        """Initialize RAG evaluator.
        
        Args:
            config: Evaluation configuration (loads from settings if None)
            retrieval_metrics: RetrievalMetrics instance (creates default if None)
            generation_metrics: GenerationMetrics instance (creates default if None)
            benchmark_runner: BenchmarkRunner instance (creates default if None)
        """
        self.config = config or self._load_config_from_settings()
        self.retrieval_metrics = retrieval_metrics or RetrievalMetrics()
        self.generation_metrics = generation_metrics or GenerationMetrics()
        self.benchmark_runner = benchmark_runner or BenchmarkRunner()
        self.alert_history: list[EvaluationAlert] = []
    
    def _load_config_from_settings(self) -> EvaluationConfig:
        """Load evaluation configuration from application settings.
        
        Returns:
            EvaluationConfig instance
        """
        settings = get_settings()
        
        # Check if evaluation settings exist in config
        eval_settings = getattr(settings, "evaluation", None)
        if eval_settings:
            return EvaluationConfig(
                enabled=getattr(eval_settings, "enabled", True),
                auto_evaluate_enabled=getattr(
                    eval_settings, "auto_evaluate_enabled", True
                ),
                sample_rate=getattr(eval_settings, "sample_rate", 0.1),
                alerting_enabled=getattr(eval_settings, "alerting_enabled", True)
            )
        
        # Return default config
        return EvaluationConfig()
    
    async def evaluate_retrieval(
        self,
        query: str,
        results: list[RetrievalResult],
        ground_truth: list[str],
        k_values: list[int] | None = None
    ) -> RetrievalEvaluation:
        """Evaluate retrieval quality for a single query.
        
        Computes MRR, NDCG@K, Recall@K, Precision@K, and Hit Rate@K.
        Optimized for < 10ms computation time.
        
        Args:
            query: The query string
            results: List of retrieved results (must have 'id' attribute)
            ground_truth: List of relevant document IDs
            k_values: List of K values for metrics (default: [5, 10])
            
        Returns:
            RetrievalEvaluation with all metrics
            
        Example:
            >>> results = [
            ...     type('R', (), {'id': 'doc_1'}),
            ...     type('R', (), {'id': 'doc_2'})
            ... ]
            >>> eval = await evaluator.evaluate_retrieval(
            ...     "What is AI?",
            ...     results,
            ...     ["doc_1"]
            ... )
        """
        if k_values is None:
            k_values = [5, 10]
        
        start_time = time.perf_counter()
        
        # Compute all metrics
        all_metrics = self.retrieval_metrics.compute_all_metrics(
            results, ground_truth, k_values
        )
        
        # Extract metrics by K value
        ndcg_at_k = {
            str(k): all_metrics.get(f"ndcg_at_{k}", 0.0)
            for k in k_values
        }
        recall_at_k = {
            str(k): all_metrics.get(f"recall_at_{k}", 0.0)
            for k in k_values
        }
        precision_at_k = {
            str(k): all_metrics.get(f"precision_at_{k}", 0.0)
            for k in k_values
        }
        hit_rate_at_k = {
            str(k): all_metrics.get(f"hit_rate_at_{k}", 0.0)
            for k in k_values
        }
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return RetrievalEvaluation(
            query=query,
            mrr=all_metrics["mrr"],
            ndcg_at_k=ndcg_at_k,
            recall_at_k=recall_at_k,
            precision_at_k=precision_at_k,
            hit_rate_at_k=hit_rate_at_k,
            k_values=k_values,
            latency_ms=latency_ms
        )
    
    async def evaluate_generation(
        self,
        query: str,
        answer: str,
        ground_truth_answer: str,
        contexts: list[str],
        include_bleu_rouge: bool = False
    ) -> GenerationEvaluation:
        """Evaluate generation quality for a single query.
        
        Computes BERTScore, faithfulness, and answer relevance.
        BERTScore target: < 100ms.
        
        Args:
            query: The query string
            answer: Generated answer
            ground_truth_answer: Ground truth answer
            contexts: List of context texts used for generation
            include_bleu_rouge: Whether to include BLEU/ROUGE scores
            
        Returns:
            GenerationEvaluation with all metrics
            
        Example:
            >>> eval = await evaluator.evaluate_generation(
            ...     "What is AI?",
            ...     "AI is artificial intelligence...",
            ...     "Artificial Intelligence is...",
            ...     ["Context about AI"]
            ... )
        """
        start_time = time.perf_counter()
        
        # Compute all generation metrics
        metrics = await self.generation_metrics.compute_all_metrics(
            answer=answer,
            ground_truth_answer=ground_truth_answer,
            question=query,
            contexts=contexts,
            include_bleu_rouge=include_bleu_rouge
        )
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return GenerationEvaluation(
            query=query,
            answer=answer,
            bertscore_precision=metrics["bertscore_precision"],
            bertscore_recall=metrics["bertscore_recall"],
            bertscore_f1=metrics["bertscore_f1"],
            faithfulness=metrics["faithfulness"],
            answer_relevance=metrics["answer_relevance"],
            bleu_score=metrics.get("bleu"),
            rouge_l_f1=metrics.get("rouge_l_f1"),
            latency_ms=latency_ms
        )
    
    async def run_benchmark(
        self,
        benchmark_name: str,
        rag_system: RAGSystem,
        config: BenchmarkConfig | None = None,
        strategy_config: dict[str, Any] | None = None
    ) -> BenchmarkResult:
        """Run a full benchmark evaluation.
        
        Evaluates a RAG system against a standard benchmark dataset.
        
        Args:
            benchmark_name: Name of benchmark (e.g., "ms_marco", "custom_qa")
            rag_system: RAG system to evaluate
            config: Benchmark configuration
            strategy_config: RAG strategy configuration used
            
        Returns:
            BenchmarkResult with aggregate and per-query metrics
            
        Example:
            >>> result = await evaluator.run_benchmark(
            ...     "ms_marco",
            ...     my_rag_system,
            ...     config=BenchmarkConfig(max_queries=100)
            ... )
            >>> print(f"Mean MRR: {result.aggregate_metrics['mean_mrr']}")
        """
        return await self.benchmark_runner.run_benchmark(
            benchmark_name=benchmark_name,
            rag_system=rag_system,
            config=config,
            strategy_config=strategy_config
        )
    
    async def run_ab_test(
        self,
        test_name: str,
        control_system: RAGSystem,
        treatment_system: RAGSystem,
        benchmark_name: str = "ms_marco",
        sample_size: int = 100,
        confidence_level: float = 0.95
    ) -> ABTestResult:
        """Run an A/B test comparing two RAG strategies.
        
        Performs statistical significance testing and generates recommendations.
        
        Args:
            test_name: Name of the A/B test
            control_system: Control (baseline) RAG system
            treatment_system: Treatment (new) RAG system
            benchmark_name: Benchmark dataset to use
            sample_size: Number of queries to test
            confidence_level: Statistical confidence level
            
        Returns:
            ABTestResult with comparison and statistical analysis
            
        Example:
            >>> ab_result = await evaluator.run_ab_test(
            ...     "reranking_test",
            ...     baseline_system,
            ...     system_with_reranking
            ... )
            >>> print(ab_result.recommendation)
        """
        return await self.benchmark_runner.run_ab_test(
            test_name=test_name,
            control_system=control_system,
            treatment_system=treatment_system,
            benchmark_name=benchmark_name,
            sample_size=sample_size,
            confidence_level=confidence_level
        )
    
    async def compare_strategies(
        self,
        strategy_a_name: str,
        strategy_b_name: str,
        strategy_a_system: RAGSystem,
        strategy_b_system: RAGSystem,
        benchmark_name: str = "ms_marco",
        sample_size: int = 100
    ) -> StrategyComparison:
        """Compare two strategies side-by-side.
        
        Args:
            strategy_a_name: Name of first strategy
            strategy_b_name: Name of second strategy
            strategy_a_system: First RAG system
            strategy_b_system: Second RAG system
            benchmark_name: Benchmark dataset to use
            sample_size: Number of queries to compare
            
        Returns:
            StrategyComparison with detailed comparison
        """
        return await self.benchmark_runner.compare_strategies(
            strategy_a_name=strategy_a_name,
            strategy_b_name=strategy_b_name,
            strategy_a_system=strategy_a_system,
            strategy_b_system=strategy_b_system,
            benchmark_name=benchmark_name,
            sample_size=sample_size
        )
    
    def check_thresholds(
        self,
        metrics: dict[str, float],
        context: dict[str, Any] | None = None
    ) -> list[EvaluationAlert]:
        """Check metrics against thresholds and generate alerts.
        
        Args:
            metrics: Dictionary of metric values
            context: Additional context for alerts
            
        Returns:
            List of triggered alerts (empty if all thresholds met)
            
        Example:
            >>> alerts = evaluator.check_thresholds({
            ...     "mrr": 0.65,
            ...     "latency_p95_ms": 1200
            ... }, context={"benchmark": "ms_marco"})
            >>> for alert in alerts:
            ...     print(alert.message)
        """
        if not self.config.alerting_enabled:
            return []
        
        alerts = []
        thresholds = self.config.thresholds
        
        # Check MRR
        if "mrr" in metrics and metrics["mrr"] < thresholds.mrr_min:
            alerts.append(EvaluationAlert(
                alert_type="threshold_violation",
                severity=AlertSeverity.WARNING,
                metric_name="mrr",
                current_value=metrics["mrr"],
                threshold_value=thresholds.mrr_min,
                message=(
                    f"MRR below threshold: {metrics['mrr']:.3f} < "
                    f"{thresholds.mrr_min:.3f}"
                ),
                context=context or {}
            ))
        
        # Check NDCG@10
        if "ndcg_at_10" in metrics and metrics["ndcg_at_10"] < thresholds.ndcg_at_10_min:
            alerts.append(EvaluationAlert(
                alert_type="threshold_violation",
                severity=AlertSeverity.WARNING,
                metric_name="ndcg_at_10",
                current_value=metrics["ndcg_at_10"],
                threshold_value=thresholds.ndcg_at_10_min,
                message=(
                    f"NDCG@10 below threshold: {metrics['ndcg_at_10']:.3f} < "
                    f"{thresholds.ndcg_at_10_min:.3f}"
                ),
                context=context or {}
            ))
        
        # Check Recall@5
        if "recall_at_5" in metrics and metrics["recall_at_5"] < thresholds.recall_at_5_min:
            alerts.append(EvaluationAlert(
                alert_type="threshold_violation",
                severity=AlertSeverity.WARNING,
                metric_name="recall_at_5",
                current_value=metrics["recall_at_5"],
                threshold_value=thresholds.recall_at_5_min,
                message=(
                    f"Recall@5 below threshold: {metrics['recall_at_5']:.3f} < "
                    f"{thresholds.recall_at_5_min:.3f}"
                ),
                context=context or {}
            ))
        
        # Check BERTScore F1
        if "bertscore_f1" in metrics:
            if metrics["bertscore_f1"] < thresholds.bertscore_f1_min:
                alerts.append(EvaluationAlert(
                    alert_type="threshold_violation",
                    severity=AlertSeverity.WARNING,
                    metric_name="bertscore_f1",
                    current_value=metrics["bertscore_f1"],
                    threshold_value=thresholds.bertscore_f1_min,
                    message=(
                        f"BERTScore F1 below threshold: "
                        f"{metrics['bertscore_f1']:.3f} < "
                        f"{thresholds.bertscore_f1_min:.3f}"
                    ),
                    context=context or {}
                ))
        
        # Check faithfulness
        if "faithfulness" in metrics:
            if metrics["faithfulness"] < thresholds.faithfulness_min:
                alerts.append(EvaluationAlert(
                    alert_type="threshold_violation",
                    severity=AlertSeverity.ERROR,
                    metric_name="faithfulness",
                    current_value=metrics["faithfulness"],
                    threshold_value=thresholds.faithfulness_min,
                    message=(
                        f"Faithfulness below threshold: "
                        f"{metrics['faithfulness']:.3f} < "
                        f"{thresholds.faithfulness_min:.3f}"
                    ),
                    context=context or {}
                ))
        
        # Check latency
        if "latency_p95_ms" in metrics:
            if metrics["latency_p95_ms"] > thresholds.latency_p95_max:
                alerts.append(EvaluationAlert(
                    alert_type="threshold_violation",
                    severity=AlertSeverity.WARNING,
                    metric_name="latency_p95_ms",
                    current_value=metrics["latency_p95_ms"],
                    threshold_value=thresholds.latency_p95_max,
                    message=(
                        f"P95 latency above threshold: "
                        f"{metrics['latency_p95_ms']:.1f}ms > "
                        f"{thresholds.latency_p95_max:.1f}ms"
                    ),
                    context=context or {}
                ))
        
        # Store alerts in history
        self.alert_history.extend(alerts)
        
        # Send alerts through configured channels
        for alert in alerts:
            self._send_alert(alert)
        
        return alerts
    
    def _send_alert(self, alert: EvaluationAlert) -> None:
        """Send alert through configured channels.
        
        Args:
            alert: Alert to send
        """
        for channel in self.config.alert_channels:
            if channel == "log":
                self._log_alert(alert)
            elif channel == "slack":
                # Slack integration would go here
                pass
            elif channel == "email":
                # Email integration would go here
                pass
    
    def _log_alert(self, alert: EvaluationAlert) -> None:
        """Log an alert.
        
        Args:
            alert: Alert to log
        """
        import logging
        
        logger = logging.getLogger(__name__)
        
        log_func = logger.warning
        if alert.severity == AlertSeverity.ERROR:
            log_func = logger.error
        elif alert.severity == AlertSeverity.CRITICAL:
            log_func = logger.critical
        elif alert.severity == AlertSeverity.INFO:
            log_func = logger.info
        
        log_func(
            "Evaluation alert: %s - %s (current: %.3f, threshold: %.3f)",
            alert.alert_type,
            alert.metric_name,
            alert.current_value,
            alert.threshold_value
        )
    
    def should_sample_for_evaluation(self) -> bool:
        """Determine if current query should be evaluated based on sample rate.
        
        Returns:
            True if query should be evaluated
        """
        if not self.config.enabled or not self.config.auto_evaluate_enabled:
            return False
        
        return random.random() < self.config.sample_rate
    
    def get_alert_history(
        self,
        severity: AlertSeverity | None = None,
        metric_name: str | None = None
    ) -> list[EvaluationAlert]:
        """Get filtered alert history.
        
        Args:
            severity: Filter by severity (optional)
            metric_name: Filter by metric name (optional)
            
        Returns:
            List of alerts matching filters
        """
        alerts = self.alert_history
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if metric_name:
            alerts = [a for a in alerts if a.metric_name == metric_name]
        
        return alerts
    
    def clear_alert_history(self) -> None:
        """Clear all alert history."""
        self.alert_history.clear()
    
    async def evaluate_end_to_end(
        self,
        query: str,
        answer: str,
        retrieved_results: list[RetrievalResult],
        ground_truth_relevant_ids: list[str],
        ground_truth_answer: str | None = None,
        contexts: list[str] | None = None,
        k_values: list[int] | None = None
    ) -> dict[str, Any]:
        """Perform end-to-end evaluation of a RAG query.
        
        This convenience method runs both retrieval and generation evaluation
        in one call and returns combined results.
        
        Args:
            query: The query string
            answer: Generated answer
            retrieved_results: List of retrieved results
            ground_truth_relevant_ids: List of relevant document IDs
            ground_truth_answer: Ground truth answer (optional)
            contexts: Context texts used (optional)
            k_values: K values for retrieval metrics
            
        Returns:
            Dictionary with retrieval and generation evaluations
        """
        # Evaluate retrieval
        retrieval_eval = await self.evaluate_retrieval(
            query=query,
            results=retrieved_results,
            ground_truth=ground_truth_relevant_ids,
            k_values=k_values
        )
        
        result: dict[str, Any] = {
            "retrieval": retrieval_eval,
            "generation": None,
            "alerts": []
        }
        
        # Evaluate generation if ground truth available
        if ground_truth_answer and contexts:
            generation_eval = await self.evaluate_generation(
                query=query,
                answer=answer,
                ground_truth_answer=ground_truth_answer,
                contexts=contexts
            )
            result["generation"] = generation_eval
            
            # Check thresholds
            metrics = {
                "mrr": retrieval_eval.mrr,
                "ndcg_at_10": retrieval_eval.ndcg_at_k.get("10", 0.0),
                "bertscore_f1": generation_eval.bertscore_f1,
                "faithfulness": generation_eval.faithfulness
            }
            result["alerts"] = self.check_thresholds(
                metrics,
                context={"query": query}
            )
        
        return result
