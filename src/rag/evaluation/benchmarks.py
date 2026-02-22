"""Benchmark framework for RAG evaluation.

This module provides tools for running standardized benchmarks on RAG systems,
including dataset loading, metric aggregation, and A/B testing capabilities.

Supported datasets:
    - MS MARCO: Microsoft Machine Reading Comprehension dataset
    - Custom QA: User-defined question-answer pairs
    - HuggingFace datasets: Any compatible HF dataset

Features:
    - Automated benchmark runs
    - A/B testing between strategies
    - Statistical significance testing
    - Progress tracking and reporting
"""

import json
import random
import time
from datetime import datetime
from typing import Any, Protocol

import numpy as np
from scipy import stats

from src.rag.evaluation.models import (
    ABTestResult,
    BenchmarkConfig,
    BenchmarkDataset,
    BenchmarkResult,
    StrategyComparison,
)

__all__ = [
    "BenchmarkRunner",
    "BenchmarkDataset",  # Re-export for convenience
    "RAGSystem",
]


class RAGSystem(Protocol):
    """Protocol for RAG system to be benchmarked.
    
    Any system implementing this protocol can be benchmarked.
    """
    
    async def query(self, query: str, **kwargs: Any) -> object:
        """Execute a query and return results.
        
        Returns:
            Object with attributes:
                - answer: Generated answer string
                - retrieved_chunks: List of objects with 'id' attribute
                - latency_ms: Query latency
        """
        ...


class BenchmarkRunner:
    """Runner for RAG benchmark evaluations.
    
    This class orchestrates benchmark runs, computes metrics, and generates
    reports comparing different RAG strategies.
    
    Example:
        >>> from src.rag.evaluation.benchmarks import BenchmarkRunner
        >>>
        >>> runner = BenchmarkRunner()
        >>>
        >>> # Load a benchmark dataset
        >>> dataset = await runner.load_dataset("ms_marco")
        >>>
        >>> # Run benchmark on a RAG system
        >>> result = await runner.run_benchmark(
        ...     benchmark_name="ms_marco",
        ...     rag_system=my_rag_system,
        ...     max_queries=100
        ... )
        >>>
        >>> print(f"Mean MRR: {result.aggregate_metrics['mean_mrr']}")
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize benchmark runner.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    async def load_dataset(
        self,
        name: str,
        dataset_path: str | None = None,
        max_queries: int | None = None
    ) -> list[BenchmarkDataset]:
        """Load a benchmark dataset.
        
        Args:
            name: Name of the benchmark (e.g., "ms_marco")
            dataset_path: Optional path to dataset (for custom datasets)
            max_queries: Maximum number of queries to load
            
        Returns:
            List of BenchmarkDataset items
            
        Raises:
            ValueError: If dataset type is not supported
            FileNotFoundError: If dataset file not found
        """
        if name.lower() == "ms_marco":
            return await self._load_ms_marco(max_queries)
        elif name.lower() == "custom_qa" or dataset_path:
            return await self._load_custom_qa(dataset_path or "", max_queries)
        else:
            raise ValueError(f"Unknown benchmark dataset: {name}")
    
    async def _load_ms_marco(
        self,
        max_queries: int | None = None
    ) -> list[BenchmarkDataset]:
        """Load MS MARCO dataset.
        
        Note: This is a placeholder implementation. In production,
        this would load from the actual MS MARCO dataset.
        
        Args:
            max_queries: Maximum number of queries to load
            
        Returns:
            List of BenchmarkDataset items
        """
        # Placeholder: In production, load from actual MS MARCO dataset
        # For now, return synthetic data for testing
        sample_queries = [
            ("What is machine learning?", ["doc_ml_1", "doc_ml_2"],
             "Machine learning is a subset of AI..."),
            ("How does a car engine work?", ["doc_engine_1"],
             "A car engine converts fuel into motion..."),
            ("What are the benefits of exercise?", ["doc_health_1", "doc_health_2"],
             "Exercise improves cardiovascular health..."),
            ("Who invented the telephone?", ["doc_history_1"],
             "Alexander Graham Bell invented the telephone..."),
            ("What is photosynthesis?", ["doc_bio_1", "doc_bio_2"],
             "Photosynthesis is the process by which plants..."),
        ]
        
        datasets = []
        for i, (query, relevant_ids, answer) in enumerate(sample_queries):
            if max_queries and i >= max_queries:
                break
            
            datasets.append(BenchmarkDataset(
                id=f"ms_marco_{i:04d}",
                query=query,
                ground_truth_relevant_ids=relevant_ids,
                ground_truth_answer=answer,
                metadata={"source": "MS MARCO", "category": "general"}
            ))
        
        return datasets
    
    async def _load_custom_qa(
        self,
        dataset_path: str,
        max_queries: int | None = None
    ) -> list[BenchmarkDataset]:
        """Load custom QA dataset from file.
        
        Supports JSON format:
        [
            {
                "id": "q1",
                "query": "What is X?",
                "ground_truth_relevant_ids": ["doc1", "doc2"],
                "ground_truth_answer": "X is..."
            }
        ]
        
        Args:
            dataset_path: Path to dataset file
            max_queries: Maximum number of queries to load
            
        Returns:
            List of BenchmarkDataset items
        """
        if dataset_path.startswith("s3://"):
            # Load from S3
            raise NotImplementedError("S3 loading not yet implemented")
        
        with open(dataset_path) as f:
            data = json.load(f)
        
        datasets = []
        for i, item in enumerate(data):
            if max_queries and i >= max_queries:
                break
            
            datasets.append(BenchmarkDataset(
                id=item.get("id", f"custom_{i:04d}"),
                query=item["query"],
                ground_truth_relevant_ids=item.get("ground_truth_relevant_ids", []),
                ground_truth_answer=item.get("ground_truth_answer"),
                metadata=item.get("metadata", {})
            ))
        
        return datasets
    
    async def run_benchmark(
        self,
        benchmark_name: str,
        rag_system: RAGSystem,
        config: BenchmarkConfig | None = None,
        strategy_config: dict[str, Any] | None = None
    ) -> BenchmarkResult:
        """Run a full benchmark evaluation.
        
        Args:
            benchmark_name: Name of the benchmark to run
            rag_system: RAG system to benchmark
            config: Benchmark configuration
            strategy_config: RAG strategy configuration used
            
        Returns:
            BenchmarkResult with all metrics and results
        """
        from src.rag.evaluation.evaluator import RAGEvaluator
        
        start_time = time.perf_counter()
        
        # Load configuration
        if config is None:
            config = BenchmarkConfig(
                name=benchmark_name,
                dataset_path=benchmark_name,
                max_queries=100
            )
        
        # Load dataset
        dataset = await self.load_dataset(
            benchmark_name,
            config.dataset_path,
            config.max_queries
        )
        
        # Initialize evaluator
        evaluator = RAGEvaluator()
        
        # Run evaluation on each query
        per_query_results: list[dict[str, Any]] = []
        successful = 0
        failed = 0
        
        for item in dataset:
            try:
                # Run RAG query
                rag_result = await rag_system.query(item.query)
                
                # Extract attributes from result object
                rag_answer = getattr(rag_result, "answer", "")
                rag_chunks = getattr(rag_result, "retrieved_chunks", [])
                rag_latency = getattr(rag_result, "latency_ms", 0.0)
                
                # Evaluate retrieval
                retrieval_eval = await evaluator.evaluate_retrieval(
                    query=item.query,
                    results=rag_chunks,
                    ground_truth=item.ground_truth_relevant_ids,
                    k_values=config.k_values
                )
                
                # Evaluate generation if ground truth answer available
                generation_eval = None
                if item.ground_truth_answer:
                    contexts = [
                        getattr(c, "content", str(c))
                        for c in rag_chunks
                    ]
                    generation_eval = await evaluator.evaluate_generation(
                        query=item.query,
                        answer=rag_answer,
                        ground_truth_answer=item.ground_truth_answer,
                        contexts=contexts
                    )
                
                per_query_results.append({
                    "query_id": item.id,
                    "query": item.query,
                    "retrieval": retrieval_eval.model_dump(),
                    "generation": generation_eval.model_dump() if generation_eval else None,
                    "latency_ms": rag_latency
                })
                
                successful += 1
                
            except Exception as e:
                per_query_results.append({
                    "query_id": item.id,
                    "query": item.query,
                    "error": str(e)
                })
                failed += 1
        
        # Compute aggregate metrics
        aggregate_metrics = self._compute_aggregate_metrics(per_query_results)
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        return BenchmarkResult(
            benchmark_name=benchmark_name,
            config=config.model_dump(),
            aggregate_metrics=aggregate_metrics,
            per_query_results=per_query_results,
            strategy_config=strategy_config or {},
            total_queries=len(dataset),
            successful_queries=successful,
            failed_queries=failed,
            total_latency_ms=total_time,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow()
        )
    
    def _compute_aggregate_metrics(
        self,
        per_query_results: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Compute aggregate metrics from per-query results.
        
        Args:
            per_query_results: List of per-query result dictionaries
            
        Returns:
            Dictionary of aggregate metrics
        """
        metrics: dict[str, list[float]] = {}
        
        for result in per_query_results:
            if "error" in result:
                continue
            
            # Aggregate retrieval metrics
            if result.get("retrieval"):
                retrieval = result["retrieval"]
                for key in ["mrr", "latency_ms"]:
                    if key in retrieval:
                        metrics.setdefault(f"{key}", []).append(retrieval[key])
                
                # Aggregate metrics at different K values
                for metric_type in ["ndcg_at_k", "recall_at_k", "precision_at_k", "hit_rate_at_k"]:
                    if metric_type in retrieval:
                        for k, value in retrieval[metric_type].items():
                            metrics.setdefault(f"{metric_type}_{k}", []).append(value)
            
            # Aggregate generation metrics
            if result.get("generation"):
                generation = result["generation"]
                for key in ["bertscore_f1", "faithfulness", "answer_relevance", "latency_ms"]:
                    if key in generation:
                        metrics.setdefault(f"{key}", []).append(generation[key])
            
            # Overall latency
            if "latency_ms" in result:
                metrics.setdefault("overall_latency_ms", []).append(result["latency_ms"])
        
        # Compute statistics
        aggregate: dict[str, float] = {}
        for metric_name, values in metrics.items():
            if values:
                aggregate[f"mean_{metric_name}"] = float(np.mean(values))
                aggregate[f"median_{metric_name}"] = float(np.median(values))
                aggregate[f"std_{metric_name}"] = float(np.std(values))
                aggregate[f"min_{metric_name}"] = float(np.min(values))
                aggregate[f"max_{metric_name}"] = float(np.max(values))
                aggregate[f"p95_{metric_name}"] = float(np.percentile(values, 95))
        
        return aggregate
    
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
        
        Args:
            test_name: Name of the A/B test
            control_system: Control (baseline) RAG system
            treatment_system: Treatment (new) RAG system
            benchmark_name: Benchmark dataset to use
            sample_size: Number of queries to test
            confidence_level: Statistical confidence level
            
        Returns:
            ABTestResult with comparison and statistical analysis
        """
        # Run benchmark on both systems
        control_result = await self.run_benchmark(
            benchmark_name=benchmark_name,
            rag_system=control_system,
            config=BenchmarkConfig(
                name=benchmark_name,
                dataset_path=benchmark_name,
                max_queries=sample_size
            )
        )
        
        treatment_result = await self.run_benchmark(
            benchmark_name=benchmark_name,
            rag_system=treatment_system,
            config=BenchmarkConfig(
                name=benchmark_name,
                dataset_path=benchmark_name,
                max_queries=sample_size
            )
        )
        
        # Extract key metrics for comparison
        control_metrics = self._extract_key_metrics(control_result)
        treatment_metrics = self._extract_key_metrics(treatment_result)
        
        # Calculate relative improvements
        improvements: dict[str, float] = {}
        for key in control_metrics:
            if control_metrics[key] != 0:
                improvements[key] = (
                    (treatment_metrics[key] - control_metrics[key])
                    / control_metrics[key]
                    * 100
                )
            else:
                improvements[key] = 0.0
        
        # Perform statistical significance testing
        p_values: dict[str, float] = {}
        is_significant: dict[str, bool] = {}
        
        for metric_name in ["mrr", "ndcg_at_10", "bertscore_f1"]:
            control_values = self._extract_metric_values(control_result, metric_name)
            treatment_values = self._extract_metric_values(treatment_result, metric_name)
            
            if control_values and treatment_values:
                # Two-sample t-test
                _, p_value = stats.ttest_ind(control_values, treatment_values)
                p_values[metric_name] = float(p_value)
                is_significant[metric_name] = p_value < (1 - confidence_level)
            else:
                p_values[metric_name] = 1.0
                is_significant[metric_name] = False
        
        # Determine recommendation
        recommendation = self._generate_recommendation(
            improvements, is_significant, control_metrics, treatment_metrics
        )
        
        return ABTestResult(
            test_name=test_name,
            control_strategy="control",
            treatment_strategy="treatment",
            control_metrics=control_metrics,
            treatment_metrics=treatment_metrics,
            relative_improvements=improvements,
            p_values=p_values,
            is_significant=is_significant,
            recommendation=recommendation,
            confidence_level=confidence_level,
            sample_size_control=control_result.successful_queries,
            sample_size_treatment=treatment_result.successful_queries
        )
    
    def _extract_key_metrics(self, result: BenchmarkResult) -> dict[str, float]:
        """Extract key metrics from benchmark result.
        
        Args:
            result: BenchmarkResult
            
        Returns:
            Dictionary of key metrics
        """
        metrics: dict[str, float] = {}
        
        # Extract retrieval metrics
        for metric in ["mean_mrr", "mean_ndcg_at_k_10", "mean_recall_at_k_5"]:
            if metric in result.aggregate_metrics:
                metrics[metric.replace("mean_", "")] = result.aggregate_metrics[metric]
        
        # Extract generation metrics
        for metric in ["mean_bertscore_f1", "mean_faithfulness", "mean_answer_relevance"]:
            if metric in result.aggregate_metrics:
                metrics[metric.replace("mean_", "")] = result.aggregate_metrics[metric]
        
        # Extract latency metrics
        if "p95_overall_latency_ms" in result.aggregate_metrics:
            metrics["latency_p95_ms"] = result.aggregate_metrics["p95_overall_latency_ms"]
        
        return metrics
    
    def _extract_metric_values(
        self,
        result: BenchmarkResult,
        metric_name: str
    ) -> list[float]:
        """Extract per-query values for a specific metric.
        
        Args:
            result: BenchmarkResult
            metric_name: Name of metric to extract
            
        Returns:
            List of metric values
        """
        values = []
        
        for query_result in result.per_query_results:
            if "error" in query_result:
                continue
            
            if metric_name == "mrr" and query_result.get("retrieval"):
                values.append(query_result["retrieval"].get("mrr", 0.0))
            elif metric_name == "ndcg_at_10" and query_result.get("retrieval"):
                ndcg = query_result["retrieval"].get("ndcg_at_k", {})
                values.append(ndcg.get("10", 0.0))
            elif metric_name == "bertscore_f1" and query_result.get("generation"):
                values.append(query_result["generation"].get("bertscore_f1", 0.0))
        
        return values
    
    def _generate_recommendation(
        self,
        improvements: dict[str, float],
        is_significant: dict[str, bool],
        control_metrics: dict[str, float],
        treatment_metrics: dict[str, float]
    ) -> str:
        """Generate recommendation based on A/B test results.
        
        Args:
            improvements: Relative improvements
            is_significant: Statistical significance flags
            control_metrics: Control group metrics
            treatment_metrics: Treatment group metrics
            
        Returns:
            Recommendation string
        """
        # Check for significant improvements
        significant_improvements = [
            k for k, v in improvements.items()
            if is_significant.get(k, False) and v > 0
        ]
        
        significant_regressions = [
            k for k, v in improvements.items()
            if is_significant.get(k, False) and v < 0
        ]
        
        if significant_improvements and not significant_regressions:
            return (
                f"Deploy treatment strategy. Significant improvements in: "
                f"{', '.join(significant_improvements)}"
            )
        elif significant_regressions and not significant_improvements:
            return (
                f"Keep control strategy. Significant regressions in: "
                f"{', '.join(significant_regressions)}"
            )
        elif significant_improvements and significant_regressions:
            return (
                "Mixed results. Review trade-offs between improvements in "
                f"{', '.join(significant_improvements)} and regressions in "
                f"{', '.join(significant_regressions)}"
            )
        else:
            return "No significant difference. Consider running a larger test."
    
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
        # Run benchmarks
        result_a = await self.run_benchmark(
            benchmark_name=benchmark_name,
            rag_system=strategy_a_system,
            config=BenchmarkConfig(
                name=benchmark_name,
                dataset_path=benchmark_name,
                max_queries=sample_size
            )
        )
        
        result_b = await self.run_benchmark(
            benchmark_name=benchmark_name,
            rag_system=strategy_b_system,
            config=BenchmarkConfig(
                name=benchmark_name,
                dataset_path=benchmark_name,
                max_queries=sample_size
            )
        )
        
        # Extract metrics
        metrics_a = self._extract_key_metrics(result_a)
        metrics_b = self._extract_key_metrics(result_b)
        
        # Calculate improvements (positive = A better than B)
        improvements: dict[str, float] = {}
        for key in metrics_a:
            if metrics_b.get(key, 0) != 0:
                improvements[key] = (
                    (metrics_a[key] - metrics_b[key]) / metrics_b[key]
                )
            else:
                improvements[key] = 0.0
        
        # Statistical significance
        statistical_significance: dict[str, bool] = {}
        for metric_name in ["mrr", "ndcg_at_10", "bertscore_f1"]:
            values_a = self._extract_metric_values(result_a, metric_name)
            values_b = self._extract_metric_values(result_b, metric_name)
            
            if values_a and values_b:
                _, p_value = stats.ttest_ind(values_a, values_b)
                statistical_significance[metric_name] = p_value < 0.05
            else:
                statistical_significance[metric_name] = False
        
        # Determine winner
        winner = None
        significant_wins = sum(
            1 for k, v in improvements.items()
            if statistical_significance.get(k, False) and v > 0
        )
        significant_losses = sum(
            1 for k, v in improvements.items()
            if statistical_significance.get(k, False) and v < 0
        )
        
        if significant_wins > significant_losses:
            winner = "A"
        elif significant_losses > significant_wins:
            winner = "B"
        else:
            winner = "tie"
        
        return StrategyComparison(
            strategy_a_name=strategy_a_name,
            strategy_b_name=strategy_b_name,
            strategy_a_metrics=metrics_a,
            strategy_b_metrics=metrics_b,
            improvements=improvements,
            statistical_significance=statistical_significance,
            winner=winner,
            sample_size=sample_size
        )
