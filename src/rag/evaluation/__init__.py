"""RAG Evaluation Framework.

This package provides comprehensive evaluation metrics and benchmarking capabilities
for measuring retrieval and generation quality in RAG systems.

Modules:
    models: Pydantic models for evaluation results and data structures
    metrics: Retrieval metrics (MRR, NDCG, Recall@K, Precision@K, Hit Rate)
    generation: Generation metrics (BERTScore, Faithfulness, Answer Relevance)
    benchmarks: Benchmark framework and dataset loaders
    evaluator: Main RAGEvaluator class

Example:
    >>> from src.rag.evaluation import RAGEvaluator
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
"""

from src.rag.evaluation.benchmarks import BenchmarkDataset, BenchmarkRunner
from src.rag.evaluation.evaluator import RAGEvaluator
from src.rag.evaluation.generation import GenerationMetrics
from src.rag.evaluation.metrics import RetrievalMetrics
from src.rag.evaluation.models import (
    ABTestResult,
    BenchmarkConfig,
    BenchmarkResult,
    EvaluationAlert,
    EvaluationConfig,
    GenerationEvaluation,
    MetricThresholds,
    RetrievalEvaluation,
    StrategyComparison,
)

__all__ = [
    # Main classes
    "BenchmarkRunner",
    "GenerationMetrics",
    "RAGEvaluator",
    "RetrievalMetrics",
    # Models
    "ABTestResult",
    "BenchmarkConfig",
    "BenchmarkDataset",
    "BenchmarkResult",
    "EvaluationAlert",
    "EvaluationConfig",
    "GenerationEvaluation",
    "MetricThresholds",
    "RetrievalEvaluation",
    "StrategyComparison",
]
