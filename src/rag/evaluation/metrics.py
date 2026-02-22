"""Retrieval metrics for RAG evaluation.

This module implements standard information retrieval metrics for evaluating
the quality of document retrieval in RAG systems.

Metrics implemented:
    - MRR (Mean Reciprocal Rank)
    - NDCG@K (Normalized Discounted Cumulative Gain)
    - Recall@K
    - Precision@K
    - Hit Rate@K

All metrics are optimized for performance with target computation times < 10ms.
"""

import math
import time
from typing import Protocol

import numpy as np


class RetrievalResult(Protocol):
    """Protocol for retrieval results.
    
    Any object with an 'id' attribute can be used as a retrieval result.
    """
    
    id: str


class RetrievalMetrics:
    """Standard retrieval metrics for RAG evaluation.
    
    This class provides static methods for computing various information
    retrieval metrics. All methods are optimized for performance.
    
    Example:
        >>> from src.rag.evaluation.metrics import RetrievalMetrics
        >>> 
        >>> results = [
        ...     type('Result', (), {'id': 'doc_1'}),
        ...     type('Result', (), {'id': 'doc_2'}),
        ...     type('Result', (), {'id': 'doc_3'}),
        ... ]
        >>> ground_truth = ['doc_2', 'doc_4']
        >>> 
        >>> mrr = RetrievalMetrics.mrr(results, ground_truth)
        >>> ndcg = RetrievalMetrics.ndcg_at_k(results, ground_truth, k=10)
        >>> recall = RetrievalMetrics.recall_at_k(results, ground_truth, k=5)
    """
    
    @staticmethod
    def mrr(results: list[RetrievalResult], ground_truth: list[str]) -> float:
        """Calculate Mean Reciprocal Rank.
        
        MRR is the average of the reciprocal ranks of the first relevant
        document for each query. Reciprocal rank is 1/rank of first relevant doc.
        
        Args:
            results: List of retrieved results (must have 'id' attribute)
            ground_truth: List of relevant document IDs
            
        Returns:
            MRR score between 0 and 1
            
        Example:
            >>> results = [{'id': 'a'}, {'id': 'b'}, {'id': 'c'}]
            >>> ground_truth = ['b']
            >>> RetrievalMetrics.mrr(results, ground_truth)
            0.5  # 'b' is at rank 2, so 1/2 = 0.5
        """
        if not ground_truth:
            return 0.0
        
        ground_truth_set = set(ground_truth)
        
        for i, result in enumerate(results, 1):
            if result.id in ground_truth_set:
                return 1.0 / i
        
        return 0.0
    
    @staticmethod
    def ndcg_at_k(
        results: list[RetrievalResult],
        ground_truth: list[str],
        k: int = 10
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain at K.
        
        NDCG measures the quality of ranking by considering the position
        of relevant documents. It uses a logarithmic discount factor to
        penalize relevant documents that appear lower in the ranking.
        
        Args:
            results: List of retrieved results (must have 'id' attribute)
            ground_truth: List of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            NDCG@K score between 0 and 1
            
        Example:
            >>> results = [{'id': 'a'}, {'id': 'b'}, {'id': 'c'}]
            >>> ground_truth = ['b', 'c']
            >>> RetrievalMetrics.ndcg_at_k(results, ground_truth, k=3)
            0.613  # NDCG calculation with logarithmic discount
        """
        if not ground_truth or k <= 0:
            return 0.0
        
        ground_truth_set = set(ground_truth)
        k = min(k, len(results))
        
        # Calculate DCG
        dcg = 0.0
        for i, result in enumerate(results[:k], 1):
            if result.id in ground_truth_set:
                # Use log base 2, add 1 because rank starts at 1
                dcg += 1.0 / math.log2(i + 1)
        
        # Calculate ideal DCG (all relevant docs at top)
        num_relevant = min(len(ground_truth_set), k)
        ideal_dcg = sum(
            1.0 / math.log2(i + 1)
            for i in range(1, num_relevant + 1)
        )
        
        return dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    
    @staticmethod
    def recall_at_k(
        results: list[RetrievalResult],
        ground_truth: list[str],
        k: int = 10
    ) -> float:
        """Calculate Recall at K.
        
        Recall@K is the proportion of relevant documents that were
        retrieved in the top K results.
        
        Args:
            results: List of retrieved results (must have 'id' attribute)
            ground_truth: List of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            Recall@K score between 0 and 1
            
        Example:
            >>> results = [{'id': 'a'}, {'id': 'b'}, {'id': 'c'}]
            >>> ground_truth = ['b', 'd']  # 'd' not in results
            >>> RetrievalMetrics.recall_at_k(results, ground_truth, k=3)
            0.5  # Only 'b' retrieved out of 2 relevant
        """
        if not ground_truth:
            return 0.0
        
        ground_truth_set = set(ground_truth)
        k = min(k, len(results))
        
        retrieved_ids = {result.id for result in results[:k]}
        relevant_retrieved = len(retrieved_ids & ground_truth_set)
        
        return relevant_retrieved / len(ground_truth_set)
    
    @staticmethod
    def precision_at_k(
        results: list[RetrievalResult],
        ground_truth: list[str],
        k: int = 10
    ) -> float:
        """Calculate Precision at K.
        
        Precision@K is the proportion of retrieved documents in the top K
        that are relevant.
        
        Args:
            results: List of retrieved results (must have 'id' attribute)
            ground_truth: List of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            Precision@K score between 0 and 1
            
        Example:
            >>> results = [{'id': 'a'}, {'id': 'b'}, {'id': 'c'}]
            >>> ground_truth = ['b']
            >>> RetrievalMetrics.precision_at_k(results, ground_truth, k=3)
            0.333  # 1 relevant out of 3 retrieved
        """
        if k <= 0 or not results:
            return 0.0
        
        ground_truth_set = set(ground_truth)
        k = min(k, len(results))
        
        retrieved_ids = {result.id for result in results[:k]}
        relevant_retrieved = len(retrieved_ids & ground_truth_set)
        
        return relevant_retrieved / k
    
    @staticmethod
    def hit_rate_at_k(
        results: list[RetrievalResult],
        ground_truth: list[str],
        k: int = 10
    ) -> float:
        """Calculate Hit Rate at K.
        
        Hit Rate@K (also known as Success@K) indicates whether at least
        one relevant document appears in the top K results.
        
        Args:
            results: List of retrieved results (must have 'id' attribute)
            ground_truth: List of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            1.0 if at least one relevant doc is in top K, 0.0 otherwise
            
        Example:
            >>> results = [{'id': 'a'}, {'id': 'b'}, {'id': 'c'}]
            >>> ground_truth = ['b']
            >>> RetrievalMetrics.hit_rate_at_k(results, ground_truth, k=2)
            1.0  # 'b' is in top 2
        """
        if not ground_truth or k <= 0:
            return 0.0
        
        ground_truth_set = set(ground_truth)
        k = min(k, len(results))
        
        for result in results[:k]:
            if result.id in ground_truth_set:
                return 1.0
        
        return 0.0
    
    @staticmethod
    def average_precision(
        results: list[RetrievalResult],
        ground_truth: list[str]
    ) -> float:
        """Calculate Average Precision (AP).
        
        AP is the average of precision values at each rank where a
        relevant document is retrieved.
        
        Args:
            results: List of retrieved results (must have 'id' attribute)
            ground_truth: List of relevant document IDs
            
        Returns:
            Average Precision score between 0 and 1
        """
        if not ground_truth:
            return 0.0
        
        ground_truth_set = set(ground_truth)
        precisions = []
        num_relevant_found = 0
        
        for i, result in enumerate(results, 1):
            if result.id in ground_truth_set:
                num_relevant_found += 1
                precisions.append(num_relevant_found / i)
        
        if not precisions:
            return 0.0
        
        return sum(precisions) / len(ground_truth_set)
    
    @staticmethod
    def compute_all_metrics(
        results: list[RetrievalResult],
        ground_truth: list[str],
        k_values: list[int] | None = None
    ) -> dict[str, float]:
        """Compute all retrieval metrics at once.
        
        This is an optimized method that computes all metrics in a single pass,
        minimizing redundant calculations.
        
        Args:
            results: List of retrieved results (must have 'id' attribute)
            ground_truth: List of relevant document IDs
            k_values: List of K values for metrics that require K (default: [5, 10])
            
        Returns:
            Dictionary containing all computed metrics
            
        Example:
            >>> results = [{'id': 'a'}, {'id': 'b'}, {'id': 'c'}]
            >>> ground_truth = ['b']
            >>> metrics = RetrievalMetrics.compute_all_metrics(results, ground_truth)
            >>> print(metrics['mrr'])
            0.5
        """
        if k_values is None:
            k_values = [5, 10]
        
        start_time = time.perf_counter()
        
        metrics: dict[str, float] = {}
        
        # Compute metrics that don't depend on K
        metrics["mrr"] = RetrievalMetrics.mrr(results, ground_truth)
        metrics["map"] = RetrievalMetrics.average_precision(results, ground_truth)
        
        # Compute metrics for each K value
        for k in k_values:
            k_str = str(k)
            metrics[f"ndcg_at_{k_str}"] = RetrievalMetrics.ndcg_at_k(
                results, ground_truth, k
            )
            metrics[f"recall_at_{k_str}"] = RetrievalMetrics.recall_at_k(
                results, ground_truth, k
            )
            metrics[f"precision_at_{k_str}"] = RetrievalMetrics.precision_at_k(
                results, ground_truth, k
            )
            metrics[f"hit_rate_at_{k_str}"] = RetrievalMetrics.hit_rate_at_k(
                results, ground_truth, k
            )
        
        # Record computation time
        metrics["latency_ms"] = (time.perf_counter() - start_time) * 1000
        
        return metrics


class BatchRetrievalMetrics:
    """Batch computation of retrieval metrics for multiple queries.
    
    This class efficiently computes metrics across multiple queries
    and provides aggregate statistics.
    
    Example:
        >>> batch = BatchRetrievalMetrics()
        >>> 
        >>> # Add results from multiple queries
        >>> batch.add_query_results(query_id="q1", results=results1, ground_truth=gt1)
        >>> batch.add_query_results(query_id="q2", results=results2, ground_truth=gt2)
        >>> 
        >>> # Get aggregate metrics
        >>> agg_metrics = batch.get_aggregate_metrics()
        >>> print(f"Mean MRR: {agg_metrics['mean_mrr']}")
    """
    
    def __init__(self, k_values: list[int] | None = None):
        """Initialize batch metrics calculator.
        
        Args:
            k_values: List of K values for metrics (default: [5, 10])
        """
        self.k_values = k_values or [5, 10]
        self.query_results: list[dict[str, float]] = []
    
    def add_query_results(
        self,
        query_id: str,
        results: list[RetrievalResult],
        ground_truth: list[str]
    ) -> dict[str, float]:
        """Add results for a single query and compute its metrics.
        
        Args:
            query_id: Unique identifier for this query
            results: List of retrieved results
            ground_truth: List of relevant document IDs
            
        Returns:
            Computed metrics for this query
        """
        metrics = RetrievalMetrics.compute_all_metrics(
            results, ground_truth, self.k_values
        )
        metrics["query_id"] = query_id  # type: ignore
        self.query_results.append(metrics)
        return metrics
    
    def get_aggregate_metrics(self) -> dict[str, float]:
        """Compute aggregate metrics across all queries.
        
        Returns:
            Dictionary with mean, median, std for each metric
        """
        if not self.query_results:
            return {}
        
        # Collect all metric names (excluding query_id)
        metric_names = [
            k for k in self.query_results[0].keys()
            if k != "query_id" and isinstance(self.query_results[0][k], (int, float))
        ]
        
        aggregate: dict[str, float] = {}
        
        for metric_name in metric_names:
            values = [r[metric_name] for r in self.query_results if metric_name in r]
            
            if values:
                aggregate[f"mean_{metric_name}"] = float(np.mean(values))
                aggregate[f"median_{metric_name}"] = float(np.median(values))
                aggregate[f"std_{metric_name}"] = float(np.std(values))
                aggregate[f"min_{metric_name}"] = float(np.min(values))
                aggregate[f"max_{metric_name}"] = float(np.max(values))
        
        aggregate["num_queries"] = len(self.query_results)
        
        return aggregate
    
    def reset(self) -> None:
        """Clear all accumulated query results."""
        self.query_results.clear()


# Convenience functions for quick metric computation
def compute_mrr(results: list[RetrievalResult], ground_truth: list[str]) -> float:
    """Convenience function to compute MRR.
    
    Args:
        results: List of retrieved results
        ground_truth: List of relevant document IDs
        
    Returns:
        MRR score
    """
    return RetrievalMetrics.mrr(results, ground_truth)


def compute_ndcg_at_k(
    results: list[RetrievalResult],
    ground_truth: list[str],
    k: int = 10
) -> float:
    """Convenience function to compute NDCG@K.
    
    Args:
        results: List of retrieved results
        ground_truth: List of relevant document IDs
        k: Number of top results to consider
        
    Returns:
        NDCG@K score
    """
    return RetrievalMetrics.ndcg_at_k(results, ground_truth, k)


def compute_recall_at_k(
    results: list[RetrievalResult],
    ground_truth: list[str],
    k: int = 10
) -> float:
    """Convenience function to compute Recall@K.
    
    Args:
        results: List of retrieved results
        ground_truth: List of relevant document IDs
        k: Number of top results to consider
        
    Returns:
        Recall@K score
    """
    return RetrievalMetrics.recall_at_k(results, ground_truth, k)


def compute_precision_at_k(
    results: list[RetrievalResult],
    ground_truth: list[str],
    k: int = 10
) -> float:
    """Convenience function to compute Precision@K.
    
    Args:
        results: List of retrieved results
        ground_truth: List of relevant document IDs
        k: Number of top results to consider
        
    Returns:
        Precision@K score
    """
    return RetrievalMetrics.precision_at_k(results, ground_truth, k)


def compute_hit_rate_at_k(
    results: list[RetrievalResult],
    ground_truth: list[str],
    k: int = 10
) -> float:
    """Convenience function to compute Hit Rate@K.
    
    Args:
        results: List of retrieved results
        ground_truth: List of relevant document IDs
        k: Number of top results to consider
        
    Returns:
        Hit Rate@K score (1.0 or 0.0)
    """
    return RetrievalMetrics.hit_rate_at_k(results, ground_truth, k)
