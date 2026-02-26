"""RAG (Retrieval-Augmented Generation) API routes.

This module provides endpoints for:
- RAG query processing with multiple strategy presets
- Listing available RAG strategies
- Evaluating strategy performance
- Retrieving RAG system metrics
- Running benchmark evaluations

All endpoints integrate with the AgenticRAG router and support various
optimization strategies including query rewriting, HyDE, reranking, and hybrid search.
"""

import time
import uuid
from datetime import datetime
from threading import Lock
from types import SimpleNamespace
from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from src.api.models.rag import (
    RAGBenchmarkRequest,
    RAGBenchmarkResponse,
    RAGMetricsResponse,
    RAGMetricsSummary,
    RAGQueryErrorResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    RAGSource,
    RAGStrategiesResponse,
    RAGStrategyEvaluateRequest,
    RAGStrategyEvaluateResponse,
    RAGStrategyInfo,
    RAGValidationErrorResponse,
)
from src.observability.logging import get_logger
from src.rag.classification import QueryClassifier
from src.rag.evaluation.evaluator import RAGEvaluator
from src.rag.evaluation.evaluator import RetrievalResult as EvaluatorRetrievalResult
from src.rag.models import QueryType, RAGResult, Source
from src.rag.router import STRATEGY_PRESETS, AgenticRAG
from src.rag.strategies.hyde import HyDERewriter
from src.rag.strategies.query_rewriting import QueryRewriter
from src.rag.strategies.reranking import ReRanker

router = APIRouter(prefix="/rag", tags=["RAG"])
logger = get_logger(__name__)

# ============================================================================
# Singleton instances for RAG components (initialized on first use)
# ============================================================================

_agentic_rag: AgenticRAG | None = None
_query_rewriter: QueryRewriter | None = None
_query_classifier: QueryClassifier | None = None
_hyde_rewriter: HyDERewriter | None = None
_reranker: ReRanker | None = None
_rag_evaluator: RAGEvaluator | None = None
_agentic_rag_lock = Lock()
_rag_evaluator_lock = Lock()


def _get_agentic_rag() -> AgenticRAG:
    """Get or initialize the AgenticRAG singleton.

    Returns:
        AgenticRAG instance
    """
    global _agentic_rag, _query_rewriter, _query_classifier, _hyde_rewriter, _reranker

    existing_rag = _agentic_rag
    if existing_rag is not None:
        return existing_rag

    with _agentic_rag_lock:
        if _agentic_rag is None:
            logger.info("Initializing AgenticRAG")

            # Initialize required components
            if _query_rewriter is None:
                _query_rewriter = QueryRewriter()

            if _query_classifier is None:
                _query_classifier = QueryClassifier()

            if _hyde_rewriter is None:
                try:
                    _hyde_rewriter = HyDERewriter.from_settings()
                    logger.info("HyDERewriter initialized")
                except Exception as e:
                    logger.warning(
                        "HyDERewriter initialization failed; disabling HyDE", error=str(e)
                    )
                    _hyde_rewriter = None

            if _reranker is None:
                try:
                    _reranker = ReRanker()
                    logger.info("ReRanker initialized")
                except Exception as e:
                    logger.warning(
                        "ReRanker initialization failed; disabling reranking", error=str(e)
                    )
                    _reranker = None

            # Initialize AgenticRAG
            _agentic_rag = AgenticRAG(
                query_rewriter=_query_rewriter,
                classifier=_query_classifier,
                hyde_rewriter=_hyde_rewriter,
                reranker=_reranker,
            )
            logger.info("AgenticRAG initialized successfully")

        assert _agentic_rag is not None
        return _agentic_rag


def _get_evaluator() -> RAGEvaluator:
    """Get or initialize the RAGEvaluator singleton.

    Returns:
        RAGEvaluator instance
    """
    global _rag_evaluator

    existing_evaluator = _rag_evaluator
    if existing_evaluator is not None:
        return existing_evaluator

    with _rag_evaluator_lock:
        if _rag_evaluator is None:
            logger.info("Initializing RAGEvaluator")
            _rag_evaluator = RAGEvaluator()
            logger.info("RAGEvaluator initialized successfully")

        assert _rag_evaluator is not None
        return _rag_evaluator


def _source_to_api(source: Source) -> RAGSource:
    """Convert internal Source to API RAGSource.

    Args:
        source: Internal Source model

    Returns:
        API RAGSource model
    """
    return RAGSource(
        chunk_id=source.chunk_id,
        content=source.content,
        score=source.score,
        metadata=source.metadata,
    )


def _result_to_response(result: RAGResult, query_id: str) -> RAGQueryResponse:
    """Convert RAGResult to API RAGQueryResponse.

    Args:
        result: RAGResult from AgenticRAG
        query_id: Unique query identifier

    Returns:
        API RAGQueryResponse
    """
    return RAGQueryResponse(
        answer=result.answer,
        sources=[_source_to_api(s) for s in result.sources],
        metrics=result.metrics,
        strategy_used=result.strategy_used,
        query_type=result.query_type,
        query_id=query_id,
    )


# ============================================================================
# Routes
# ============================================================================


@router.post(
    "/query",
    response_model=RAGQueryResponse,
    responses={
        400: {"model": RAGValidationErrorResponse, "description": "Invalid request parameters"},
        500: {"model": RAGQueryErrorResponse, "description": "RAG processing failed"},
    },
    summary="Execute RAG query",
    description="""
    Execute a Retrieval-Augmented Generation query with the specified strategy.
    
    The endpoint:
    1. Classifies the query type (factual, analytical, comparative, vague, multi_hop)
    2. Selects optimal strategies based on query type and preset
    3. Rewrites the query for better retrieval
    4. Retrieves relevant documents
    5. Re-ranks results (if enabled)
    6. Evaluates quality and self-corrects if needed
    7. Generates the final answer with source attribution
    
    **Strategy Presets:**
    - `auto`: Automatically selects strategies based on query classification
    - `fast`: Prioritizes speed (query_rewrite + hybrid_search only)
    - `balanced`: Balanced approach with reranking (default)
    - `thorough`: Maximum quality with HyDE and all optimizations
    """,
)
async def rag_query(
    request: Request,
    query_request: RAGQueryRequest,
) -> RAGQueryResponse:
    """Execute a RAG query.

    Args:
        request: FastAPI request object
        query_request: RAG query parameters

    Returns:
        RAG query response with answer, sources, and metrics

    Raises:
        HTTPException: If query processing fails
    """
    query_id = f"rag_{uuid.uuid4().hex[:12]}"
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    logger.info(
        "rag_query_request",
        request_id=request_id,
        query_id=query_id,
        query=query_request.query[:100],
        strategy=query_request.strategy,
    )

    try:
        # Get AgenticRAG instance
        agentic_rag = _get_agentic_rag()

        # Process the query
        result = await agentic_rag.process(
            query=query_request.query,
            strategy_preset=query_request.strategy,
            context=query_request.context,
        )

        # Convert to API response
        response = _result_to_response(result, query_id)

        logger.info(
            "rag_query_completed",
            request_id=request_id,
            query_id=query_id,
            latency_ms=result.metrics.latency_ms,
            strategy_used=result.strategy_used,
            query_type=result.query_type.value,
            chunks_used=len(result.sources),
        )

        return response

    except Exception as e:
        logger.error(
            "rag_query_failed",
            request_id=request_id,
            query_id=query_id,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "RAG_PROCESSING_FAILED",
                    "message": f"Failed to process RAG query: {e!s}",
                },
                "query_id": query_id,
            },
        ) from e


@router.get(
    "/strategies",
    response_model=RAGStrategiesResponse,
    summary="List available RAG strategies",
    description="""
    Get information about all available RAG strategy presets.
    
    Returns details about each strategy including:
    - Configuration settings (query_rewrite, hyde, reranking, hybrid_search)
    - Recommended use cases
    - Estimated latency
    """,
)
async def list_strategies(request: Request) -> RAGStrategiesResponse:
    """List available RAG strategies.

    Args:
        request: FastAPI request object

    Returns:
        List of available strategies with details
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    logger.info("list_strategies_request", request_id=request_id)

    # Define strategy information
    strategies = [
        RAGStrategyInfo(
            name="fast",
            description="Prioritizes speed over quality. Uses query rewriting and hybrid search only.",
            config=STRATEGY_PRESETS["fast"],
            use_cases=["Simple factual queries", "High-traffic scenarios", "Mobile apps"],
            estimated_latency_ms=100,
        ),
        RAGStrategyInfo(
            name="balanced",
            description="Balanced approach with reranking for improved quality at moderate latency.",
            config=STRATEGY_PRESETS["balanced"],
            use_cases=["General purpose queries", "Production workloads", "Recommended default"],
            estimated_latency_ms=250,
        ),
        RAGStrategyInfo(
            name="thorough",
            description="Maximum quality with HyDE, reranking, and all optimizations.",
            config=STRATEGY_PRESETS["thorough"],
            use_cases=["Complex analytical queries", "Research tasks", "Low-latency requirements"],
            estimated_latency_ms=500,
        ),
    ]

    # Check if auto is available (it's always available)
    auto_config = {
        "query_rewrite": True,
        "hyde": False,  # Determined dynamically
        "reranking": True,  # Determined dynamically
        "hybrid_search": True,
    }

    strategies.insert(
        0,
        RAGStrategyInfo(
            name="auto",
            description="Automatically selects optimal strategies based on query classification.",
            config=auto_config,
            use_cases=["Unknown query types", "Mixed workloads", "Self-optimizing systems"],
            estimated_latency_ms=250,
        ),
    )

    logger.info(
        "list_strategies_completed",
        request_id=request_id,
        count=len(strategies),
    )

    return RAGStrategiesResponse(
        strategies=strategies,
        default_strategy="balanced",
        total_count=len(strategies),
    )


@router.post(
    "/strategies/{name}/evaluate",
    response_model=RAGStrategyEvaluateResponse,
    responses={
        400: {"model": RAGValidationErrorResponse, "description": "Invalid request parameters"},
        404: {"description": "Strategy not found"},
        500: {"model": RAGQueryErrorResponse, "description": "Evaluation failed"},
    },
    summary="Evaluate a RAG strategy",
    description="""
    Evaluate a specific RAG strategy against a test query with optional ground truth.
    
    This endpoint is useful for:
    - Comparing strategy performance on specific query types
    - Validating strategy configurations
    - Benchmarking before production deployment
    
    Provide ground_truth_relevant_ids and ground_truth_answer for comprehensive evaluation
    including both retrieval and generation metrics.
    """,
)
async def evaluate_strategy(
    name: str,
    request: Request,
    eval_request: RAGStrategyEvaluateRequest,
) -> RAGStrategyEvaluateResponse:
    """Evaluate a specific RAG strategy.

    Args:
        name: Strategy name (fast, balanced, thorough, auto)
        request: FastAPI request object
        eval_request: Evaluation parameters

    Returns:
        Evaluation results with metrics

    Raises:
        HTTPException: If strategy not found or evaluation fails
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    logger.info(
        "evaluate_strategy_request",
        request_id=request_id,
        strategy=name,
        query=eval_request.query[:100],
        iterations=eval_request.iterations,
    )

    # Validate strategy name
    valid_strategies = ["auto", "fast", "balanced", "thorough"]
    if name not in valid_strategies:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "STRATEGY_NOT_FOUND",
                    "message": f"Strategy '{name}' not found. Valid strategies: {valid_strategies}",
                }
            },
        )

    try:
        start_time = time.perf_counter()

        # Get components
        agentic_rag = _get_agentic_rag()
        evaluator = _get_evaluator()

        # Run multiple iterations
        iteration_results = []
        all_results = []

        for iteration in range(eval_request.iterations):
            # Execute RAG query
            result = await agentic_rag.process(
                query=eval_request.query,
                strategy_preset=name,
                context=eval_request.context,
            )
            all_results.append(result)

            # Evaluate retrieval if ground truth provided
            if eval_request.ground_truth_relevant_ids:
                # Create retrieval results from sources
                retrieval_results: list[EvaluatorRetrievalResult] = [
                    cast("EvaluatorRetrievalResult", SimpleNamespace(id=s.chunk_id))
                    for s in result.sources
                ]

                retrieval_eval = await evaluator.evaluate_retrieval(
                    query=eval_request.query,
                    results=retrieval_results,
                    ground_truth=eval_request.ground_truth_relevant_ids,
                )

                iteration_results.append(
                    {
                        "iteration": iteration + 1,
                        "retrieval": {
                            "mrr": retrieval_eval.mrr,
                            "ndcg_at_10": retrieval_eval.ndcg_at_k.get("10", 0.0),
                            "recall_at_5": retrieval_eval.recall_at_k.get("5", 0.0),
                        },
                        "latency_ms": result.metrics.latency_ms,
                        "strategy_used": result.strategy_used,
                    }
                )
            else:
                iteration_results.append(
                    {
                        "iteration": iteration + 1,
                        "latency_ms": result.metrics.latency_ms,
                        "strategy_used": result.strategy_used,
                        "chunks_used": len(result.sources),
                    }
                )

        # Calculate aggregate metrics
        avg_latency = sum(r.metrics.latency_ms for r in all_results) / len(all_results)
        avg_retrieval_score = sum(r.metrics.retrieval_score for r in all_results) / len(all_results)

        retrieval_metrics = {
            "avg_latency_ms": round(avg_latency, 2),
            "avg_retrieval_score": round(avg_retrieval_score, 3),
            "avg_chunks_used": sum(len(r.sources) for r in all_results) / len(all_results),
            "avg_tokens_used": sum(r.metrics.tokens_used for r in all_results) / len(all_results),
        }

        # Evaluate generation if ground truth provided
        generation_metrics = None
        if eval_request.ground_truth_answer and all_results:
            # Use the last result for generation evaluation
            last_result = all_results[-1]
            contexts = [s.content for s in last_result.sources]

            generation_eval = await evaluator.evaluate_generation(
                query=eval_request.query,
                answer=last_result.answer,
                ground_truth_answer=eval_request.ground_truth_answer,
                contexts=contexts,
            )

            generation_metrics = {
                "bertscore_f1": generation_eval.bertscore_f1,
                "faithfulness": generation_eval.faithfulness,
                "answer_relevance": generation_eval.answer_relevance,
            }

            retrieval_metrics["bertscore_f1"] = generation_eval.bertscore_f1

        total_latency_ms = (time.perf_counter() - start_time) * 1000

        # Get query type from first result
        query_type = all_results[0].query_type if all_results else QueryType.FACTUAL

        logger.info(
            "evaluate_strategy_completed",
            request_id=request_id,
            strategy=name,
            latency_ms=total_latency_ms,
            iterations=eval_request.iterations,
        )

        return RAGStrategyEvaluateResponse(
            strategy_name=name,
            query=eval_request.query,
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            latency_ms=total_latency_ms,
            query_type=query_type,
            per_iteration_results=iteration_results,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "evaluate_strategy_failed",
            request_id=request_id,
            strategy=name,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "EVALUATION_FAILED",
                    "message": f"Failed to evaluate strategy: {e!s}",
                }
            },
        ) from e


@router.get(
    "/metrics",
    response_model=RAGMetricsResponse,
    summary="Get RAG system metrics",
    description="""
    Retrieve current metrics for the RAG system.
    
    Returns:
    - Summary statistics (total queries, average latency, quality scores)
    - Component health status
    - Recent evaluation alerts
    - Performance trends
    
    Note: This endpoint returns real-time metrics. For historical data,
    use the audit logs or monitoring dashboards.
    """,
)
async def get_metrics(request: Request) -> RAGMetricsResponse:
    """Get RAG system metrics.

    Args:
        request: FastAPI request object

    Returns:
        RAG metrics response
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    logger.info("get_rag_metrics_request", request_id=request_id)

    try:
        # Get component health
        agentic_rag = _get_agentic_rag()
        health = await agentic_rag.health_check()

        # Get evaluator for alerts
        evaluator = _get_evaluator()
        alerts = evaluator.get_alert_history()

        # Format recent alerts
        recent_alerts = [
            {
                "type": alert.alert_type,
                "severity": alert.severity.value,
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
            }
            for alert in alerts[-10:]  # Last 10 alerts
        ]

        # Create summary (in a real implementation, these would come from metrics storage)
        summary = RAGMetricsSummary(
            total_queries=0,  # Would be populated from metrics store
            avg_latency_ms=0.0,
            avg_retrieval_score=0.0,
            avg_classification_confidence=0.0,
            strategy_usage={"auto": 0, "fast": 0, "balanced": 0, "thorough": 0},
            query_type_distribution={
                "factual": 0,
                "analytical": 0,
                "comparative": 0,
                "vague": 0,
                "multi_hop": 0,
            },
        )

        logger.info(
            "get_rag_metrics_completed",
            request_id=request_id,
            health_status=health.get("healthy", False),
            alert_count=len(recent_alerts),
        )

        return RAGMetricsResponse(
            summary=summary,
            component_health=health,
            recent_alerts=recent_alerts,
            performance_trends={},
        )

    except Exception as e:
        logger.error(
            "get_rag_metrics_failed",
            request_id=request_id,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "METRICS_FAILED",
                    "message": f"Failed to retrieve metrics: {e!s}",
                }
            },
        ) from e


@router.post(
    "/evaluate",
    response_model=RAGBenchmarkResponse,
    responses={
        400: {"model": RAGValidationErrorResponse, "description": "Invalid request parameters"},
        500: {"model": RAGQueryErrorResponse, "description": "Benchmark failed"},
    },
    summary="Run benchmark evaluation",
    description="""
    Run a benchmark evaluation of the RAG system against a standard dataset.
    
    This endpoint is useful for:
    - Assessing overall RAG system quality
    - Comparing strategy performance
    - Regression testing
    - Capacity planning
    
    **Benchmarks:**
    - `ms_marco`: Microsoft MARCO dataset (if available)
    - `custom_qa`: Custom question-answer pairs (if configured)
    
    Note: Benchmark runs can take several minutes depending on max_queries.
    Consider using async processing for large benchmarks.
    """,
)
async def run_benchmark(
    request: Request,
    benchmark_request: RAGBenchmarkRequest,
) -> RAGBenchmarkResponse:
    """Run benchmark evaluation.

    Args:
        request: FastAPI request object
        benchmark_request: Benchmark parameters

    Returns:
        Benchmark results with aggregate and per-query metrics

    Raises:
        HTTPException: If benchmark fails
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    logger.info(
        "run_benchmark_request",
        request_id=request_id,
        benchmark=benchmark_request.benchmark_name,
        max_queries=benchmark_request.max_queries,
        strategy=benchmark_request.strategy_preset,
    )

    try:
        started_at = datetime.now()

        # Get evaluator
        evaluator = _get_evaluator()
        agentic_rag = _get_agentic_rag()

        # Create a simple RAG system wrapper for evaluation
        class SimpleRAGSystem:
            """Simple RAG system wrapper for benchmarking."""

            def __init__(self, agentic_rag: AgenticRAG, strategy_preset: str):
                self.agentic_rag = agentic_rag
                self.strategy_preset = strategy_preset

            async def query(self, query: str, **kwargs: Any) -> Any:
                """Execute query."""
                return await self.agentic_rag.process(
                    query=query,
                    strategy_preset=self.strategy_preset,
                )

        rag_system = SimpleRAGSystem(agentic_rag, benchmark_request.strategy_preset)

        # Run benchmark
        from src.rag.evaluation.models import BenchmarkConfig

        config = BenchmarkConfig(
            name=benchmark_request.benchmark_name,
            dataset_path=benchmark_request.benchmark_name,
            max_queries=benchmark_request.max_queries,
            k_values=benchmark_request.k_values,
        )

        # Note: In a full implementation, this would run against actual benchmark data
        # For now, we return a placeholder response
        benchmark_result = await evaluator.run_benchmark(
            benchmark_name=benchmark_request.benchmark_name,
            rag_system=rag_system,
            config=config,
            strategy_config={"preset": benchmark_request.strategy_preset},
        )

        completed_at = datetime.now()

        logger.info(
            "run_benchmark_completed",
            request_id=request_id,
            benchmark=benchmark_request.benchmark_name,
            total_queries=benchmark_result.total_queries,
            successful_queries=benchmark_result.successful_queries,
            latency_ms=benchmark_result.total_latency_ms,
        )

        return RAGBenchmarkResponse(
            benchmark_name=benchmark_request.benchmark_name,
            strategy_preset=benchmark_request.strategy_preset,
            aggregate_metrics=benchmark_result.aggregate_metrics,
            per_query_results=benchmark_result.per_query_results,
            total_queries=benchmark_result.total_queries,
            successful_queries=benchmark_result.successful_queries,
            failed_queries=benchmark_result.failed_queries,
            total_latency_ms=benchmark_result.total_latency_ms,
            started_at=started_at,
            completed_at=completed_at,
        )

    except Exception as e:
        logger.error(
            "run_benchmark_failed",
            request_id=request_id,
            benchmark=benchmark_request.benchmark_name,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "BENCHMARK_FAILED",
                    "message": f"Failed to run benchmark: {e!s}",
                }
            },
        ) from e


# Import datetime here to avoid circular imports
def datetime_now() -> datetime:
    """Get current datetime."""
    from datetime import datetime

    return datetime.now()
