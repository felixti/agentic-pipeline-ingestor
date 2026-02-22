"""Pydantic models for RAG API requests and responses.

These models define the request/response schemas for the RAG (Retrieval-Augmented Generation)
endpoints, including query processing, strategy management, and evaluation.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.rag.models import QueryType, RAGMetrics, Source


# ============================================================================
# Enums
# ============================================================================

class StrategyPreset(str):
    """Available strategy presets for RAG queries."""
    
    AUTO = "auto"
    FAST = "fast"
    BALANCED = "balanced"
    THOROUGH = "thorough"


# ============================================================================
# Request Models
# ============================================================================

class RAGQueryRequest(BaseModel):
    """Request model for RAG query endpoint.
    
    Attributes:
        query: The user's natural language query
        strategy: Strategy preset to use (auto, fast, balanced, thorough)
        context: Optional conversation context for follow-up queries
        filters: Optional metadata filters for document retrieval
        top_k: Number of sources to retrieve (default: 5)
    """
    model_config = ConfigDict(populate_by_name=True, json_schema_extra={
        "example": {
            "query": "What is vibe coding and what are its pros and cons?",
            "strategy": "balanced",
            "context": {
                "previous_queries": ["Tell me about programming trends"],
                "session_id": "session_123"
            },
            "filters": {"source_type": "documentation"},
            "top_k": 5
        }
    })
    
    query: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The user's natural language query"
    )
    strategy: str = Field(
        default="auto",
        pattern="^(auto|fast|balanced|thorough)$",
        description="Strategy preset: auto, fast, balanced, or thorough"
    )
    context: dict[str, Any] | None = Field(
        default=None,
        description="Optional conversation context with previous_queries, previous_responses, session_id"
    )
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata filters for document retrieval"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of sources to retrieve"
    )
    
    @field_validator("query")
    @classmethod
    def validate_query_not_empty(cls, v: str) -> str:
        """Validate query is not empty or whitespace only."""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()


class RAGStrategyEvaluateRequest(BaseModel):
    """Request model for evaluating a specific RAG strategy.
    
    Attributes:
        query: The test query to evaluate
        ground_truth_relevant_ids: List of expected relevant chunk IDs
        ground_truth_answer: Expected answer text (optional)
        iterations: Number of iterations to run for averaging
    """
    model_config = ConfigDict(populate_by_name=True, json_schema_extra={
        "example": {
            "query": "What is machine learning?",
            "ground_truth_relevant_ids": ["chunk_1", "chunk_2"],
            "ground_truth_answer": "Machine learning is a subset of AI...",
            "iterations": 3
        }
    })
    
    query: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The test query to evaluate"
    )
    ground_truth_relevant_ids: list[str] = Field(
        default_factory=list,
        description="List of expected relevant chunk IDs"
    )
    ground_truth_answer: str | None = Field(
        default=None,
        description="Expected answer text for generation evaluation"
    )
    iterations: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of iterations to run for averaging"
    )
    context: dict[str, Any] | None = Field(
        default=None,
        description="Optional conversation context"
    )


class RAGBenchmarkRequest(BaseModel):
    """Request model for running benchmark evaluation.
    
    Attributes:
        benchmark_name: Name of the benchmark dataset (e.g., ms_marco, custom_qa)
        max_queries: Maximum number of queries to evaluate
        k_values: K values for retrieval metrics
        strategy_preset: Strategy preset to evaluate
    """
    model_config = ConfigDict(populate_by_name=True, json_schema_extra={
        "example": {
            "benchmark_name": "ms_marco",
            "max_queries": 100,
            "k_values": [5, 10],
            "strategy_preset": "balanced"
        }
    })
    
    benchmark_name: str = Field(
        ...,
        min_length=1,
        description="Name of benchmark dataset (ms_marco, custom_qa, etc.)"
    )
    max_queries: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of queries to evaluate"
    )
    k_values: list[int] = Field(
        default_factory=lambda: [5, 10],
        description="K values for retrieval metrics"
    )
    strategy_preset: str = Field(
        default="balanced",
        pattern="^(auto|fast|balanced|thorough)$",
        description="Strategy preset to evaluate"
    )
    
    @field_validator("k_values")
    @classmethod
    def validate_k_values(cls, v: list[int]) -> list[int]:
        """Validate k_values are positive integers."""
        if not all(isinstance(k, int) and k > 0 for k in v):
            raise ValueError("All k_values must be positive integers")
        return v


# ============================================================================
# Response Models
# ============================================================================

class RAGSource(BaseModel):
    """Source document in RAG response.
    
    Attributes:
        chunk_id: Unique identifier of the source chunk
        content: Text content from the source
        score: Relevance score (0-1)
        metadata: Additional metadata about the source
    """
    model_config = ConfigDict(populate_by_name=True)
    
    chunk_id: str = Field(..., description="Unique identifier of the source chunk")
    content: str = Field(..., description="Text content from the source")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score (0-1)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RAGQueryResponse(BaseModel):
    """Response model for RAG query endpoint.
    
    Attributes:
        answer: Generated answer text
        sources: List of source documents used
        metrics: Execution metrics
        strategy_used: Strategy preset used for this query
        query_type: Type of query classified
        query_id: Unique identifier for this query
    """
    model_config = ConfigDict(populate_by_name=True, json_schema_extra={
        "example": {
            "answer": "Vibe coding is a programming approach that emphasizes...",
            "sources": [
                {
                    "chunk_id": "550e8400-e29b-41d4-a716-446655440000",
                    "content": "Vibe coding is a programming approach...",
                    "score": 0.92,
                    "metadata": {"source": "doc1.pdf", "page": 3}
                }
            ],
            "metrics": {
                "latency_ms": 245.5,
                "tokens_used": 850,
                "retrieval_score": 0.82,
                "classification_confidence": 0.95,
                "rewrite_time_ms": 45.2,
                "retrieval_time_ms": 120.0,
                "reranking_time_ms": 35.5,
                "generation_time_ms": 44.8,
                "chunks_retrieved": 20,
                "chunks_used": 5,
                "self_correction_iterations": 0
            },
            "strategy_used": "balanced",
            "query_type": "factual",
            "query_id": "rag_query_123"
        }
    })
    
    answer: str = Field(..., description="Generated answer text")
    sources: list[RAGSource] = Field(default_factory=list, description="List of source documents used")
    metrics: RAGMetrics = Field(..., description="Execution metrics")
    strategy_used: str = Field(..., description="Strategy preset used for this query")
    query_type: QueryType = Field(..., description="Type of query classified")
    query_id: str = Field(..., description="Unique identifier for this query")


class RAGStrategyInfo(BaseModel):
    """Information about a RAG strategy.
    
    Attributes:
        name: Strategy name (fast, balanced, thorough)
        description: Human-readable description
        config: Strategy configuration settings
        use_cases: Recommended use cases
    """
    model_config = ConfigDict(populate_by_name=True)
    
    name: str = Field(..., description="Strategy name")
    description: str = Field(..., description="Human-readable description")
    config: dict[str, bool] = Field(..., description="Strategy configuration settings")
    use_cases: list[str] = Field(default_factory=list, description="Recommended use cases")
    estimated_latency_ms: int = Field(default=0, description="Estimated latency in milliseconds")


class RAGStrategiesResponse(BaseModel):
    """Response model listing available RAG strategies.
    
    Attributes:
        strategies: List of available strategies
        default_strategy: Default strategy name
        total_count: Total number of strategies
    """
    model_config = ConfigDict(populate_by_name=True)
    
    strategies: list[RAGStrategyInfo] = Field(..., description="List of available strategies")
    default_strategy: str = Field(default="balanced", description="Default strategy name")
    total_count: int = Field(..., description="Total number of strategies")


class RAGStrategyEvaluateResponse(BaseModel):
    """Response model for strategy evaluation.
    
    Attributes:
        strategy_name: Name of the evaluated strategy
        query: The test query used
        retrieval_metrics: Retrieval quality metrics
        generation_metrics: Generation quality metrics (if ground truth provided)
        latency_ms: Total evaluation time
        query_type: Classified query type
        per_iteration_results: Results from each iteration
    """
    model_config = ConfigDict(populate_by_name=True)
    
    strategy_name: str = Field(..., description="Name of the evaluated strategy")
    query: str = Field(..., description="The test query used")
    retrieval_metrics: dict[str, Any] = Field(..., description="Retrieval quality metrics")
    generation_metrics: dict[str, Any] | None = Field(
        default=None, description="Generation quality metrics (if ground truth provided)"
    )
    latency_ms: float = Field(..., description="Total evaluation time in milliseconds")
    query_type: QueryType = Field(..., description="Classified query type")
    per_iteration_results: list[dict[str, Any]] = Field(
        default_factory=list, description="Results from each iteration"
    )


class RAGMetricsSummary(BaseModel):
    """Summary of RAG system metrics.
    
    Attributes:
        total_queries: Total number of queries processed
        avg_latency_ms: Average query latency
        avg_retrieval_score: Average retrieval quality score
        strategy_usage: Usage count per strategy
        query_type_distribution: Distribution of query types
        timestamp: When metrics were collected
    """
    model_config = ConfigDict(populate_by_name=True)
    
    total_queries: int = Field(default=0, description="Total number of queries processed")
    avg_latency_ms: float = Field(default=0.0, description="Average query latency")
    avg_retrieval_score: float = Field(default=0.0, description="Average retrieval quality score")
    avg_classification_confidence: float = Field(default=0.0, description="Average classification confidence")
    strategy_usage: dict[str, int] = Field(default_factory=dict, description="Usage count per strategy")
    query_type_distribution: dict[str, int] = Field(default_factory=dict, description="Distribution of query types")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When metrics were collected")


class RAGMetricsResponse(BaseModel):
    """Response model for RAG metrics endpoint.
    
    Attributes:
        summary: Summary metrics
        component_health: Health status of RAG components
        recent_alerts: Recent evaluation alerts
        performance_trends: Performance trend data
    """
    model_config = ConfigDict(populate_by_name=True)
    
    summary: RAGMetricsSummary = Field(..., description="Summary metrics")
    component_health: dict[str, Any] = Field(default_factory=dict, description="Health status of RAG components")
    recent_alerts: list[dict[str, Any]] = Field(default_factory=list, description="Recent evaluation alerts")
    performance_trends: dict[str, Any] = Field(default_factory=dict, description="Performance trend data")


class RAGBenchmarkResponse(BaseModel):
    """Response model for benchmark evaluation.
    
    Attributes:
        benchmark_name: Name of the benchmark
        strategy_preset: Strategy preset evaluated
        aggregate_metrics: Aggregated metrics across all queries
        per_query_results: Individual results for each query
        total_queries: Total number of queries processed
        successful_queries: Number of successfully processed queries
        failed_queries: Number of failed queries
        total_latency_ms: Total time for benchmark run
        started_at: When benchmark started
        completed_at: When benchmark completed
    """
    model_config = ConfigDict(populate_by_name=True)
    
    benchmark_name: str = Field(..., description="Name of the benchmark")
    strategy_preset: str = Field(..., description="Strategy preset evaluated")
    aggregate_metrics: dict[str, float] = Field(..., description="Aggregated metrics across all queries")
    per_query_results: list[dict[str, Any]] = Field(default_factory=list, description="Individual results")
    total_queries: int = Field(..., description="Total number of queries processed")
    successful_queries: int = Field(..., description="Number of successfully processed queries")
    failed_queries: int = Field(..., description="Number of failed queries")
    total_latency_ms: float = Field(..., description="Total time for benchmark run")
    started_at: datetime = Field(..., description="When benchmark started")
    completed_at: datetime | None = Field(default=None, description="When benchmark completed")


# ============================================================================
# Error Response Models
# ============================================================================

class RAGQueryErrorResponse(BaseModel):
    """Error response for RAG query endpoint."""
    model_config = ConfigDict(populate_by_name=True)
    
    error: dict[str, Any] = Field(..., description="Error details")
    query_id: str | None = Field(default=None, description="Query ID if available")


class RAGValidationErrorResponse(BaseModel):
    """Validation error response."""
    model_config = ConfigDict(populate_by_name=True)
    
    error: dict[str, Any] = Field(..., description="Error details")
    validation_errors: list[dict[str, Any]] = Field(default_factory=list, description="Validation errors")
