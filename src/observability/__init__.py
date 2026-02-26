"""Observability package for the Agentic Data Pipeline Ingestor.

This package provides comprehensive observability features including:
- OpenTelemetry distributed tracing
- GenAI-specific span attributes for LLM operations
- Prometheus metrics
- Structured JSON logging
- FastAPI middleware for automatic instrumentation
"""

from src.observability.genai_spans import GenAISpanAttributes
from src.observability.logging import StructuredLogger, get_logger
from src.observability.metrics import (
    EmbeddingMetrics,
    MetricsManager,
    SearchMetrics,
    get_metrics_manager,
)
from src.observability.middleware import ObservabilityMiddleware  # pyright: ignore[reportAttributeAccessIssue]
from src.observability.tracing import TelemetryManager, get_telemetry_manager, get_tracer

__all__ = [
    "GenAISpanAttributes",
    "MetricsManager",
    "SearchMetrics",
    "EmbeddingMetrics",
    "ObservabilityMiddleware",
    "StructuredLogger",
    "TelemetryManager",
    "get_logger",
    "get_metrics_manager",
    "get_telemetry_manager",
    "get_tracer",
]
