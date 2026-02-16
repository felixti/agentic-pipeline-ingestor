"""Observability package for the Agentic Data Pipeline Ingestor.

This package provides comprehensive observability features including:
- OpenTelemetry distributed tracing
- GenAI-specific span attributes for LLM operations
- Prometheus metrics
- Structured JSON logging
- FastAPI middleware for automatic instrumentation
"""

from src.observability.genai_spans import GenAISpanAttributes
from src.observability.metrics import MetricsManager, get_metrics_manager
from src.observability.tracing import TelemetryManager, get_telemetry_manager, get_tracer
from src.observability.logging import StructuredLogger, get_logger
from src.observability.middleware import ObservabilityMiddleware

__all__ = [
    "GenAISpanAttributes",
    "MetricsManager",
    "get_metrics_manager",
    "TelemetryManager",
    "get_telemetry_manager",
    "get_tracer",
    "StructuredLogger",
    "get_logger",
    "ObservabilityMiddleware",
]
