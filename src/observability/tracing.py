"""OpenTelemetry tracing configuration for the Agentic Data Pipeline Ingestor.

This module provides distributed tracing capabilities using OpenTelemetry,
with support for Jaeger, Azure Monitor, and other OTLP-compatible backends.
"""

import os
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional, Union

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION, DEPLOYMENT_ENVIRONMENT
from opentelemetry.trace import Status, StatusCode, SpanKind

# Optional exporters - these may not be installed
try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    OTLP_GRPC_AVAILABLE = True
except ImportError:
    OTLP_GRPC_AVAILABLE = False

try:
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    JAEGER_AVAILABLE = True
except ImportError:
    JAEGER_AVAILABLE = False

try:
    from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
    AZURE_MONITOR_AVAILABLE = True
except ImportError:
    AZURE_MONITOR_AVAILABLE = False

from src.config import settings

# Global tracer provider instance
_tracer_provider: Optional[TracerProvider] = None
_telemetry_manager: Optional["TelemetryManager"] = None


def get_tracer(name: str = "pipeline-ingestor") -> trace.Tracer:
    """Get a tracer instance.
    
    Args:
        name: The tracer name (usually module name)
        
    Returns:
        OpenTelemetry Tracer instance
    """
    return trace.get_tracer(name)


class TelemetryManager:
    """Manages OpenTelemetry tracing configuration.
    
    This class handles the initialization and configuration of distributed
    tracing for the pipeline system, supporting multiple backends.
    
    Supported Backends:
    - OTLP (OpenTelemetry Protocol) - Generic gRPC/HTTP
    - Jaeger - Popular open-source tracing
    - Azure Monitor - Native Azure cloud tracing
    - Console - Development/debugging output
    
    Example:
        >>> manager = TelemetryManager()
        >>> manager.setup_tracing(
        ...     service_name="pipeline-api",
        ...     jaeger_endpoint="http://jaeger:14268/api/traces"
        ... )
    """
    
    def __init__(self):
        """Initialize the telemetry manager."""
        self._provider: Optional[TracerProvider] = None
        self._initialized = False
        self._exporters: list = []
    
    def setup_tracing(
        self,
        service_name: str,
        service_version: str = "1.0.0",
        environment: str = "development",
        jaeger_endpoint: Optional[str] = None,
        otlp_endpoint: Optional[str] = None,
        azure_connection_string: Optional[str] = None,
        console_export: bool = False,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> TracerProvider:
        """Setup distributed tracing with configured exporters.
        
        Args:
            service_name: Name of the service being traced
            service_version: Version of the service
            environment: Deployment environment (development, staging, production)
            jaeger_endpoint: Optional Jaeger collector endpoint
            otlp_endpoint: Optional OTLP collector endpoint
            azure_connection_string: Optional Azure Monitor connection string
            console_export: Whether to export spans to console (for debugging)
            attributes: Additional resource attributes
            
        Returns:
            Configured TracerProvider instance
            
        Raises:
            RuntimeError: If tracing is already initialized
        """
        global _tracer_provider
        
        if self._initialized:
            raise RuntimeError("Tracing is already initialized")
        
        # Build resource attributes
        resource_attrs = {
            SERVICE_NAME: service_name,
            SERVICE_VERSION: service_version,
            DEPLOYMENT_ENVIRONMENT: environment,
        }
        
        if attributes:
            resource_attrs.update(attributes)
        
        # Add custom attributes for our pipeline
        resource_attrs.update({
            "service.namespace": "agentic-pipeline",
            "service.instance.id": os.getenv("HOSTNAME", "unknown"),
            "host.name": os.getenv("HOSTNAME", "localhost"),
        })
        
        # Create resource and provider
        resource = Resource.create(resource_attrs)
        self._provider = TracerProvider(resource=resource)
        
        # Add exporters based on configuration
        self._add_exporters(
            jaeger_endpoint=jaeger_endpoint,
            otlp_endpoint=otlp_endpoint,
            azure_connection_string=azure_connection_string,
            console_export=console_export,
        )
        
        # Set as global provider
        trace.set_tracer_provider(self._provider)
        _tracer_provider = self._provider
        self._initialized = True
        
        return self._provider
    
    def _add_exporters(
        self,
        jaeger_endpoint: Optional[str] = None,
        otlp_endpoint: Optional[str] = None,
        azure_connection_string: Optional[str] = None,
        console_export: bool = False,
    ) -> None:
        """Add span exporters based on configuration.
        
        Args:
            jaeger_endpoint: Jaeger collector endpoint
            otlp_endpoint: OTLP collector endpoint
            azure_connection_string: Azure Monitor connection string
            console_export: Enable console export
        """
        # Jaeger exporter
        if jaeger_endpoint and JAEGER_AVAILABLE:
            jaeger_exporter = JaegerExporter(
                collector_endpoint=jaeger_endpoint,
            )
            self._provider.add_span_processor(
                BatchSpanProcessor(jaeger_exporter)
            )
            self._exporters.append("jaeger")
        elif jaeger_endpoint and not JAEGER_AVAILABLE:
            import logging
            logging.getLogger(__name__).warning(
                "Jaeger endpoint configured but jaeger exporter not installed. "
                "Install with: pip install opentelemetry-exporter-jaeger"
            )
        
        # OTLP exporter
        if otlp_endpoint and OTLP_GRPC_AVAILABLE:
            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            self._provider.add_span_processor(
                BatchSpanProcessor(otlp_exporter)
            )
            self._exporters.append("otlp")
        elif otlp_endpoint and not OTLP_GRPC_AVAILABLE:
            import logging
            logging.getLogger(__name__).warning(
                "OTLP endpoint configured but OTLP exporter not installed. "
                "Install with: pip install opentelemetry-exporter-otlp"
            )
        
        # Azure Monitor exporter
        if azure_connection_string and AZURE_MONITOR_AVAILABLE:
            azure_exporter = AzureMonitorTraceExporter(
                connection_string=azure_connection_string
            )
            self._provider.add_span_processor(
                BatchSpanProcessor(azure_exporter)
            )
            self._exporters.append("azure_monitor")
        elif azure_connection_string and not AZURE_MONITOR_AVAILABLE:
            import logging
            logging.getLogger(__name__).warning(
                "Azure Monitor configured but exporter not installed. "
                "Install with: pip install azure-monitor-opentelemetry-exporter"
            )
        
        # Console exporter for debugging
        if console_export or (not self._exporters and settings.debug):
            console_exporter = ConsoleSpanExporter()
            self._provider.add_span_processor(
                BatchSpanProcessor(console_exporter)
            )
            self._exporters.append("console")
    
    def shutdown(self) -> None:
        """Shutdown the tracer provider and flush pending spans."""
        if self._provider:
            self._provider.shutdown()
            self._initialized = False
            self._exporters.clear()
    
    @property
    def is_initialized(self) -> bool:
        """Check if tracing is initialized."""
        return self._initialized
    
    @property
    def active_exporters(self) -> list:
        """List of active exporter names."""
        return self._exporters.copy()


def get_telemetry_manager() -> TelemetryManager:
    """Get or create the global telemetry manager instance.
    
    Returns:
        TelemetryManager singleton instance
    """
    global _telemetry_manager
    if _telemetry_manager is None:
        _telemetry_manager = TelemetryManager()
    return _telemetry_manager


def setup_tracing_from_settings() -> Optional[TracerProvider]:
    """Setup tracing from application settings.
    
    This convenience function reads tracing configuration from
    environment variables and settings.
    
    Returns:
        Configured TracerProvider or None if disabled
    """
    manager = get_telemetry_manager()
    
    # Check if tracing is enabled
    if os.getenv("OTEL_ENABLED", "true").lower() == "false":
        return None
    
    service_name = os.getenv("OTEL_SERVICE_NAME", "pipeline-api")
    service_version = settings.app_version
    environment = os.getenv("OTEL_ENVIRONMENT", settings.env)
    
    jaeger_endpoint = os.getenv("OTEL_EXPORTER_JAEGER_ENDPOINT")
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    azure_connection = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    console_export = os.getenv("OTEL_CONSOLE_EXPORT", "false").lower() == "true"
    
    return manager.setup_tracing(
        service_name=service_name,
        service_version=service_version,
        environment=environment,
        jaeger_endpoint=jaeger_endpoint,
        otlp_endpoint=otlp_endpoint,
        azure_connection_string=azure_connection,
        console_export=console_export,
    )


@contextmanager
def start_span(
    name: str,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None,
    parent=None,
) -> Generator[trace.Span, None, None]:
    """Context manager for starting a span.
    
    Args:
        name: Span name
        kind: Span kind (SERVER, CLIENT, INTERNAL, etc.)
        attributes: Initial span attributes
        parent: Parent span context
        
    Yields:
        Active span
        
    Example:
        >>> with start_span("process_document", SpanKind.INTERNAL) as span:
        ...     span.set_attribute("document.id", doc_id)
        ...     process_document(doc_id)
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(
        name=name,
        kind=kind,
        attributes=attributes,
        parent=parent,
    ) as span:
        yield span


@contextmanager
def start_pipeline_stage_span(
    stage_name: str,
    job_id: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> Generator[trace.Span, None, None]:
    """Context manager for pipeline stage spans.
    
    Args:
        stage_name: Name of the pipeline stage
        job_id: Associated job ID
        attributes: Additional span attributes
        
    Yields:
        Active span with pipeline context
    """
    attrs = {
        "pipeline.stage": stage_name,
    }
    if job_id:
        attrs["job.id"] = job_id
    if attributes:
        attrs.update(attributes)
    
    with start_span(
        name=f"pipeline.stage.{stage_name}",
        kind=SpanKind.INTERNAL,
        attributes=attrs,
    ) as span:
        yield span
