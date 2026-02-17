"""Structured logging configuration for the Agentic Data Pipeline Ingestor.

This module provides structured JSON logging with correlation ID tracking,
request context, and integration with OpenTelemetry trace context.
"""

import logging
import sys
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import structlog
from structlog.types import FilteringBoundLogger

# Context variables for request tracking
_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)
_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)
_span_id: ContextVar[str | None] = ContextVar("span_id", default=None)
_request_context: ContextVar[dict[str, Any]] = ContextVar("request_context", default={})


class StructuredLogger:
    """Structured JSON logging manager.
    
    This class configures and manages structured logging using structlog,
    with support for JSON output, correlation IDs, and OpenTelemetry integration.
    
    Example:
        >>> logger = StructuredLogger()
        >>> logger.setup_logging(json_format=True)
        >>> log = logger.get_logger("my_module")
        >>> log.info("event_occurred", key="value")
    """

    def __init__(self):
        """Initialize the structured logger."""
        self._configured = False

    def setup_logging(
        self,
        json_format: bool = True,
        log_level: str = "INFO",
        log_file: str | None = None,
        include_trace_context: bool = True,
        extra_processors: list | None = None,
    ) -> None:
        """Setup structured logging configuration.
        
        Args:
            json_format: Whether to output JSON format (vs. console)
            log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional file path to log to
            include_trace_context: Whether to include OpenTelemetry trace context
            extra_processors: Additional structlog processors
        """
        if self._configured:
            return

        # Configure standard library logging
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=getattr(logging, log_level.upper()),
        )

        # Build processor chain
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
        ]

        # Add trace context processor if enabled
        if include_trace_context:
            processors.append(self._add_trace_context)

        # Add correlation ID processor
        processors.append(self._add_correlation_id)

        # Add request context processor
        processors.append(self._add_request_context)

        # Add extra processors
        if extra_processors:
            processors.extend(extra_processors)

        # Add common processors
        processors.extend([
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ])

        # Add formatter
        if json_format:
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())

        # Configure structlog
        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        self._configured = True

    def _add_trace_context(
        self,
        logger: FilteringBoundLogger,
        method_name: str,
        event_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """Add OpenTelemetry trace context to log events.
        
        Args:
            logger: Logger instance
            method_name: Method being called
            event_dict: Event dictionary being built
            
        Returns:
            Updated event dictionary
        """
        try:
            from opentelemetry import trace

            current_span = trace.get_current_span()
            if current_span:
                span_context = current_span.get_span_context()
                if span_context.is_valid:
                    event_dict["trace_id"] = format(span_context.trace_id, "032x")
                    event_dict["span_id"] = format(span_context.span_id, "016x")
                    event_dict["trace_flags"] = str(span_context.trace_flags)
        except ImportError:
            pass

        # Also add from context vars as fallback
        trace_id = _trace_id.get()
        span_id = _span_id.get()
        if trace_id and "trace_id" not in event_dict:
            event_dict["trace_id"] = trace_id
        if span_id and "span_id" not in event_dict:
            event_dict["span_id"] = span_id

        return event_dict

    def _add_correlation_id(
        self,
        logger: FilteringBoundLogger,
        method_name: str,
        event_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """Add correlation ID to log events.
        
        Args:
            logger: Logger instance
            method_name: Method being called
            event_dict: Event dictionary being built
            
        Returns:
            Updated event dictionary
        """
        corr_id = _correlation_id.get()
        if corr_id:
            event_dict["correlation_id"] = corr_id
        return event_dict

    def _add_request_context(
        self,
        logger: FilteringBoundLogger,
        method_name: str,
        event_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """Add request context to log events.
        
        Args:
            logger: Logger instance
            method_name: Method being called
            event_dict: Event dictionary being built
            
        Returns:
            Updated event dictionary
        """
        context = _request_context.get()
        if context:
            event_dict.update(context)
        return event_dict

    def get_logger(self, name: str) -> FilteringBoundLogger:
        """Get a logger instance.
        
        Args:
            name: Logger name (usually module name)
            
        Returns:
            Configured structlog logger
        """
        if not self._configured:
            self.setup_logging()
        return structlog.get_logger(name)


# Global logger instance
_structured_logger: StructuredLogger | None = None


def get_logger(name: str) -> FilteringBoundLogger:
    """Get a structured logger.
    
    Args:
        name: Logger name
        
    Returns:
        Structured logger instance
    """
    global _structured_logger
    if _structured_logger is None:
        _structured_logger = StructuredLogger()
    return _structured_logger.get_logger(name)


def setup_logging(
    json_format: bool = True,
    log_level: str = "INFO",
    **kwargs: Any,
) -> StructuredLogger:
    """Setup structured logging globally.
    
    Args:
        json_format: Whether to output JSON
        log_level: Minimum log level
        **kwargs: Additional configuration
        
    Returns:
        Configured StructuredLogger
    """
    global _structured_logger
    _structured_logger = StructuredLogger()
    _structured_logger.setup_logging(
        json_format=json_format,
        log_level=log_level,
        **kwargs,
    )
    return _structured_logger


# Context management functions

def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID for the current context.
    
    Args:
        correlation_id: Correlation identifier
    """
    _correlation_id.set(correlation_id)


def get_correlation_id() -> str | None:
    """Get the current correlation ID.
    
    Returns:
        Current correlation ID or None
    """
    return _correlation_id.get()


def set_trace_context(trace_id: str, span_id: str) -> None:
    """Set the trace context for the current context.
    
    Args:
        trace_id: Trace ID
        span_id: Span ID
    """
    _trace_id.set(trace_id)
    _span_id.set(span_id)


def clear_trace_context() -> None:
    """Clear the trace context."""
    _trace_id.set(None)
    _span_id.set(None)


def set_request_context(context: dict[str, Any]) -> None:
    """Set request context for logging.
    
    Args:
        context: Dictionary of context values
    """
    _request_context.set(context)


def update_request_context(**kwargs: Any) -> None:
    """Update request context with new values.
    
    Args:
        **kwargs: Context values to add
    """
    current = _request_context.get()
    current.update(kwargs)
    _request_context.set(current)


def clear_request_context() -> None:
    """Clear the request context."""
    _request_context.set({})


@contextmanager
def correlation_id_scope(correlation_id: str) -> Generator[None, None, None]:
    """Context manager for correlation ID scope.
    
    Args:
        correlation_id: Correlation ID to set
        
    Yields:
        None
        
    Example:
        >>> with correlation_id_scope("abc-123"):
        ...     logger.info("processing")
    """
    token = _correlation_id.set(correlation_id)
    try:
        yield
    finally:
        _correlation_id.reset(token)


@contextmanager
def request_context_scope(**context: Any) -> Generator[None, None, None]:
    """Context manager for request context scope.
    
    Args:
        **context: Context values
        
    Yields:
        None
        
    Example:
        >>> with request_context_scope(user_id="123", request_path="/api/jobs"):
        ...     logger.info("handling_request")
    """
    token = _request_context.set(context)
    try:
        yield
    finally:
        _request_context.reset(token)


class LogContext:
    """Context manager for combined logging context.
    
    This provides a convenient way to set correlation ID, trace context,
    and request context in one operation.
    
    Example:
        >>> with LogContext(
        ...     correlation_id="abc-123",
        ...     trace_id="def-456",
        ...     user_id="user-789",
        ... ):
        ...     logger.info("processing_request")
    """

    def __init__(
        self,
        correlation_id: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
        **context: Any,
    ):
        """Initialize log context.
        
        Args:
            correlation_id: Correlation ID
            trace_id: Trace ID
            span_id: Span ID
            **context: Additional context values
        """
        self.correlation_id = correlation_id
        self.trace_id = trace_id
        self.span_id = span_id
        self.context = context
        self.tokens = []

    def __enter__(self) -> "LogContext":
        """Enter the context."""
        if self.correlation_id:
            self.tokens.append(("corr", _correlation_id.set(self.correlation_id)))
        if self.trace_id:
            self.tokens.append(("trace", _trace_id.set(self.trace_id)))
        if self.span_id:
            self.tokens.append(("span", _span_id.set(self.span_id)))
        if self.context:
            self.tokens.append(("ctx", _request_context.set(self.context)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context."""
        for name, token in reversed(self.tokens):
            if name == "corr":
                _correlation_id.reset(token)
            elif name == "trace":
                _trace_id.reset(token)
            elif name == "span":
                _span_id.reset(token)
            elif name == "ctx":
                _request_context.reset(token)
