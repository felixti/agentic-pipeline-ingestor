"""Unit tests for structured logging configuration."""

import logging
import sys
from unittest.mock import MagicMock, patch

import pytest
import structlog

from src.observability.logging import (
    LogContext,
    StructuredLogger,
    clear_request_context,
    clear_trace_context,
    correlation_id_scope,
    get_correlation_id,
    get_logger,
    request_context_scope,
    set_correlation_id,
    set_request_context,
    set_trace_context,
    setup_logging,
    update_request_context,
)


@pytest.mark.unit
class TestStructuredLogger:
    """Tests for StructuredLogger class."""

    def test_init(self):
        """Test StructuredLogger initialization."""
        logger = StructuredLogger()
        assert logger._configured is False

    @patch("src.observability.logging.logging.basicConfig")
    @patch("src.observability.logging.structlog.configure")
    def test_setup_logging_json_format(self, mock_configure, mock_basic_config):
        """Test setup with JSON format."""
        logger = StructuredLogger()
        logger.setup_logging(json_format=True, log_level="INFO")

        assert logger._configured is True
        mock_basic_config.assert_called_once()
        mock_configure.assert_called_once()

        # Verify JSON renderer is used (check by class name since it's a bound method)
        call_args = mock_configure.call_args[1]
        processors = call_args["processors"]
        # Check for JSONRenderer by module path since we need to handle both class and instance
        has_json_renderer = any(
            "JSONRenderer" in str(type(p)) or str(p) == "<class 'structlog.processors.JSONRenderer'>"
            for p in processors
        )
        assert has_json_renderer, f"Expected JSONRenderer in processors, got: {processors}"

    @patch("src.observability.logging.logging.basicConfig")
    @patch("src.observability.logging.structlog.configure")
    def test_setup_logging_console_format(self, mock_configure, mock_basic_config):
        """Test setup with console format."""
        logger = StructuredLogger()
        logger.setup_logging(json_format=False, log_level="DEBUG")

        # Verify ConsoleRenderer is used
        call_args = mock_configure.call_args[1]
        processors = call_args["processors"]
        # Check for ConsoleRenderer by module path
        has_console_renderer = any(
            "ConsoleRenderer" in str(type(p)) or str(p) == "<class 'structlog.dev.ConsoleRenderer'>"
            for p in processors
        )
        assert has_console_renderer, f"Expected ConsoleRenderer in processors, got: {processors}"

    @patch("src.observability.logging.logging.basicConfig")
    @patch("src.observability.logging.structlog.configure")
    def test_setup_logging_with_log_file(self, mock_configure, mock_basic_config):
        """Test setup with log file."""
        logger = StructuredLogger()

        with patch("src.observability.logging.logging.FileHandler") as mock_file_handler:
            mock_handler = MagicMock()
            mock_file_handler.return_value = mock_handler

            logger.setup_logging(json_format=True, log_level="INFO", log_file="/var/log/app.log")

            # The basicConfig should handle file logging
            mock_basic_config.assert_called_once()

    @patch("src.observability.logging.logging.basicConfig")
    @patch("src.observability.logging.structlog.configure")
    def test_setup_logging_with_extra_processors(self, mock_configure, mock_basic_config):
        """Test setup with extra processors."""
        logger = StructuredLogger()
        extra_processor = MagicMock()

        logger.setup_logging(
            json_format=True,
            log_level="INFO",
            extra_processors=[extra_processor]
        )

        call_args = mock_configure.call_args[1]
        processors = call_args["processors"]
        assert extra_processor in processors

    @patch("src.observability.logging.logging.basicConfig")
    @patch("src.observability.logging.structlog.configure")
    def test_setup_logging_already_configured(self, mock_configure, mock_basic_config):
        """Test setup when already configured."""
        logger = StructuredLogger()
        logger._configured = True

        logger.setup_logging(json_format=True)

        mock_configure.assert_not_called()

    @patch("src.observability.logging.logging.basicConfig")
    @patch("src.observability.logging.structlog.configure")
    def test_setup_logging_without_trace_context(self, mock_configure, mock_basic_config):
        """Test setup without trace context."""
        logger = StructuredLogger()
        logger.setup_logging(json_format=True, include_trace_context=False)

        call_args = mock_configure.call_args[1]
        processors = call_args["processors"]
        # Should not have the _add_trace_context processor
        assert logger._add_trace_context not in processors

    def test_add_trace_context_with_valid_span(self):
        """Test adding trace context with valid span."""
        logger = StructuredLogger()

        mock_span = MagicMock()
        mock_span_context = MagicMock()
        mock_span_context.is_valid = True
        mock_span_context.trace_id = 12345
        mock_span_context.span_id = 67890
        mock_span_context.trace_flags = 1
        mock_span.get_span_context.return_value = mock_span_context

        with patch("opentelemetry.trace.get_current_span", return_value=mock_span):
            event_dict = {}
            result = logger._add_trace_context(None, "info", event_dict)

        assert result["trace_id"] == "00000000000000000000000000003039"
        assert result["span_id"] == "0000000000010932"
        assert result["trace_flags"] == "1"

    def test_add_trace_context_with_invalid_span(self):
        """Test adding trace context with invalid span."""
        logger = StructuredLogger()

        mock_span = MagicMock()
        mock_span_context = MagicMock()
        mock_span_context.is_valid = False
        mock_span.get_span_context.return_value = mock_span_context

        with patch("opentelemetry.trace.get_current_span", return_value=mock_span):
            event_dict = {}
            result = logger._add_trace_context(None, "info", event_dict)

        # Should not add trace info for invalid span
        assert "trace_id" not in result

    def test_add_trace_context_with_fallback_vars(self):
        """Test adding trace context from context vars as fallback."""
        logger = StructuredLogger()

        # Set context vars
        from src.observability.logging import _span_id, _trace_id
        _trace_id.set("fallback-trace-id")
        _span_id.set("fallback-span-id")

        try:
            with patch("opentelemetry.trace.get_current_span", return_value=None):
                event_dict = {}
                result = logger._add_trace_context(None, "info", event_dict)

            assert result["trace_id"] == "fallback-trace-id"
            assert result["span_id"] == "fallback-span-id"
        finally:
            _trace_id.set(None)
            _span_id.set(None)

    def test_add_trace_context_import_error(self):
        """Test adding trace context when opentelemetry not available."""
        logger = StructuredLogger()

        # Simulate ImportError by temporarily removing trace module
        import src.observability.logging as logging_module
        original_trace = getattr(logging_module, "trace", None)

        try:
            if original_trace:
                delattr(logging_module, "trace")
            event_dict = {}
            result = logger._add_trace_context(None, "info", event_dict)

            # Should not raise, just return event_dict without trace info
            assert "trace_id" not in result or result.get("trace_id") is None
        finally:
            if original_trace:
                logging_module.trace = original_trace

    def test_add_correlation_id(self):
        """Test adding correlation ID."""
        logger = StructuredLogger()

        # Set correlation ID
        from src.observability.logging import _correlation_id
        _correlation_id.set("corr-123")

        try:
            event_dict = {}
            result = logger._add_correlation_id(None, "info", event_dict)

            assert result["correlation_id"] == "corr-123"
        finally:
            _correlation_id.set(None)

    def test_add_correlation_id_none(self):
        """Test adding correlation ID when none set."""
        logger = StructuredLogger()

        from src.observability.logging import _correlation_id
        _correlation_id.set(None)

        event_dict = {}
        result = logger._add_correlation_id(None, "info", event_dict)

        assert "correlation_id" not in result

    def test_add_request_context(self):
        """Test adding request context."""
        logger = StructuredLogger()

        from src.observability.logging import _request_context
        _request_context.set({"user_id": "123", "request_path": "/api/jobs"})

        try:
            event_dict = {}
            result = logger._add_request_context(None, "info", event_dict)

            assert result["user_id"] == "123"
            assert result["request_path"] == "/api/jobs"
        finally:
            _request_context.set({})

    def test_add_request_context_empty(self):
        """Test adding request context when empty."""
        logger = StructuredLogger()

        from src.observability.logging import _request_context
        _request_context.set({})

        event_dict = {"existing": "value"}
        result = logger._add_request_context(None, "info", event_dict)

        assert result == {"existing": "value"}

    @patch("src.observability.logging.structlog.get_logger")
    def test_get_logger(self, mock_get_logger):
        """Test getting logger instance."""
        logger = StructuredLogger()
        logger._configured = True

        mock_structlog_logger = MagicMock()
        mock_get_logger.return_value = mock_structlog_logger

        result = logger.get_logger("test_module")

        mock_get_logger.assert_called_once_with("test_module")
        assert result == mock_structlog_logger

    @patch("src.observability.logging.logging.basicConfig")
    @patch("src.observability.logging.structlog.configure")
    def test_get_logger_auto_configure(self, mock_configure, mock_basic_config):
        """Test getting logger auto-configures if not configured."""
        logger = StructuredLogger()
        assert logger._configured is False

        with patch.object(logger, "setup_logging") as mock_setup:
            logger.get_logger("test_module")
            mock_setup.assert_called_once()


@pytest.mark.unit
class TestGetLogger:
    """Tests for get_logger function."""

    @patch("src.observability.logging._structured_logger", None)
    @patch("src.observability.logging.StructuredLogger")
    def test_get_logger_creates_new_instance(self, mock_structured_logger_class):
        """Test that get_logger creates new instance if needed."""
        mock_logger_instance = MagicMock()
        mock_structlog_logger = MagicMock()
        mock_logger_instance.get_logger.return_value = mock_structlog_logger
        mock_structured_logger_class.return_value = mock_logger_instance

        result = get_logger("test_module")

        mock_structured_logger_class.assert_called_once()
        mock_logger_instance.get_logger.assert_called_once_with("test_module")
        assert result == mock_structlog_logger


@pytest.mark.unit
class TestSetupLogging:
    """Tests for setup_logging function."""

    @patch("src.observability.logging.StructuredLogger")
    def test_setup_logging_returns_instance(self, mock_structured_logger_class):
        """Test that setup_logging returns StructuredLogger."""
        mock_logger_instance = MagicMock()
        mock_structured_logger_class.return_value = mock_logger_instance

        result = setup_logging(json_format=True, log_level="DEBUG")

        assert result == mock_logger_instance
        mock_logger_instance.setup_logging.assert_called_once_with(
            json_format=True,
            log_level="DEBUG",
        )


@pytest.mark.unit
class TestCorrelationIdFunctions:
    """Tests for correlation ID context functions."""

    def test_set_and_get_correlation_id(self):
        """Test setting and getting correlation ID."""
        set_correlation_id("test-correlation-id")
        assert get_correlation_id() == "test-correlation-id"

    def test_get_correlation_id_none(self):
        """Test getting correlation ID when not set."""
        from src.observability.logging import _correlation_id
        _correlation_id.set(None)
        assert get_correlation_id() is None


@pytest.mark.unit
class TestTraceContextFunctions:
    """Tests for trace context functions."""

    def test_set_trace_context(self):
        """Test setting trace context."""
        from src.observability.logging import _span_id, _trace_id

        set_trace_context("trace-123", "span-456")

        assert _trace_id.get() == "trace-123"
        assert _span_id.get() == "span-456"

    def test_clear_trace_context(self):
        """Test clearing trace context."""
        from src.observability.logging import _span_id, _trace_id

        set_trace_context("trace-123", "span-456")
        clear_trace_context()

        assert _trace_id.get() is None
        assert _span_id.get() is None


@pytest.mark.unit
class TestRequestContextFunctions:
    """Tests for request context functions."""

    def test_set_request_context(self):
        """Test setting request context."""
        from src.observability.logging import _request_context

        context = {"user_id": "123", "tenant": "abc"}
        set_request_context(context)

        assert _request_context.get() == context

    def test_update_request_context(self):
        """Test updating request context."""
        from src.observability.logging import _request_context

        set_request_context({"user_id": "123"})
        update_request_context(tenant="abc", ip="192.168.1.1")

        result = _request_context.get()
        assert result["user_id"] == "123"
        assert result["tenant"] == "abc"
        assert result["ip"] == "192.168.1.1"

    def test_clear_request_context(self):
        """Test clearing request context."""
        from src.observability.logging import _request_context

        set_request_context({"user_id": "123"})
        clear_request_context()

        assert _request_context.get() == {}


@pytest.mark.unit
class TestCorrelationIdScope:
    """Tests for correlation_id_scope context manager."""

    def test_correlation_id_scope_sets_id(self):
        """Test that correlation_id_scope sets the ID."""
        from src.observability.logging import _correlation_id
        _correlation_id.set(None)

        with correlation_id_scope("scoped-corr-id"):
            assert get_correlation_id() == "scoped-corr-id"

        # After exiting, should be None again
        assert get_correlation_id() is None

    def test_correlation_id_scope_restores_previous(self):
        """Test that correlation_id_scope restores previous ID."""
        set_correlation_id("original-id")

        with correlation_id_scope("new-id"):
            assert get_correlation_id() == "new-id"

        # Should restore to original
        assert get_correlation_id() == "original-id"


@pytest.mark.unit
class TestRequestContextScope:
    """Tests for request_context_scope context manager."""

    def test_request_context_scope_sets_context(self):
        """Test that request_context_scope sets context."""
        from src.observability.logging import _request_context
        _request_context.set({})

        with request_context_scope(user_id="123", action="create"):
            ctx = _request_context.get()
            assert ctx["user_id"] == "123"
            assert ctx["action"] == "create"

        # After exiting, should be empty
        assert _request_context.get() == {}

    def test_request_context_scope_restores_previous(self):
        """Test that request_context_scope restores previous context."""
        from src.observability.logging import _request_context
        _request_context.set({"original": "data"})

        with request_context_scope(new_data="test"):
            ctx = _request_context.get()
            assert ctx == {"new_data": "test"}

        # Should restore to original
        assert _request_context.get() == {"original": "data"}


@pytest.mark.unit
class TestLogContext:
    """Tests for LogContext class."""

    def test_init(self):
        """Test LogContext initialization."""
        ctx = LogContext(
            correlation_id="corr-123",
            trace_id="trace-456",
            span_id="span-789",
            user_id="user-abc",
        )

        assert ctx.correlation_id == "corr-123"
        assert ctx.trace_id == "trace-456"
        assert ctx.span_id == "span-789"
        assert ctx.context == {"user_id": "user-abc"}
        assert ctx.tokens == []

    def test_enter_exit_correlation_id(self):
        """Test entering and exiting context with correlation ID."""
        from src.observability.logging import _correlation_id
        _correlation_id.set(None)

        ctx = LogContext(correlation_id="corr-123")

        with ctx:
            assert get_correlation_id() == "corr-123"

        assert get_correlation_id() is None

    def test_enter_exit_trace_context(self):
        """Test entering and exiting context with trace context."""
        from src.observability.logging import _span_id, _trace_id

        ctx = LogContext(trace_id="trace-123", span_id="span-456")

        with ctx:
            assert _trace_id.get() == "trace-123"
            assert _span_id.get() == "span-456"

        assert _trace_id.get() is None
        assert _span_id.get() is None

    def test_enter_exit_request_context(self):
        """Test entering and exiting context with request context."""
        from src.observability.logging import _request_context
        _request_context.set({})

        ctx = LogContext(user_id="123", action="test")

        with ctx:
            result_ctx = _request_context.get()
            assert result_ctx["user_id"] == "123"
            assert result_ctx["action"] == "test"

        assert _request_context.get() == {}

    def test_enter_exit_combined(self):
        """Test entering and exiting context with all values."""
        from src.observability.logging import _correlation_id, _request_context, _span_id, _trace_id

        _correlation_id.set(None)
        _trace_id.set(None)
        _span_id.set(None)
        _request_context.set({})

        ctx = LogContext(
            correlation_id="corr-123",
            trace_id="trace-456",
            span_id="span-789",
            user_id="user-abc",
            request_path="/api/test",
        )

        with ctx:
            assert get_correlation_id() == "corr-123"
            assert _trace_id.get() == "trace-456"
            assert _span_id.get() == "span-789"
            assert _request_context.get()["user_id"] == "user-abc"
            assert _request_context.get()["request_path"] == "/api/test"

        # All should be reset
        assert get_correlation_id() is None
        assert _trace_id.get() is None
        assert _span_id.get() is None
        assert _request_context.get() == {}

    def test_enter_no_values(self):
        """Test entering context with no values set."""
        ctx = LogContext()

        result = ctx.__enter__()
        assert result == ctx

        # Exit should work without errors
        ctx.__exit__(None, None, None)
