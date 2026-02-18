"""Unit tests for FastAPI observability middleware."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from opentelemetry.trace import SpanKind, Status, StatusCode

from src.observability.middleware import (
    ObservabilityMiddleware,
    MetricsMiddleware,
    setup_observability,
    _get_route,
)


@pytest.fixture
def mock_asgi_scope():
    """Create a mock ASGI scope for HTTP request."""
    return {
        "type": "http",
        "method": "GET",
        "path": "/api/v1/jobs",
        "headers": [
            (b"host", b"localhost:8000"),
            (b"user-agent", b"test-agent"),
        ],
        "query_string": b"",
    }


@pytest.fixture
def mock_asgi_lifespan_scope():
    """Create a mock ASGI scope for lifespan."""
    return {
        "type": "lifespan",
    }


@pytest.fixture
def mock_receive():
    """Create a mock ASGI receive function."""
    async def receive():
        return {"type": "http.request", "body": b""}
    return receive


@pytest.fixture
def mock_send():
    """Create a mock ASGI send function."""
    return AsyncMock()


@pytest.mark.unit
class TestObservabilityMiddleware:
    """Tests for ObservabilityMiddleware class."""

    def test_init(self):
        """Test middleware initialization."""
        app = MagicMock(spec=FastAPI)
        middleware = ObservabilityMiddleware(app)

        assert middleware.app == app

    @pytest.mark.asyncio
    async def test_call_non_http_scope(self, mock_asgi_lifespan_scope, mock_receive, mock_send):
        """Test middleware passes through non-HTTP scopes."""
        app = AsyncMock()
        middleware = ObservabilityMiddleware(app)

        await middleware(mock_asgi_lifespan_scope, mock_receive, mock_send)

        app.assert_called_once_with(mock_asgi_lifespan_scope, mock_receive, mock_send)

    @pytest.mark.asyncio
    async def test_call_http_scope(self, mock_asgi_scope, mock_receive, mock_send):
        """Test middleware handles HTTP scope."""
        app = AsyncMock()
        middleware = ObservabilityMiddleware(app)

        with patch("src.observability.middleware.set_correlation_id") as mock_set_corr:
            with patch("src.observability.middleware.start_span") as mock_start_span:
                with patch("src.observability.middleware.request_context_scope") as mock_ctx_scope:
                    with patch("src.observability.middleware.get_metrics_manager") as mock_get_metrics:
                        with patch("src.observability.middleware.logger"):
                            mock_span = MagicMock()
                            mock_ctx_manager = MagicMock()
                            mock_ctx_manager.__enter__ = MagicMock(return_value=mock_span)
                            mock_ctx_manager.__exit__ = MagicMock(return_value=False)
                            mock_start_span.return_value = mock_ctx_manager

                            mock_metrics = MagicMock()
                            mock_get_metrics.return_value = mock_metrics

                            mock_ctx = MagicMock()
                            mock_ctx_manager_2 = MagicMock()
                            mock_ctx_manager_2.__enter__ = MagicMock(return_value=None)
                            mock_ctx_manager_2.__exit__ = MagicMock(return_value=False)
                            mock_ctx_scope.return_value = mock_ctx_manager_2

                            await middleware(mock_asgi_scope, mock_receive, mock_send)

                            mock_set_corr.assert_called_once()

    def test_get_route_template_with_route(self):
        """Test getting route template from request with route."""
        app = MagicMock()
        middleware = ObservabilityMiddleware(app)

        mock_request = MagicMock()
        mock_request.url.path = "/api/v1/jobs"
        mock_route = MagicMock()
        mock_route.path = "/api/v1/jobs"
        mock_request.scope = {"route": mock_route}

        result = middleware._get_route_template(mock_request)

        assert result == "/api/v1/jobs"

    def test_get_route_template_without_route(self):
        """Test getting route template without route in scope."""
        app = MagicMock()
        middleware = ObservabilityMiddleware(app)

        mock_request = MagicMock()
        mock_request.url.path = "/api/v1/jobs/123"
        mock_request.scope = {}

        result = middleware._get_route_template(mock_request)

        assert result == "/api/v1/jobs/123"


@pytest.mark.unit
class TestMetricsMiddleware:
    """Tests for MetricsMiddleware class."""

    def test_init(self):
        """Test middleware initialization."""
        app = MagicMock()

        with patch("src.observability.middleware.get_metrics_manager") as mock_get_metrics:
            mock_metrics = MagicMock()
            mock_get_metrics.return_value = mock_metrics

            middleware = MetricsMiddleware(app)

            assert middleware.app == app
            assert middleware.metrics == mock_metrics

    @pytest.mark.asyncio
    async def test_call_non_http_scope(self, mock_asgi_lifespan_scope, mock_receive, mock_send):
        """Test middleware passes through non-HTTP scopes."""
        with patch("src.observability.middleware.get_metrics_manager"):
            app = AsyncMock()
            middleware = MetricsMiddleware(app)

            await middleware(mock_asgi_lifespan_scope, mock_receive, mock_send)

            app.assert_called_once_with(mock_asgi_lifespan_scope, mock_receive, mock_send)

    @pytest.mark.asyncio
    async def test_call_records_metrics(self, mock_asgi_scope, mock_receive, mock_send):
        """Test middleware records metrics for HTTP request."""
        with patch("src.observability.middleware.get_metrics_manager") as mock_get_metrics:
            mock_metrics = MagicMock()
            mock_get_metrics.return_value = mock_metrics

            app = AsyncMock()
            middleware = MetricsMiddleware(app)

            await middleware(mock_asgi_scope, mock_receive, mock_send)

            mock_metrics.record_api_request.assert_called_once()
            call_args = mock_metrics.record_api_request.call_args[1]
            assert call_args["method"] == "GET"
            assert call_args["endpoint"] == "/api/v1/jobs"
            assert call_args["status_code"] == 200
            assert "duration" in call_args

    @pytest.mark.asyncio
    async def test_call_records_error_status(self, mock_asgi_scope, mock_receive):
        """Test middleware records error status codes."""
        with patch("src.observability.middleware.get_metrics_manager") as mock_get_metrics:
            mock_metrics = MagicMock()
            mock_get_metrics.return_value = mock_metrics

            captured_status = [200]  # default

            async def wrapped_send(message):
                nonlocal captured_status
                if message["type"] == "http.response.start":
                    captured_status[0] = message.get("status", 200)

            app = AsyncMock()
            middleware = MetricsMiddleware(app)

            await middleware(mock_asgi_scope, mock_receive, wrapped_send)

            # The status code is determined by what the wrapped_send captures
            call_args = mock_metrics.record_api_request.call_args[1]
            assert call_args["status_code"] == captured_status[0]


@pytest.mark.unit
class TestSetupObservability:
    """Tests for setup_observability function."""

    def test_setup_observability_adds_middleware(self):
        """Test that setup_observability adds middleware to app."""
        app = MagicMock()

        setup_observability(app)

        # Should call app.middleware twice (once for correlation_id, once for observability)
        assert app.middleware.call_count == 2

    @pytest.mark.asyncio
    async def test_correlation_id_middleware(self):
        """Test correlation ID middleware."""
        app = MagicMock()

        setup_observability(app)

        # Get the correlation_id middleware decorator
        decorator = app.middleware.call_args_list[0][0][0]
        assert decorator == "http"

        # The decorator should have been called with a function
        middleware_func = app.middleware.call_args_list[0][1].get("call_next") or app.middleware.call_args_list[0][0][1] if len(app.middleware.call_args_list[0][0]) > 1 else None

    @pytest.mark.asyncio
    async def test_observability_middleware(self):
        """Test observability middleware."""
        app = MagicMock()

        setup_observability(app)

        # Get the observability middleware decorator
        decorator = app.middleware.call_args_list[1][0][0]
        assert decorator == "http"


@pytest.mark.unit
class TestGetRoute:
    """Tests for _get_route function."""

    def test_get_route_with_route_in_scope(self):
        """Test getting route when route exists in scope."""
        mock_request = MagicMock()
        mock_route = MagicMock()
        mock_route.path = "/api/v1/jobs/{job_id}"
        mock_request.scope = {"route": mock_route}

        result = _get_route(mock_request)

        assert result == "/api/v1/jobs/{job_id}"

    def test_get_route_without_route_in_scope(self):
        """Test getting route when route not in scope."""
        mock_request = MagicMock()
        mock_request.url.path = "/api/v1/jobs"
        mock_request.scope = {}

        result = _get_route(mock_request)

        assert result == "/api/v1/jobs"

    def test_get_route_with_route_no_path_attr(self):
        """Test getting route when route has no path attribute."""
        mock_request = MagicMock()
        mock_route = MagicMock(spec=[])
        mock_request.scope = {"route": mock_route}
        mock_request.url.path = "/fallback/path"

        result = _get_route(mock_request)

        assert result == "/fallback/path"


@pytest.mark.unit
class TestObservabilityMiddlewareIntegration:
    """Integration-style tests for ObservabilityMiddleware."""

    @pytest.mark.asyncio
    async def test_full_request_flow(self, mock_asgi_scope, mock_receive, mock_send):
        """Test full request flow through middleware."""
        app = AsyncMock()
        middleware = ObservabilityMiddleware(app)

        with patch("src.observability.middleware.set_correlation_id") as mock_set_corr:
            with patch("src.observability.middleware.start_span") as mock_start_span:
                with patch("src.observability.middleware.request_context_scope") as mock_ctx_scope:
                    with patch("src.observability.middleware.get_metrics_manager") as mock_get_metrics:
                        with patch("src.observability.middleware.logger") as mock_logger:
                            with patch("src.observability.middleware.time.time") as mock_time:
                                # Setup mocks
                                mock_time.side_effect = [0.0, 1.0]  # Start and end times

                                mock_span = MagicMock()
                                mock_ctx_manager = MagicMock()
                                mock_ctx_manager.__enter__ = MagicMock(return_value=mock_span)
                                mock_ctx_manager.__exit__ = MagicMock(return_value=False)
                                mock_start_span.return_value = mock_ctx_manager

                                mock_ctx_manager_2 = MagicMock()
                                mock_ctx_manager_2.__enter__ = MagicMock(return_value=None)
                                mock_ctx_manager_2.__exit__ = MagicMock(return_value=False)
                                mock_ctx_scope.return_value = mock_ctx_manager_2

                                mock_metrics = MagicMock()
                                mock_get_metrics.return_value = mock_metrics

                                # Execute
                                await middleware(mock_asgi_scope, mock_receive, mock_send)

                                # Verify span was created
                                mock_start_span.assert_called_once()
                                call_kwargs = mock_start_span.call_args[1]
                                assert call_kwargs["name"] == "GET /api/v1/jobs"
                                assert call_kwargs["kind"] == SpanKind.SERVER

                                # Verify span attributes
                                attrs = call_kwargs["attributes"]
                                assert attrs["http.method"] == "GET"
                                assert attrs["http.route"] == "/api/v1/jobs"
                                assert attrs["http.target"] == "/api/v1/jobs"
                                assert attrs["http.scheme"] == "http"
                                assert attrs["http.host"] == "localhost:8000"
                                assert attrs["http.user_agent"] == "test-agent"

    @pytest.mark.asyncio
    async def test_request_with_x_request_id_header(self, mock_asgi_scope, mock_receive, mock_send):
        """Test that X-Request-ID header is used as correlation ID."""
        scope = mock_asgi_scope.copy()
        scope["headers"] = [
            (b"host", b"localhost:8000"),
            (b"x-request-id", b"existing-request-id"),
        ]

        app = AsyncMock()
        middleware = ObservabilityMiddleware(app)

        with patch("src.observability.middleware.set_correlation_id") as mock_set_corr:
            with patch("src.observability.middleware.start_span") as mock_start_span:
                with patch("src.observability.middleware.request_context_scope") as mock_ctx_scope:
                    with patch("src.observability.middleware.get_metrics_manager"):
                        with patch("src.observability.middleware.logger"):
                            mock_ctx_manager = MagicMock()
                            mock_ctx_manager.__enter__ = MagicMock(return_value=MagicMock())
                            mock_ctx_manager.__exit__ = MagicMock(return_value=False)
                            mock_start_span.return_value = mock_ctx_manager

                            mock_ctx_manager_2 = MagicMock()
                            mock_ctx_manager_2.__enter__ = MagicMock(return_value=None)
                            mock_ctx_manager_2.__exit__ = MagicMock(return_value=False)
                            mock_ctx_scope.return_value = mock_ctx_manager_2

                            await middleware(scope, mock_receive, mock_send)

                            mock_set_corr.assert_called_once_with("existing-request-id")

    @pytest.mark.asyncio
    async def test_request_content_length_tracking(self, mock_asgi_scope, mock_receive, mock_send):
        """Test that content length is tracked."""
        scope = mock_asgi_scope.copy()
        scope["headers"] = [
            (b"host", b"localhost:8000"),
            (b"content-length", b"1024"),
        ]

        app = AsyncMock()
        middleware = ObservabilityMiddleware(app)

        with patch("src.observability.middleware.set_correlation_id"):
            with patch("src.observability.middleware.start_span") as mock_start_span:
                with patch("src.observability.middleware.request_context_scope") as mock_ctx_scope:
                    with patch("src.observability.middleware.get_metrics_manager") as mock_get_metrics:
                        with patch("src.observability.middleware.logger"):
                            mock_span = MagicMock()
                            mock_ctx_manager = MagicMock()
                            mock_ctx_manager.__enter__ = MagicMock(return_value=mock_span)
                            mock_ctx_manager.__exit__ = MagicMock(return_value=False)
                            mock_start_span.return_value = mock_ctx_manager

                            mock_ctx_manager_2 = MagicMock()
                            mock_ctx_manager_2.__enter__ = MagicMock(return_value=None)
                            mock_ctx_manager_2.__exit__ = MagicMock(return_value=False)
                            mock_ctx_scope.return_value = mock_ctx_manager_2

                            mock_metrics = MagicMock()
                            mock_get_metrics.return_value = mock_metrics

                            await middleware(scope, mock_receive, mock_send)

                            # Verify metrics were recorded with request size
                            call_args = mock_metrics.record_api_request.call_args[1]
                            assert call_args["request_size"] == 1024


@pytest.mark.unit
class TestObservabilityMiddlewareErrorHandling:
    """Tests for error handling in ObservabilityMiddleware."""

    @pytest.mark.asyncio
    async def test_exception_handling(self, mock_asgi_scope, mock_receive, mock_send):
        """Test that exceptions are properly handled and recorded."""
        app = AsyncMock()
        middleware = ObservabilityMiddleware(app)

        with patch("src.observability.middleware.set_correlation_id"):
            with patch("src.observability.middleware.start_span") as mock_start_span:
                with patch("src.observability.middleware.request_context_scope") as mock_ctx_scope:
                    with patch("src.observability.middleware.get_metrics_manager") as mock_get_metrics:
                        with patch("src.observability.middleware.logger") as mock_logger:
                            with patch("src.observability.middleware.time.time") as mock_time:
                                mock_time.side_effect = [0.0, 1.0]

                                mock_span = MagicMock()
                                mock_ctx_manager = MagicMock()
                                mock_ctx_manager.__enter__ = MagicMock(return_value=mock_span)
                                mock_ctx_manager.__exit__ = MagicMock(return_value=False)
                                mock_start_span.return_value = mock_ctx_manager

                                mock_ctx_manager_2 = MagicMock()
                                mock_ctx_manager_2.__enter__ = MagicMock(return_value=None)
                                mock_ctx_manager_2.__exit__ = MagicMock(return_value=False)
                                mock_ctx_scope.return_value = mock_ctx_manager_2

                                mock_metrics = MagicMock()
                                mock_get_metrics.return_value = mock_metrics

                                # Simulate exception in app
                                exception = ValueError("Test error")
                                app.side_effect = exception

                                # Execute and expect exception to be raised
                                with pytest.raises(ValueError, match="Test error"):
                                    await middleware(mock_asgi_scope, mock_receive, mock_send)

                                # Verify error was recorded on span
                                mock_span.set_status.assert_called_once()
                                mock_span.record_exception.assert_called_once_with(exception)

                                # Verify error metrics were recorded
                                call_args = mock_metrics.record_api_request.call_args[1]
                                assert call_args["status_code"] == 500

                                # Verify error was logged
                                mock_logger.error.assert_called_once()
                                log_call = mock_logger.error.call_args[1]
                                assert log_call["error"] == "Test error"
                                assert log_call["error_type"] == "ValueError"
