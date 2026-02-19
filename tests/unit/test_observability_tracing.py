"""Unit tests for OpenTelemetry tracing configuration."""

import os
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import SpanKind

from src.observability.tracing import (
    TelemetryManager,
    get_telemetry_manager,
    get_tracer,
    setup_tracing_from_settings,
    start_pipeline_stage_span,
    start_span,
)


@pytest.mark.unit
class TestGetTracer:
    """Tests for get_tracer function."""

    def test_get_tracer_default_name(self):
        """Test getting tracer with default name."""
        with patch("src.observability.tracing.trace.get_tracer") as mock_get_tracer:
            mock_tracer = MagicMock()
            mock_get_tracer.return_value = mock_tracer

            tracer = get_tracer()

            mock_get_tracer.assert_called_once_with("pipeline-ingestor")
            assert tracer == mock_tracer

    def test_get_tracer_custom_name(self):
        """Test getting tracer with custom name."""
        with patch("src.observability.tracing.trace.get_tracer") as mock_get_tracer:
            mock_tracer = MagicMock()
            mock_get_tracer.return_value = mock_tracer

            tracer = get_tracer("custom-module")

            mock_get_tracer.assert_called_once_with("custom-module")
            assert tracer == mock_tracer


@pytest.mark.unit
class TestTelemetryManager:
    """Tests for TelemetryManager class."""

    def test_init(self):
        """Test TelemetryManager initialization."""
        manager = TelemetryManager()

        assert manager._provider is None
        assert manager._initialized is False
        assert manager._exporters == []

    def test_is_initialized_property(self):
        """Test is_initialized property."""
        manager = TelemetryManager()
        assert manager.is_initialized is False

        manager._initialized = True
        assert manager.is_initialized is True

    def test_active_exporters_property(self):
        """Test active_exporters property returns copy."""
        manager = TelemetryManager()
        manager._exporters = ["jaeger", "console"]

        exporters = manager.active_exporters
        assert exporters == ["jaeger", "console"]

        # Verify it's a copy
        exporters.append("otlp")
        assert manager._exporters == ["jaeger", "console"]

    @patch("src.observability.tracing.trace.set_tracer_provider")
    @patch("src.observability.tracing.Resource")
    @patch("src.observability.tracing.TracerProvider")
    def test_setup_tracing_basic(self, mock_provider_class, mock_resource_class, mock_set_tracer):
        """Test basic tracing setup."""
        mock_resource = MagicMock()
        mock_resource_class.create.return_value = mock_resource

        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        manager = TelemetryManager()
        result = manager.setup_tracing(
            service_name="test-service",
            service_version="1.2.3",
            environment="testing",
        )

        assert manager._initialized is True
        assert result == mock_provider
        mock_set_tracer.assert_called_once_with(mock_provider)

    def test_setup_tracing_already_initialized(self):
        """Test setup_tracing raises error when already initialized."""
        manager = TelemetryManager()
        manager._initialized = True

        with pytest.raises(RuntimeError, match="Tracing is already initialized"):
            manager.setup_tracing(service_name="test-service")

    @patch("src.observability.tracing.trace.set_tracer_provider")
    @patch("src.observability.tracing.Resource.create")
    @patch("src.observability.tracing.TracerProvider")
    def test_setup_tracing_with_attributes(self, mock_provider_class, mock_resource_create, mock_set_tracer):
        """Test tracing setup with custom attributes."""
        mock_resource = MagicMock()
        mock_resource_create.return_value = mock_resource

        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        with patch.dict(os.environ, {"HOSTNAME": "test-host"}):
            manager = TelemetryManager()
            manager.setup_tracing(
                service_name="test-service",
                service_version="1.0.0",
                environment="production",
                attributes={"custom.attr": "value"},
            )

        # Check that Resource.create was called with expected attributes
        call_args = mock_resource_create.call_args
        # Resource.create is called with positional args (resource_attrs,)
        resource_attrs = call_args[0][0] if call_args[0] else call_args[1]
        assert resource_attrs["custom.attr"] == "value"
        assert resource_attrs["service.namespace"] == "agentic-pipeline"
        assert resource_attrs["service.instance.id"] == "test-host"

    @patch("src.observability.tracing.trace.set_tracer_provider")
    @patch("src.observability.tracing.Resource")
    @patch("src.observability.tracing.TracerProvider")
    @patch("src.observability.tracing.BatchSpanProcessor")
    @patch("src.observability.tracing.ConsoleSpanExporter")
    def test_setup_tracing_console_exporter(
        self,
        mock_console_exporter_class,
        mock_processor_class,
        mock_provider_class,
        mock_resource_class,
        mock_set_tracer,
    ):
        """Test tracing setup with console exporter."""
        mock_resource = MagicMock()
        mock_resource_class.create.return_value = mock_resource

        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        mock_console_exporter = MagicMock()
        mock_console_exporter_class.return_value = mock_console_exporter

        manager = TelemetryManager()
        manager.setup_tracing(
            service_name="test-service",
            console_export=True,
        )

        mock_console_exporter_class.assert_called_once()
        mock_processor_class.assert_called()
        assert "console" in manager._exporters

    @patch("src.observability.tracing.trace.set_tracer_provider")
    @patch("src.observability.tracing.Resource.create")
    @patch("src.observability.tracing.TracerProvider")
    def test_setup_tracing_jaeger_exporter(
        self,
        mock_provider_class,
        mock_resource_create,
        mock_set_tracer,
    ):
        """Test tracing setup with Jaeger exporter."""
        mock_resource = MagicMock()
        mock_resource_create.return_value = mock_resource

        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        # Mock the JaegerExporter class directly in the module
        mock_jaeger_class = MagicMock()
        mock_jaeger_instance = MagicMock()
        mock_jaeger_class.return_value = mock_jaeger_instance

        with patch.dict("src.observability.tracing.__dict__", {"JaegerExporter": mock_jaeger_class, "JAEGER_AVAILABLE": True}):
            with patch("src.observability.tracing.BatchSpanProcessor") as mock_processor:
                manager = TelemetryManager()
                manager.setup_tracing(
                    service_name="test-service",
                    jaeger_endpoint="http://jaeger:14268/api/traces",
                )

                mock_jaeger_class.assert_called_once_with(
                    collector_endpoint="http://jaeger:14268/api/traces"
                )
                assert "jaeger" in manager._exporters

    @patch("src.observability.tracing.trace.set_tracer_provider")
    @patch("src.observability.tracing.Resource")
    @patch("src.observability.tracing.TracerProvider")
    @patch("src.observability.tracing.JAEGER_AVAILABLE", False)
    def test_setup_tracing_jaeger_not_available(
        self,
        mock_provider_class,
        mock_resource_class,
        mock_set_tracer,
    ):
        """Test tracing setup with Jaeger when not installed."""
        mock_resource = MagicMock()
        mock_resource_class.create.return_value = mock_resource

        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            manager = TelemetryManager()
            manager.setup_tracing(
                service_name="test-service",
                jaeger_endpoint="http://jaeger:14268/api/traces",
            )

            mock_logger.warning.assert_called_once()
            assert "jaeger" not in manager._exporters

    @patch("src.observability.tracing.trace.set_tracer_provider")
    @patch("src.observability.tracing.Resource")
    @patch("src.observability.tracing.TracerProvider")
    @patch("src.observability.tracing.OTLP_GRPC_AVAILABLE", True)
    def test_setup_tracing_otlp_exporter(
        self,
        mock_provider_class,
        mock_resource_class,
        mock_set_tracer,
    ):
        """Test tracing setup with OTLP exporter."""
        mock_resource = MagicMock()
        mock_resource_class.create.return_value = mock_resource

        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        with patch("src.observability.tracing.OTLPSpanExporter") as mock_otlp:
            with patch("src.observability.tracing.BatchSpanProcessor") as mock_processor:
                mock_otlp_instance = MagicMock()
                mock_otlp.return_value = mock_otlp_instance

                manager = TelemetryManager()
                manager.setup_tracing(
                    service_name="test-service",
                    otlp_endpoint="http://otel-collector:4317",
                )

                mock_otlp.assert_called_once_with(endpoint="http://otel-collector:4317")
                assert "otlp" in manager._exporters

    @patch("src.observability.tracing.trace.set_tracer_provider")
    @patch("src.observability.tracing.Resource.create")
    @patch("src.observability.tracing.TracerProvider")
    def test_setup_tracing_azure_monitor_exporter(
        self,
        mock_provider_class,
        mock_resource_create,
        mock_set_tracer,
    ):
        """Test tracing setup with Azure Monitor exporter."""
        mock_resource = MagicMock()
        mock_resource_create.return_value = mock_resource

        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        # Mock the AzureMonitorTraceExporter class directly in the module
        mock_azure_class = MagicMock()
        mock_azure_instance = MagicMock()
        mock_azure_class.return_value = mock_azure_instance

        with patch.dict("src.observability.tracing.__dict__", {"AzureMonitorTraceExporter": mock_azure_class, "AZURE_MONITOR_AVAILABLE": True}):
            with patch("src.observability.tracing.BatchSpanProcessor") as mock_processor:
                manager = TelemetryManager()
                manager.setup_tracing(
                    service_name="test-service",
                    azure_connection_string="InstrumentationKey=12345",
                )

                mock_azure_class.assert_called_once_with(
                    connection_string="InstrumentationKey=12345"
                )
                assert "azure_monitor" in manager._exporters

    def test_shutdown(self):
        """Test shutdown method."""
        manager = TelemetryManager()
        mock_provider = MagicMock()
        manager._provider = mock_provider
        manager._initialized = True
        manager._exporters = ["console", "jaeger"]

        manager.shutdown()

        mock_provider.shutdown.assert_called_once()
        assert manager._initialized is False
        assert manager._exporters == []

    def test_shutdown_no_provider(self):
        """Test shutdown when no provider exists."""
        manager = TelemetryManager()
        # Should not raise any error
        manager.shutdown()


@pytest.mark.unit
class TestGetTelemetryManager:
    """Tests for get_telemetry_manager function."""

    def test_get_telemetry_manager_singleton(self):
        """Test that get_telemetry_manager returns singleton."""
        with patch("src.observability.tracing._telemetry_manager", None):
            manager1 = get_telemetry_manager()
            manager2 = get_telemetry_manager()

            assert manager1 is manager2

    def test_get_telemetry_manager_returns_existing(self):
        """Test that existing manager is returned."""
        existing_manager = MagicMock()
        with patch("src.observability.tracing._telemetry_manager", existing_manager):
            manager = get_telemetry_manager()

            assert manager is existing_manager


@pytest.mark.unit
class TestSetupTracingFromSettings:
    """Tests for setup_tracing_from_settings function."""

    @patch("src.observability.tracing.get_telemetry_manager")
    def test_setup_tracing_disabled(self, mock_get_manager):
        """Test tracing setup when disabled via environment."""
        with patch.dict(os.environ, {"OTEL_ENABLED": "false"}):
            result = setup_tracing_from_settings()
            assert result is None

    @patch("src.observability.tracing.get_telemetry_manager")
    @patch("src.observability.tracing.settings")
    def test_setup_tracing_enabled(self, mock_settings, mock_get_manager):
        """Test tracing setup when enabled."""
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager

        mock_settings.app_version = "1.0.0"
        mock_settings.env = "development"

        with patch.dict(os.environ, {}, clear=True):
            setup_tracing_from_settings()

        mock_manager.setup_tracing.assert_called_once()
        call_kwargs = mock_manager.setup_tracing.call_args[1]
        assert call_kwargs["service_name"] == "pipeline-api"
        assert call_kwargs["service_version"] == "1.0.0"

    @patch("src.observability.tracing.get_telemetry_manager")
    @patch("src.observability.tracing.settings")
    def test_setup_tracing_with_env_vars(self, mock_settings, mock_get_manager):
        """Test tracing setup with environment variables."""
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager

        mock_settings.app_version = "2.0.0"
        mock_settings.env = "production"

        env_vars = {
            "OTEL_SERVICE_NAME": "custom-service",
            "OTEL_ENVIRONMENT": "staging",
            "OTEL_EXPORTER_JAEGER_ENDPOINT": "http://jaeger:14268/api/traces",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel:4317",
            "OTEL_CONSOLE_EXPORT": "true",
        }

        with patch.dict(os.environ, env_vars):
            setup_tracing_from_settings()

        call_kwargs = mock_manager.setup_tracing.call_args[1]
        assert call_kwargs["service_name"] == "custom-service"
        assert call_kwargs["environment"] == "staging"
        assert call_kwargs["jaeger_endpoint"] == "http://jaeger:14268/api/traces"
        assert call_kwargs["otlp_endpoint"] == "http://otel:4317"
        assert call_kwargs["console_export"] is True

    @patch("src.observability.tracing.get_telemetry_manager")
    @patch("src.observability.tracing.settings")
    def test_setup_tracing_with_azure_connection(self, mock_settings, mock_get_manager):
        """Test tracing setup with Azure Monitor connection string."""
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager

        mock_settings.app_version = "1.0.0"
        mock_settings.env = "development"

        with patch.dict(os.environ, {"APPLICATIONINSIGHTS_CONNECTION_STRING": "InstrumentationKey=xyz"}):
            setup_tracing_from_settings()

        call_kwargs = mock_manager.setup_tracing.call_args[1]
        assert call_kwargs["azure_connection_string"] == "InstrumentationKey=xyz"


@pytest.mark.unit
class TestStartSpan:
    """Tests for start_span context manager."""

    def test_start_span_basic(self):
        """Test basic span creation."""
        mock_tracer = MagicMock()
        mock_span = MagicMock()

        with patch("src.observability.tracing.get_tracer", return_value=mock_tracer):
            mock_context_manager = MagicMock()
            mock_context_manager.__enter__ = MagicMock(return_value=mock_span)
            mock_context_manager.__exit__ = MagicMock(return_value=False)
            mock_tracer.start_as_current_span.return_value = mock_context_manager

            with start_span("test-operation") as span:
                assert span == mock_span

            mock_tracer.start_as_current_span.assert_called_once_with(
                name="test-operation",
                kind=SpanKind.INTERNAL,
                attributes=None,
            )

    def test_start_span_with_attributes(self):
        """Test span creation with attributes."""
        mock_tracer = MagicMock()
        mock_span = MagicMock()

        with patch("src.observability.tracing.get_tracer", return_value=mock_tracer):
            mock_context_manager = MagicMock()
            mock_context_manager.__enter__ = MagicMock(return_value=mock_span)
            mock_context_manager.__exit__ = MagicMock(return_value=False)
            mock_tracer.start_as_current_span.return_value = mock_context_manager

            attrs = {"key": "value", "number": 42}
            with start_span("test-operation", attributes=attrs) as span:
                assert span == mock_span

            call_kwargs = mock_tracer.start_as_current_span.call_args[1]
            assert call_kwargs["attributes"] == attrs

    def test_start_span_with_kind(self):
        """Test span creation with specific kind."""
        mock_tracer = MagicMock()
        mock_span = MagicMock()

        with patch("src.observability.tracing.get_tracer", return_value=mock_tracer):
            mock_context_manager = MagicMock()
            mock_context_manager.__enter__ = MagicMock(return_value=mock_span)
            mock_context_manager.__exit__ = MagicMock(return_value=False)
            mock_tracer.start_as_current_span.return_value = mock_context_manager

            with start_span("test-operation", kind=SpanKind.SERVER) as span:
                assert span == mock_span

            call_kwargs = mock_tracer.start_as_current_span.call_args[1]
            assert call_kwargs["kind"] == SpanKind.SERVER

    def test_start_span_with_parent(self):
        """Test span creation with parent context."""
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_parent = MagicMock()

        with patch("src.observability.tracing.get_tracer", return_value=mock_tracer):
            mock_context_manager = MagicMock()
            mock_context_manager.__enter__ = MagicMock(return_value=mock_span)
            mock_context_manager.__exit__ = MagicMock(return_value=False)
            mock_tracer.start_as_current_span.return_value = mock_context_manager

            with start_span("test-operation", parent=mock_parent) as span:
                assert span == mock_span

            call_kwargs = mock_tracer.start_as_current_span.call_args[1]
            assert call_kwargs["context"] == mock_parent


@pytest.mark.unit
class TestStartPipelineStageSpan:
    """Tests for start_pipeline_stage_span context manager."""

    def test_start_pipeline_stage_span_basic(self):
        """Test basic pipeline stage span creation."""
        mock_tracer = MagicMock()
        mock_span = MagicMock()

        with patch("src.observability.tracing.get_tracer", return_value=mock_tracer):
            mock_context_manager = MagicMock()
            mock_context_manager.__enter__ = MagicMock(return_value=mock_span)
            mock_context_manager.__exit__ = MagicMock(return_value=False)
            mock_tracer.start_as_current_span.return_value = mock_context_manager

            with start_pipeline_stage_span("parse") as span:
                assert span == mock_span

            call_kwargs = mock_tracer.start_as_current_span.call_args[1]
            assert call_kwargs["name"] == "pipeline.stage.parse"
            assert call_kwargs["kind"] == SpanKind.INTERNAL
            assert call_kwargs["attributes"]["pipeline.stage"] == "parse"

    def test_start_pipeline_stage_span_with_job_id(self):
        """Test pipeline stage span with job ID."""
        mock_tracer = MagicMock()
        mock_span = MagicMock()

        with patch("src.observability.tracing.get_tracer", return_value=mock_tracer):
            mock_context_manager = MagicMock()
            mock_context_manager.__enter__ = MagicMock(return_value=mock_span)
            mock_context_manager.__exit__ = MagicMock(return_value=False)
            mock_tracer.start_as_current_span.return_value = mock_context_manager

            with start_pipeline_stage_span("enrich", job_id="job-123") as span:
                assert span == mock_span

            call_kwargs = mock_tracer.start_as_current_span.call_args[1]
            assert call_kwargs["attributes"]["pipeline.stage"] == "enrich"
            assert call_kwargs["attributes"]["job.id"] == "job-123"

    def test_start_pipeline_stage_span_with_extra_attributes(self):
        """Test pipeline stage span with additional attributes."""
        mock_tracer = MagicMock()
        mock_span = MagicMock()

        with patch("src.observability.tracing.get_tracer", return_value=mock_tracer):
            mock_context_manager = MagicMock()
            mock_context_manager.__enter__ = MagicMock(return_value=mock_span)
            mock_context_manager.__exit__ = MagicMock(return_value=False)
            mock_tracer.start_as_current_span.return_value = mock_context_manager

            extra_attrs = {"document.type": "pdf", "document.size": 1024}
            with start_pipeline_stage_span("transform", job_id="job-456", attributes=extra_attrs) as span:
                assert span == mock_span

            call_kwargs = mock_tracer.start_as_current_span.call_args[1]
            attrs = call_kwargs["attributes"]
            assert attrs["pipeline.stage"] == "transform"
            assert attrs["job.id"] == "job-456"
            assert attrs["document.type"] == "pdf"
            assert attrs["document.size"] == 1024
