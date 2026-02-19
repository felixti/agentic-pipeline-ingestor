"""Unit tests for destination routing module."""

import asyncio
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import UUID, uuid4

import pytest

from src.api.models import DestinationConfig, DestinationFilter, DestinationType, FilterOperator
from src.core.routing import (
    CircuitBreaker,
    CircuitBreakerState,
    DestinationRouter,
    DestinationRouteResult,
    MultiDestinationResult,
    RoutingFilterBuilder,
    RoutingStatus,
    get_router,
)
from src.plugins.base import TransformedData, WriteResult
from src.plugins.registry import PluginRegistry


@pytest.mark.unit
class TestRoutingStatus:
    """Tests for RoutingStatus enum."""

    def test_enum_values(self):
        """Test that enum values are correct."""
        assert RoutingStatus.PENDING.value == "pending"
        assert RoutingStatus.IN_PROGRESS.value == "in_progress"
        assert RoutingStatus.SUCCESS.value == "success"
        assert RoutingStatus.PARTIAL_SUCCESS.value == "partial_success"
        assert RoutingStatus.FAILED.value == "failed"
        assert RoutingStatus.SKIPPED.value == "skipped"


@pytest.mark.unit
class TestDestinationRouteResult:
    """Tests for DestinationRouteResult dataclass."""

    def test_creation(self):
        """Test creating a DestinationRouteResult."""
        result = DestinationRouteResult(
            destination_id="dest_1",
            destination_type="webhook",
            status=RoutingStatus.SUCCESS,
            success=True,
            latency_ms=100,
        )

        assert result.destination_id == "dest_1"
        assert result.destination_type == "webhook"
        assert result.status == RoutingStatus.SUCCESS
        assert result.success is True
        assert result.latency_ms == 100
        assert result.write_result is None
        assert result.error is None
        assert result.filtered is False
        assert result.filter_reason is None

    def test_with_error(self):
        """Test creating a result with an error."""
        result = DestinationRouteResult(
            destination_id="dest_2",
            destination_type="s3",
            status=RoutingStatus.FAILED,
            success=False,
            error="Connection timeout",
            latency_ms=5000,
        )

        assert result.success is False
        assert result.error == "Connection timeout"

    def test_filtered(self):
        """Test creating a filtered result."""
        result = DestinationRouteResult(
            destination_id="dest_3",
            destination_type="cognee",
            status=RoutingStatus.SKIPPED,
            success=True,
            filtered=True,
            filter_reason="Mime type filter mismatch",
        )

        assert result.filtered is True
        assert result.filter_reason == "Mime type filter mismatch"


@pytest.mark.unit
class TestMultiDestinationResult:
    """Tests for MultiDestinationResult dataclass."""

    def test_creation(self):
        """Test creating a MultiDestinationResult."""
        job_id = uuid4()
        result = MultiDestinationResult(
            job_id=job_id,
            overall_status=RoutingStatus.SUCCESS,
            destination_results=[],
        )

        assert result.job_id == job_id
        assert result.overall_status == RoutingStatus.SUCCESS
        assert result.destination_results == []
        assert result.success_count == 0
        assert result.failure_count == 0
        assert result.skipped_count == 0
        assert result.total_latency_ms == 0
        assert result.partial_failure is False
        assert result.errors == []

    def test_to_dict(self):
        """Test converting result to dictionary."""
        job_id = uuid4()
        dest_result = DestinationRouteResult(
            destination_id="dest_1",
            destination_type="webhook",
            status=RoutingStatus.SUCCESS,
            success=True,
        )
        result = MultiDestinationResult(
            job_id=job_id,
            overall_status=RoutingStatus.SUCCESS,
            destination_results=[dest_result],
            success_count=1,
        )

        dict_result = result.to_dict()

        assert dict_result["job_id"] == str(job_id)
        assert dict_result["overall_status"] == "success"
        assert dict_result["success_count"] == 1
        assert len(dict_result["destination_results"]) == 1
        assert dict_result["destination_results"][0]["destination_id"] == "dest_1"


@pytest.mark.unit
class TestDestinationRouter:
    """Tests for DestinationRouter class."""

    @pytest.fixture
    def mock_registry(self):
        """Create a mock plugin registry."""
        return MagicMock(spec=PluginRegistry)

    @pytest.fixture
    def router(self, mock_registry):
        """Create a DestinationRouter instance."""
        return DestinationRouter(plugin_registry=mock_registry)

    @pytest.fixture
    def sample_data(self):
        """Create sample transformed data."""
        return TransformedData(
            job_id=uuid4(),
            chunks=[{"text": "Test content"}],
            metadata={"mime_type": "application/pdf"},
        )

    @pytest.fixture
    def sample_destination(self):
        """Create a sample destination config."""
        return DestinationConfig(
            id=uuid4(),
            type=DestinationType.WEBHOOK,
            name="test_webhook",
            config={"url": "https://example.com/webhook"},
            enabled=True,
        )

    @pytest.mark.asyncio
    async def test_route_to_multiple_empty_destinations(self, router, sample_data):
        """Test routing with no destinations returns skipped status."""
        result = await router.route_to_multiple(sample_data, [])

        assert result.overall_status == RoutingStatus.SKIPPED
        assert result.skipped_count == 1

    @pytest.mark.asyncio
    async def test_route_to_single_disabled_destination(self, router, sample_data):
        """Test that disabled destinations are skipped."""
        destination = DestinationConfig(
            id=uuid4(),
            type=DestinationType.WEBHOOK,
            name="disabled_webhook",
            config={},
            enabled=False,
        )

        result = await router._route_to_single(sample_data, destination)

        assert result.success is True
        assert result.filtered is True
        assert result.filter_reason == "Destination disabled"
        assert result.status == RoutingStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_route_to_single_circuit_breaker_open(self, router, sample_data):
        """Test that open circuit breaker causes skip."""
        destination = DestinationConfig(
            id=uuid4(),
            type=DestinationType.WEBHOOK,
            name="test_webhook",
            config={},
            enabled=True,
        )

        # Manually open the circuit breaker
        dest_id = str(destination.id)
        router._circuit_breakers[dest_id] = CircuitBreaker()
        router._circuit_breakers[dest_id].state = CircuitBreakerState.OPEN
        router._circuit_breakers[dest_id].last_state_change = datetime.utcnow()

        result = await router._route_to_single(sample_data, destination)

        assert result.filtered is True
        assert "Circuit breaker open" in result.filter_reason

    @pytest.mark.asyncio
    async def test_route_to_single_filter_mismatch(
        self, router, sample_data, mock_registry
    ):
        """Test that filter mismatch causes skip."""
        destination = DestinationConfig(
            id=uuid4(),
            type=DestinationType.WEBHOOK,
            name="test_webhook",
            config={},
            enabled=True,
            filters=[
                DestinationFilter(
                    field="mime_type",
                    operator=FilterOperator.EQUALS,
                    value="image/png",
                )
            ],
        )

        result = await router._route_to_single(sample_data, destination)

        assert result.filtered is True
        assert "Filter failed" in result.filter_reason

    @pytest.mark.asyncio
    async def test_route_to_single_success(
        self, router, sample_data, mock_registry, sample_destination
    ):
        """Test successful routing to a destination."""
        # Setup mock destination plugin
        mock_plugin = AsyncMock()
        mock_write_result = WriteResult(success=True, destination_id="dest_1")
        mock_plugin.write = AsyncMock(return_value=mock_write_result)

        mock_registry.get_destination.return_value = mock_plugin

        result = await router._route_to_single(sample_data, sample_destination)

        assert result.success is True
        assert result.status == RoutingStatus.SUCCESS
        assert result.write_result is not None
        mock_plugin.initialize.assert_called_once()
        mock_plugin.connect.assert_called_once()
        mock_plugin.write.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_to_single_plugin_not_found(
        self, router, sample_data, mock_registry, sample_destination
    ):
        """Test routing when plugin is not found."""
        mock_registry.get_destination.return_value = None

        result = await router._route_to_single(sample_data, sample_destination)

        assert result.success is False
        assert result.status == RoutingStatus.FAILED
        assert "Destination plugin not found" in result.error

    @pytest.mark.asyncio
    async def test_route_to_single_plugin_exception(
        self, router, sample_data, mock_registry, sample_destination
    ):
        """Test routing when plugin raises an exception."""
        mock_plugin = AsyncMock()
        mock_plugin.initialize.side_effect = Exception("Connection failed")

        mock_registry.get_destination.return_value = mock_plugin

        # Patch the logger to avoid structlog issues with kwargs
        with patch.object(router.logger, "error"):
            result = await router._route_to_single(sample_data, sample_destination)

        assert result.success is False
        assert result.status == RoutingStatus.FAILED
        assert "Connection failed" in result.error

    @pytest.mark.asyncio
    async def test_route_to_multiple_parallel(
        self, router, sample_data, mock_registry
    ):
        """Test parallel routing to multiple destinations."""
        destinations = [
            DestinationConfig(
                id=uuid4(),
                type=DestinationType.WEBHOOK,
                name="webhook1",
                config={},
                enabled=True,
            ),
            DestinationConfig(
                id=uuid4(),
                type=DestinationType.WEBHOOK,
                name="webhook2",
                config={},
                enabled=True,
            ),
        ]

        mock_plugin = AsyncMock()
        mock_write_result = WriteResult(success=True, destination_id="dest")
        mock_plugin.write = AsyncMock(return_value=mock_write_result)

        mock_registry.get_destination.return_value = mock_plugin

        result = await router.route_to_multiple(
            sample_data, destinations, parallel=True, max_concurrent=5
        )

        assert result.success_count == 2
        assert result.overall_status == RoutingStatus.SUCCESS
        assert result.partial_failure is False

    @pytest.mark.asyncio
    async def test_route_to_multiple_sequential(
        self, router, sample_data, mock_registry
    ):
        """Test sequential routing to multiple destinations."""
        destinations = [
            DestinationConfig(
                id=uuid4(),
                type=DestinationType.WEBHOOK,
                name="webhook1",
                config={},
                enabled=True,
            ),
            DestinationConfig(
                id=uuid4(),
                type=DestinationType.WEBHOOK,
                name="webhook2",
                config={},
                enabled=True,
            ),
        ]

        mock_plugin = AsyncMock()
        mock_write_result = WriteResult(success=True, destination_id="dest")
        mock_plugin.write = AsyncMock(return_value=mock_write_result)

        mock_registry.get_destination.return_value = mock_plugin

        result = await router.route_to_multiple(
            sample_data, destinations, parallel=False
        )

        assert result.success_count == 2
        assert result.overall_status == RoutingStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_route_to_multiple_partial_failure(
        self, router, sample_data, mock_registry
    ):
        """Test routing with some destinations failing."""
        destinations = [
            DestinationConfig(
                id=uuid4(),
                type=DestinationType.WEBHOOK,
                name="webhook1",
                config={},
                enabled=True,
            ),
            DestinationConfig(
                id=uuid4(),
                type=DestinationType.WEBHOOK,
                name="webhook2",
                config={},
                enabled=True,
            ),
        ]

        # First call succeeds, second fails
        mock_plugin_success = AsyncMock()
        mock_plugin_success.write = AsyncMock(
            return_value=WriteResult(success=True, destination_id="dest1")
        )

        mock_plugin_fail = AsyncMock()
        mock_plugin_fail.write = AsyncMock(
            return_value=WriteResult(success=False, destination_id="dest2", error="Failed")
        )

        mock_registry.get_destination.side_effect = [
            mock_plugin_success,
            mock_plugin_fail,
        ]

        result = await router.route_to_multiple(sample_data, destinations)

        assert result.success_count == 1
        assert result.failure_count == 1
        assert result.overall_status == RoutingStatus.PARTIAL_SUCCESS
        assert result.partial_failure is True

    @pytest.mark.asyncio
    async def test_route_to_multiple_all_fail(
        self, router, sample_data, mock_registry
    ):
        """Test routing when all destinations fail."""
        destinations = [
            DestinationConfig(
                id=uuid4(),
                type=DestinationType.WEBHOOK,
                name="webhook1",
                config={},
                enabled=True,
            ),
        ]

        mock_plugin = AsyncMock()
        mock_plugin.write = AsyncMock(
            return_value=WriteResult(success=False, error="Failed")
        )

        mock_registry.get_destination.return_value = mock_plugin

        result = await router.route_to_multiple(sample_data, destinations)

        assert result.success_count == 0
        assert result.failure_count == 1
        assert result.overall_status == RoutingStatus.FAILED

    def test_apply_filters_empty(self, router, sample_data):
        """Test applying empty filters passes."""
        result = router._apply_filters(sample_data, [])

        assert result["passed"] is True
        assert result["reason"] is None

    def test_apply_filters_equals_pass(self, router, sample_data):
        """Test equals filter that passes."""
        filters = [
            DestinationFilter(
                field="mime_type",
                operator=FilterOperator.EQUALS,
                value="application/pdf",
            )
        ]

        result = router._apply_filters(sample_data, filters)

        assert result["passed"] is True

    def test_apply_filters_equals_fail(self, router, sample_data):
        """Test equals filter that fails."""
        filters = [
            DestinationFilter(
                field="mime_type",
                operator=FilterOperator.EQUALS,
                value="image/png",
            )
        ]

        result = router._apply_filters(sample_data, filters)

        assert result["passed"] is False
        assert "Filter failed" in result["reason"]

    def test_apply_filters_contains_pass(self, router, sample_data):
        """Test contains filter that passes."""
        filters = [
            DestinationFilter(
                field="mime_type",
                operator=FilterOperator.CONTAINS,
                value="pdf",
            )
        ]

        result = router._apply_filters(sample_data, filters)

        assert result["passed"] is True

    def test_apply_filters_not_equals_pass(self, router, sample_data):
        """Test not equals filter that passes."""
        filters = [
            DestinationFilter(
                field="mime_type",
                operator=FilterOperator.NOT_EQUALS,
                value="image/png",
            )
        ]

        result = router._apply_filters(sample_data, filters)

        assert result["passed"] is True

    def test_apply_filters_starts_with_pass(self, router, sample_data):
        """Test starts_with filter that passes."""
        filters = [
            DestinationFilter(
                field="mime_type",
                operator=FilterOperator.STARTS_WITH,
                value="application",
            )
        ]

        result = router._apply_filters(sample_data, filters)

        assert result["passed"] is True

    def test_apply_filters_ends_with_pass(self, router, sample_data):
        """Test ends_with filter that passes."""
        filters = [
            DestinationFilter(
                field="mime_type",
                operator=FilterOperator.ENDS_WITH,
                value="pdf",
            )
        ]

        result = router._apply_filters(sample_data, filters)

        assert result["passed"] is True

    def test_apply_filters_regex_pass(self, router, sample_data):
        """Test regex filter that passes."""
        filters = [
            DestinationFilter(
                field="mime_type",
                operator=FilterOperator.REGEX,
                value="application/.*",
            )
        ]

        result = router._apply_filters(sample_data, filters)

        assert result["passed"] is True

    def test_apply_filters_regex_invalid(self, router, sample_data):
        """Test regex filter with invalid pattern."""
        filters = [
            DestinationFilter(
                field="mime_type",
                operator=FilterOperator.REGEX,
                value="[invalid",
            )
        ]

        result = router._apply_filters(sample_data, filters)

        assert result["passed"] is False

    def test_apply_filters_multiple(self, router, sample_data):
        """Test multiple filters - all must pass."""
        filters = [
            DestinationFilter(
                field="mime_type",
                operator=FilterOperator.EQUALS,
                value="application/pdf",
            ),
            DestinationFilter(
                field="mime_type",
                operator=FilterOperator.CONTAINS,
                value="pdf",
            ),
        ]

        result = router._apply_filters(sample_data, filters)

        assert result["passed"] is True

    def test_apply_filters_multiple_one_fails(self, router, sample_data):
        """Test multiple filters where one fails."""
        filters = [
            DestinationFilter(
                field="mime_type",
                operator=FilterOperator.EQUALS,
                value="application/pdf",
            ),
            DestinationFilter(
                field="mime_type",
                operator=FilterOperator.EQUALS,
                value="image/png",
            ),
        ]

        result = router._apply_filters(sample_data, filters)

        assert result["passed"] is False

    def test_evaluate_filter_equals(self, router):
        """Test equals operator."""
        assert router._evaluate_filter("test", "test", FilterOperator.EQUALS) is True
        assert router._evaluate_filter("test", "other", FilterOperator.EQUALS) is False

    def test_evaluate_filter_not_equals(self, router):
        """Test not_equals operator."""
        assert (
            router._evaluate_filter("test", "other", FilterOperator.NOT_EQUALS) is True
        )
        assert (
            router._evaluate_filter("test", "test", FilterOperator.NOT_EQUALS) is False
        )

    def test_evaluate_filter_contains(self, router):
        """Test contains operator."""
        assert (
            router._evaluate_filter("hello world", "world", FilterOperator.CONTAINS)
            is True
        )
        assert (
            router._evaluate_filter("hello world", "foo", FilterOperator.CONTAINS)
            is False
        )

    def test_evaluate_filter_starts_with(self, router):
        """Test starts_with operator."""
        assert (
            router._evaluate_filter("hello world", "hello", FilterOperator.STARTS_WITH)
            is True
        )
        assert (
            router._evaluate_filter("hello world", "world", FilterOperator.STARTS_WITH)
            is False
        )

    def test_evaluate_filter_ends_with(self, router):
        """Test ends_with operator."""
        assert (
            router._evaluate_filter("hello world", "world", FilterOperator.ENDS_WITH)
            is True
        )
        assert (
            router._evaluate_filter("hello world", "hello", FilterOperator.ENDS_WITH)
            is False
        )

    def test_evaluate_filter_regex(self, router):
        """Test regex operator."""
        assert (
            router._evaluate_filter("hello123", r"\d+", FilterOperator.REGEX) is True
        )
        assert (
            router._evaluate_filter("hello", r"\d+", FilterOperator.REGEX) is False
        )

    def test_is_circuit_open_no_breaker(self, router):
        """Test is_circuit_open when no breaker exists."""
        assert router._is_circuit_open("nonexistent") is False

    def test_is_circuit_open_exists_closed(self, router):
        """Test is_circuit_open when breaker exists and is closed."""
        router._circuit_breakers["dest1"] = CircuitBreaker()
        assert router._is_circuit_open("dest1") is False

    def test_is_circuit_open_exists_open(self, router):
        """Test is_circuit_open when breaker exists and is open."""
        router._circuit_breakers["dest1"] = CircuitBreaker()
        router._circuit_breakers["dest1"].state = CircuitBreakerState.OPEN
        router._circuit_breakers["dest1"].last_state_change = datetime.utcnow()
        assert router._is_circuit_open("dest1") is True

    def test_record_success_new_breaker(self, router):
        """Test recording success creates new breaker."""
        router._record_success("dest1")

        assert "dest1" in router._circuit_breakers
        assert router._circuit_breakers["dest1"].success_count == 1

    def test_record_success_existing_breaker(self, router):
        """Test recording success on existing breaker."""
        router._record_success("dest1")
        router._record_success("dest1")

        assert router._circuit_breakers["dest1"].success_count == 2

    def test_record_failure_new_breaker(self, router):
        """Test recording failure creates new breaker."""
        router._record_failure("dest1")

        assert "dest1" in router._circuit_breakers
        assert router._circuit_breakers["dest1"].failure_count == 1

    @pytest.mark.asyncio
    async def test_get_destination_health(self, router):
        """Test getting health status for destinations."""
        # Setup some circuit breakers
        router._record_success("dest1")
        router._record_failure("dest2")

        health = await router.get_destination_health()

        assert "dest1" in health
        assert "dest2" in health
        assert health["dest1"]["success_count"] == 1
        assert health["dest2"]["failure_count"] == 1


@pytest.mark.unit
class TestCircuitBreakerState:
    """Tests for CircuitBreakerState enum."""

    def test_enum_values(self):
        """Test that enum values are correct."""
        assert CircuitBreakerState.CLOSED.value == "closed"
        assert CircuitBreakerState.OPEN.value == "open"
        assert CircuitBreakerState.HALF_OPEN.value == "half_open"


@pytest.mark.unit
class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_default_initialization(self):
        """Test default circuit breaker initialization."""
        breaker = CircuitBreaker()

        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.failure_threshold == 5
        assert breaker.success_threshold == 3
        assert breaker.timeout_seconds == 60
        assert breaker.failure_count == 0
        assert breaker.success_count == 0

    def test_custom_initialization(self):
        """Test circuit breaker with custom parameters."""
        breaker = CircuitBreaker(
            failure_threshold=3, success_threshold=2, timeout_seconds=30
        )

        assert breaker.failure_threshold == 3
        assert breaker.success_threshold == 2
        assert breaker.timeout_seconds == 30

    def test_is_open_closed_state(self):
        """Test is_open returns False when closed."""
        breaker = CircuitBreaker()
        assert breaker.is_open() is False

    def test_is_open_open_state(self):
        """Test is_open returns True when open."""
        breaker = CircuitBreaker()
        breaker.state = CircuitBreakerState.OPEN
        breaker.last_state_change = datetime.utcnow()
        assert breaker.is_open() is True

    def test_is_open_timeout_transition(self):
        """Test is_open transitions to half-open after timeout."""
        from datetime import timedelta
        breaker = CircuitBreaker()
        breaker.state = CircuitBreakerState.OPEN
        breaker.last_state_change = datetime.utcnow()

        # Should still be open immediately
        assert breaker.is_open() is True

        # Simulate timeout by setting last change to past (need timedelta)
        breaker.last_state_change = datetime.utcnow() - timedelta(seconds=breaker.timeout_seconds + 1)

        # After timeout, should transition to half-open and return False
        assert breaker.is_open() is False
        assert breaker.state == CircuitBreakerState.HALF_OPEN

    def test_record_success_closed(self):
        """Test recording success in closed state."""
        breaker = CircuitBreaker()
        breaker.record_success()

        assert breaker.success_count == 1
        assert breaker.last_success_time is not None

    def test_record_success_half_open_threshold_met(self):
        """Test recording success in half-open closes the circuit."""
        breaker = CircuitBreaker(success_threshold=2)
        breaker.state = CircuitBreakerState.HALF_OPEN

        breaker.record_success()
        assert breaker.state == CircuitBreakerState.HALF_OPEN

        breaker.record_success()
        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.failure_count == 0

    def test_record_failure_closed_threshold_not_met(self):
        """Test recording failure in closed state doesn't open circuit."""
        breaker = CircuitBreaker(failure_threshold=5)
        breaker.record_failure()

        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.failure_count == 1

    def test_record_failure_closed_threshold_met(self):
        """Test recording enough failures opens the circuit."""
        breaker = CircuitBreaker(failure_threshold=2)

        # Patch logger to avoid structlog issues
        with patch("src.core.routing.logger"):
            breaker.record_failure()
            assert breaker.state == CircuitBreakerState.CLOSED

            breaker.record_failure()
            assert breaker.state == CircuitBreakerState.OPEN
            assert breaker.last_state_change is not None

    def test_record_failure_half_open(self):
        """Test recording failure in half-open reopens the circuit."""
        breaker = CircuitBreaker()
        breaker.state = CircuitBreakerState.HALF_OPEN
        breaker.last_state_change = None

        breaker.record_failure()

        assert breaker.state == CircuitBreakerState.OPEN


@pytest.mark.unit
class TestRoutingFilterBuilder:
    """Tests for RoutingFilterBuilder class."""

    def test_builder_initialization(self):
        """Test filter builder initialization."""
        builder = RoutingFilterBuilder()
        assert builder.filters == []

    def test_equals_filter(self):
        """Test adding equals filter."""
        builder = RoutingFilterBuilder()
        result = builder.equals("mime_type", "application/pdf")

        assert result is builder  # Should return self for chaining
        assert len(builder.filters) == 1
        assert builder.filters[0].field == "mime_type"
        assert builder.filters[0].operator == FilterOperator.EQUALS
        assert builder.filters[0].value == "application/pdf"

    def test_not_equals_filter(self):
        """Test adding not_equals filter."""
        builder = RoutingFilterBuilder()
        builder.not_equals("status", "failed")

        assert builder.filters[0].operator == FilterOperator.NOT_EQUALS

    def test_contains_filter(self):
        """Test adding contains filter."""
        builder = RoutingFilterBuilder()
        builder.contains("text", "keyword")

        assert builder.filters[0].operator == FilterOperator.CONTAINS

    def test_starts_with_filter(self):
        """Test adding starts_with filter."""
        builder = RoutingFilterBuilder()
        builder.starts_with("path", "/docs/")

        assert builder.filters[0].operator == FilterOperator.STARTS_WITH

    def test_regex_filter(self):
        """Test adding regex filter."""
        builder = RoutingFilterBuilder()
        builder.regex("mime_type", "application/.*")

        assert builder.filters[0].operator == FilterOperator.REGEX

    def test_chaining(self):
        """Test filter chaining."""
        builder = (
            RoutingFilterBuilder()
            .equals("mime_type", "application/pdf")
            .contains("text", "important")
            .starts_with("path", "/docs/")
        )

        assert len(builder.filters) == 3

    def test_build_returns_copy(self):
        """Test that build returns a copy of filters."""
        builder = RoutingFilterBuilder()
        builder.equals("mime_type", "application/pdf")

        filters1 = builder.build()
        filters2 = builder.build()

        assert filters1 is not filters2
        assert filters1 == filters2

    def test_build_empty(self):
        """Test building empty filters."""
        builder = RoutingFilterBuilder()
        result = builder.build()

        assert result == []


@pytest.mark.unit
class TestGetRouter:
    """Tests for get_router function."""

    def test_get_router_singleton(self):
        """Test that get_router returns a singleton."""
        router1 = get_router()
        router2 = get_router()

        assert router1 is router2

    def test_get_router_returns_router(self):
        """Test that get_router returns a DestinationRouter."""
        router = get_router()

        assert isinstance(router, DestinationRouter)
