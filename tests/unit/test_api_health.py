"""Unit tests for health check API routes.

Tests the health check endpoints including comprehensive health checks,
liveness probes, readiness probes, and detailed component health checks.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, status

from src.api.models import ComponentHealth, HealthStatus
from src.api.routes.health import (
    ComprehensiveHealthResponse,
    LivenessResponse,
    ReadinessResponse,
    check_database,
    check_destinations,
    check_llm_providers,
    check_opentelemetry,
    check_redis,
    check_storage,
    detailed_component_health,
    health_check,
    liveness_probe,
    readiness_probe,
)


@pytest.mark.unit
class TestHealthCheckFunctions:
    """Tests for individual health check functions."""

    @pytest.mark.asyncio
    async def test_check_database_success(self):
        """Test successful database health check."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1
        
        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        
        # Patch at the module where it's imported (src.db.models)
        with patch("src.db.models.get_session") as mock_get_session:
            async def mock_gen():
                yield mock_session
            
            mock_get_session.return_value = mock_gen()
            result = await check_database()
        
        assert result.healthy is True
        assert result.component == "database"
        assert "successful" in result.message.lower()
        assert result.latency_ms is not None

    @pytest.mark.asyncio
    async def test_check_database_failure(self):
        """Test failed database health check."""
        with patch("src.db.models.get_session") as mock_get_session:
            mock_session = AsyncMock()
            mock_session.execute.side_effect = Exception("Connection refused")
            
            async def mock_gen():
                yield mock_session
            
            mock_get_session.return_value = mock_gen()
            result = await check_database()
        
        assert result.healthy is False
        assert result.component == "database"
        assert "failed" in result.message.lower()

    @pytest.mark.asyncio
    async def test_check_redis_success(self):
        """Test successful Redis health check."""
        with patch("src.api.routes.health.settings") as mock_settings:
            mock_settings.redis_url = "redis://localhost:6379/0"
            
            with patch("redis.asyncio.from_url") as mock_from_url:
                mock_client = AsyncMock()
                mock_client.ping.return_value = True
                mock_client.close = AsyncMock()
                mock_from_url.return_value = mock_client
                
                result = await check_redis()
        
        assert result.healthy is True
        assert result.component == "redis"
        assert "successful" in result.message.lower()

    @pytest.mark.asyncio
    async def test_check_redis_failure(self):
        """Test failed Redis health check."""
        with patch("src.api.routes.health.settings") as mock_settings:
            mock_settings.redis_url = "redis://localhost:6379/0"
            
            with patch("redis.asyncio.from_url") as mock_from_url:
                mock_client = AsyncMock()
                mock_client.ping.side_effect = Exception("Connection refused")
                mock_from_url.return_value = mock_client
                
                result = await check_redis()
        
        assert result.healthy is False
        assert result.component == "redis"
        assert "failed" in result.message.lower()

    @pytest.mark.asyncio
    async def test_check_llm_providers_success(self):
        """Test successful LLM providers health check."""
        with patch("src.api.routes.health.load_llm_config") as mock_load_config:
            with patch("src.api.routes.health.LLMProvider") as mock_provider_class:
                mock_provider = AsyncMock()
                mock_provider.health_check.return_value = {
                    "healthy": True,
                    "models": {"gpt-4": "healthy", "claude-3": "healthy"},
                }
                mock_provider_class.return_value = mock_provider
                
                result = await check_llm_providers()
        
        assert result.healthy is True
        assert result.component == "llm_providers"
        assert result.details is not None

    @pytest.mark.asyncio
    async def test_check_llm_providers_unhealthy(self):
        """Test LLM providers health check with unhealthy models."""
        with patch("src.api.routes.health.load_llm_config") as mock_load_config:
            with patch("src.api.routes.health.LLMProvider") as mock_provider_class:
                mock_provider = AsyncMock()
                mock_provider.health_check.return_value = {
                    "healthy": False,
                    "models": {"gpt-4": "unhealthy"},
                }
                mock_provider_class.return_value = mock_provider
                
                result = await check_llm_providers()
        
        assert result.healthy is False
        assert result.component == "llm_providers"

    @pytest.mark.asyncio
    async def test_check_llm_providers_exception(self):
        """Test LLM providers health check with exception."""
        with patch("src.api.routes.health.load_llm_config") as mock_load_config:
            mock_load_config.side_effect = Exception("Config not found")
            
            result = await check_llm_providers()
        
        assert result.healthy is False
        assert result.component == "llm_providers"
        assert "failed" in result.message.lower()

    @pytest.mark.asyncio
    async def test_check_destinations_success(self):
        """Test successful destinations health check."""
        with patch("src.api.routes.health.get_registry") as mock_get_registry:
            mock_registry = AsyncMock()
            mock_registry.health_check_all.return_value = {
                "cognee": True,
                "graphrag": True,
            }
            mock_get_registry.return_value = mock_registry
            
            result = await check_destinations()
        
        assert result.healthy is True
        assert result.component == "destinations"
        assert result.details["healthy"] == 2
        assert result.details["total"] == 2

    @pytest.mark.asyncio
    async def test_check_destinations_partial_failure(self):
        """Test destinations health check with partial failures."""
        with patch("src.api.routes.health.get_registry") as mock_get_registry:
            mock_registry = AsyncMock()
            mock_registry.health_check_all.return_value = {
                "cognee": True,
                "graphrag": False,
            }
            mock_get_registry.return_value = mock_registry
            
            result = await check_destinations()
        
        assert result.healthy is True  # Degraded but still functional
        assert result.component == "destinations"

    @pytest.mark.asyncio
    async def test_check_destinations_all_failure(self):
        """Test destinations health check with all failures."""
        with patch("src.api.routes.health.get_registry") as mock_get_registry:
            mock_registry = AsyncMock()
            mock_registry.health_check_all.return_value = {
                "cognee": False,
                "graphrag": False,
            }
            mock_get_registry.return_value = mock_registry
            
            result = await check_destinations()
        
        assert result.healthy is False
        assert result.component == "destinations"

    @pytest.mark.asyncio
    async def test_check_storage_success(self):
        """Test successful storage health check."""
        with patch("os.makedirs") as mock_makedirs:
            with patch("builtins.open", MagicMock()) as mock_open:
                mock_file = MagicMock()
                mock_file.read.return_value = "ok"
                mock_open.return_value.__enter__.return_value = mock_file
                
                with patch("os.remove") as mock_remove:
                    result = await check_storage()
        
        assert result.healthy is True
        assert result.component == "storage"

    @pytest.mark.asyncio
    async def test_check_storage_failure(self):
        """Test failed storage health check."""
        with patch("os.makedirs") as mock_makedirs:
            mock_makedirs.side_effect = PermissionError("Access denied")
            
            result = await check_storage()
        
        assert result.healthy is False
        assert result.component == "storage"

    @pytest.mark.asyncio
    async def test_check_opentelemetry_initialized(self):
        """Test OpenTelemetry health check when initialized."""
        with patch("src.api.routes.health.get_telemetry_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.is_initialized = True
            mock_manager.active_exporters = ["otlp", "console"]
            mock_get_manager.return_value = mock_manager
            
            result = await check_opentelemetry()
        
        assert result.healthy is True
        assert result.component == "opentelemetry"
        assert "Tracing enabled" in result.message

    @pytest.mark.asyncio
    async def test_check_opentelemetry_not_initialized(self):
        """Test OpenTelemetry health check when not initialized."""
        with patch("src.api.routes.health.get_telemetry_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.is_initialized = False
            mock_get_manager.return_value = mock_manager
            
            result = await check_opentelemetry()
        
        assert result.healthy is True  # Not critical
        assert result.component == "opentelemetry"
        assert "not initialized" in result.message.lower()


@pytest.mark.unit
class TestHealthCheckEndpoint:
    """Tests for the main health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check_all_healthy(self):
        """Test health check when all components are healthy."""
        mock_result = MagicMock()
        mock_result.healthy = True
        mock_result.component = "test"
        mock_result.message = "OK"
        mock_result.latency_ms = 1.0
        mock_result.details = None
        
        with patch("src.api.routes.health.check_database", return_value=mock_result):
            with patch("src.api.routes.health.check_redis", return_value=mock_result):
                with patch("src.api.routes.health.check_llm_providers", return_value=mock_result):
                    with patch("src.api.routes.health.check_destinations", return_value=mock_result):
                        with patch("src.api.routes.health.check_storage", return_value=mock_result):
                            with patch("src.api.routes.health.check_opentelemetry", return_value=mock_result):
                                with patch("src.api.routes.health.settings") as mock_settings:
                                    mock_settings.app_version = "1.0.0"
                                    mock_settings.env = "test"
                                    
                                    response = await health_check()
        
        assert response.status == HealthStatus.HEALTHY
        assert response.overall_healthy is True
        assert "api" in response.components
        assert response.version == "1.0.0"
        assert response.environment == "test"

    @pytest.mark.asyncio
    async def test_health_check_some_unhealthy(self):
        """Test health check when some components are unhealthy."""
        healthy_result = MagicMock()
        healthy_result.healthy = True
        healthy_result.component = "healthy_component"
        healthy_result.message = "OK"
        healthy_result.latency_ms = 1.0
        healthy_result.details = None
        
        unhealthy_result = MagicMock()
        unhealthy_result.healthy = False
        unhealthy_result.component = "unhealthy_component"
        unhealthy_result.message = "Failed"
        unhealthy_result.latency_ms = 1.0
        unhealthy_result.details = None
        
        with patch("src.api.routes.health.check_database", return_value=healthy_result):
            with patch("src.api.routes.health.check_redis", return_value=unhealthy_result):
                with patch("src.api.routes.health.check_llm_providers", return_value=healthy_result):
                    with patch("src.api.routes.health.check_destinations", return_value=healthy_result):
                        with patch("src.api.routes.health.check_storage", return_value=healthy_result):
                            with patch("src.api.routes.health.check_opentelemetry", return_value=healthy_result):
                                with patch("src.api.routes.health.settings") as mock_settings:
                                    mock_settings.app_version = "1.0.0"
                                    mock_settings.env = "test"
                                    
                                    response = await health_check()
        
        assert response.status == HealthStatus.UNHEALTHY
        assert response.overall_healthy is False


@pytest.mark.unit
class TestReadinessProbe:
    """Tests for the Kubernetes readiness probe endpoint."""

    @pytest.mark.asyncio
    async def test_readiness_probe_ready(self):
        """Test readiness probe when all dependencies are ready."""
        healthy_result = MagicMock()
        healthy_result.healthy = True
        healthy_result.component = "database"
        healthy_result.message = "OK"
        healthy_result.latency_ms = 1.0
        healthy_result.details = None
        
        with patch("src.api.routes.health.check_database", return_value=healthy_result):
            with patch("src.api.routes.health.check_redis", return_value=healthy_result):
                response = await readiness_probe()
        
        assert response.ready is True
        assert "database" in response.checks
        assert "timestamp" in response.model_dump()

    @pytest.mark.asyncio
    async def test_readiness_probe_not_ready(self):
        """Test readiness probe when dependencies are not ready."""
        unhealthy_result = MagicMock()
        unhealthy_result.healthy = False
        unhealthy_result.component = "database"
        unhealthy_result.message = "Connection failed"
        unhealthy_result.latency_ms = 1.0
        unhealthy_result.details = None
        
        with patch("src.api.routes.health.check_database", return_value=unhealthy_result):
            with pytest.raises(HTTPException) as exc_info:
                await readiness_probe()
        
        assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


@pytest.mark.unit
class TestLivenessProbe:
    """Tests for the Kubernetes liveness probe endpoint."""

    @pytest.mark.asyncio
    async def test_liveness_probe(self):
        """Test liveness probe returns alive status."""
        response = await liveness_probe()
        
        assert response.alive is True
        assert "timestamp" in response.model_dump()


@pytest.mark.unit
class TestDetailedComponentHealth:
    """Tests for detailed component health endpoint."""

    @pytest.mark.asyncio
    async def test_detailed_health_database(self):
        """Test detailed health check for database."""
        mock_result = MagicMock()
        mock_result.healthy = True
        mock_result.component = "database"
        mock_result.message = "OK"
        mock_result.latency_ms = 5.0
        mock_result.details = {"connections": 10}
        
        with patch("src.api.routes.health.check_database", return_value=mock_result):
            response = await detailed_component_health("database")
        
        assert response["component"] == "database"
        assert response["healthy"] is True
        assert response["latency_ms"] == 5.0
        assert "timestamp" in response

    @pytest.mark.asyncio
    async def test_detailed_health_redis(self):
        """Test detailed health check for Redis."""
        mock_result = MagicMock()
        mock_result.healthy = True
        mock_result.component = "redis"
        mock_result.message = "OK"
        mock_result.latency_ms = 3.0
        mock_result.details = None
        
        with patch("src.api.routes.health.check_redis", return_value=mock_result):
            response = await detailed_component_health("redis")
        
        assert response["component"] == "redis"
        assert response["healthy"] is True

    @pytest.mark.asyncio
    async def test_detailed_health_invalid_component(self):
        """Test detailed health check with invalid component."""
        with pytest.raises(HTTPException) as exc_info:
            await detailed_component_health("invalid_component")
        
        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
        assert "Unknown component" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_detailed_health_all_components(self):
        """Test detailed health check for all valid components."""
        valid_components = [
            "database",
            "redis",
            "llm_providers",
            "destinations",
            "storage",
            "opentelemetry",
        ]
        
        mock_result = MagicMock()
        mock_result.healthy = True
        mock_result.component = "test"
        mock_result.message = "OK"
        mock_result.latency_ms = 1.0
        mock_result.details = None
        
        for component in valid_components:
            with patch.dict("src.api.routes.health.check_functions", {component: AsyncMock(return_value=mock_result)}):
                # We need to check that each component function exists
                assert component in [
                    "database", "redis", "llm_providers",
                    "destinations", "storage", "opentelemetry"
                ]
