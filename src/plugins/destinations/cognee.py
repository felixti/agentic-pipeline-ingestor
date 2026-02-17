"""Cognee destination plugin for processed document output.

This module provides the Cognee API integration for storing processed
documents, chunks, and metadata in the Cognee knowledge graph system.
"""

import json
import logging
import os
from typing import Any
from uuid import UUID

import httpx

from src.plugins.base import (
    Connection,
    DestinationPlugin,
    HealthStatus,
    PluginMetadata,
    PluginType,
    TransformedData,
    ValidationResult,
    WriteResult,
)

logger = logging.getLogger(__name__)


class CogneeDestination(DestinationPlugin):
    """Cognee destination plugin for knowledge graph storage.
    
    Cognee is the primary destination for processed documents,
    providing graph-based storage with semantic search capabilities.
    
    Example:
        >>> destination = CogneeDestination()
        >>> await destination.initialize({
        ...     "api_url": "https://api.cognee.example.com",
        ...     "api_key": "my-api-key"
        ... })
        >>> conn = await destination.connect({"dataset_id": "my-dataset"})
        >>> result = await destination.write(conn, transformed_data)
    """

    def __init__(self) -> None:
        """Initialize the Cognee destination."""
        self._api_url: str | None = None
        self._api_key: str | None = None
        self._config: dict[str, Any] = {}
        self._client: httpx.AsyncClient | None = None

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            id="cognee",
            name="Cognee Knowledge Graph",
            version="1.0.0",
            type=PluginType.DESTINATION,
            description="Primary destination for processed documents in Cognee knowledge graph",
            author="Pipeline Team",
            supported_formats=["json", "markdown", "text"],
            requires_auth=True,
            config_schema={
                "type": "object",
                "properties": {
                    "api_url": {
                        "type": "string",
                        "description": "Cognee API base URL",
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Cognee API key",
                    },
                    "dataset_id": {
                        "type": "string",
                        "description": "Default dataset ID",
                    },
                    "timeout": {
                        "type": "integer",
                        "default": 60,
                        "description": "API request timeout in seconds",
                    },
                },
                "required": ["api_url"],
            },
        )

    async def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the destination with configuration.
        
        Args:
            config: Destination configuration with:
                - api_url: Cognee API base URL
                - api_key: Cognee API key
                - timeout: Request timeout (default: 60)
        """
        self._config = config
        self._api_url = config.get("api_url") or os.getenv("COGNEE_API_URL")
        self._api_key = config.get("api_key") or os.getenv("COGNEE_API_KEY")

        timeout = config.get("timeout", 60)

        if not self._api_url:
            logger.warning(
                "Cognee API URL not configured. "
                "Set COGNEE_API_URL environment variable or pass in config."
            )

        # Create HTTP client
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        self._client = httpx.AsyncClient(
            base_url=self._api_url.rstrip("/") if self._api_url else "",
            headers=headers,
            timeout=timeout,
        )

        logger.info("Cognee destination initialized")

    async def connect(self, config: dict[str, Any]) -> Connection:
        """Establish a connection to Cognee.
        
        Args:
            config: Connection configuration with:
                - dataset_id: Target dataset ID
                - graph_name: Optional graph name
                
        Returns:
            Connection handle
        """
        dataset_id = config.get("dataset_id", "default")

        # Verify dataset exists or create it
        await self._ensure_dataset(dataset_id)

        return Connection(
            id=UUID(int=hash(dataset_id) % (2**32)),
            plugin_id="cognee",
            config={
                "dataset_id": dataset_id,
                "graph_name": config.get("graph_name", "default"),
            },
        )

    async def write(
        self,
        conn: Connection,
        data: TransformedData,
    ) -> WriteResult:
        """Write transformed data to Cognee.
        
        Args:
            conn: Connection handle
            data: Transformed data to write
            
        Returns:
            WriteResult with operation status
        """
        import time

        start_time = time.time()

        if not self._client:
            return WriteResult(
                success=False,
                error="Cognee client not initialized",
            )

        dataset_id = conn.config.get("dataset_id", "default")

        try:
            # Build document payload
            payload = self._build_payload(data, conn.config)

            # Send to Cognee API
            response = await self._client.post(
                f"/v1/datasets/{dataset_id}/documents",
                json=payload,
            )

            response.raise_for_status()
            result = response.json()

            processing_time = int((time.time() - start_time) * 1000)

            return WriteResult(
                success=True,
                destination_id="cognee",
                destination_uri=f"/datasets/{dataset_id}/documents/{result.get('id', 'unknown')}",
                records_written=len(data.chunks),
                bytes_written=len(json.dumps(payload)),
                processing_time_ms=processing_time,
                metadata={
                    "document_id": result.get("id"),
                    "dataset_id": dataset_id,
                    "chunks_count": len(data.chunks),
                },
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"Cognee API error: {e.response.status_code} - {e.response.text}")
            return WriteResult(
                success=False,
                error=f"Cognee API error: {e.response.status_code}",
            )
        except Exception as e:
            logger.error(f"Failed to write to Cognee: {e}")
            return WriteResult(
                success=False,
                error=f"Write failed: {e!s}",
            )

    def _build_payload(
        self,
        data: TransformedData,
        conn_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the API payload for Cognee.
        
        Args:
            data: Transformed data
            conn_config: Connection configuration
            
        Returns:
            API payload dictionary
        """
        payload: dict[str, Any] = {
            "job_id": str(data.job_id),
            "original_format": data.original_format,
            "output_format": data.output_format,
            "metadata": data.metadata,
            "lineage": data.lineage,
        }

        # Add chunks
        if data.chunks:
            payload["chunks"] = [
                {
                    "index": i,
                    "content": chunk.get("content", ""),
                    "metadata": chunk.get("metadata", {}),
                }
                for i, chunk in enumerate(data.chunks)
            ]

        # Add embeddings if present
        if data.embeddings:
            payload["embeddings"] = data.embeddings

        return payload

    async def _ensure_dataset(self, dataset_id: str) -> None:
        """Ensure the dataset exists in Cognee.
        
        Args:
            dataset_id: Dataset ID to ensure
        """
        if not self._client:
            return

        try:
            # Check if dataset exists
            response = await self._client.get(f"/v1/datasets/{dataset_id}")

            if response.status_code == 404:
                # Create dataset
                await self._client.post(
                    "/v1/datasets",
                    json={
                        "id": dataset_id,
                        "name": dataset_id,
                        "description": f"Dataset created by pipeline for {dataset_id}",
                    },
                )
                logger.info(f"Created Cognee dataset: {dataset_id}")

        except Exception as e:
            logger.warning(f"Failed to ensure dataset exists: {e}")

    async def validate_config(self, config: dict[str, Any]) -> ValidationResult:
        """Validate destination configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            ValidationResult with validation status
        """
        errors: list[str] = []
        warnings: list[str] = []

        api_url = config.get("api_url") or os.getenv("COGNEE_API_URL")
        api_key = config.get("api_key") or os.getenv("COGNEE_API_KEY")

        if not api_url:
            errors.append("api_url is required (or set COGNEE_API_URL)")
        elif not api_url.startswith(("http://", "https://")):
            errors.append("api_url must be a valid HTTP(S) URL")

        if not api_key:
            warnings.append("api_key not provided - authentication may fail")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    async def health_check(self, config: dict[str, Any] | None = None) -> HealthStatus:
        """Check the health of the destination.
        
        Args:
            config: Optional configuration for health check
            
        Returns:
            HealthStatus indicating destination health
        """
        if not self._client:
            return HealthStatus.UNHEALTHY

        if not self._api_url:
            return HealthStatus.UNHEALTHY

        try:
            # Simple health check - try to list datasets
            response = await self._client.get("/v1/health", timeout=10.0)

            if response.status_code == 200:
                return HealthStatus.HEALTHY
            elif response.status_code < 500:
                return HealthStatus.DEGRADED
            else:
                return HealthStatus.UNHEALTHY

        except httpx.TimeoutException:
            return HealthStatus.DEGRADED
        except Exception as e:
            logger.warning(f"Cognee health check failed: {e}")
            return HealthStatus.UNHEALTHY

    async def shutdown(self) -> None:
        """Shutdown the destination and cleanup resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("Cognee destination shutdown")


class CogneeMockDestination(DestinationPlugin):
    """Mock Cognee destination for testing without actual Cognee server.
    
    This implementation stores data in memory for testing purposes.
    """

    def __init__(self) -> None:
        """Initialize the mock destination."""
        self._storage: dict[str, list[dict[str, Any]]] = {}
        self._config: dict[str, Any] = {}

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            id="cognee_mock",
            name="Cognee Mock (Testing)",
            version="1.0.0",
            type=PluginType.DESTINATION,
            description="Mock Cognee destination for testing",
            author="Pipeline Team",
            supported_formats=["json"],
            requires_auth=False,
        )

    async def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the mock destination."""
        self._config = config
        logger.info("Cognee mock destination initialized")

    async def connect(self, config: dict[str, Any]) -> Connection:
        """Create a mock connection."""
        dataset_id = config.get("dataset_id", "default")

        if dataset_id not in self._storage:
            self._storage[dataset_id] = []

        return Connection(
            id=UUID(int=hash(dataset_id) % (2**32)),
            plugin_id="cognee_mock",
            config={"dataset_id": dataset_id},
        )

    async def write(
        self,
        conn: Connection,
        data: TransformedData,
    ) -> WriteResult:
        """Write data to mock storage."""
        import time

        start_time = time.time()
        dataset_id = conn.config.get("dataset_id", "default")

        # Store the data
        payload = {
            "job_id": str(data.job_id),
            "chunks": data.chunks,
            "metadata": data.metadata,
            "timestamp": time.time(),
        }

        self._storage[dataset_id].append(payload)

        processing_time = int((time.time() - start_time) * 1000)

        return WriteResult(
            success=True,
            destination_id="cognee_mock",
            destination_uri=f"/mock/datasets/{dataset_id}",
            records_written=len(data.chunks),
            bytes_written=len(str(payload)),
            processing_time_ms=processing_time,
        )

    async def health_check(self, config: dict[str, Any] | None = None) -> HealthStatus:
        """Always healthy for mock."""
        return HealthStatus.HEALTHY

    def get_stored_data(self, dataset_id: str) -> list[dict[str, Any]]:
        """Get stored data for testing.
        
        Args:
            dataset_id: Dataset ID
            
        Returns:
            List of stored documents
        """
        return self._storage.get(dataset_id, [])

    def clear_storage(self) -> None:
        """Clear all stored data."""
        self._storage.clear()
