"""Pinecone vector database destination plugin.

This module provides Pinecone integration for storing document embeddings
with semantic search capabilities.
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


class PineconeDestination(DestinationPlugin):
    """Pinecone vector database destination plugin.
    
    This plugin stores document chunks with their embeddings in Pinecone
    for semantic search and similarity matching.
    
    Example:
        >>> destination = PineconeDestination()
        >>> await destination.initialize({
        ...     "api_key": "your-api-key",
        ...     "environment": "us-east1-gcp",
        ...     "index_name": "documents"
        ... })
        >>> conn = await destination.connect({"namespace": "default"})
        >>> result = await destination.write(conn, transformed_data)
    """

    def __init__(self) -> None:
        """Initialize the Pinecone destination."""
        self._api_key: str | None = None
        self._environment: str | None = None
        self._index_name: str | None = None
        self._config: dict[str, Any] = {}
        self._client: httpx.AsyncClient | None = None
        self._base_url: str | None = None

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            id="pinecone",
            name="Pinecone Vector Database",
            version="1.0.0",
            type=PluginType.DESTINATION,
            description="Pinecone vector database for semantic search and embeddings",
            author="Pipeline Team",
            supported_formats=["json"],
            requires_auth=True,
            config_schema={
                "type": "object",
                "properties": {
                    "api_key": {
                        "type": "string",
                        "description": "Pinecone API key",
                    },
                    "environment": {
                        "type": "string",
                        "description": "Pinecone environment (e.g., us-east1-gcp)",
                    },
                    "index_name": {
                        "type": "string",
                        "description": "Pinecone index name",
                    },
                    "batch_size": {
                        "type": "integer",
                        "default": 100,
                        "description": "Batch size for upserts",
                    },
                    "timeout": {
                        "type": "integer",
                        "default": 60,
                        "description": "Request timeout in seconds",
                    },
                },
                "required": ["api_key", "index_name"],
            },
        )

    async def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the destination with configuration.
        
        Args:
            config: Destination configuration with:
                - api_key: Pinecone API key
                - environment: Pinecone environment
                - index_name: Pinecone index name
                - batch_size: Batch size for upserts (default: 100)
                - timeout: Request timeout (default: 60)
        """
        self._config = config
        self._api_key = config.get("api_key") or os.getenv("PINECONE_API_KEY")
        self._environment = config.get("environment") or os.getenv("PINECONE_ENVIRONMENT", "us-east1-gcp")
        self._index_name = config.get("index_name") or os.getenv("PINECONE_INDEX_NAME")

        timeout = config.get("timeout", 60)

        if not self._api_key:
            logger.warning(
                "Pinecone API key not configured. "
                "Set PINECONE_API_KEY environment variable or pass in config."
            )

        if not self._index_name:
            logger.warning(
                "Pinecone index name not configured. "
                "Set PINECONE_INDEX_NAME environment variable or pass in config."
            )

        # Build base URL
        if self._index_name:
            self._base_url = f"https://{self._index_name}.svc.{self._environment}.pinecone.io"

        # Create HTTP client
        headers: dict[str, str] = {
            "Api-Key": self._api_key or "",
            "Content-Type": "application/json",
        }

        self._client = httpx.AsyncClient(
            base_url=self._base_url or "",
            headers=headers,
            timeout=timeout,
        )

        logger.info("Pinecone destination initialized")

    async def connect(self, config: dict[str, Any]) -> Connection:
        """Establish a connection to Pinecone.
        
        Args:
            config: Connection configuration with:
                - namespace: Pinecone namespace
                
        Returns:
            Connection handle
        """
        namespace = config.get("namespace", "default")

        # Verify index exists
        await self._verify_index()

        return Connection(
            id=UUID(int=hash(namespace) % (2**32)),
            plugin_id="pinecone",
            config={
                "namespace": namespace,
                "batch_size": config.get("batch_size", self._config.get("batch_size", 100)),
            },
        )

    async def _verify_index(self) -> None:
        """Verify the Pinecone index exists."""
        if not self._client or not self._api_key:
            return

        try:
            response = await self._client.get("/describe_index_stats")
            response.raise_for_status()
            logger.info(f"Pinecone index verified: {self._index_name}")
        except Exception as e:
            logger.warning(f"Failed to verify Pinecone index: {e}")

    async def write(
        self,
        conn: Connection,
        data: TransformedData,
    ) -> WriteResult:
        """Write transformed data to Pinecone.
        
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
                error="Pinecone client not initialized",
            )

        if not data.embeddings:
            return WriteResult(
                success=False,
                error="No embeddings provided - Pinecone requires embeddings",
            )

        if len(data.embeddings) != len(data.chunks):
            return WriteResult(
                success=False,
                error=f"Embeddings count ({len(data.embeddings)}) does not match chunks count ({len(data.chunks)})",
            )

        namespace = conn.config.get("namespace", "default")
        batch_size = conn.config.get("batch_size", 100)

        try:
            # Build vectors
            vectors = self._build_vectors(data, conn)

            # Upsert in batches
            total_upserted = 0
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                upserted = await self._upsert_batch(namespace, batch)
                total_upserted += upserted

            processing_time = int((time.time() - start_time) * 1000)

            return WriteResult(
                success=True,
                destination_id="pinecone",
                destination_uri=f"pinecone://{self._index_name}/{namespace}",
                records_written=total_upserted,
                bytes_written=len(json.dumps(vectors)),
                processing_time_ms=processing_time,
                metadata={
                    "index_name": self._index_name,
                    "namespace": namespace,
                    "vectors_upserted": total_upserted,
                    "dimension": len(data.embeddings[0]) if data.embeddings else 0,
                },
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"Pinecone API error: {e.response.status_code} - {e.response.text}")
            return WriteResult(
                success=False,
                error=f"Pinecone API error: {e.response.status_code}",
            )
        except Exception as e:
            logger.error(f"Failed to write to Pinecone: {e}")
            return WriteResult(
                success=False,
                error=f"Write failed: {e!s}",
            )

    def _build_vectors(
        self,
        data: TransformedData,
        conn: Connection,
    ) -> list[dict[str, Any]]:
        """Build vector records for Pinecone.
        
        Args:
            data: Transformed data
            conn: Connection handle
            
        Returns:
            List of vector records
        """
        vectors = []

        for i, (chunk, embedding) in enumerate(zip(data.chunks, data.embeddings or [])):
            vector_id = f"{data.job_id}_chunk_{i}"

            vector = {
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "job_id": str(data.job_id),
                    "chunk_index": i,
                    "content": chunk.get("content", "")[:1000],  # Truncate for metadata
                    "original_format": data.original_format,
                    **{k: v for k, v in chunk.get("metadata", {}).items()
                       if isinstance(v, (str, int, float, bool))}
                }
            }

            vectors.append(vector)

        return vectors

    async def _upsert_batch(
        self,
        namespace: str,
        vectors: list[dict[str, Any]],
    ) -> int:
        """Upsert a batch of vectors to Pinecone.
        
        Args:
            namespace: Pinecone namespace
            vectors: List of vector records
            
        Returns:
            Number of vectors upserted
        """
        if not self._client:
            raise RuntimeError("Pinecone client not initialized")

        response = await self._client.post(
            "/vectors/upsert",
            json={
                "vectors": vectors,
                "namespace": namespace,
            },
        )
        response.raise_for_status()

        return len(vectors)

    async def query(
        self,
        namespace: str,
        vector: list[float],
        top_k: int = 10,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Query vectors by similarity.
        
        Args:
            namespace: Pinecone namespace
            vector: Query vector
            top_k: Number of results
            filter_dict: Optional metadata filter
            
        Returns:
            Query results
        """
        if not self._client:
            raise RuntimeError("Pinecone client not initialized")

        payload = {
            "vector": vector,
            "topK": top_k,
            "namespace": namespace,
            "includeMetadata": True,
        }

        if filter_dict:
            payload["filter"] = filter_dict

        response = await self._client.post("/query", json=payload)
        response.raise_for_status()

        result = response.json()
        return result.get("matches", [])

    async def delete(
        self,
        namespace: str,
        ids: list[str] | None = None,
        filter_dict: dict[str, Any] | None = None,
        delete_all: bool = False,
    ) -> bool:
        """Delete vectors from Pinecone.
        
        Args:
            namespace: Pinecone namespace
            ids: Optional list of vector IDs to delete
            filter_dict: Optional metadata filter
            delete_all: Delete all vectors in namespace
            
        Returns:
            True if successful
        """
        if not self._client:
            raise RuntimeError("Pinecone client not initialized")

        payload = {"namespace": namespace}

        if delete_all:
            payload["deleteAll"] = True
        elif ids:
            payload["ids"] = ids
        elif filter_dict:
            payload["filter"] = filter_dict

        response = await self._client.post("/vectors/delete", json=payload)
        response.raise_for_status()

        return True

    async def validate_config(self, config: dict[str, Any]) -> ValidationResult:
        """Validate destination configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            ValidationResult with validation status
        """
        errors: list[str] = []
        warnings: list[str] = []

        api_key = config.get("api_key") or os.getenv("PINECONE_API_KEY")
        index_name = config.get("index_name") or os.getenv("PINECONE_INDEX_NAME")

        if not api_key:
            errors.append("api_key is required (or set PINECONE_API_KEY)")

        if not index_name:
            errors.append("index_name is required (or set PINECONE_INDEX_NAME)")

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
        if not self._client or not self._api_key:
            return HealthStatus.UNHEALTHY

        try:
            response = await self._client.get("/describe_index_stats", timeout=10.0)

            if response.status_code == 200:
                return HealthStatus.HEALTHY
            elif response.status_code < 500:
                return HealthStatus.DEGRADED
            else:
                return HealthStatus.UNHEALTHY

        except httpx.TimeoutException:
            return HealthStatus.DEGRADED
        except Exception as e:
            logger.warning(f"Pinecone health check failed: {e}")
            return HealthStatus.UNHEALTHY

    async def shutdown(self) -> None:
        """Shutdown the destination and cleanup resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("Pinecone destination shutdown")


class PineconeMockDestination(PineconeDestination):
    """Mock Pinecone destination for testing."""

    def __init__(self) -> None:
        """Initialize the mock destination."""
        super().__init__()
        self._storage: dict[str, dict[str, Any]] = {}

    async def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the mock destination."""
        self._config = config
        self._index_name = config.get("index_name", "mock-index")
        logger.info("Pinecone mock destination initialized")

    async def connect(self, config: dict[str, Any]) -> Connection:
        """Create a mock connection."""
        namespace = config.get("namespace", "default")

        if namespace not in self._storage:
            self._storage[namespace] = {}

        return Connection(
            id=UUID(int=hash(namespace) % (2**32)),
            plugin_id="pinecone_mock",
            config={"namespace": namespace},
        )

    async def write(
        self,
        conn: Connection,
        data: TransformedData,
    ) -> WriteResult:
        """Write data to mock storage."""
        import time

        start_time = time.time()
        namespace = conn.config.get("namespace", "default")

        if not data.embeddings:
            return WriteResult(
                success=False,
                error="No embeddings provided",
            )

        vectors = self._build_vectors(data, conn)

        for vector in vectors:
            self._storage[namespace][vector["id"]] = vector

        processing_time = int((time.time() - start_time) * 1000)

        return WriteResult(
            success=True,
            destination_id="pinecone_mock",
            destination_uri=f"mock://pinecone/{namespace}",
            records_written=len(vectors),
            bytes_written=len(str(vectors)),
            processing_time_ms=processing_time,
            metadata={
                "namespace": namespace,
                "vectors_upserted": len(vectors),
            },
        )

    async def query(
        self,
        namespace: str,
        vector: list[float],
        top_k: int = 10,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Mock query vectors."""
        vectors = list(self._storage.get(namespace, {}).values())

        # Mock similarity calculation - just return first top_k
        matches = [
            {
                "id": v["id"],
                "score": 0.95 - (i * 0.01),
                "metadata": v.get("metadata", {}),
            }
            for i, v in enumerate(vectors[:top_k])
        ]

        return matches

    def get_stored_vectors(self, namespace: str) -> dict[str, Any]:
        """Get stored vectors for testing."""
        return self._storage.get(namespace, {})

    def clear_storage(self) -> None:
        """Clear all stored data."""
        self._storage.clear()
