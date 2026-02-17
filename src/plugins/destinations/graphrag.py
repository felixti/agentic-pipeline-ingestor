"""Microsoft GraphRAG destination plugin for knowledge graph construction.

This module provides integration with Microsoft's GraphRAG (Graph Retrieval-Augmented Generation)
for building and querying knowledge graphs from processed documents.
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


class GraphRAGDestination(DestinationPlugin):
    """GraphRAG destination plugin for knowledge graph construction.
    
    GraphRAG builds knowledge graphs using:
    - Entity extraction and linking
    - Relationship extraction
    - Community detection
    - Graph indexing for RAG
    
    Example:
        >>> destination = GraphRAGDestination()
        >>> await destination.initialize({
        ...     "api_url": "https://api.graphrag.example.com",
        ...     "api_key": "my-api-key"
        ... })
        >>> conn = await destination.connect({"graph_id": "my-graph"})
        >>> result = await destination.write(conn, transformed_data)
    """

    def __init__(self) -> None:
        """Initialize the GraphRAG destination."""
        self._api_url: str | None = None
        self._api_key: str | None = None
        self._config: dict[str, Any] = {}
        self._client: httpx.AsyncClient | None = None

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            id="graphrag",
            name="Microsoft GraphRAG",
            version="1.0.0",
            type=PluginType.DESTINATION,
            description="Microsoft GraphRAG integration for knowledge graph construction and RAG",
            author="Pipeline Team",
            supported_formats=["json", "text", "markdown"],
            requires_auth=True,
            config_schema={
                "type": "object",
                "properties": {
                    "api_url": {
                        "type": "string",
                        "description": "GraphRAG API base URL",
                    },
                    "api_key": {
                        "type": "string",
                        "description": "GraphRAG API key",
                    },
                    "timeout": {
                        "type": "integer",
                        "default": 120,
                        "description": "API request timeout in seconds",
                    },
                    "auto_extract_entities": {
                        "type": "boolean",
                        "default": True,
                        "description": "Automatically extract entities from documents",
                    },
                    "community_detection": {
                        "type": "boolean",
                        "default": True,
                        "description": "Enable community detection on graph",
                    },
                },
                "required": ["api_url"],
            },
        )

    async def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the destination with configuration.
        
        Args:
            config: Destination configuration with:
                - api_url: GraphRAG API base URL
                - api_key: GraphRAG API key
                - timeout: Request timeout (default: 120)
                - auto_extract_entities: Enable entity extraction
                - community_detection: Enable community detection
        """
        self._config = config
        self._api_url = config.get("api_url") or os.getenv("GRAPHRAG_API_URL")
        self._api_key = config.get("api_key") or os.getenv("GRAPHRAG_API_KEY")

        timeout = config.get("timeout", 120)

        if not self._api_url:
            logger.warning(
                "GraphRAG API URL not configured. "
                "Set GRAPHRAG_API_URL environment variable or pass in config."
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

        logger.info("GraphRAG destination initialized")

    async def connect(self, config: dict[str, Any]) -> Connection:
        """Establish a connection to GraphRAG.
        
        Args:
            config: Connection configuration with:
                - graph_id: Target graph ID
                - index_name: Optional index name
                
        Returns:
            Connection handle
        """
        graph_id = config.get("graph_id", "default")

        # Ensure graph exists
        await self._ensure_graph(graph_id)

        return Connection(
            id=UUID(int=hash(graph_id) % (2**32)),
            plugin_id="graphrag",
            config={
                "graph_id": graph_id,
                "index_name": config.get("index_name", "default"),
                "auto_extract_entities": config.get(
                    "auto_extract_entities",
                    self._config.get("auto_extract_entities", True)
                ),
                "community_detection": config.get(
                    "community_detection",
                    self._config.get("community_detection", True)
                ),
            },
        )

    async def write(
        self,
        conn: Connection,
        data: TransformedData,
    ) -> WriteResult:
        """Write transformed data to GraphRAG.
        
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
                error="GraphRAG client not initialized",
            )

        graph_id = conn.config.get("graph_id", "default")

        try:
            # Build document payload for GraphRAG
            payload = self._build_payload(data, conn.config)

            # Send to GraphRAG API
            response = await self._client.post(
                f"/v1/graphs/{graph_id}/documents",
                json=payload,
            )

            response.raise_for_status()
            result = response.json()

            # Trigger entity extraction if enabled
            if conn.config.get("auto_extract_entities", True):
                await self._trigger_entity_extraction(graph_id, result.get("document_id"))

            # Trigger community detection if enabled
            if conn.config.get("community_detection", True):
                await self._trigger_community_detection(graph_id)

            processing_time = int((time.time() - start_time) * 1000)

            return WriteResult(
                success=True,
                destination_id="graphrag",
                destination_uri=f"/graphs/{graph_id}/documents/{result.get('document_id', 'unknown')}",
                records_written=len(data.chunks),
                bytes_written=len(json.dumps(payload)),
                processing_time_ms=processing_time,
                metadata={
                    "document_id": result.get("document_id"),
                    "graph_id": graph_id,
                    "chunks_count": len(data.chunks),
                    "entities_extracted": result.get("entities_count"),
                    "relationships_created": result.get("relationships_count"),
                },
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"GraphRAG API error: {e.response.status_code} - {e.response.text}")
            return WriteResult(
                success=False,
                error=f"GraphRAG API error: {e.response.status_code}",
            )
        except Exception as e:
            logger.error(f"Failed to write to GraphRAG: {e}")
            return WriteResult(
                success=False,
                error=f"Write failed: {e!s}",
            )

    def _build_payload(
        self,
        data: TransformedData,
        conn_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the API payload for GraphRAG.
        
        Args:
            data: Transformed data
            conn_config: Connection configuration
            
        Returns:
            API payload dictionary
        """
        payload: dict[str, Any] = {
            "job_id": str(data.job_id),
            "text_chunks": [
                {
                    "index": i,
                    "content": chunk.get("content", ""),
                    "metadata": chunk.get("metadata", {}),
                }
                for i, chunk in enumerate(data.chunks)
            ],
            "metadata": data.metadata,
            "lineage": data.lineage,
            "original_format": data.original_format,
        }

        # Add embeddings if present
        if data.embeddings:
            payload["embeddings"] = data.embeddings

        # Add GraphRAG-specific options
        payload["options"] = {
            "extract_entities": conn_config.get("auto_extract_entities", True),
            "extract_relationships": True,
            "build_communities": conn_config.get("community_detection", True),
        }

        return payload

    async def _ensure_graph(self, graph_id: str) -> None:
        """Ensure the graph exists in GraphRAG.
        
        Args:
            graph_id: Graph ID to ensure
        """
        if not self._client:
            return

        try:
            response = await self._client.get(f"/v1/graphs/{graph_id}")

            if response.status_code == 404:
                # Create graph
                await self._client.post(
                    "/v1/graphs",
                    json={
                        "id": graph_id,
                        "name": graph_id,
                        "description": "Knowledge graph created by pipeline",
                    },
                )
                logger.info(f"Created GraphRAG graph: {graph_id}")

        except Exception as e:
            logger.warning(f"Failed to ensure graph exists: {e}")

    async def _trigger_entity_extraction(self, graph_id: str, document_id: str | None) -> None:
        """Trigger entity extraction for a document.
        
        Args:
            graph_id: Graph ID
            document_id: Document ID
        """
        if not self._client or not document_id:
            return

        try:
            await self._client.post(
                f"/v1/graphs/{graph_id}/extract-entities",
                json={"document_id": document_id},
            )
            logger.debug(f"Triggered entity extraction for {document_id}")
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")

    async def _trigger_community_detection(self, graph_id: str) -> None:
        """Trigger community detection on the graph.
        
        Args:
            graph_id: Graph ID
        """
        if not self._client:
            return

        try:
            await self._client.post(
                f"/v1/graphs/{graph_id}/detect-communities",
                json={},
            )
            logger.debug(f"Triggered community detection for {graph_id}")
        except Exception as e:
            logger.warning(f"Community detection failed: {e}")

    async def search(
        self,
        graph_id: str,
        query: str,
        search_type: str = "hybrid",
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Search the knowledge graph.
        
        Args:
            graph_id: Graph ID to search
            query: Search query
            search_type: Type of search (semantic, graph, hybrid)
            top_k: Number of results to return
            
        Returns:
            Search results
        """
        if not self._client:
            raise RuntimeError("GraphRAG client not initialized")

        try:
            response = await self._client.post(
                f"/v1/graphs/{graph_id}/search",
                json={
                    "query": query,
                    "search_type": search_type,
                    "top_k": top_k,
                },
            )
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error(f"GraphRAG search failed: {e}")
            raise

    async def get_entities(
        self,
        graph_id: str,
        entity_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get entities from the graph.
        
        Args:
            graph_id: Graph ID
            entity_type: Optional filter by entity type
            limit: Maximum number of entities
            
        Returns:
            List of entities
        """
        if not self._client:
            raise RuntimeError("GraphRAG client not initialized")

        try:
            params = {"limit": limit}
            if entity_type:
                params["type"] = entity_type

            response = await self._client.get(
                f"/v1/graphs/{graph_id}/entities",
                params=params,
            )
            response.raise_for_status()
            return response.json().get("entities", [])

        except Exception as e:
            logger.error(f"Failed to get entities: {e}")
            return []

    async def get_relationships(
        self,
        graph_id: str,
        entity_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get relationships from the graph.
        
        Args:
            graph_id: Graph ID
            entity_id: Optional entity ID to filter by
            limit: Maximum number of relationships
            
        Returns:
            List of relationships
        """
        if not self._client:
            raise RuntimeError("GraphRAG client not initialized")

        try:
            params = {"limit": limit}
            if entity_id:
                params["entity_id"] = entity_id

            response = await self._client.get(
                f"/v1/graphs/{graph_id}/relationships",
                params=params,
            )
            response.raise_for_status()
            return response.json().get("relationships", [])

        except Exception as e:
            logger.error(f"Failed to get relationships: {e}")
            return []

    async def validate_config(self, config: dict[str, Any]) -> ValidationResult:
        """Validate destination configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            ValidationResult with validation status
        """
        errors: list[str] = []
        warnings: list[str] = []

        api_url = config.get("api_url") or os.getenv("GRAPHRAG_API_URL")
        api_key = config.get("api_key") or os.getenv("GRAPHRAG_API_KEY")

        if not api_url:
            errors.append("api_url is required (or set GRAPHRAG_API_URL)")
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
            logger.warning(f"GraphRAG health check failed: {e}")
            return HealthStatus.UNHEALTHY

    async def shutdown(self) -> None:
        """Shutdown the destination and cleanup resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("GraphRAG destination shutdown")


class GraphRAGMockDestination(GraphRAGDestination):
    """Mock GraphRAG destination for testing.
    
    This implementation stores data in memory for testing purposes
    without requiring an actual GraphRAG server.
    """

    def __init__(self) -> None:
        """Initialize the mock destination."""
        super().__init__()
        self._storage: dict[str, list[dict[str, Any]]] = {}
        self._entities: dict[str, list[dict[str, Any]]] = {}
        self._relationships: dict[str, list[dict[str, Any]]] = {}

    async def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the mock destination."""
        self._config = config
        logger.info("GraphRAG mock destination initialized")

    async def connect(self, config: dict[str, Any]) -> Connection:
        """Create a mock connection."""
        graph_id = config.get("graph_id", "default")

        if graph_id not in self._storage:
            self._storage[graph_id] = []
            self._entities[graph_id] = []
            self._relationships[graph_id] = []

        return Connection(
            id=UUID(int=hash(graph_id) % (2**32)),
            plugin_id="graphrag_mock",
            config={"graph_id": graph_id},
        )

    async def write(
        self,
        conn: Connection,
        data: TransformedData,
    ) -> WriteResult:
        """Write data to mock storage."""
        import time

        start_time = time.time()
        graph_id = conn.config.get("graph_id", "default")

        document_id = str(UUID(int=hash(str(data.job_id)) % (2**32)))

        # Store the data
        payload = {
            "document_id": document_id,
            "job_id": str(data.job_id),
            "chunks": data.chunks,
            "metadata": data.metadata,
            "timestamp": time.time(),
        }

        self._storage[graph_id].append(payload)

        # Generate mock entities
        mock_entities = [
            {"id": f"ent_{i}", "name": f"Entity {i}", "type": "mock"}
            for i in range(min(len(data.chunks), 5))
        ]
        self._entities[graph_id].extend(mock_entities)

        processing_time = int((time.time() - start_time) * 1000)

        return WriteResult(
            success=True,
            destination_id="graphrag_mock",
            destination_uri=f"/mock/graphs/{graph_id}/documents/{document_id}",
            records_written=len(data.chunks),
            bytes_written=len(str(payload)),
            processing_time_ms=processing_time,
            metadata={
                "document_id": document_id,
                "entities_count": len(mock_entities),
            },
        )

    async def get_entities(
        self,
        graph_id: str,
        entity_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get mock entities."""
        entities = self._entities.get(graph_id, [])
        if entity_type:
            entities = [e for e in entities if e.get("type") == entity_type]
        return entities[:limit]

    async def get_relationships(
        self,
        graph_id: str,
        entity_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get mock relationships."""
        return self._relationships.get(graph_id, [])[:limit]

    def get_stored_data(self, graph_id: str) -> list[dict[str, Any]]:
        """Get stored data for testing."""
        return self._storage.get(graph_id, [])

    def clear_storage(self) -> None:
        """Clear all stored data."""
        self._storage.clear()
        self._entities.clear()
        self._relationships.clear()
