"""Weaviate vector database destination plugin.

This module provides Weaviate integration for storing document embeddings
with semantic search, GraphQL queries, and schema-based storage.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

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


class WeaviateDestination(DestinationPlugin):
    """Weaviate vector database destination plugin.
    
    This plugin stores document chunks with their embeddings in Weaviate
    for semantic search using GraphQL queries and schema-based storage.
    
    Example:
        >>> destination = WeaviateDestination()
        >>> await destination.initialize({
        ...     "url": "https://my-weaviate-instance.weaviate.network",
        ...     "api_key": "your-api-key",
        ...     "class_name": "Document"
        ... })
        >>> conn = await destination.connect({})
        >>> result = await destination.write(conn, transformed_data)
    """
    
    def __init__(self) -> None:
        """Initialize the Weaviate destination."""
        self._url: Optional[str] = None
        self._api_key: Optional[str] = None
        self._class_name: str = "Document"
        self._config: Dict[str, Any] = {}
        self._client: Optional[httpx.AsyncClient] = None
    
    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            id="weaviate",
            name="Weaviate Vector Database",
            version="1.0.0",
            type=PluginType.DESTINATION,
            description="Weaviate vector database with GraphQL and semantic search",
            author="Pipeline Team",
            supported_formats=["json"],
            requires_auth=True,
            config_schema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Weaviate instance URL",
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Weaviate API key",
                    },
                    "class_name": {
                        "type": "string",
                        "default": "Document",
                        "description": "Weaviate class/schema name",
                    },
                    "batch_size": {
                        "type": "integer",
                        "default": 100,
                        "description": "Batch size for imports",
                    },
                    "timeout": {
                        "type": "integer",
                        "default": 60,
                        "description": "Request timeout in seconds",
                    },
                },
                "required": ["url"],
            },
        )
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the destination with configuration.
        
        Args:
            config: Destination configuration with:
                - url: Weaviate instance URL
                - api_key: Weaviate API key
                - class_name: Weaviate class name (default: Document)
                - batch_size: Batch size (default: 100)
                - timeout: Request timeout (default: 60)
        """
        self._config = config
        self._url = config.get("url") or os.getenv("WEAVIATE_URL")
        self._api_key = config.get("api_key") or os.getenv("WEAVIATE_API_KEY")
        self._class_name = config.get("class_name") or os.getenv("WEAVIATE_CLASS_NAME", "Document")
        
        timeout = config.get("timeout", 60)
        
        if not self._url:
            logger.warning(
                "Weaviate URL not configured. "
                "Set WEAVIATE_URL environment variable or pass in config."
            )
        
        # Create HTTP client
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        
        self._client = httpx.AsyncClient(
            base_url=self._url.rstrip("/") if self._url else "",
            headers=headers,
            timeout=timeout,
        )
        
        logger.info("Weaviate destination initialized")
    
    async def connect(self, config: Dict[str, Any]) -> Connection:
        """Establish a connection to Weaviate.
        
        Args:
            config: Connection configuration
            
        Returns:
            Connection handle
        """
        # Ensure schema exists
        await self._ensure_schema()
        
        return Connection(
            id=UUID(int=hash(self._class_name) % (2**32)),
            plugin_id="weaviate",
            config={
                "class_name": self._class_name,
                "batch_size": config.get("batch_size", self._config.get("batch_size", 100)),
            },
        )
    
    async def _ensure_schema(self) -> None:
        """Ensure the Weaviate schema exists."""
        if not self._client:
            return
        
        try:
            # Check if class exists
            response = await self._client.get(f"/v1/schema/{self._class_name}")
            
            if response.status_code == 404:
                # Create class
                schema = {
                    "class": self._class_name,
                    "description": "Document chunks for semantic search",
                    "vectorizer": "none",  # We provide our own vectors
                    "properties": [
                        {
                            "name": "content",
                            "dataType": ["text"],
                            "description": "Chunk content",
                            "moduleConfig": {
                                "text2vec-contextionary": {"skip": True}
                            }
                        },
                        {
                            "name": "job_id",
                            "dataType": ["text"],
                            "description": "Source job ID",
                        },
                        {
                            "name": "chunk_index",
                            "dataType": ["int"],
                            "description": "Chunk index in document",
                        },
                        {
                            "name": "original_format",
                            "dataType": ["text"],
                            "description": "Original document format",
                        },
                        {
                            "name": "metadata",
                            "dataType": ["text"],
                            "description": "JSON metadata",
                        },
                    ],
                }
                
                response = await self._client.post("/v1/schema", json=schema)
                response.raise_for_status()
                logger.info(f"Created Weaviate schema: {self._class_name}")
                
        except Exception as e:
            logger.warning(f"Schema creation warning: {e}")
    
    async def write(
        self,
        conn: Connection,
        data: TransformedData,
    ) -> WriteResult:
        """Write transformed data to Weaviate.
        
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
                error="Weaviate client not initialized",
            )
        
        if not data.embeddings:
            return WriteResult(
                success=False,
                error="No embeddings provided - Weaviate requires embeddings",
            )
        
        if len(data.embeddings) != len(data.chunks):
            return WriteResult(
                success=False,
                error=f"Embeddings count ({len(data.embeddings)}) does not match chunks count ({len(data.chunks)})",
            )
        
        class_name = conn.config.get("class_name", self._class_name)
        batch_size = conn.config.get("batch_size", 100)
        
        try:
            # Import objects
            objects = self._build_objects(data, conn)
            
            # Import in batches
            total_imported = 0
            for i in range(0, len(objects), batch_size):
                batch = objects[i:i + batch_size]
                imported = await self._import_batch(class_name, batch)
                total_imported += imported
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return WriteResult(
                success=True,
                destination_id="weaviate",
                destination_uri=f"weaviate://{class_name}",
                records_written=total_imported,
                bytes_written=len(json.dumps(objects)),
                processing_time_ms=processing_time,
                metadata={
                    "class_name": class_name,
                    "objects_imported": total_imported,
                    "dimension": len(data.embeddings[0]) if data.embeddings else 0,
                },
            )
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Weaviate API error: {e.response.status_code} - {e.response.text}")
            return WriteResult(
                success=False,
                error=f"Weaviate API error: {e.response.status_code}",
            )
        except Exception as e:
            logger.error(f"Failed to write to Weaviate: {e}")
            return WriteResult(
                success=False,
                error=f"Write failed: {str(e)}",
            )
    
    def _build_objects(
        self,
        data: TransformedData,
        conn: Connection,
    ) -> List[Dict[str, Any]]:
        """Build Weaviate objects from transformed data.
        
        Args:
            data: Transformed data
            conn: Connection handle
            
        Returns:
            List of Weaviate objects
        """
        objects = []
        
        for i, (chunk, embedding) in enumerate(zip(data.chunks, data.embeddings or [])):
            obj = {
                "id": str(uuid4()),
                "vector": embedding,
                "properties": {
                    "content": chunk.get("content", ""),
                    "job_id": str(data.job_id),
                    "chunk_index": i,
                    "original_format": data.original_format,
                    "metadata": json.dumps(chunk.get("metadata", {})),
                }
            }
            
            objects.append(obj)
        
        return objects
    
    async def _import_batch(
        self,
        class_name: str,
        objects: List[Dict[str, Any]],
    ) -> int:
        """Import a batch of objects to Weaviate.
        
        Args:
            class_name: Weaviate class name
            objects: List of objects to import
            
        Returns:
            Number of objects imported
        """
        if not self._client:
            raise RuntimeError("Weaviate client not initialized")
        
        # Use batch endpoint for efficiency
        batch_objects = []
        for obj in objects:
            batch_objects.append({
                "class": class_name,
                "id": obj["id"],
                "vector": obj["vector"],
                "properties": obj["properties"],
            })
        
        response = await self._client.post(
            "/v1/batch/objects",
            json={"objects": batch_objects},
        )
        response.raise_for_status()
        
        return len(objects)
    
    async def search(
        self,
        class_name: str,
        query: Optional[str] = None,
        vector: Optional[List[float]] = None,
        limit: int = 10,
        certainty: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Search for objects in Weaviate.
        
        Args:
            class_name: Weaviate class name
            query: Text query (optional)
            vector: Vector for similarity search (optional)
            limit: Maximum results
            certainty: Minimum certainty threshold
            
        Returns:
            Search results
        """
        if not self._client:
            raise RuntimeError("Weaviate client not initialized")
        
        # Build GraphQL query
        if vector:
            # Vector search
            graphql_query = {
                "query": f"""
                {{
                    Get {{
                        {class_name}(
                            limit: {limit}
                            nearVector: {{
                                vector: {json.dumps(vector)}
                                certainty: {certainty}
                            }}
                        ) {{
                            content
                            job_id
                            chunk_index
                            original_format
                            metadata
                            _additional {{
                                certainty
                                distance
                            }}
                        }}
                    }}
                }}
                """
            }
        elif query:
            # BM25 text search
            graphql_query = {
                "query": f"""
                {{
                    Get {{
                        {class_name}(
                            limit: {limit}
                            bm25: {{
                                query: "{query}"
                            }}
                        ) {{
                            content
                            job_id
                            chunk_index
                            original_format
                            metadata
                            _additional {{
                                score
                            }}
                        }}
                    }}
                }}
                """
            }
        else:
            raise ValueError("Either query or vector must be provided")
        
        response = await self._client.post("/v1/graphql", json=graphql_query)
        response.raise_for_status()
        
        result = response.json()
        data = result.get("data", {}).get("Get", {}).get(class_name, [])
        
        return data
    
    async def delete(
        self,
        class_name: str,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Delete objects from Weaviate.
        
        Args:
            class_name: Weaviate class name
            where_filter: Optional filter for deletion
            
        Returns:
            True if successful
        """
        if not self._client:
            raise RuntimeError("Weaviate client not initialized")
        
        # Build GraphQL delete query
        if where_filter:
            graphql_query = {
                "query": f"""
                mutation {{
                    Delete {{
                        {class_name}(
                            where: {json.dumps(where_filter)}
                        ) {{
                            objects
                        }}
                    }}
                }}
                """
            }
        else:
            # Delete all - use match all filter
            graphql_query = {
                "query": f"""
                mutation {{
                    Delete {{
                        {class_name}(
                            where: {{
                                path: ["job_id"]
                                operator: Like
                                valueText: "*"
                            }}
                        ) {{
                            objects
                        }}
                    }}
                }}
                """
            }
        
        response = await self._client.post("/v1/graphql", json=graphql_query)
        response.raise_for_status()
        
        return True
    
    async def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate destination configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            ValidationResult with validation status
        """
        errors: List[str] = []
        warnings: List[str] = []
        
        url = config.get("url") or os.getenv("WEAVIATE_URL")
        
        if not url:
            errors.append("url is required (or set WEAVIATE_URL)")
        elif not url.startswith(("http://", "https://")):
            errors.append("url must be a valid HTTP(S) URL")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )
    
    async def health_check(self, config: Optional[Dict[str, Any]] = None) -> HealthStatus:
        """Check the health of the destination.
        
        Args:
            config: Optional configuration for health check
            
        Returns:
            HealthStatus indicating destination health
        """
        if not self._client or not self._url:
            return HealthStatus.UNHEALTHY
        
        try:
            response = await self._client.get("/v1/.well-known/live", timeout=10.0)
            
            if response.status_code == 200:
                return HealthStatus.HEALTHY
            elif response.status_code < 500:
                return HealthStatus.DEGRADED
            else:
                return HealthStatus.UNHEALTHY
                
        except httpx.TimeoutException:
            return HealthStatus.DEGRADED
        except Exception as e:
            logger.warning(f"Weaviate health check failed: {e}")
            return HealthStatus.UNHEALTHY
    
    async def shutdown(self) -> None:
        """Shutdown the destination and cleanup resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("Weaviate destination shutdown")


class WeaviateMockDestination(WeaviateDestination):
    """Mock Weaviate destination for testing."""
    
    def __init__(self) -> None:
        """Initialize the mock destination."""
        super().__init__()
        self._storage: Dict[str, List[Dict[str, Any]]] = {}
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the mock destination."""
        self._config = config
        self._class_name = config.get("class_name", "Document")
        logger.info("Weaviate mock destination initialized")
    
    async def connect(self, config: Dict[str, Any]) -> Connection:
        """Create a mock connection."""
        if self._class_name not in self._storage:
            self._storage[self._class_name] = []
        
        return Connection(
            id=UUID(int=hash(self._class_name) % (2**32)),
            plugin_id="weaviate_mock",
            config={"class_name": self._class_name},
        )
    
    async def write(
        self,
        conn: Connection,
        data: TransformedData,
    ) -> WriteResult:
        """Write data to mock storage."""
        import time
        
        start_time = time.time()
        
        if not data.embeddings:
            return WriteResult(
                success=False,
                error="No embeddings provided",
            )
        
        objects = self._build_objects(data, conn)
        
        for obj in objects:
            self._storage[self._class_name].append(obj)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return WriteResult(
            success=True,
            destination_id="weaviate_mock",
            destination_uri=f"mock://weaviate/{self._class_name}",
            records_written=len(objects),
            bytes_written=len(str(objects)),
            processing_time_ms=processing_time,
            metadata={
                "class_name": self._class_name,
                "objects_imported": len(objects),
            },
        )
    
    async def search(
        self,
        class_name: str,
        query: Optional[str] = None,
        vector: Optional[List[float]] = None,
        limit: int = 10,
        certainty: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Mock search."""
        objects = self._storage.get(class_name, [])[:limit]
        
        return [
            {
                "content": obj["properties"].get("content", ""),
                "job_id": obj["properties"].get("job_id", ""),
                "chunk_index": obj["properties"].get("chunk_index", 0),
                "original_format": obj["properties"].get("original_format", ""),
                "metadata": obj["properties"].get("metadata", ""),
                "_additional": {"certainty": 0.95},
            }
            for obj in objects
        ]
    
    def get_stored_objects(self, class_name: str) -> List[Dict[str, Any]]:
        """Get stored objects for testing."""
        return self._storage.get(class_name, [])
    
    def clear_storage(self) -> None:
        """Clear all stored data."""
        self._storage.clear()
