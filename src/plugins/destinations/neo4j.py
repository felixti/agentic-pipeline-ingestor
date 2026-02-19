"""Neo4j graph database destination plugin.

This module provides Neo4j integration for storing processed documents
as nodes and relationships in a graph database.
"""

import json
import logging
import os
from typing import Any
from uuid import UUID

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


class Neo4jDestination(DestinationPlugin):
    """Neo4j destination plugin for graph database storage.
    
    This plugin stores processed documents as nodes in Neo4j with:
    - Document nodes with metadata
    - Chunk nodes linked to documents
    - Entity nodes extracted from content
    - Relationship extraction
    - Cypher query generation
    
    Example:
        >>> destination = Neo4jDestination()
        >>> await destination.initialize({
        ...     "uri": "bolt://localhost:7687",
        ...     "username": "neo4j",
        ...     "password": "password"
        ... })
        >>> conn = await destination.connect({"database": "pipeline"})
        >>> result = await destination.write(conn, transformed_data)
    """

    def __init__(self) -> None:
        """Initialize the Neo4j destination."""
        self._uri: str | None = None
        self._username: str | None = None
        self._password: str | None = None
        self._config: dict[str, Any] = {}
        self._driver: Any | None = None

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            id="neo4j",
            name="Neo4j Graph Database",
            version="1.0.0",
            type=PluginType.DESTINATION,
            description="Neo4j graph database for document and entity storage",
            author="Pipeline Team",
            supported_formats=["json", "text", "markdown"],
            requires_auth=True,
            config_schema={
                "type": "object",
                "properties": {
                    "uri": {
                        "type": "string",
                        "description": "Neo4j connection URI (bolt:// or neo4j://)",
                    },
                    "username": {
                        "type": "string",
                        "description": "Neo4j username",
                    },
                    "password": {
                        "type": "string",
                        "description": "Neo4j password",
                    },
                    "database": {
                        "type": "string",
                        "default": "neo4j",
                        "description": "Database name",
                    },
                    "create_entities": {
                        "type": "boolean",
                        "default": True,
                        "description": "Create entity nodes from content",
                    },
                    "create_relationships": {
                        "type": "boolean",
                        "default": True,
                        "description": "Extract and create relationships",
                    },
                },
                "required": ["uri"],
            },
        )

    async def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the destination with configuration.
        
        Args:
            config: Destination configuration with:
                - uri: Neo4j connection URI
                - username: Neo4j username
                - password: Neo4j password
                - database: Database name (default: neo4j)
                - create_entities: Enable entity creation
                - create_relationships: Enable relationship creation
        """
        self._config = config
        self._uri = config.get("uri") or os.getenv("NEO4J_URI")
        self._username = config.get("username") or os.getenv("NEO4J_USERNAME", "neo4j")
        self._password = config.get("password") or os.getenv("NEO4J_PASSWORD")

        if not self._uri:
            logger.warning(
                "Neo4j URI not configured. "
                "Set NEO4J_URI environment variable or pass in config."
            )

        # Try to import neo4j driver
        try:
            from neo4j import AsyncGraphDatabase
            logger.info("Neo4j driver available")
        except ImportError:
            logger.warning(
                "neo4j driver not installed. "
                "Install with: pip install neo4j"
            )

        logger.info("Neo4j destination initialized")

    async def connect(self, config: dict[str, Any]) -> Connection:
        """Establish a connection to Neo4j.
        
        Args:
            config: Connection configuration with:
                - database: Database name
                
        Returns:
            Connection handle
        """
        from neo4j import AsyncGraphDatabase

        database = config.get("database", self._config.get("database", "neo4j"))

        if not self._uri:
            raise ConnectionError("Neo4j URI not configured")

        # Create driver
        self._driver = AsyncGraphDatabase.driver(
            self._uri,
            auth=(self._username, self._password) if self._password else None,
        )

        # Test connection
        await self._driver.verify_connectivity()

        # Ensure schema exists
        await self._ensure_schema(database)

        return Connection(
            id=UUID(int=hash(database) % (2**32)),
            plugin_id="neo4j",
            config={
                "database": database,
                "create_entities": config.get(
                    "create_entities",
                    self._config.get("create_entities", True)
                ),
                "create_relationships": config.get(
                    "create_relationships",
                    self._config.get("create_relationships", True)
                ),
            },
        )

    async def _ensure_schema(self, database: str) -> None:
        """Ensure the database schema exists.
        
        Args:
            database: Database name
        """
        if not self._driver:
            return

        # Create constraints and indexes
        constraints = [
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE INDEX document_job_id IF NOT EXISTS FOR (d:Document) ON (d.job_id)",
            "CREATE INDEX chunk_document_id IF NOT EXISTS FOR (c:Chunk) ON (c.document_id)",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
        ]

        async with self._driver.session(database=database) as session:
            for constraint in constraints:
                try:
                    await session.run(constraint)
                except Exception as e:
                    logger.warning(f"Schema creation warning: {e}")

    async def write(
        self,
        conn: Connection,
        data: TransformedData,
    ) -> WriteResult:
        """Write transformed data to Neo4j.
        
        Args:
            conn: Connection handle
            data: Transformed data to write
            
        Returns:
            WriteResult with operation status
        """
        import time

        start_time = time.time()

        if not self._driver:
            return WriteResult(
                success=False,
                error="Neo4j driver not initialized",
            )

        database = conn.config.get("database", "neo4j")

        try:
            # Create document node
            document_id = await self._create_document(conn, data, database)

            # Create chunk nodes
            chunk_ids = await self._create_chunks(conn, data, document_id, database)

            # Create entities if enabled
            entity_count = 0
            if conn.config.get("create_entities", True):
                entity_count = await self._create_entities(conn, data, chunk_ids, database)

            # Create relationships if enabled
            rel_count = 0
            if conn.config.get("create_relationships", True):
                rel_count = await self._create_relationships(conn, data, chunk_ids, database)

            processing_time = int((time.time() - start_time) * 1000)

            return WriteResult(
                success=True,
                destination_id="neo4j",
                destination_uri=f"neo4j://{database}/Document/{document_id}",
                records_written=len(data.chunks) + 1,
                bytes_written=len(json.dumps(data.chunks)),
                processing_time_ms=processing_time,
                metadata={
                    "document_id": document_id,
                    "chunks_created": len(chunk_ids),
                    "entities_created": entity_count,
                    "relationships_created": rel_count,
                    "database": database,
                },
            )

        except Exception as e:
            logger.error(f"Failed to write to Neo4j: {e}")
            return WriteResult(
                success=False,
                error=f"Write failed: {e!s}",
            )

    async def _create_document(
        self,
        conn: Connection,
        data: TransformedData,
        database: str,
    ) -> str:
        """Create a document node in Neo4j.
        
        Args:
            conn: Connection handle
            data: Transformed data
            database: Database name
            
        Returns:
            Document node ID
        """
        document_id = str(data.job_id)

        query = """
        MERGE (d:Document {id: $id})
        SET d.job_id = $job_id,
            d.original_format = $original_format,
            d.output_format = $output_format,
            d.chunk_count = $chunk_count,
            d.metadata = $metadata,
            d.lineage = $lineage,
            d.created_at = datetime(),
            d.updated_at = datetime()
        RETURN d.id as id
        """

        params = {
            "id": document_id,
            "job_id": str(data.job_id),
            "original_format": data.original_format,
            "output_format": data.output_format,
            "chunk_count": len(data.chunks),
            "metadata": json.dumps(data.metadata),
            "lineage": json.dumps(data.lineage),
        }

        if self._driver is None:
            raise RuntimeError("Neo4j driver not initialized")
        async with self._driver.session(database=database) as session:
            result = await session.run(query, params)
            record = await result.single()
            return str(record["id"])

    async def _create_chunks(
        self,
        conn: Connection,
        data: TransformedData,
        document_id: str,
        database: str,
    ) -> list[str]:
        """Create chunk nodes linked to the document.
        
        Args:
            conn: Connection handle
            data: Transformed data
            document_id: Parent document ID
            database: Database name
            
        Returns:
            List of chunk IDs
        """
        chunk_ids = []

        query = """
        MATCH (d:Document {id: $document_id})
        MERGE (c:Chunk {id: $chunk_id})
        SET c.index = $index,
            c.content = $content,
            c.metadata = $metadata,
            c.created_at = datetime()
        MERGE (d)-[:HAS_CHUNK {index: $index}]->(c)
        RETURN c.id as id
        """

        if self._driver is None:
            raise RuntimeError("Neo4j driver not initialized")
        async with self._driver.session(database=database) as session:
            for i, chunk in enumerate(data.chunks):
                chunk_id = f"{document_id}_chunk_{i}"
                params = {
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "index": i,
                    "content": chunk.get("content", "")[:10000],  # Limit content size
                    "metadata": json.dumps(chunk.get("metadata", {})),
                }

                result = await session.run(query, params)
                record = await result.single()
                chunk_ids.append(record["id"])

        return chunk_ids

    async def _create_entities(
        self,
        conn: Connection,
        data: TransformedData,
        chunk_ids: list[str],
        database: str,
    ) -> int:
        """Create entity nodes from document content.
        
        Args:
            conn: Connection handle
            data: Transformed data
            chunk_ids: List of chunk IDs
            database: Database name
            
        Returns:
            Number of entities created
        """
        # This is a placeholder - in a real implementation, you would:
        # 1. Use an NER model to extract entities
        # 2. Create Entity nodes
        # 3. Link them to chunks

        # For now, create some basic entities from metadata
        entity_count = 0

        query = """
        MATCH (c:Chunk {id: $chunk_id})
        MERGE (e:Entity {id: $entity_id, name: $entity_name})
        SET e.type = $entity_type,
            e.created_at = datetime()
        MERGE (c)-[:CONTAINS_ENTITY {confidence: $confidence}]->(e)
        RETURN e.id as id
        """

        # Extract simple entities from metadata if available
        metadata = data.metadata
        doc_entities = []

        if "title" in metadata:
            doc_entities.append(("title", metadata["title"], 0.9))
        if "author" in metadata:
            doc_entities.append(("author", metadata["author"], 0.9))
        if "organization" in metadata:
            doc_entities.append(("organization", metadata["organization"], 0.9))

        if self._driver is None:
            return 0
        async with self._driver.session(database=database) as session:
            for chunk_id in chunk_ids[:3]:  # Limit to first 3 chunks
                for entity_type, entity_name, confidence in doc_entities:
                    entity_id = f"{chunk_id}_ent_{entity_type}"
                    params = {
                        "chunk_id": chunk_id,
                        "entity_id": entity_id,
                        "entity_name": str(entity_name)[:100],
                        "entity_type": entity_type,
                        "confidence": confidence,
                    }

                    try:
                        await session.run(query, params)
                        entity_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to create entity: {e}")

        return entity_count

    async def _create_relationships(
        self,
        conn: Connection,
        data: TransformedData,
        chunk_ids: list[str],
        database: str,
    ) -> int:
        """Create relationships between chunks.
        
        Args:
            conn: Connection handle
            data: Transformed data
            chunk_ids: List of chunk IDs
            database: Database name
            
        Returns:
            Number of relationships created
        """
        # Create sequential relationships between chunks
        if len(chunk_ids) < 2:
            return 0

        query = """
        MATCH (c1:Chunk {id: $chunk1_id})
        MATCH (c2:Chunk {id: $chunk2_id})
        MERGE (c1)-[:NEXT_CHUNK {document_id: $document_id}]->(c2)
        """

        document_id = str(data.job_id)
        rel_count = 0

        if self._driver is None:
            return 0
        async with self._driver.session(database=database) as session:
            for i in range(len(chunk_ids) - 1):
                params = {
                    "chunk1_id": chunk_ids[i],
                    "chunk2_id": chunk_ids[i + 1],
                    "document_id": document_id,
                }

                try:
                    await session.run(query, params)
                    rel_count += 1
                except Exception as e:
                    logger.warning(f"Failed to create relationship: {e}")

        return rel_count

    async def execute_cypher(
        self,
        database: str,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query.
        
        Args:
            database: Database name
            query: Cypher query string
            params: Query parameters
            
        Returns:
            Query results
        """
        if not self._driver:
            raise RuntimeError("Neo4j driver not initialized")

        results = []

        async with self._driver.session(database=database) as session:
            result = await session.run(query, params or {})
            async for record in result:
                results.append(dict(record))

        return results

    async def search(
        self,
        database: str,
        query: str,
        search_type: str = "content",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search for documents or chunks.
        
        Args:
            database: Database name
            query: Search query
            search_type: Type of search (content, entity, metadata)
            limit: Maximum results
            
        Returns:
            Search results
        """
        if search_type == "content":
            cypher = """
            MATCH (c:Chunk)
            WHERE c.content CONTAINS $query
            RETURN c.id as id, c.content as content, c.index as index
            LIMIT $limit
            """
        elif search_type == "entity":
            cypher = """
            MATCH (e:Entity)
            WHERE e.name CONTAINS $query
            MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e)
            RETURN DISTINCT c.id as id, c.content as content, e.name as entity
            LIMIT $limit
            """
        else:
            cypher = """
            MATCH (d:Document)
            WHERE d.metadata CONTAINS $query
            RETURN d.id as id, d.metadata as metadata
            LIMIT $limit
            """

        return await self.execute_cypher(database, cypher, {
            "query": query,
            "limit": limit,
        })

    async def validate_config(self, config: dict[str, Any]) -> ValidationResult:
        """Validate destination configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            ValidationResult with validation status
        """
        errors: list[str] = []
        warnings: list[str] = []

        uri = config.get("uri") or os.getenv("NEO4J_URI")
        password = config.get("password") or os.getenv("NEO4J_PASSWORD")

        if not uri:
            errors.append("uri is required (or set NEO4J_URI)")
        elif not uri.startswith(("bolt://", "bolt+s://", "neo4j://", "neo4j+s://")):
            errors.append("uri must be a valid Neo4j URI (bolt:// or neo4j://)")

        if not password:
            warnings.append("password not provided - may fail to connect")

        # Check for neo4j driver
        try:
            import neo4j
        except ImportError:
            errors.append("neo4j driver not installed (pip install neo4j)")

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
        if not self._driver:
            return HealthStatus.UNHEALTHY

        try:
            await self._driver.verify_connectivity()
            return HealthStatus.HEALTHY
        except Exception as e:
            logger.warning(f"Neo4j health check failed: {e}")
            return HealthStatus.UNHEALTHY

    async def shutdown(self) -> None:
        """Shutdown the destination and cleanup resources."""
        if self._driver:
            await self._driver.close()
            self._driver = None
            logger.info("Neo4j destination shutdown")


class Neo4jMockDestination(Neo4jDestination):
    """Mock Neo4j destination for testing.
    
    This implementation stores data in memory for testing purposes
    without requiring an actual Neo4j server.
    """

    def __init__(self) -> None:
        """Initialize the mock destination."""
        super().__init__()
        self._storage: dict[str, Any] = {
            "documents": {},
            "chunks": {},
            "entities": {},
            "relationships": [],
        }

    async def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the mock destination."""
        self._config = config
        logger.info("Neo4j mock destination initialized")

    async def connect(self, config: dict[str, Any]) -> Connection:
        """Create a mock connection."""
        database = config.get("database", "neo4j")

        return Connection(
            id=UUID(int=hash(database) % (2**32)),
            plugin_id="neo4j_mock",
            config={"database": database},
        )

    async def write(
        self,
        conn: Connection,
        data: TransformedData,
    ) -> WriteResult:
        """Write data to mock storage."""
        import time

        start_time = time.time()
        document_id = str(data.job_id)

        # Store document
        self._storage["documents"][document_id] = {
            "id": document_id,
            "job_id": str(data.job_id),
            "chunks": data.chunks,
            "metadata": data.metadata,
        }

        # Store chunks
        chunk_ids = []
        for i, chunk in enumerate(data.chunks):
            chunk_id = f"{document_id}_chunk_{i}"
            self._storage["chunks"][chunk_id] = {
                "id": chunk_id,
                "document_id": document_id,
                "index": i,
                "content": chunk.get("content", ""),
            }
            chunk_ids.append(chunk_id)

        processing_time = int((time.time() - start_time) * 1000)

        return WriteResult(
            success=True,
            destination_id="neo4j_mock",
            destination_uri=f"mock://neo4j/Document/{document_id}",
            records_written=len(data.chunks) + 1,
            bytes_written=len(str(data.chunks)),
            processing_time_ms=processing_time,
            metadata={
                "document_id": document_id,
                "chunks_created": len(chunk_ids),
            },
        )

    async def execute_cypher(
        self,
        database: str,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a mock Cypher query."""
        # Simple mock implementation
        return [{"mock": "result"}]

    def get_stored_documents(self) -> dict[str, Any]:
        """Get stored documents for testing."""
        documents: dict[str, Any] = self._storage["documents"]
        return documents

    def clear_storage(self) -> None:
        """Clear all stored data."""
        self._storage = {
            "documents": {},
            "chunks": {},
            "entities": {},
            "relationships": [],
        }
