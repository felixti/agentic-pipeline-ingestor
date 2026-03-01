"""Cognee local destination plugin using the actual Cognee library APIs.

This module provides the CogneeLocalDestination which uses the local Cognee
Python library (not the API) for graph-based document storage with:
- Cognee's internal Neo4j integration for graph storage
- Cognee's internal pgvector integration for vector storage
- Cognee's LLM adapter (configured to use litellm)

Usage flow:
1. cognee.add() - Add document chunks to Cognee
2. cognee.cognify() - Process documents into knowledge graph
3. cognee.search() - Search the knowledge graph

This is distinct from CogneeDestination which uses HTTP API calls.
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from src.infrastructure.neo4j.client import close_neo4j_client, get_neo4j_client
from src.observability.logging import get_logger
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

logger = get_logger(__name__)

# Cognee search type mapping - will be populated when cognee is imported
_SEARCH_TYPE_MAP: dict[str, Any] = {}


def _get_search_type(search_type: str) -> Any:
    """Get Cognee SearchType for the given search type string.
    
    Args:
        search_type: Search type string ("graph", "vector", "hybrid")
        
    Returns:
        Cognee SearchType enum value
    """
    try:
        from cognee.api.v1.search import SearchType
        
        mapping = {
            "graph": SearchType.GRAPH_COMPLETION,
            "vector": SearchType.RAG_COMPLETION,
            "hybrid": SearchType.FEELING_LUCKY,
            "summary": SearchType.SUMMARIES,
            "chunks": SearchType.CHUNKS,
        }
        return mapping.get(search_type, SearchType.GRAPH_COMPLETION)
    except ImportError:
        raise RuntimeError("cognee library not installed")


@dataclass
class SearchResult:
    """Result from a graph search operation.
    
    Attributes:
        chunk_id: Unique identifier of the chunk
        content: Text content of the chunk
        score: Relevance score (0.0 to 1.0)
        source_document: ID of the parent document
        entities: List of entity names mentioned in the chunk
        relationships: List of relationships involving mentioned entities
        metadata: Additional metadata about the result
    """
    chunk_id: str
    content: str
    score: float
    source_document: str
    entities: list[str] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class CogneeLocalDestination(DestinationPlugin):
    """Local Cognee destination using the actual Cognee library APIs.
    
    Uses Cognee's internal integrations for:
    - Neo4j for graph storage (nodes and relationships)
    - PostgreSQL/pgvector for vector embeddings
    - litellm for LLM integration (via Cognee's adapter)
    
    This is the primary GraphRAG destination that provides:
    - Document ingestion into knowledge graph via cognee.add()
    - Automatic entity extraction and relationship mapping via cognee.cognify()
    - Vector storage in pgvector for hybrid search
    - Graph traversal for contextual retrieval via cognee.search()
    
    Example:
        >>> destination = CogneeLocalDestination()
        >>> await destination.initialize({
        ...     "dataset_id": "my-dataset",
        ...     "graph_name": "knowledge-graph",
        ... })
        >>> conn = await destination.connect({"dataset_id": "my-dataset"})
        >>> result = await destination.write(conn, transformed_data)
        >>> # After all documents are added, process them
        >>> await destination.process_dataset(conn)
        >>> # Search the knowledge graph
        >>> results = await destination.search(conn, "your query", search_type="hybrid")
    
    Environment Variables:
        # Neo4j (used internally by Cognee)
        NEO4J_URI: Neo4j connection URI (default: bolt://neo4j:7687)
        NEO4J_USER: Neo4j username (default: neo4j)
        NEO4J_PASSWORD: Neo4j password (default: cognee-graph-db)
        
        # PostgreSQL/pgvector (used internally by Cognee)
        DB_URL: PostgreSQL connection URL
        
        # LLM Configuration (for Cognee's litellm adapter)
        LLM_API_KEY: API key for LLM provider (or OPENAI_API_KEY, AZURE_OPENAI_API_KEY)
        LLM_MODEL: Model to use (default: gpt-4.1)
        EMBEDDING_MODEL: Embedding model (default: text-embedding-3-small)
        
        # Cognee-specific
        COGNEE_LLM_PROVIDER: LLM provider for Cognee (default: litellm)
        COGNEE_LLM_MODEL: LLM model to use (default: azure/gpt-4.1)
        COGNEE_EMBEDDING_MODEL: Embedding model (default: azure/text-embedding-3-small)
    """

    def __init__(self) -> None:
        """Initialize the Cognee local destination."""
        self._config: dict[str, Any] = {}
        self._dataset_id: str | None = None
        self._graph_name: str = "default"
        self._neo4j_client = None
        self._is_initialized = False
        self._cognee_module: Any = None
        self._search_type_module: Any = None

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            id="cognee_local",
            name="Cognee Local (Native Cognee Library)",
            version="1.0.0",
            type=PluginType.DESTINATION,
            description="Local Cognee GraphRAG using native cognee.add(), cognee.cognify(), and cognee.search() APIs",
            author="Pipeline Team",
            supported_formats=["json", "markdown", "text"],
            requires_auth=False,
            config_schema={
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "Default dataset ID for document storage",
                    },
                    "graph_name": {
                        "type": "string",
                        "default": "default",
                        "description": "Name of the knowledge graph",
                    },
                    "auto_cognify": {
                        "type": "boolean",
                        "default": False,
                        "description": "Automatically call cognify after each write",
                    },
                    "neo4j_uri": {
                        "type": "string",
                        "description": "Neo4j connection URI (overrides env var)",
                    },
                    "neo4j_user": {
                        "type": "string",
                        "description": "Neo4j username (overrides env var)",
                    },
                    "neo4j_password": {
                        "type": "string",
                        "description": "Neo4j password (overrides env var)",
                    },
                },
                "required": [],
            },
        )

    async def initialize(self, config: dict[str, Any]) -> None:
        """Initialize Cognee with local storage backends.
        
        This method:
        1. Verifies the Cognee library is available
        2. Sets up environment variables for Cognee's internal integrations
        3. Optionally verifies Neo4j connectivity (for health checks)
        
        Args:
            config: Destination configuration with:
                - dataset_id: Default dataset ID
                - graph_name: Knowledge graph name
                - auto_cognify: Auto-process after each write (default: False)
                - neo4j_uri: Neo4j URI (optional, overrides env)
                - neo4j_user: Neo4j username (optional, overrides env)
                - neo4j_password: Neo4j password (optional, overrides env)
        """
        import_start = time.time()
        
        # Check if cognee library is available
        try:
            import cognee
            from cognee.api.v1.search import SearchType
            
            self._cognee_module = cognee
            self._search_type_module = SearchType
            
            logger.info(
                "cognee_library_available",
                version=getattr(cognee, "__version__", "unknown"),
            )
        except ImportError:
            logger.warning(
                "cognee_library_not_installed",
                message="Cognee library not found. Install with: pip install cognee",
            )
            raise RuntimeError(
                "Cognee library is required for CogneeLocalDestination. "
                "Install with: pip install cognee"
            )

        self._config = config
        self._dataset_id = config.get("dataset_id", "default")
        self._graph_name = config.get("graph_name", "default")

        # Set up environment variables for Cognee if not already set
        # Cognee reads these internally for its Neo4j and pgvector integrations
        self._setup_cognee_environment(config)

        # Read Neo4j configuration for health check connectivity
        neo4j_uri = config.get("neo4j_uri") or os.getenv("NEO4J_URI", "bolt://neo4j:7687")
        neo4j_user = config.get("neo4j_user") or os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = config.get("neo4j_password") or os.getenv(
            "NEO4J_PASSWORD", "cognee-graph-db"
        )

        logger.info(
            "cognee_local_initializing",
            dataset_id=self._dataset_id,
            graph_name=self._graph_name,
            neo4j_uri=neo4j_uri,
            has_db_url=os.getenv("DB_URL") is not None,
        )

        try:
            # Initialize Neo4j connection for health checks only
            # Cognee manages its own Neo4j connections internally
            self._neo4j_client = await get_neo4j_client(
                uri=neo4j_uri,
                user=neo4j_user,
                password=neo4j_password,
            )

            self._is_initialized = True

            init_duration = time.time() - import_start
            logger.info(
                "cognee_local_initialized",
                dataset_id=self._dataset_id,
                graph_name=self._graph_name,
                duration_ms=round(init_duration * 1000, 2),
            )

        except Exception as e:
            logger.error(
                "cognee_local_initialization_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    def _setup_cognee_environment(self, config: dict[str, Any]) -> None:
        """Set up environment variables for Cognee's internal integrations.
        
        Cognee reads configuration from environment variables for:
        - Neo4j graph database
        - PostgreSQL/pgvector
        - LLM provider (via litellm)
        
        Args:
            config: Configuration dictionary
        """
        # Ensure Neo4j environment variables are set for Cognee
        if not os.getenv("NEO4J_URI"):
            os.environ["NEO4J_URI"] = config.get("neo4j_uri", "bolt://neo4j:7687")
        if not os.getenv("NEO4J_USER"):
            os.environ["NEO4J_USER"] = config.get("neo4j_user", "neo4j")
        if not os.getenv("NEO4J_PASSWORD"):
            os.environ["NEO4J_PASSWORD"] = config.get("neo4j_password", "cognee-graph-db")

        # Parse DB_URL for pgvector settings if available
        db_url = os.getenv("DB_URL", "")
        if db_url and "postgresql" in db_url:
            try:
                # Parse postgresql://user:password@host:port/dbname
                # or postgresql+asyncpg://user:password@host:port/dbname
                import re
                
                # Remove asyncpg driver prefix for parsing
                clean_url = db_url.replace("+asyncpg", "")
                
                # Extract connection details
                match = re.match(
                    r"postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/(.*)",
                    clean_url
                )
                if match:
                    user, password, host, port, dbname = match.groups()
                    
                    # Set pgvector environment variables for Cognee
                    if not os.getenv("PGVECTOR_HOST"):
                        os.environ["PGVECTOR_HOST"] = host
                    if not os.getenv("PGVECTOR_PORT"):
                        os.environ["PGVECTOR_PORT"] = port
                    if not os.getenv("PGVECTOR_USER"):
                        os.environ["PGVECTOR_USER"] = user
                    if not os.getenv("PGVECTOR_PASSWORD"):
                        os.environ["PGVECTOR_PASSWORD"] = password
                    if not os.getenv("PGVECTOR_DB"):
                        os.environ["PGVECTOR_DB"] = dbname
                    
                    logger.debug(
                        "cognee_pgvector_env_configured",
                        host=host,
                        port=port,
                        db=dbname,
                    )
            except Exception as e:
                logger.warning(
                    "cognee_pgvector_env_parse_warning",
                    error=str(e),
                )

        # Set LLM environment variables for Cognee's litellm adapter
        # Priority: explicit config > existing env vars > defaults
        if not os.getenv("LLM_API_KEY"):
            # Try to use existing LLM provider keys
            if os.getenv("OPENAI_API_KEY"):
                os.environ["LLM_API_KEY"] = os.getenv("OPENAI_API_KEY")
            elif os.getenv("AZURE_OPENAI_API_KEY"):
                os.environ["LLM_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")

        if not os.getenv("LLM_MODEL"):
            os.environ["LLM_MODEL"] = os.getenv(
                "COGNEE_LLM_MODEL", "gpt-4.1"
            )

        if not os.getenv("EMBEDDING_MODEL"):
            os.environ["EMBEDDING_MODEL"] = os.getenv(
                "COGNEE_EMBEDDING_MODEL", "text-embedding-3-small"
            )

        logger.debug(
            "cognee_environment_configured",
            neo4j_uri=os.getenv("NEO4J_URI"),
            pgvector_host=os.getenv("PGVECTOR_HOST"),
            llm_model=os.getenv("LLM_MODEL"),
        )

    async def connect(self, config: dict[str, Any]) -> Connection:
        """Create connection to Cognee dataset.
        
        Args:
            config: Connection configuration with:
                - dataset_id: Target dataset ID
                - graph_name: Optional graph name override
                
        Returns:
            Connection handle with dataset_id and graph_name
            
        Raises:
            ConnectionError: If not initialized or connection fails
        """
        if not self._is_initialized:
            raise ConnectionError("CogneeLocalDestination not initialized. Call initialize() first.")

        dataset_id = config.get("dataset_id", self._dataset_id or "default")
        graph_name = config.get("graph_name", self._graph_name)

        logger.info(
            "cognee_local_connected",
            dataset_id=dataset_id,
            graph_name=graph_name,
        )

        return Connection(
            id=UUID(int=hash(f"{dataset_id}:{graph_name}") % (2**32)),
            plugin_id="cognee_local",
            config={
                "dataset_id": dataset_id,
                "graph_name": graph_name,
                "auto_cognify": config.get(
                    "auto_cognify",
                    self._config.get("auto_cognify", False)
                ),
            },
        )

    async def write(
        self,
        conn: Connection,
        data: TransformedData,
    ) -> WriteResult:
        """Write transformed data to Cognee using cognee.add().
        
        This method:
        1. Adds each chunk to Cognee using cognee.add()
        2. Optionally calls cognee.cognify() if auto_cognify is enabled
        
        Note: Documents are not processed into a knowledge graph until
        cognee.cognify() is called. Call process_dataset() after all
        documents are added, or enable auto_cognify.
        
        Args:
            conn: Connection handle from connect()
            data: Transformed data to write
            
        Returns:
            WriteResult with operation status and metadata
        """
        start_time = time.time()

        if not self._is_initialized or not self._cognee_module:
            return WriteResult(
                success=False,
                error="CogneeLocalDestination not initialized",
            )

        dataset_id = conn.config.get("dataset_id", "default")
        auto_cognify = conn.config.get("auto_cognify", False)

        try:
            # Track added documents
            documents_added = 0
            chunks_added = 0

            # Add each chunk to Cognee using cognee.add()
            for i, chunk in enumerate(data.chunks):
                content = chunk.get("content", "")
                if not content:
                    continue

                # Add text to Cognee dataset
                # This stores the raw content that will be processed by cognify()
                await self._cognee_module.add(
                    content,
                    dataset_name=dataset_id,
                )
                chunks_added += 1

            # Optionally auto-process into knowledge graph
            if auto_cognify and chunks_added > 0:
                await self.process_dataset(conn)

            processing_time = int((time.time() - start_time) * 1000)

            logger.info(
                "cognee_write_completed",
                job_id=str(data.job_id),
                dataset_id=dataset_id,
                chunks_added=chunks_added,
                auto_cognify=auto_cognify,
                duration_ms=processing_time,
            )

            return WriteResult(
                success=True,
                destination_id="cognee_local",
                destination_uri=f"cognee://{dataset_id}/Document/{data.job_id}",
                records_written=chunks_added,
                bytes_written=len(json.dumps(data.chunks)),
                processing_time_ms=processing_time,
                metadata={
                    "job_id": str(data.job_id),
                    "dataset_id": dataset_id,
                    "chunks_added": chunks_added,
                    "auto_cognify": auto_cognify,
                },
            )

        except Exception as e:
            logger.error(
                "cognee_write_failed",
                error=str(e),
                error_type=type(e).__name__,
                dataset_id=dataset_id,
            )
            return WriteResult(
                success=False,
                error=f"Write failed: {e!s}",
            )

    async def process_dataset(self, conn: Connection) -> dict[str, Any]:
        """Process dataset into knowledge graph using cognee.cognify().
        
        This method processes all documents added to the dataset using
        Cognee's knowledge graph construction pipeline. It:
        - Extracts entities from text
        - Creates relationships between entities
        - Stores vectors in pgvector
        - Builds the graph in Neo4j
        
        Args:
            conn: Connection handle from connect()
            
        Returns:
            Processing result with metadata
            
        Raises:
            RuntimeError: If processing fails
        """
        if not self._is_initialized or not self._cognee_module:
            raise RuntimeError("CogneeLocalDestination not initialized")

        dataset_id = conn.config.get("dataset_id", "default")
        start_time = time.time()

        try:
            logger.info(
                "cognee_cognify_started",
                dataset_id=dataset_id,
            )

            # Process the dataset using cognee.cognify()
            # This builds the knowledge graph from added documents
            await self._cognee_module.cognify(datasets=[dataset_id])

            processing_time = int((time.time() - start_time) * 1000)

            logger.info(
                "cognee_cognify_completed",
                dataset_id=dataset_id,
                duration_ms=processing_time,
            )

            return {
                "success": True,
                "dataset_id": dataset_id,
                "processing_time_ms": processing_time,
            }

        except Exception as e:
            logger.error(
                "cognee_cognify_failed",
                dataset_id=dataset_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise RuntimeError(f"Failed to process dataset: {e}") from e

    async def search(
        self,
        conn: Connection,
        query: str,
        search_type: str = "hybrid",
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Search the knowledge graph using cognee.search().
        
        Uses Cognee's native search API with appropriate SearchType:
        - "graph": GRAPH_COMPLETION - Graph traversal search
        - "vector": RAG_COMPLETION - Vector similarity search
        - "hybrid": FEELING_LUCKY - Combined search (default)
        - "summary": SUMMARIES - Search for summaries
        - "chunks": CHUNKS - Raw chunk search
        
        Args:
            conn: Connection handle from connect()
            query: Search query string
            search_type: Type of search ("vector", "graph", "hybrid", "summary", "chunks")
            top_k: Maximum number of results to return
            
        Returns:
            List of SearchResult objects ordered by relevance
            
        Raises:
            ValueError: If search_type is invalid
            RuntimeError: If search fails
            
        Example:
            >>> results = await destination.search(
            ...     conn,
            ...     "machine learning applications",
            ...     search_type="hybrid",
            ...     top_k=5
            ... )
            >>> for result in results:
            ...     print(f"{result.chunk_id}: {result.score}")
        """
        if not self._is_initialized or not self._cognee_module:
            raise RuntimeError("CogneeLocalDestination not initialized")

        valid_search_types = ("vector", "graph", "hybrid", "summary", "chunks")
        if search_type not in valid_search_types:
            raise ValueError(
                f"Invalid search_type: {search_type}. "
                f"Use one of: {', '.join(valid_search_types)}"
            )

        dataset_id = conn.config.get("dataset_id", "default")
        start_time = time.time()

        try:
            # Get the appropriate Cognee SearchType
            cognee_search_type = _get_search_type(search_type)

            logger.info(
                "cognee_search_started",
                query=query[:100],
                search_type=search_type,
                dataset_id=dataset_id,
            )

            # Execute search using cognee.search()
            cognee_results = await self._cognee_module.search(
                query_text=query,
                query_type=cognee_search_type,
                datasets=[dataset_id],
            )

            # Convert Cognee results to SearchResult objects
            results = self._convert_cognee_results(cognee_results, top_k)

            query_time = int((time.time() - start_time) * 1000)

            logger.info(
                "cognee_search_completed",
                search_type=search_type,
                query=query[:100],
                result_count=len(results),
                dataset_id=dataset_id,
                query_time_ms=query_time,
            )

            return results

        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__

            # Check if this is a "no data" error from Cognee
            # Cognee raises SearchPreconditionError when no documents have been added
            if "SearchPreconditionError" in error_type or "prerequisites not met" in error_msg.lower() or "no database" in error_msg.lower():
                logger.warning(
                    "cognee_search_no_data",
                    search_type=search_type,
                    query=query[:100],
                    dataset_id=dataset_id,
                    message="No documents in knowledge graph. Process documents with cognee_local destination first.",
                )
                # Return empty results instead of raising error
                return []

            logger.error(
                "cognee_search_failed",
                search_type=search_type,
                query=query[:100],
                error=error_msg,
                error_type=error_type,
            )
            raise RuntimeError(f"Search failed: {e}") from e

    def _convert_cognee_results(
        self,
        cognee_results: list[Any],
        top_k: int,
    ) -> list[SearchResult]:
        """Convert Cognee search results to SearchResult objects.
        
        Args:
            cognee_results: Raw results from cognee.search()
            top_k: Maximum number of results
            
        Returns:
            List of SearchResult objects
        """
        results = []

        for i, item in enumerate(cognee_results[:top_k]):
            # Cognee results can be strings, dicts, or custom objects
            # Handle different result types
            
            # Debug: Log the result type and available fields
            if isinstance(item, dict):
                logger.debug(
                    "cognee_search_result_dict",
                    keys=list(item.keys()),
                    has_metadata="metadata" in item,
                    metadata_keys=list(item.get("metadata", {}).keys()) if "metadata" in item else [],
                )
            elif not isinstance(item, str):
                logger.debug(
                    "cognee_search_result_object",
                    type=type(item).__name__,
                    attrs=[attr for attr in dir(item) if not attr.startswith("_")],
                )
            if isinstance(item, str):
                # Simple text result
                result = SearchResult(
                    chunk_id=f"result_{i}",
                    content=item,
                    score=1.0 - (i * 0.1),  # Simple ranking
                    source_document="unknown",
                    metadata={"result_type": "text"},
                )
            elif isinstance(item, dict):
                # Dictionary result with potential metadata
                # Try multiple possible field names for source document
                source_document = (
                    item.get("source_document") or
                    item.get("document_id") or
                    item.get("file_name") or
                    item.get("source") or
                    item.get("document_name") or
                    item.get("metadata", {}).get("file_name") or
                    item.get("metadata", {}).get("source_document") or
                    "unknown"
                )
                # Try multiple possible field names for entities
                entities = (
                    item.get("entities") or
                    item.get("extracted_entities") or
                    item.get("nodes") or
                    item.get("concepts") or
                    item.get("mentioned_entities") or
                    item.get("metadata", {}).get("entities") or
                    []
                )
                result = SearchResult(
                    chunk_id=item.get("id", f"result_{i}"),
                    content=item.get("text", item.get("content", str(item))),
                    score=item.get("score", 1.0 - (i * 0.1)),
                    source_document=source_document,
                    entities=entities,
                    metadata=item.get("metadata", {}),
                )
            else:
                # Object result - try to extract attributes
                content = str(item)
                chunk_id = f"result_{i}"
                score = 1.0 - (i * 0.1)
                source = "unknown"
                entities = []
                metadata = {}

                # Try to extract attributes if available
                if hasattr(item, "text"):
                    content = item.text
                if hasattr(item, "id"):
                    chunk_id = item.id
                if hasattr(item, "score"):
                    score = item.score
                # Try multiple possible attribute names for source document
                for attr in ["source_document", "document_id", "file_name", "source", "document_name"]:
                    if hasattr(item, attr) and getattr(item, attr):
                        source = getattr(item, attr)
                        break
                # Try multiple possible attribute names for entities
                for attr in ["entities", "extracted_entities", "nodes", "concepts", "mentioned_entities"]:
                    if hasattr(item, attr) and getattr(item, attr):
                        entities = getattr(item, attr)
                        break

                result = SearchResult(
                    chunk_id=chunk_id,
                    content=content,
                    score=score,
                    source_document=source,
                    entities=entities,
                    metadata=metadata,
                )

            results.append(result)

        return results

    async def get_dataset_stats(
        self,
        conn: Connection,
    ) -> dict[str, Any]:
        """Get statistics for a dataset.
        
        Args:
            conn: Connection handle from connect()
            
        Returns:
            Dataset statistics
        """
        if not self._is_initialized:
            return {"error": "Not initialized"}

        dataset_id = conn.config.get("dataset_id", "default")

        try:
            # Try to get stats from Neo4j for health check
            if self._neo4j_client:
                result = await self._neo4j_client.execute_query(
                    """
                    MATCH (d:Document {dataset_id: $dataset_id})
                    RETURN count(d) as document_count
                    """,
                    {"dataset_id": dataset_id}
                )
                document_count = result[0].get("document_count", 0) if result else 0
            else:
                document_count = 0

            return {
                "dataset_id": dataset_id,
                "document_count": document_count,
                "initialized": self._is_initialized,
            }

        except Exception as e:
            logger.warning(
                "cognee_dataset_stats_warning",
                dataset_id=dataset_id,
                error=str(e),
            )
            return {
                "dataset_id": dataset_id,
                "error": str(e),
            }

    async def extract_entities(
        self,
        conn: Connection,
        text: str,
    ) -> dict[str, Any]:
        """Extract entities and relationships from text.
        
        Uses Cognee's entity extraction capabilities to identify
        entities and their relationships from the provided text.
        
        Args:
            conn: Connection handle from connect()
            text: Text content to extract entities from
            
        Returns:
            Dictionary with 'entities' and 'relationships' lists
            
        Raises:
            RuntimeError: If extraction fails
        """
        if not self._is_initialized or not self._cognee_module:
            raise RuntimeError("CogneeLocalDestination not initialized")

        dataset_id = conn.config.get("dataset_id", "default")
        start_time = time.time()

        try:
            logger.info(
                "cognee_extract_entities_started",
                dataset_id=dataset_id,
                text_length=len(text),
            )

            # Try to use Cognee's entity extraction if available
            # For now, we use a simple implementation that can be enhanced
            # when Cognee provides direct entity extraction APIs
            entities: list[dict[str, Any]] = []
            relationships: list[dict[str, Any]] = []

            # Check if cognee has entity extraction capability
            if hasattr(self._cognee_module, 'extract_entities'):
                cognee_result = await self._cognee_module.extract_entities(
                    text=text,
                    dataset_name=dataset_id,
                )
                entities = cognee_result.get('entities', [])
                relationships = cognee_result.get('relationships', [])
            else:
                # Fallback: simple entity extraction using common patterns
                # This is a placeholder until Cognee provides full entity extraction
                entities, relationships = self._simple_entity_extraction(text)

            processing_time = int((time.time() - start_time) * 1000)

            logger.info(
                "cognee_extract_entities_completed",
                dataset_id=dataset_id,
                entity_count=len(entities),
                relationship_count=len(relationships),
                duration_ms=processing_time,
            )

            return {
                "entities": entities,
                "relationships": relationships,
                "dataset_id": dataset_id,
                "processing_time_ms": processing_time,
            }

        except Exception as e:
            logger.error(
                "cognee_extract_entities_failed",
                dataset_id=dataset_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise RuntimeError(f"Entity extraction failed: {e}") from e

    def _simple_entity_extraction(
        self,
        text: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Simple entity extraction fallback implementation.
        
        This is a basic implementation that looks for capitalized words
        as potential entities. In production, this should use a proper
        NER model or Cognee's native extraction.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Tuple of (entities list, relationships list)
        """
        import re

        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []

        # Simple pattern: look for capitalized words/phrases
        # This is a naive implementation for demonstration
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        matches = re.findall(pattern, text)

        # Deduplicate and create entity objects
        seen = set()
        for match in matches:
            if match not in seen and len(match) > 2:
                seen.add(match)
                entities.append({
                    "name": match,
                    "type": "UNKNOWN",
                    "description": f"Entity found in text",
                })

        return entities, relationships

    async def get_stats(
        self,
        conn: Connection,
    ) -> dict[str, Any]:
        """Get comprehensive graph statistics.
        
        Args:
            conn: Connection handle from connect()
            
        Returns:
            Dictionary with dataset statistics including:
            - document_count: Number of documents
            - chunk_count: Number of chunks
            - entity_count: Number of entities in graph
            - relationship_count: Number of relationships
            - graph_density: Graph density metric
            - last_updated: Last update timestamp
        """
        if not self._is_initialized:
            raise RuntimeError("CogneeLocalDestination not initialized")

        dataset_id = conn.config.get("dataset_id", "default")

        try:
            logger.info(
                "cognee_get_stats_started",
                dataset_id=dataset_id,
            )

            # Get basic stats from Neo4j
            document_count = 0
            entity_count = 0
            relationship_count = 0
            chunk_count = 0

            if self._neo4j_client:
                # Document count
                doc_result = await self._neo4j_client.execute_query(
                    """
                    MATCH (d:Document {dataset_id: $dataset_id})
                    RETURN count(d) as document_count
                    """,
                    {"dataset_id": dataset_id}
                )
                document_count = doc_result[0].get("document_count", 0) if doc_result else 0

                # Entity count (nodes that are not documents or chunks)
                entity_result = await self._neo4j_client.execute_query(
                    """
                    MATCH (e)
                    WHERE e.dataset_id = $dataset_id
                    AND NOT e:Document AND NOT e:Chunk
                    RETURN count(e) as entity_count
                    """,
                    {"dataset_id": dataset_id}
                )
                entity_count = entity_result[0].get("entity_count", 0) if entity_result else 0

                # Relationship count
                rel_result = await self._neo4j_client.execute_query(
                    """
                    MATCH ()-[r]->()
                    WHERE r.dataset_id = $dataset_id
                    RETURN count(r) as relationship_count
                    """,
                    {"dataset_id": dataset_id}
                )
                relationship_count = rel_result[0].get("relationship_count", 0) if rel_result else 0

                # Chunk count
                chunk_result = await self._neo4j_client.execute_query(
                    """
                    MATCH (c:Chunk {dataset_id: $dataset_id})
                    RETURN count(c) as chunk_count
                    """,
                    {"dataset_id": dataset_id}
                )
                chunk_count = chunk_result[0].get("chunk_count", 0) if chunk_result else 0

            # Calculate graph density
            graph_density = 0.0
            if entity_count > 0:
                max_edges = entity_count * (entity_count - 1)
                if max_edges > 0:
                    graph_density = relationship_count / max_edges

            from datetime import datetime, timezone

            result = {
                "dataset_id": dataset_id,
                "document_count": document_count,
                "chunk_count": chunk_count,
                "entity_count": entity_count,
                "relationship_count": relationship_count,
                "graph_density": round(min(1.0, max(0.0, graph_density)), 4),
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }

            logger.info(
                "cognee_get_stats_completed",
                dataset_id=dataset_id,
                document_count=document_count,
                entity_count=entity_count,
                relationship_count=relationship_count,
            )

            return result

        except Exception as e:
            logger.error(
                "cognee_get_stats_failed",
                dataset_id=dataset_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise RuntimeError(f"Failed to get stats: {e}") from e

    async def validate_config(self, config: dict[str, Any]) -> ValidationResult:
        """Validate destination configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            ValidationResult with validation status and any errors/warnings
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Check Neo4j configuration (for health checks and Cognee internal use)
        neo4j_uri = config.get("neo4j_uri") or os.getenv("NEO4J_URI")
        neo4j_password = config.get("neo4j_password") or os.getenv("NEO4J_PASSWORD")

        if not neo4j_uri:
            warnings.append("Neo4j URI not configured (set NEO4J_URI or pass neo4j_uri)")
        elif not neo4j_uri.startswith(("bolt://", "bolt+s://", "neo4j://", "neo4j+s://")):
            errors.append("Neo4j URI must be a valid bolt:// or neo4j:// URL")

        if not neo4j_password:
            warnings.append("Neo4j password not configured - may fail to connect")

        # Check required dependencies
        try:
            import cognee
            from cognee.api.v1.search import SearchType  # noqa: F401
        except ImportError as e:
            errors.append(f"cognee library not installed or incomplete: {e}")

        # Check PostgreSQL for pgvector (Cognee uses this internally)
        db_url = os.getenv("DB_URL")
        if not db_url:
            warnings.append("DB_URL not set - Cognee's pgvector integration may not work")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    async def health_check(self, config: dict[str, Any] | None = None) -> HealthStatus:
        """Check the health of the Cognee local destination.
        
        Checks:
        - Cognee library availability
        - Neo4j connectivity (used by Cognee internally)
        
        Args:
            config: Optional configuration for health check
            
        Returns:
            HealthStatus indicating destination health
        """
        if not self._is_initialized:
            return HealthStatus.UNHEALTHY

        try:
            # Check Cognee library is available
            if not self._cognee_module:
                return HealthStatus.UNHEALTHY

            # Check Neo4j connectivity (for health check only)
            if self._neo4j_client:
                neo4j_healthy = await self._neo4j_client.health_check()
                if not neo4j_healthy:
                    return HealthStatus.DEGRADED

            logger.debug("cognee_health_check_passed")
            return HealthStatus.HEALTHY

        except Exception as e:
            logger.warning(
                "cognee_health_check_failed",
                error=str(e),
            )
            return HealthStatus.UNHEALTHY

    async def shutdown(self) -> None:
        """Shutdown the destination and cleanup resources.
        
        Closes Neo4j connection (used for health checks) and cleans up resources.
        Note: Cognee manages its own connections internally.
        """
        logger.info("cognee_local_shutting_down")

        # Close Neo4j client (used for health checks only)
        if self._neo4j_client:
            await close_neo4j_client()
            self._neo4j_client = None

        self._cognee_module = None
        self._search_type_module = None
        self._is_initialized = False

        logger.info("cognee_local_shutdown_complete")


class CogneeLocalMockDestination(CogneeLocalDestination):
    """Mock Cognee local destination for testing.
    
    This implementation stores data in memory for testing purposes
    without requiring actual Neo4j or PostgreSQL connections.
    
    It mimics the Cognee API but stores everything in memory.
    
    Example:
        >>> destination = CogneeLocalMockDestination()
        >>> await destination.initialize({})
        >>> conn = await destination.connect({"dataset_id": "test"})
        >>> result = await destination.write(conn, data)
        >>> await destination.process_dataset(conn)
        >>> results = await destination.search(conn, "query")
    """

    def __init__(self) -> None:
        """Initialize the mock destination."""
        super().__init__()
        self._storage: dict[str, Any] = {
            "documents": {},
            "chunks": {},
            "datasets": {},
        }

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            id="cognee_local_mock",
            name="Cognee Local Mock (Testing)",
            version="1.0.0",
            type=PluginType.DESTINATION,
            description="Mock Cognee local destination for testing without external services",
            author="Pipeline Team",
            supported_formats=["json"],
            requires_auth=False,
        )

    async def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the mock destination.
        
        Args:
            config: Mock configuration (stored but not used)
        """
        self._config = config
        self._dataset_id = config.get("dataset_id", "default")
        self._graph_name = config.get("graph_name", "default")
        self._is_initialized = True

        # Don't require actual cognee library for mock
        self._cognee_module = True  # type: ignore

        logger.info(
            "cognee_local_mock_initialized",
            dataset_id=self._dataset_id,
            graph_name=self._graph_name,
        )

    async def connect(self, config: dict[str, Any]) -> Connection:
        """Create a mock connection.
        
        Args:
            config: Connection configuration
            
        Returns:
            Mock connection handle
        """
        dataset_id = config.get("dataset_id", self._dataset_id or "default")

        if dataset_id not in self._storage["datasets"]:
            self._storage["datasets"][dataset_id] = {
                "documents": [],
                "processed": False,
            }

        return Connection(
            id=UUID(int=hash(dataset_id) % (2**32)),
            plugin_id="cognee_local_mock",
            config={
                "dataset_id": dataset_id,
                "graph_name": config.get("graph_name", self._graph_name),
                "auto_cognify": config.get("auto_cognify", False),
            },
        )

    async def write(
        self,
        conn: Connection,
        data: TransformedData,
    ) -> WriteResult:
        """Write data to mock storage.
        
        Args:
            conn: Connection handle
            data: Transformed data to write
            
        Returns:
            WriteResult with operation status
        """
        start_time = time.time()
        dataset_id = conn.config.get("dataset_id", "default")
        document_id = str(data.job_id)
        auto_cognify = conn.config.get("auto_cognify", False)

        # Store document
        chunks_data = []
        for i, chunk in enumerate(data.chunks):
            chunk_id = f"{document_id}_chunk_{i}"
            chunks_data.append({
                "id": chunk_id,
                "content": chunk.get("content", ""),
                "index": i,
                "metadata": chunk.get("metadata", {}),
            })

        self._storage["documents"][document_id] = {
            "id": document_id,
            "dataset_id": dataset_id,
            "job_id": str(data.job_id),
            "chunks": chunks_data,
            "metadata": data.metadata,
        }

        self._storage["datasets"][dataset_id]["documents"].append(document_id)

        # Auto-process if enabled
        if auto_cognify:
            await self.process_dataset(conn)

        processing_time = int((time.time() - start_time) * 1000)

        logger.debug(
            "cognee_local_mock_write",
            dataset_id=dataset_id,
            document_id=document_id,
            chunks=len(chunks_data),
        )

        return WriteResult(
            success=True,
            destination_id="cognee_local_mock",
            destination_uri=f"mock://cognee/{dataset_id}/Document/{document_id}",
            records_written=len(data.chunks) + 1,
            bytes_written=len(str(data.chunks)),
            processing_time_ms=processing_time,
            metadata={
                "document_id": document_id,
                "dataset_id": dataset_id,
                "chunks_added": len(chunks_data),
                "mock": True,
            },
        )

    async def process_dataset(self, conn: Connection) -> dict[str, Any]:
        """Process dataset (mock implementation).
        
        Args:
            conn: Connection handle
            
        Returns:
            Processing result
        """
        dataset_id = conn.config.get("dataset_id", "default")
        
        # Mark dataset as processed
        if dataset_id in self._storage["datasets"]:
            self._storage["datasets"][dataset_id]["processed"] = True

        logger.debug(
            "cognee_local_mock_cognify",
            dataset_id=dataset_id,
        )

        return {
            "success": True,
            "dataset_id": dataset_id,
            "mock": True,
        }

    async def search(
        self,
        conn: Connection,
        query: str,
        search_type: str = "hybrid",
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Search mock storage.
        
        Args:
            conn: Connection handle
            query: Search query
            search_type: Type of search
            top_k: Maximum results
            
        Returns:
            List of SearchResult objects
        """
        dataset_id = conn.config.get("dataset_id", "default")
        results = []

        # Simple mock search - find chunks containing query terms
        query_terms = query.lower().split()
        
        for doc_id, doc in self._storage["documents"].items():
            if doc.get("dataset_id") != dataset_id:
                continue

            for chunk in doc.get("chunks", []):
                content = chunk.get("content", "").lower()
                score = sum(1 for term in query_terms if term in content) / len(query_terms) if query_terms else 0
                
                if score > 0:
                    results.append(SearchResult(
                        chunk_id=chunk["id"],
                        content=chunk["content"],
                        score=min(1.0, score),
                        source_document=doc_id,
                        metadata={"search_type": search_type, "mock": True},
                    ))

        # Sort by score and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    async def health_check(self, config: dict[str, Any] | None = None) -> HealthStatus:
        """Always healthy for mock.
        
        Returns:
            HealthStatus.HEALTHY
        """
        return HealthStatus.HEALTHY

    async def validate_config(self, config: dict[str, Any]) -> ValidationResult:
        """Validate mock configuration.
        
        Mock always validates successfully.
        
        Returns:
            ValidationResult(valid=True)
        """
        return ValidationResult(valid=True)

    async def shutdown(self) -> None:
        """Shutdown mock destination."""
        self._is_initialized = False
        self._storage = {
            "documents": {},
            "chunks": {},
            "datasets": {},
        }
        logger.info("cognee_local_mock_shutdown")

    def get_stored_documents(self, dataset_id: str | None = None) -> dict[str, Any]:
        """Get stored documents for testing.
        
        Args:
            dataset_id: Optional dataset ID to filter by
            
        Returns:
            Dictionary of stored documents
        """
        if dataset_id:
            return {
                k: v for k, v in self._storage["documents"].items()
                if v.get("dataset_id") == dataset_id
            }
        return self._storage["documents"]

    def clear_storage(self) -> None:
        """Clear all stored data."""
        self._storage = {
            "documents": {},
            "chunks": {},
            "datasets": {},
        }
        logger.debug("cognee_local_mock_storage_cleared")
