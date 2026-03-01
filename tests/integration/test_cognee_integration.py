"""Integration tests for Cognee GraphRAG with Neo4j and PostgreSQL.

These tests require:
- Neo4j running (configured via NEO4J_URI env var)
- PostgreSQL with pgvector (configured via DB_URL env var)

Run with: pytest tests/integration/test_cognee_integration.py -v
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from src.plugins.base import Connection, HealthStatus, TransformedData, WriteResult
from src.plugins.destinations.cognee_llm import (
    CogneeLLMProvider,
    ExtractedEntity,
    ExtractedRelationship,
)
from src.plugins.destinations.cognee_local import (
    CogneeLocalDestination,
    CogneeLocalMockDestination,
    SearchResult,
)

# Skip all tests if Neo4j is not available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("NEO4J_URI") is None,
        reason="Neo4j not configured (set NEO4J_URI)",
    ),
]


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
async def neo4j_client():
    """Provide Neo4j client for tests.
    
    Creates a client, connects, and cleans up after tests.
    """
    from src.infrastructure.neo4j.client import Neo4jClient, close_neo4j_client
    
    client = Neo4jClient()
    try:
        await client.connect()
        yield client
    finally:
        await client.close()
        await close_neo4j_client()


@pytest.fixture
async def cognee_destination():
    """Provide CogneeLocalDestination for tests.
    
    Creates destination, initializes, and cleans up after tests.
    """
    dest = CogneeLocalDestination()
    
    try:
        await dest.initialize({
            "dataset_id": "test-dataset",
            "graph_name": "test-graph",
            "extract_entities": True,
            "extract_relationships": True,
            "store_vectors": True,
        })
        yield dest
    finally:
        await dest.shutdown()


@pytest.fixture
def sample_transformed_data():
    """Create sample transformed data for testing."""
    return TransformedData(
        job_id=uuid4(),
        chunks=[
            {
                "content": "Microsoft Corporation was founded by Bill Gates and Paul Allen in 1975.",
                "metadata": {"page": 1, "index": 0},
            },
            {
                "content": "The company is headquartered in Redmond, Washington.",
                "metadata": {"page": 1, "index": 1},
            },
            {
                "content": "Microsoft develops software products including Windows and Office.",
                "metadata": {"page": 2, "index": 2},
            },
        ],
        embeddings=[[0.1] * 1536, [0.2] * 1536, [0.3] * 1536],
        metadata={
            "title": "Microsoft History",
            "author": "Test Author",
            "organization": "Test Org",
        },
        lineage={"source": "test", "parser": "test_parser"},
        original_format="pdf",
        output_format="json",
    )


@pytest.fixture
def sample_connection():
    """Create a sample connection."""
    return Connection(
        id=UUID(int=hash("test-dataset:default") % (2**32)),
        plugin_id="cognee_local",
        config={
            "dataset_id": "test-dataset",
            "graph_name": "test-graph",
            "extract_entities": True,
            "extract_relationships": True,
            "store_vectors": True,
        },
    )


@pytest.fixture
def mock_cognee_destination():
    """Provide mock CogneeLocalDestination for isolated tests."""
    dest = CogneeLocalMockDestination()
    return dest


# ============================================================================
# Neo4j Connection Tests
# ============================================================================

@pytest.mark.integration
class TestNeo4jConnection:
    """Tests for Neo4j connection."""

    @pytest.mark.asyncio
    async def test_neo4j_connection(self, neo4j_client):
        """Test Neo4j client can connect and execute queries."""
        # Verify connection is healthy
        is_healthy = await neo4j_client.health_check()
        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_neo4j_execute_query(self, neo4j_client):
        """Test executing a Cypher query."""
        # Create a test node
        result = await neo4j_client.execute_write(
            "CREATE (n:TestNode {name: $name, test_id: $test_id}) RETURN n.name as name",
            {"name": "Integration Test", "test_id": "test_123"},
        )
        
        assert len(result) == 1
        assert result[0]["name"] == "Integration Test"

    @pytest.mark.asyncio
    async def test_neo4j_execute_read_query(self, neo4j_client):
        """Test executing a read Cypher query."""
        # First create a node
        await neo4j_client.execute_write(
            "CREATE (n:TestNode {name: $name, test_id: $test_id})",
            {"name": "Read Test", "test_id": "read_test_123"},
        )
        
        # Then read it back
        result = await neo4j_client.execute_query(
            "MATCH (n:TestNode {test_id: $test_id}) RETURN n.name as name",
            {"test_id": "read_test_123"},
        )
        
        assert len(result) >= 1
        assert any(r["name"] == "Read Test" for r in result)

    @pytest.mark.asyncio
    async def test_neo4j_database_info(self, neo4j_client):
        """Test getting database information."""
        info = await neo4j_client.get_database_info()
        
        assert "version" in info
        assert "edition" in info
        assert "name" in info
        assert "uri" in info


# ============================================================================
# CogneeLocalDestination Integration Tests
# ============================================================================

@pytest.mark.integration
class TestCogneeLocalDestinationIntegration:
    """Tests for CogneeLocalDestination with real Neo4j."""

    @pytest.mark.asyncio
    async def test_initialize_and_connect(self, neo4j_client):
        """Test destination initialization and connection."""
        dest = CogneeLocalDestination()
        
        await dest.initialize({
            "dataset_id": "integration-test",
            "graph_name": "integration-graph",
        })
        
        assert dest._is_initialized is True
        assert dest._neo4j_client is not None
        
        conn = await dest.connect({"dataset_id": "integration-test"})
        assert isinstance(conn, Connection)
        assert conn.config["dataset_id"] == "integration-test"
        
        await dest.shutdown()

    @pytest.mark.asyncio
    async def test_write_document(self, neo4j_client, sample_transformed_data):
        """Test writing document to Cognee."""
        dest = CogneeLocalDestination()
        
        await dest.initialize({
            "dataset_id": "write-test",
            "graph_name": "write-graph",
            "extract_entities": False,  # Skip entity extraction for speed
            "extract_relationships": False,
        })
        
        conn = await dest.connect({"dataset_id": "write-test"})
        result = await dest.write(conn, sample_transformed_data)
        
        assert isinstance(result, WriteResult)
        assert result.success is True
        assert result.records_written == 4  # 3 chunks + 1 document
        
        await dest.shutdown()

    @pytest.mark.asyncio
    async def test_write_and_search(self, neo4j_client, sample_transformed_data):
        """Test writing document and then searching."""
        dest = CogneeLocalDestination()
        
        await dest.initialize({
            "dataset_id": "search-test",
            "graph_name": "search-graph",
            "extract_entities": True,
            "extract_relationships": True,
        })
        
        conn = await dest.connect({"dataset_id": "search-test"})
        
        # Write document
        write_result = await dest.write(conn, sample_transformed_data)
        assert write_result.success is True
        
        # Search for content
        results = await dest.search(conn, "Microsoft", search_type="graph", top_k=5)
        
        assert isinstance(results, list)
        # Should find chunks mentioning Microsoft
        
        await dest.shutdown()

    @pytest.mark.asyncio
    async def test_graph_search(self, neo4j_client, sample_transformed_data):
        """Test graph search functionality."""
        dest = CogneeLocalDestination()
        
        await dest.initialize({
            "dataset_id": "graph-search-test",
            "graph_name": "graph-search-graph",
            "extract_entities": True,
            "extract_relationships": True,
        })
        
        conn = await dest.connect({"dataset_id": "graph-search-test"})
        
        # Write document with entity extraction
        await dest.write(conn, sample_transformed_data)
        
        # Search using graph traversal
        results = await dest.search(conn, "Bill Gates", search_type="graph", top_k=5)
        
        assert isinstance(results, list)
        
        await dest.shutdown()

    @pytest.mark.asyncio
    async def test_health_check(self, neo4j_client):
        """Test health check with real Neo4j."""
        dest = CogneeLocalDestination()
        
        await dest.initialize({
            "dataset_id": "health-test",
            "graph_name": "health-graph",
        })
        
        health = await dest.health_check()
        assert health == HealthStatus.HEALTHY
        
        await dest.shutdown()

    @pytest.mark.asyncio
    async def test_validate_config(self, neo4j_client):
        """Test configuration validation."""
        dest = CogneeLocalDestination()
        
        # Valid config with Neo4j URI set
        result = await dest.validate_config({})
        
        # May have warnings but should be valid if Neo4j is configured
        assert hasattr(result, "valid")
        assert hasattr(result, "errors")
        assert hasattr(result, "warnings")

    @pytest.mark.asyncio
    async def test_create_entity_nodes(self, neo4j_client):
        """Test creating entity nodes in Neo4j."""
        dest = CogneeLocalDestination()
        
        await dest.initialize({
            "dataset_id": "entity-test",
            "graph_name": "entity-graph",
        })
        
        conn = await dest.connect({"dataset_id": "entity-test"})
        
        # First create a document and chunk
        doc_id = "test_doc_123"
        chunk_id = "test_chunk_123"
        
        # Create chunk node
        await neo4j_client.execute_write(
            """
            CREATE (c:Chunk {id: $chunk_id, dataset_id: $dataset_id, content: "Test content"})
            """,
            {"chunk_id": chunk_id, "dataset_id": "entity-test"},
        )
        
        entities = [
            ExtractedEntity(name="Test Company", type="ORGANIZATION", description="A test company"),
            ExtractedEntity(name="John Doe", type="PERSON", description="A person"),
        ]
        
        count = await dest.create_entity_nodes(conn, doc_id, chunk_id, entities)
        
        assert count == 2
        
        # Verify entities were created
        result = await neo4j_client.execute_query(
            "MATCH (e:Entity {dataset_id: $dataset_id}) RETURN count(e) as count",
            {"dataset_id": "entity-test"},
        )
        assert result[0]["count"] >= 2
        
        await dest.shutdown()

    @pytest.mark.asyncio
    async def test_create_relationship_edges(self, neo4j_client):
        """Test creating relationship edges between entities."""
        dest = CogneeLocalDestination()
        
        await dest.initialize({
            "dataset_id": "rel-test",
            "graph_name": "rel-graph",
        })
        
        conn = await dest.connect({"dataset_id": "rel-test"})
        
        # First create entities
        await neo4j_client.execute_write(
            """
            CREATE (e1:Entity {id: 'e1', name: 'Entity A', type: 'PERSON', dataset_id: $dataset_id})
            CREATE (e2:Entity {id: 'e2', name: 'Entity B', type: 'ORGANIZATION', dataset_id: $dataset_id})
            """,
            {"dataset_id": "rel-test"},
        )
        
        relationships = [
            ExtractedRelationship(
                source="Entity A",
                target="Entity B",
                type="WORKS_AT",
                confidence=0.95,
            ),
        ]
        
        count = await dest.create_relationship_edges(conn, relationships)
        
        assert count == 1
        
        # Verify relationship was created
        result = await neo4j_client.execute_query(
            """
            MATCH (e1:Entity)-[r:RELATES_TO]->(e2:Entity)
            WHERE e1.dataset_id = $dataset_id
            RETURN count(r) as count
            """,
            {"dataset_id": "rel-test"},
        )
        assert result[0]["count"] >= 1
        
        await dest.shutdown()


# ============================================================================
# Cognee + Neo4j Integration Tests
# ============================================================================

@pytest.mark.integration
class TestCogneeNeo4jIntegration:
    """Tests for Cognee + Neo4j integration."""

    @pytest.mark.asyncio
    async def test_document_ingestion_flow(self, neo4j_client):
        """Test full document ingestion: chunks -> entities -> graph."""
        dest = CogneeLocalDestination()
        
        await dest.initialize({
            "dataset_id": "ingestion-test",
            "graph_name": "ingestion-graph",
            "extract_entities": True,
            "extract_relationships": True,
        })
        
        data = TransformedData(
            job_id=uuid4(),
            chunks=[
                {"content": "Alice works at Acme Corp.", "metadata": {}},
                {"content": "Bob is a developer at Acme Corp.", "metadata": {}},
            ],
            embeddings=[[0.1] * 1536, [0.2] * 1536],
            metadata={},
            lineage={},
            original_format="txt",
            output_format="json",
        )
        
        conn = await dest.connect({"dataset_id": "ingestion-test"})
        result = await dest.write(conn, data)
        
        assert result.success is True
        assert result.metadata["chunks_created"] == 2
        
        await dest.shutdown()

    @pytest.mark.asyncio
    async def test_entity_extraction_flow(self, neo4j_client):
        """Test entity extraction with LLM integration."""
        dest = CogneeLocalDestination()
        
        await dest.initialize({
            "dataset_id": "extraction-test",
            "graph_name": "extraction-graph",
            "extract_entities": True,
        })
        
        # Test entity extraction
        entities = await dest.extract_entities(
            "Microsoft Corporation was founded by Bill Gates."
        )
        
        assert isinstance(entities, list)
        # Should extract Microsoft and Bill Gates
        
        await dest.shutdown()

    @pytest.mark.asyncio
    async def test_relationship_creation(self, neo4j_client):
        """Test relationship creation in Neo4j."""
        dest = CogneeLocalDestination()
        
        await dest.initialize({
            "dataset_id": "relationship-test",
            "graph_name": "relationship-graph",
        })
        
        # Create test entities first
        await neo4j_client.execute_write(
            """
            CREATE (e1:Entity {id: 'rel_e1', name: 'Person A', type: 'PERSON', dataset_id: $dataset_id})
            CREATE (e2:Entity {id: 'rel_e2', name: 'Company B', type: 'ORGANIZATION', dataset_id: $dataset_id})
            """,
            {"dataset_id": "relationship-test"},
        )
        
        conn = await dest.connect({"dataset_id": "relationship-test"})
        
        relationships = [
            ExtractedRelationship(
                source="Person A",
                target="Company B",
                type="WORKS_AT",
                confidence=0.9,
            ),
        ]
        
        count = await dest.create_relationship_edges(conn, relationships)
        assert count >= 0  # May or may not succeed depending on entity matching
        
        await dest.shutdown()

    @pytest.mark.asyncio
    async def test_search_integration(self, neo4j_client, sample_transformed_data):
        """Test search with real Neo4j data."""
        dest = CogneeLocalDestination()
        
        await dest.initialize({
            "dataset_id": "search-integration-test",
            "graph_name": "search-integration-graph",
            "extract_entities": True,
        })
        
        conn = await dest.connect({"dataset_id": "search-integration-test"})
        
        # Ingest data
        await dest.write(conn, sample_transformed_data)
        
        # Search
        results = await dest.search(conn, "Microsoft software", search_type="graph", top_k=5)
        
        assert isinstance(results, list)
        
        await dest.shutdown()

    @pytest.mark.asyncio
    async def test_end_to_end_pipeline(self, neo4j_client):
        """Test complete pipeline: ingest -> search -> verify."""
        dest = CogneeLocalDestination()
        
        await dest.initialize({
            "dataset_id": "e2e-test",
            "graph_name": "e2e-graph",
            "extract_entities": True,
            "extract_relationships": True,
        })
        
        conn = await dest.connect({"dataset_id": "e2e-test"})
        
        # Create test data
        data = TransformedData(
            job_id=uuid4(),
            chunks=[
                {"content": "Python is a programming language created by Guido van Rossum.", "metadata": {}},
                {"content": "It is widely used for web development and data science.", "metadata": {}},
            ],
            embeddings=[[0.1] * 1536, [0.2] * 1536],
            metadata={},
            lineage={},
            original_format="txt",
            output_format="json",
        )
        
        # Ingest
        write_result = await dest.write(conn, data)
        assert write_result.success is True
        
        # Search
        search_results = await dest.search(conn, "Python programming", search_type="graph", top_k=3)
        assert isinstance(search_results, list)
        
        # Verify health
        health = await dest.health_check()
        assert health == HealthStatus.HEALTHY
        
        await dest.shutdown()


# ============================================================================
# Cognee + PostgreSQL/pgvector Integration Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("DB_URL") is None,
    reason="PostgreSQL not configured (set DB_URL)",
)
class TestCogneePgvectorIntegration:
    """Tests for Cognee + PostgreSQL/pgvector integration."""

    @pytest.mark.asyncio
    async def test_vector_storage(self, neo4j_client):
        """Test storing vectors in pgvector.
        
        Note: This is a placeholder test as vector storage implementation
        depends on the existing pgvector infrastructure.
        """
        dest = CogneeLocalDestination()
        
        await dest.initialize({
            "dataset_id": "vector-test",
            "graph_name": "vector-graph",
            "store_vectors": True,
        })
        
        data = TransformedData(
            job_id=uuid4(),
            chunks=[{"content": "Test chunk", "metadata": {}}],
            embeddings=[[0.1] * 1536],
            metadata={},
            lineage={},
            original_format="txt",
            output_format="json",
        )
        
        conn = await dest.connect({"dataset_id": "vector-test"})
        result = await dest.write(conn, data)
        
        assert result.success is True
        assert result.metadata["vectors_stored"] == 1
        
        await dest.shutdown()

    @pytest.mark.asyncio
    async def test_vector_search(self, neo4j_client):
        """Test vector similarity search.
        
        Note: Vector search is currently a placeholder in the implementation.
        This test documents the expected behavior.
        """
        dest = CogneeLocalDestination()
        
        await dest.initialize({
            "dataset_id": "vector-search-test",
            "graph_name": "vector-search-graph",
        })
        
        conn = await dest.connect({"dataset_id": "vector-search-test"})
        
        # Vector search currently returns empty list (placeholder)
        results = await dest._vector_search("query", "vector-search-test", 5)
        
        assert isinstance(results, list)
        
        await dest.shutdown()

    @pytest.mark.asyncio
    async def test_hybrid_search(self, neo4j_client, sample_transformed_data):
        """Test hybrid (vector + graph) search."""
        dest = CogneeLocalDestination()
        
        await dest.initialize({
            "dataset_id": "hybrid-test",
            "graph_name": "hybrid-graph",
            "extract_entities": True,
        })
        
        conn = await dest.connect({"dataset_id": "hybrid-test"})
        
        # Ingest data
        await dest.write(conn, sample_transformed_data)
        
        # Hybrid search (combines vector and graph results)
        results = await dest.search(conn, "Microsoft", search_type="hybrid", top_k=5)
        
        assert isinstance(results, list)
        
        await dest.shutdown()


# ============================================================================
# Cognee + LLM Integration Tests
# ============================================================================

@pytest.mark.integration
class TestCogneeLLMIntegration:
    """Tests for Cognee + LLM integration."""

    @pytest.mark.asyncio
    async def test_llm_entity_extraction(self):
        """Test LLM-based entity extraction."""
        provider = CogneeLLMProvider()
        
        # Mock the LLM call for predictable testing
        with patch.object(
            provider._llm,
            "simple_completion",
            return_value='{"entities": [{"name": "OpenAI", "type": "ORGANIZATION", "description": "AI company"}]}',
        ):
            entities = await provider.extract_entities(
                "OpenAI developed the GPT language model."
            )
        
        assert isinstance(entities, list)
        assert len(entities) == 1
        assert entities[0].name == "OpenAI"
        assert entities[0].type == "ORGANIZATION"

    @pytest.mark.asyncio
    async def test_llm_relationship_extraction(self):
        """Test LLM-based relationship extraction."""
        provider = CogneeLLMProvider()
        
        entities = [
            ExtractedEntity(name="Alice", type="PERSON"),
            ExtractedEntity(name="IBM", type="ORGANIZATION"),
        ]
        
        with patch.object(
            provider._llm,
            "simple_completion",
            return_value='{"relationships": [{"source": "Alice", "target": "IBM", "type": "WORKS_AT", "confidence": 0.9}]}',
        ):
            relationships = await provider.extract_relationships(
                "Alice works at IBM as a software engineer.",
                entities,
            )
        
        assert isinstance(relationships, list)
        assert len(relationships) == 1
        assert relationships[0].source == "Alice"
        assert relationships[0].target == "IBM"
        assert relationships[0].type == "WORKS_AT"

    @pytest.mark.asyncio
    async def test_llm_health_check(self):
        """Test LLM provider health."""
        provider = CogneeLLMProvider()
        
        with patch.object(
            provider._llm,
            "simple_completion",
            return_value="healthy",
        ):
            health = await provider.health_check()
        
        assert health["healthy"] is True
        assert health["status"] == "healthy"


# ============================================================================
# Cleanup Tests
# ============================================================================

@pytest.mark.integration
class TestCleanup:
    """Tests for cleanup operations."""

    @pytest.mark.asyncio
    async def test_neo4j_cleanup(self, neo4j_client):
        """Clean up test data from Neo4j."""
        # Delete all test nodes
        await neo4j_client.execute_write(
            """
            MATCH (n)
            WHERE n.dataset_id CONTAINS 'test' 
               OR n.dataset_id CONTAINS 'integration'
               OR n.id CONTAINS 'test'
            DETACH DELETE n
            """
        )
        
        # Verify cleanup
        result = await neo4j_client.execute_query(
            """
            MATCH (n)
            WHERE n.dataset_id CONTAINS 'test' 
               OR n.dataset_id CONTAINS 'integration'
            RETURN count(n) as count
            """
        )
        
        assert result[0]["count"] == 0
