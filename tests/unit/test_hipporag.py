"""Unit tests for HippoRAG integration.

Tests for HippoRAGDestination, HippoRAGLLMProvider, and HippoRAGMockDestination
using mocked dependencies for isolation.
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import numpy as np
import pytest

from src.plugins.base import (
    Connection,
    HealthStatus,
    TransformedData,
    WriteResult,
)
from src.plugins.destinations.hipporag import (
    HippoRAGDestination,
    HippoRAGMockDestination,
    KnowledgeGraph,
    QAResult,
    RetrievalResult,
)
from src.plugins.destinations.hipporag_llm import HippoRAGLLMProvider

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_save_dir(tmp_path):
    """Create a temporary save directory."""
    return str(tmp_path / "hipporag_test")


@pytest.fixture
async def hipporag_destination(temp_save_dir):
    """Create an initialized HippoRAG destination."""
    with patch("src.plugins.destinations.hipporag.os.makedirs"):
        dest = HippoRAGDestination()
    
    # Mock the LLM provider to avoid external calls
    mock_llm = MagicMock()
    mock_llm.extract_triples = AsyncMock(return_value=[])
    mock_llm.embed_text = AsyncMock(return_value=np.array([0.1, 0.2, 0.3]))
    mock_llm.extract_query_entities = AsyncMock(return_value=["Steve Jobs"])
    mock_llm.answer_question = AsyncMock(return_value="Apple Inc.")
    mock_llm.health_check = AsyncMock(return_value={"healthy": True})
    
    dest._llm_provider = mock_llm
    
    with patch.object(dest, '_load_graph', new_callable=AsyncMock):
        await dest.initialize({
            "save_dir": temp_save_dir,
            "llm_model": "azure/gpt-4.1",
            "embedding_model": "azure/text-embedding-3-small",
            "retrieval_k": 10,
        })
    
    # Store mock for use in tests
    dest._mock_llm = mock_llm
    yield dest
    
    # Cleanup
    await dest.shutdown()


@pytest.fixture
def sample_transformed_data():
    """Create sample transformed data for testing."""
    return TransformedData(
        job_id=UUID("12345678-1234-1234-1234-123456789abc"),
        chunks=[
            {
                "content": "Steve Jobs founded Apple in 1976.",
                "metadata": {"page": 1, "index": 0},
            },
            {
                "content": "Apple is headquartered in Cupertino, California.",
                "metadata": {"page": 1, "index": 1},
            },
        ],
        metadata={"title": "Apple History", "author": "Test Author"},
        lineage={"source": "test", "parser": "test_parser"},
        original_format="pdf",
        output_format="json",
    )


@pytest.fixture
def sample_connection():
    """Create a sample connection."""
    return Connection(
        id=UUID(int=hash("hipporag") % (2**32)),
        plugin_id="hipporag",
        config={"save_dir": "/tmp/hipporag"},
    )


@pytest.fixture
def sample_triples():
    """Create sample OpenIE triples."""
    return [
        ("Steve Jobs", "founded", "Apple"),
        ("Apple", "headquartered in", "Cupertino"),
        ("Cupertino", "located in", "California"),
    ]


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider."""
    mock = MagicMock()
    mock.extract_triples = AsyncMock(return_value=[])
    mock.extract_query_entities = AsyncMock(return_value=[])
    mock.answer_question = AsyncMock(return_value="Test answer")
    mock.embed_text = AsyncMock(return_value=np.array([0.1, 0.2, 0.3]))
    mock.health_check = AsyncMock(return_value={"healthy": True})
    mock.simple_completion = AsyncMock(return_value="{}")
    return mock


# ============================================================================
# HippoRAGDestination Tests
# ============================================================================

@pytest.mark.unit
class TestHippoRAGDestination:
    """Tests for HippoRAGDestination plugin."""

    @patch("src.plugins.destinations.hipporag.os.makedirs")
    def test_init(self, mock_makedirs):
        """Test destination initialization."""
        dest = HippoRAGDestination()
        
        assert dest._config == {}
        assert dest._save_dir == "/data/hipporag"  # Default from env
        assert dest._llm_model == "azure/gpt-4.1"
        assert dest._embedding_model == "azure/text-embedding-3-small"
        assert dest._retrieval_k == 10
        assert dest._llm_provider is None
        assert dest._is_initialized is False
        mock_makedirs.assert_called()

    @patch("src.plugins.destinations.hipporag.os.makedirs")
    def test_metadata(self, mock_makedirs):
        """Test destination metadata."""
        dest = HippoRAGDestination()
        metadata = dest.metadata
        
        assert metadata.id == "hipporag"
        assert metadata.name == "HippoRAG Multi-Hop Reasoning"
        assert metadata.version == "1.0.0"
        assert "json" in metadata.supported_formats
        assert "text" in metadata.supported_formats
        assert "markdown" in metadata.supported_formats
        assert metadata.requires_auth is False
        assert "save_dir" in metadata.config_schema["properties"]
        assert "llm_model" in metadata.config_schema["properties"]
        assert "embedding_model" in metadata.config_schema["properties"]

    @pytest.mark.asyncio
    @patch("src.plugins.destinations.hipporag.os.makedirs")
    async def test_initialize_with_config(self, mock_makedirs, temp_save_dir):
        """Test initialization with config parameters."""
        dest = HippoRAGDestination()
        
        with patch.object(dest, '_load_graph', new_callable=AsyncMock):
            await dest.initialize({
                "save_dir": temp_save_dir,
                "llm_model": "custom/model",
                "embedding_model": "custom/embedding",
                "retrieval_k": 20,
            })
        
        assert dest._save_dir == temp_save_dir
        assert dest._llm_model == "custom/model"
        assert dest._embedding_model == "custom/embedding"
        assert dest._retrieval_k == 20
        assert dest._is_initialized is True
        assert dest._llm_provider is not None

    @pytest.mark.asyncio
    @patch("src.plugins.destinations.hipporag.os.makedirs")
    async def test_initialize_with_env_vars(self, mock_makedirs, temp_save_dir):
        """Test initialization with environment variables."""
        with patch.dict(os.environ, {
            "HIPPO_SAVE_DIR": temp_save_dir,
            "HIPPO_LLM_MODEL": "env/llm-model",
            "HIPPO_EMBEDDING_MODEL": "env/embedding-model",
            "HIPPO_RETRIEVAL_K": "15",
        }):
            dest = HippoRAGDestination()
            
            with patch.object(dest, '_load_graph', new_callable=AsyncMock):
                await dest.initialize({})
            
            assert dest._save_dir == temp_save_dir
            assert dest._llm_model == "env/llm-model"
            assert dest._embedding_model == "env/embedding-model"
            assert dest._retrieval_k == 15
            
            await dest.shutdown()

    @pytest.mark.asyncio
    @patch("src.plugins.destinations.hipporag.os.makedirs")
    async def test_initialize_creates_save_dir(self, mock_makedirs, temp_save_dir):
        """Test that initialization creates save directory."""
        dest = HippoRAGDestination()
        test_dir = os.path.join(temp_save_dir, "nested", "dir")
        
        with patch.object(dest, '_load_graph', new_callable=AsyncMock):
            await dest.initialize({"save_dir": test_dir})
        
        assert os.path.exists(test_dir)
        
        await dest.shutdown()


@pytest.mark.unit
class TestHippoRAGConnection:
    """Tests for HippoRAG connection handling."""

    @pytest.mark.asyncio
    async def test_connect_creates_connection(self, hipporag_destination):
        """Test connect creates a connection."""
        conn = await hipporag_destination.connect({})
        
        assert isinstance(conn, Connection)
        assert conn.plugin_id == "hipporag"
        assert conn.config["save_dir"] == hipporag_destination._save_dir

    @pytest.mark.asyncio
    @patch("src.plugins.destinations.hipporag.os.makedirs")
    async def test_connect_not_initialized(self, mock_makedirs):
        """Test connect fails when not initialized."""
        dest = HippoRAGDestination()
        
        with pytest.raises(ConnectionError, match="not initialized"):
            await dest.connect({})


@pytest.mark.unit
class TestHippoRAGWrite:
    """Tests for HippoRAG write operations."""

    @pytest.mark.asyncio
    async def test_write_success(self, hipporag_destination, sample_connection, sample_transformed_data):
        """Test successful write operation."""
        # Configure mock to return triples
        hipporag_destination._mock_llm.extract_triples.return_value = [
            ("Steve Jobs", "founded", "Apple"),
        ]
        
        result = await hipporag_destination.write(sample_connection, sample_transformed_data)
        
        assert isinstance(result, WriteResult)
        assert result.success is True
        assert result.destination_id == "hipporag"
        assert result.records_written == 2
        assert result.metadata["chunks_indexed"] == 2
        assert result.metadata["triples_total"] == 1

    @pytest.mark.asyncio
    @patch("src.plugins.destinations.hipporag.os.makedirs")
    async def test_write_not_initialized(self, mock_makedirs, sample_connection, sample_transformed_data):
        """Test write when not initialized."""
        dest = HippoRAGDestination()
        
        result = await dest.write(sample_connection, sample_transformed_data)
        
        assert isinstance(result, WriteResult)
        assert result.success is False
        assert "not initialized" in result.error.lower()

    @pytest.mark.asyncio
    async def test_write_no_content(self, hipporag_destination, sample_connection):
        """Test write with empty content."""
        data = TransformedData(
            job_id=uuid4(),
            chunks=[{"content": "", "metadata": {}}],
        )
        
        result = await hipporag_destination.write(sample_connection, data)
        
        assert result.success is True
        assert result.records_written == 0
        assert "No text content" in result.metadata["message"]

    @pytest.mark.asyncio
    async def test_write_extracts_text_from_chunks(self, hipporag_destination, sample_connection):
        """Test that write extracts text from chunks."""
        hipporag_destination._mock_llm.extract_triples.return_value = [
            ("Entity", "relation", "Object"),
        ]
        
        data = TransformedData(
            job_id=uuid4(),
            chunks=[
                {"content": "First chunk text", "metadata": {"index": 0}},
                {"content": "Second chunk text", "metadata": {"index": 1}},
            ],
        )
        
        result = await hipporag_destination.write(sample_connection, data)
        
        assert result.records_written == 2
        # Verify LLM was called with the text
        assert hipporag_destination._mock_llm.extract_triples.call_count == 2


@pytest.mark.unit
class TestHippoRAGIndexing:
    """Tests for HippoRAG document indexing."""

    @pytest.mark.asyncio
    async def test_index_documents(self, hipporag_destination, sample_triples):
        """Test document indexing."""
        hipporag_destination._mock_llm.extract_triples.return_value = sample_triples
        
        texts = ["Steve Jobs founded Apple.", "Apple is in Cupertino."]
        metadatas = [{"job_id": "test1", "chunk_index": 0}, {"job_id": "test1", "chunk_index": 1}]
        
        await hipporag_destination.index_documents(texts, metadatas)
        
        # Check graph state
        assert len(hipporag_destination._graph.triples) == 3
        assert "Steve Jobs" in hipporag_destination._graph.entities
        assert "Apple" in hipporag_destination._graph.entities

    @pytest.mark.asyncio
    async def test_index_documents_empty_triples(self, hipporag_destination):
        """Test indexing when no triples are extracted."""
        hipporag_destination._mock_llm.extract_triples.return_value = []
        
        texts = ["This is a simple sentence with no clear triples."]
        metadatas = [{"job_id": "test", "chunk_index": 0}]
        
        await hipporag_destination.index_documents(texts, metadatas)
        
        # Should still add passage without triples
        assert len(hipporag_destination._graph.passages) == 1

    @pytest.mark.asyncio
    async def test_run_openie(self, hipporag_destination, sample_triples):
        """Test OpenIE triple extraction."""
        hipporag_destination._mock_llm.extract_triples.return_value = sample_triples
        
        triples = await hipporag_destination._run_openie("Some text")
        
        assert triples == sample_triples

    @pytest.mark.asyncio
    async def test_run_openie_error(self, hipporag_destination):
        """Test OpenIE handles errors gracefully."""
        hipporag_destination._mock_llm.extract_triples.side_effect = Exception("LLM error")
        
        triples = await hipporag_destination._run_openie("Some text")
        
        assert triples == []


@pytest.mark.unit
class TestHippoRAGRetrieval:
    """Tests for HippoRAG multi-hop retrieval."""

    @pytest.mark.asyncio
    async def test_retrieve(self, hipporag_destination):
        """Test multi-hop retrieval."""
        # Setup graph with data
        hipporag_destination._graph.add_triple(
            "Steve Jobs", "founded", "Apple",
            "passage_1", "Steve Jobs founded Apple in 1976."
        )
        hipporag_destination._graph.add_triple(
            "Apple", "headquartered in", "Cupertino",
            "passage_2", "Apple is headquartered in Cupertino."
        )
        
        hipporag_destination._mock_llm.extract_query_entities.return_value = ["Steve Jobs"]
        
        results = await hipporag_destination.retrieve(["What did Steve Jobs found?"], num_to_retrieve=2)
        
        assert len(results) == 1
        assert isinstance(results[0], RetrievalResult)
        assert results[0].query == "What did Steve Jobs found?"
        assert len(results[0].passages) > 0
        assert "Steve Jobs" in results[0].entities

    @pytest.mark.asyncio
    @patch("src.plugins.destinations.hipporag.os.makedirs")
    async def test_retrieve_not_initialized(self, mock_makedirs):
        """Test retrieval when not initialized."""
        dest = HippoRAGDestination()
        
        with pytest.raises(RuntimeError, match="not initialized"):
            await dest.retrieve(["query"])

    @pytest.mark.asyncio
    async def test_retrieve_empty_graph(self, hipporag_destination):
        """Test retrieval with empty graph."""
        hipporag_destination._mock_llm.extract_query_entities.return_value = ["Unknown Entity"]
        
        results = await hipporag_destination.retrieve(["Who is Unknown Entity?"])
        
        assert len(results) == 1
        # Should return empty result gracefully

    @pytest.mark.asyncio
    async def test_extract_query_entities(self, hipporag_destination):
        """Test query entity extraction."""
        hipporag_destination._mock_llm.extract_query_entities.return_value = ["Steve Jobs", "Apple"]
        
        # Add entity to graph
        hipporag_destination._graph.entities["Steve Jobs"] = {"name": "Steve Jobs"}
        
        entities = await hipporag_destination._extract_query_entities("What did Steve Jobs found?")
        
        assert "Steve Jobs" in entities

    @pytest.mark.asyncio
    async def test_extract_query_entities_fallback(self, hipporag_destination):
        """Test entity extraction fallback to keyword matching."""
        hipporag_destination._mock_llm.extract_query_entities.return_value = []
        
        # Add entity to graph
        hipporag_destination._graph.entities["Steve Jobs"] = {"name": "Steve Jobs"}
        
        entities = await hipporag_destination._extract_query_entities("Tell me about Steve Jobs")
        
        assert "Steve Jobs" in entities

    @pytest.mark.asyncio
    async def test_run_ppr(self, hipporag_destination):
        """Test Personalized PageRank scoring."""
        # Setup graph
        hipporag_destination._graph.add_triple(
            "A", "connects_to", "B",
            "p1", "A connects to B"
        )
        hipporag_destination._graph.add_triple(
            "B", "connects_to", "C",
            "p2", "B connects to C"
        )
        
        results = await hipporag_destination._run_ppr(["A"], top_k=2)
        
        assert len(results) <= 2
        # A's passage should have high score
        passage_ids = [r[0] for r in results]
        assert "p1" in passage_ids


@pytest.mark.unit
class TestHippoRAGRAGQA:
    """Tests for HippoRAG RAG QA."""

    @pytest.mark.asyncio
    async def test_rag_qa(self, hipporag_destination):
        """Test RAG QA."""
        # Setup graph
        hipporag_destination._graph.add_triple(
            "Steve Jobs", "founded", "Apple",
            "passage_1", "Steve Jobs founded Apple in 1976."
        )
        
        hipporag_destination._mock_llm.extract_query_entities.return_value = ["Steve Jobs"]
        hipporag_destination._mock_llm.answer_question.return_value = "Apple Inc."
        
        results = await hipporag_destination.rag_qa(["What company did Steve Jobs found?"])
        
        assert len(results) == 1
        assert isinstance(results[0], QAResult)
        assert results[0].answer == "Apple Inc."
        assert len(results[0].sources) > 0

    @pytest.mark.asyncio
    async def test_rag_qa_no_passages(self, hipporag_destination):
        """Test RAG QA when no passages found."""
        hipporag_destination._mock_llm.extract_query_entities.return_value = []
        
        results = await hipporag_destination.rag_qa(["Unknown question?"])
        
        assert len(results) == 1
        assert "couldn't find" in results[0].answer.lower()

    @pytest.mark.asyncio
    @patch("src.plugins.destinations.hipporag.os.makedirs")
    async def test_rag_qa_not_initialized(self, mock_makedirs):
        """Test RAG QA when not initialized."""
        dest = HippoRAGDestination()
        
        results = await dest.rag_qa(["What is AI?"])
        
        assert len(results) == 1
        assert "not initialized" in results[0].answer.lower()


@pytest.mark.unit
class TestHippoRAGHealthCheck:
    """Tests for HippoRAG health check."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, hipporag_destination):
        """Test health check returns healthy."""
        health = await hipporag_destination.health_check()
        
        assert health == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    @patch("src.plugins.destinations.hipporag.os.makedirs")
    async def test_health_check_not_initialized(self, mock_makedirs):
        """Test health check when not initialized."""
        dest = HippoRAGDestination()
        
        health = await dest.health_check()
        
        assert health == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_health_check_unwritable_dir(self, hipporag_destination):
        """Test health check with unwritable save directory."""
        # Make save dir read-only
        os.chmod(hipporag_destination._save_dir, 0o555)
        
        try:
            health = await hipporag_destination.health_check()
            assert health == HealthStatus.UNHEALTHY
        finally:
            # Restore permissions for cleanup
            os.chmod(hipporag_destination._save_dir, 0o755)

    @pytest.mark.asyncio
    async def test_health_check_llm_unhealthy(self, hipporag_destination):
        """Test health check when LLM is unhealthy."""
        hipporag_destination._mock_llm.health_check.return_value = {"healthy": False}
        
        health = await hipporag_destination.health_check()
        
        assert health == HealthStatus.DEGRADED


@pytest.mark.unit
class TestHippoRAGPersistence:
    """Tests for HippoRAG graph persistence."""

    @pytest.mark.asyncio
    @patch("src.plugins.destinations.hipporag.os.makedirs")
    async def test_save_and_load_graph(self, mock_makedirs, temp_save_dir):
        """Test graph persistence."""
        # First session - create data
        dest1 = HippoRAGDestination()
        
        mock_llm = MagicMock()
        mock_llm.extract_triples = AsyncMock(return_value=[
            ("Subject", "predicate", "Object"),
        ])
        mock_llm.embed_text = AsyncMock(return_value=np.array([0.1]))
        mock_llm.health_check = AsyncMock(return_value={"healthy": True})
        dest1._llm_provider = mock_llm
        
        with patch.object(dest1, '_load_graph', new_callable=AsyncMock):
            await dest1.initialize({"save_dir": temp_save_dir})
        
        conn = await dest1.connect({})
        data = TransformedData(
            job_id=uuid4(),
            chunks=[{"content": "Subject predicate Object.", "metadata": {}}],
        )
        await dest1.write(conn, data)
        await dest1.shutdown()
        
        # Second session - verify data persisted
        dest2 = HippoRAGDestination()
        dest2._llm_provider = mock_llm
        await dest2.initialize({"save_dir": temp_save_dir})
        
        assert "Subject" in dest2._graph.entities
        assert "Object" in dest2._graph.entities
        assert ("Subject", "predicate", "Object") in dest2._graph.triples
        
        await dest2.shutdown()


@pytest.mark.unit
class TestKnowledgeGraph:
    """Tests for KnowledgeGraph dataclass."""

    def test_add_triple(self):
        """Test adding triple to knowledge graph."""
        kg = KnowledgeGraph()
        
        kg.add_triple("Steve Jobs", "founded", "Apple", "p1", "Steve Jobs founded Apple.")
        
        assert ("Steve Jobs", "founded", "Apple") in kg.triples
        assert "Steve Jobs" in kg.entities
        assert "Apple" in kg.entities
        assert kg.passages["p1"] == "Steve Jobs founded Apple."
        assert "p1" in kg.entity_passages["Steve Jobs"]
        assert "p1" in kg.entity_passages["Apple"]

    def test_add_triple_duplicate(self):
        """Test that duplicate triples are not added."""
        kg = KnowledgeGraph()
        
        kg.add_triple("A", "rel", "B", "p1", "Text")
        kg.add_triple("A", "rel", "B", "p2", "Different text")
        
        assert len(kg.triples) == 1  # Duplicate not added

    def test_get_entity_passages(self):
        """Test getting passages for entity."""
        kg = KnowledgeGraph()
        kg.add_triple("A", "rel", "B", "p1", "Passage 1")
        kg.add_triple("A", "rel", "C", "p2", "Passage 2")
        
        passages = kg.get_entity_passages("A")
        
        assert len(passages) == 2
        assert "Passage 1" in passages
        assert "Passage 2" in passages

    def test_get_related_entities(self):
        """Test getting related entities."""
        kg = KnowledgeGraph()
        kg.add_triple("A", "rel1", "B", "p1", "Text")
        kg.add_triple("B", "rel2", "C", "p2", "Text")
        kg.add_triple("D", "rel3", "E", "p3", "Text")
        
        related = kg.get_related_entities("B")
        
        assert len(related) == 2  # A->B and B->C


# ============================================================================
# HippoRAGLLMProvider Tests
# ============================================================================

@pytest.mark.unit
class TestHippoRAGLLMProvider:
    """Tests for HippoRAGLLMProvider."""

    def test_init_default(self):
        """Test initialization with default LLM provider."""
        with patch("src.plugins.destinations.hipporag_llm.LLMProvider") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.return_value = mock_llm
            
            provider = HippoRAGLLMProvider()
            
            assert provider._llm is mock_llm
            assert provider.llm_model == "azure/gpt-4.1"
            assert provider.embedding_model == "azure/text-embedding-3-small"

    def test_init_with_custom_provider(self, mock_llm_provider):
        """Test initialization with custom LLM provider."""
        provider = HippoRAGLLMProvider(llm_provider=mock_llm_provider)
        
        assert provider._llm is mock_llm_provider

    @pytest.mark.asyncio
    async def test_extract_triples_success(self, mock_llm_provider):
        """Test successful OpenIE triple extraction."""
        mock_llm_provider.simple_completion.return_value = json.dumps({
            "triples": [
                ["Steve Jobs", "founded", "Apple"],
                ["Apple", "located in", "Cupertino"],
            ]
        })
        
        provider = HippoRAGLLMProvider(llm_provider=mock_llm_provider)
        triples = await provider.extract_triples("Steve Jobs founded Apple in Cupertino.")
        
        assert len(triples) == 2
        assert triples[0] == ("Steve Jobs", "founded", "Apple")

    @pytest.mark.asyncio
    async def test_extract_triples_empty_text(self, mock_llm_provider):
        """Test triple extraction with empty text."""
        provider = HippoRAGLLMProvider(llm_provider=mock_llm_provider)
        
        triples = await provider.extract_triples("")
        assert triples == []
        
        triples = await provider.extract_triples("   ")
        assert triples == []
        
        mock_llm_provider.simple_completion.assert_not_called()

    @pytest.mark.asyncio
    async def test_extract_triples_json_in_markdown(self, mock_llm_provider):
        """Test extraction when JSON is wrapped in markdown."""
        mock_llm_provider.simple_completion.return_value = '''```json
        {"triples": [["A", "rel", "B"]]}
        ```'''
        
        provider = HippoRAGLLMProvider(llm_provider=mock_llm_provider)
        triples = await provider.extract_triples("Text")
        
        assert len(triples) == 1
        assert triples[0] == ("A", "rel", "B")

    @pytest.mark.asyncio
    async def test_extract_triples_llm_error(self, mock_llm_provider):
        """Test triple extraction handles LLM errors."""
        mock_llm_provider.simple_completion.side_effect = Exception("LLM error")
        
        provider = HippoRAGLLMProvider(llm_provider=mock_llm_provider)
        triples = await provider.extract_triples("Text")
        
        assert triples == []

    @pytest.mark.asyncio
    async def test_extract_query_entities_success(self, mock_llm_provider):
        """Test successful query entity extraction."""
        mock_llm_provider.simple_completion.return_value = json.dumps({
            "entities": ["Steve Jobs", "Apple", "iPhone"]
        })
        
        provider = HippoRAGLLMProvider(llm_provider=mock_llm_provider)
        entities = await provider.extract_query_entities("What iPhone did Steve Jobs create at Apple?")
        
        assert len(entities) == 3
        assert "Steve Jobs" in entities

    @pytest.mark.asyncio
    async def test_extract_query_entities_empty_query(self, mock_llm_provider):
        """Test query entity extraction with empty query."""
        provider = HippoRAGLLMProvider(llm_provider=mock_llm_provider)
        
        entities = await provider.extract_query_entities("")
        assert entities == []

    @pytest.mark.asyncio
    async def test_extract_query_entities_fallback(self, mock_llm_provider):
        """Test fallback when entity extraction fails."""
        mock_llm_provider.simple_completion.side_effect = Exception("LLM error")
        
        provider = HippoRAGLLMProvider(llm_provider=mock_llm_provider)
        entities = await provider.extract_query_entities("What about machine learning?")
        
        # Should fall back to word extraction
        assert len(entities) > 0

    @pytest.mark.asyncio
    async def test_answer_question_success(self, mock_llm_provider):
        """Test successful question answering."""
        mock_llm_provider.simple_completion.return_value = "Steve Jobs founded Apple Inc."
        
        provider = HippoRAGLLMProvider(llm_provider=mock_llm_provider)
        answer = await provider.answer_question(
            "What company did Steve Jobs found?",
            ["Steve Jobs founded Apple Inc. in 1976."]
        )
        
        assert "Apple" in answer

    @pytest.mark.asyncio
    async def test_answer_question_empty_context(self, mock_llm_provider):
        """Test question answering with empty context."""
        provider = HippoRAGLLMProvider(llm_provider=mock_llm_provider)
        
        answer = await provider.answer_question("Question?", [])
        
        assert "No relevant context" in answer

    @pytest.mark.asyncio
    async def test_answer_question_error(self, mock_llm_provider):
        """Test question answering handles errors."""
        mock_llm_provider.simple_completion.side_effect = Exception("LLM error")
        
        provider = HippoRAGLLMProvider(llm_provider=mock_llm_provider)
        answer = await provider.answer_question("Question?", ["Context passage."])
        
        assert "Context" in answer  # Fallback to concatenating contexts

    @pytest.mark.asyncio
    async def test_embed_text(self, mock_llm_provider):
        """Test text embedding."""
        with patch("src.plugins.destinations.hipporag_llm.litellm") as mock_litellm:
            mock_litellm.aembedding = AsyncMock(return_value=MagicMock(
                data=[{"embedding": [0.1, 0.2, 0.3]}]
            ))
            
            provider = HippoRAGLLMProvider(llm_provider=mock_llm_provider)
            embedding = await provider.embed_text("Test text")
            
            assert embedding is not None
            assert len(embedding) == 3

    @pytest.mark.asyncio
    async def test_embed_text_empty(self, mock_llm_provider):
        """Test embedding empty text."""
        provider = HippoRAGLLMProvider(llm_provider=mock_llm_provider)
        
        embedding = await provider.embed_text("")
        assert embedding is None

    @pytest.mark.asyncio
    async def test_embed_text_error(self, mock_llm_provider):
        """Test embedding handles errors."""
        with patch("src.plugins.destinations.hipporag_llm.litellm") as mock_litellm:
            mock_litellm.aembedding = AsyncMock(side_effect=Exception("Embedding error"))
            
            provider = HippoRAGLLMProvider(llm_provider=mock_llm_provider)
            embedding = await provider.embed_text("Test text")
            
            assert embedding is None

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, mock_llm_provider):
        """Test health check returns healthy."""
        mock_llm_provider.simple_completion.return_value = "healthy"
        
        provider = HippoRAGLLMProvider(llm_provider=mock_llm_provider)
        health = await provider.health_check()
        
        assert health["healthy"] is True
        assert health["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, mock_llm_provider):
        """Test health check returns unhealthy."""
        mock_llm_provider.simple_completion.return_value = "unexpected"
        
        provider = HippoRAGLLMProvider(llm_provider=mock_llm_provider)
        health = await provider.health_check()
        
        assert health["healthy"] is False
        assert health["status"] == "degraded"


@pytest.mark.unit
class TestHippoRAGLLMProviderHelpers:
    """Tests for HippoRAGLLMProvider helper methods."""

    def test_extract_json_from_markdown_with_json_tag(self):
        """Test extracting JSON from markdown with json tag."""
        provider = HippoRAGLLMProvider()
        text = '```json\n{"key": "value"}\n```'
        
        result = provider._extract_json_from_markdown(text)
        
        assert result == '{"key": "value"}'

    def test_extract_json_from_markdown_without_tag(self):
        """Test extracting JSON from markdown without json tag."""
        provider = HippoRAGLLMProvider()
        text = '```\n{"key": "value"}\n```'
        
        result = provider._extract_json_from_markdown(text)
        
        assert result == '{"key": "value"}'

    def test_extract_json_from_markdown_no_code_block(self):
        """Test returning original when no code block."""
        provider = HippoRAGLLMProvider()
        text = '{"key": "value"}'
        
        result = provider._extract_json_from_markdown(text)
        
        assert result == '{"key": "value"}'


# ============================================================================
# HippoRAGMockDestination Tests
# ============================================================================

@pytest.mark.unit
class TestHippoRAGMockDestination:
    """Tests for HippoRAGMockDestination."""

    def test_init(self):
        """Test mock destination initialization."""
        dest = HippoRAGMockDestination()
        
        assert dest._config == {}
        assert dest._documents == []
        assert dest._is_initialized is False

    def test_metadata(self):
        """Test mock destination metadata."""
        dest = HippoRAGMockDestination()
        metadata = dest.metadata
        
        assert metadata.id == "hipporag_mock"
        assert metadata.name == "HippoRAG Mock (Testing)"
        assert metadata.requires_auth is False

    @pytest.mark.asyncio
    async def test_mock_initialize(self):
        """Test mock initialization."""
        dest = HippoRAGMockDestination()
        await dest.initialize({"test": "config"})
        
        assert dest._config == {"test": "config"}
        assert dest._is_initialized is True

    @pytest.mark.asyncio
    async def test_mock_connect(self):
        """Test mock connect."""
        dest = HippoRAGMockDestination()
        await dest.initialize({})
        
        conn = await dest.connect({})
        
        assert isinstance(conn, Connection)
        assert conn.plugin_id == "hipporag_mock"

    @pytest.mark.asyncio
    async def test_mock_write(self):
        """Test mock write."""
        dest = HippoRAGMockDestination()
        await dest.initialize({})
        
        data = TransformedData(
            job_id=UUID("12345678-1234-1234-1234-123456789abc"),
            chunks=[
                {"content": "Chunk 1", "metadata": {}},
                {"content": "Chunk 2", "metadata": {}},
            ],
        )
        
        result = await dest.write(None, data)
        
        assert isinstance(result, WriteResult)
        assert result.success is True
        assert result.destination_id == "hipporag_mock"
        assert result.records_written == 2
        assert len(dest._documents) == 2

    @pytest.mark.asyncio
    async def test_mock_storage(self):
        """Test mock data storage."""
        dest = HippoRAGMockDestination()
        await dest.initialize({})
        
        data1 = TransformedData(
            job_id=uuid4(),
            chunks=[{"content": "Document 1", "metadata": {}}],
        )
        data2 = TransformedData(
            job_id=uuid4(),
            chunks=[{"content": "Document 2", "metadata": {}}],
        )
        
        await dest.write(None, data1)
        await dest.write(None, data2)
        
        assert len(dest._documents) == 2
        assert dest._documents[0]["content"] == "Document 1"
        assert dest._documents[1]["content"] == "Document 2"

    @pytest.mark.asyncio
    async def test_mock_retrieve(self):
        """Test mock retrieval."""
        dest = HippoRAGMockDestination()
        await dest.initialize({})
        
        # Add documents
        data = TransformedData(
            job_id=uuid4(),
            chunks=[
                {"content": "Apple is a technology company", "metadata": {}},
                {"content": "Microsoft is also a tech company", "metadata": {}},
            ],
        )
        await dest.write(None, data)
        
        results = await dest.retrieve(["Apple technology"])
        
        assert len(results) == 1
        assert len(results[0].passages) > 0

    @pytest.mark.asyncio
    async def test_mock_rag_qa(self):
        """Test mock RAG QA."""
        dest = HippoRAGMockDestination()
        await dest.initialize({})
        
        results = await dest.rag_qa(["What is AI?"])
        
        assert len(results) == 1
        assert results[0].query == "What is AI?"
        assert "Mock answer" in results[0].answer

    @pytest.mark.asyncio
    async def test_mock_health_check(self):
        """Test mock health check."""
        dest = HippoRAGMockDestination()
        
        # Not initialized - mock returns HEALTHY always
        health = await dest.health_check()
        assert health == HealthStatus.HEALTHY
        
        # After initialization
        await dest.initialize({})
        health = await dest.health_check()
        assert health == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_mock_shutdown(self):
        """Test mock shutdown."""
        dest = HippoRAGMockDestination()
        await dest.initialize({})
        await dest.write(None, TransformedData(job_id=uuid4(), chunks=[{"content": "Test"}]))
        
        await dest.shutdown()
        
        assert dest._is_initialized is False
        assert dest._documents == []

    @pytest.mark.asyncio
    async def test_mock_write_not_initialized(self):
        """Test mock write when not initialized."""
        dest = HippoRAGMockDestination()
        
        data = TransformedData(job_id=uuid4(), chunks=[{"content": "Test"}])
        result = await dest.write(None, data)
        
        assert result.success is False
        assert "not initialized" in result.error.lower()


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.unit
class TestHippoRAGErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_write_handles_exception(self, hipporag_destination, sample_connection, sample_transformed_data):
        """Test write handles unexpected exceptions."""
        # Cause an error by making extract_triples fail
        hipporag_destination._mock_llm.extract_triples.side_effect = Exception("Unexpected error")
        
        result = await hipporag_destination.write(sample_connection, sample_transformed_data)
        
        # Should still succeed (errors are logged but not raised)
        assert isinstance(result, WriteResult)
        # The operation continues even if individual passages fail

    @pytest.mark.asyncio
    async def test_retrieve_handles_exception(self, hipporag_destination):
        """Test retrieve handles exceptions gracefully."""
        hipporag_destination._mock_llm.extract_query_entities.side_effect = Exception("LLM error")
        
        results = await hipporag_destination.retrieve(["Query?"])
        
        # Should return empty result, not raise
        assert len(results) == 1
        assert results[0].passages == []

    @pytest.mark.asyncio
    async def test_rag_qa_handles_exception(self, hipporag_destination):
        """Test RAG QA handles exceptions."""
        hipporag_destination._mock_llm.extract_query_entities.side_effect = Exception("Error")
        
        results = await hipporag_destination.rag_qa(["Question?"])
        
        assert len(results) == 1
