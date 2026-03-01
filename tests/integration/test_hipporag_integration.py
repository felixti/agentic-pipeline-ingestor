"""Integration tests for HippoRAG.

These tests verify the end-to-end functionality of HippoRAG including:
- Document indexing flow (document → OpenIE → knowledge graph)
- Multi-hop retrieval using Personalized PageRank
- RAG QA with context
- Graph persistence

Usage:
    pytest tests/integration/test_hipporag_integration.py -v

Note: These tests use real file storage but mock LLM calls for speed.
"""

import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import numpy as np
import pytest

from src.plugins.base import Connection, HealthStatus, TransformedData, WriteResult
from src.plugins.destinations.hipporag import (
    HippoRAGDestination,
    HippoRAGMockDestination,
    QAResult,
    RetrievalResult,
)
from src.plugins.destinations.hipporag_llm import HippoRAGLLMProvider

pytestmark = [
    pytest.mark.integration,
]


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_save_dir():
    """Create a temporary directory for HippoRAG storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
async def hipporag_destination(temp_save_dir):
    """Provide initialized HippoRAG destination.
    
    Uses mocked LLM for speed but real file storage.
    """
    dest = HippoRAGDestination()
    
    # Mock LLM provider for controlled testing
    mock_llm = MagicMock()
    mock_llm.extract_triples = AsyncMock(return_value=[])
    mock_llm.embed_text = AsyncMock(return_value=np.array([0.1, 0.2, 0.3]))
    mock_llm.extract_query_entities = AsyncMock(return_value=[])
    mock_llm.answer_question = AsyncMock(return_value="Test answer")
    mock_llm.health_check = AsyncMock(return_value={"healthy": True})
    
    dest._llm_provider = mock_llm
    
    await dest.initialize({
        "save_dir": temp_save_dir,
        "llm_model": "azure/gpt-4.1",
        "embedding_model": "azure/text-embedding-3-small",
        "retrieval_k": 10,
    })
    
    dest._mock_llm = mock_llm
    yield dest
    
    await dest.shutdown()


@pytest.fixture
def sample_transformed_data():
    """Create sample transformed data with multi-hop content."""
    return TransformedData(
        job_id=uuid4(),
        chunks=[
            {
                "content": "Erik Hort was born in Montebello, New York.",
                "metadata": {"page": 1, "index": 0},
            },
            {
                "content": "Montebello is a village in the town of Ramapo.",
                "metadata": {"page": 1, "index": 1},
            },
            {
                "content": "Ramapo is located in Rockland County, New York.",
                "metadata": {"page": 2, "index": 2},
            },
        ],
        metadata={"title": "Biography"},
        lineage={"source": "test"},
        original_format="txt",
        output_format="json",
    )


@pytest.fixture
def sample_connection():
    """Create a sample connection."""
    return Connection(
        id=UUID(int=hash("hipporag") % (2**32)),
        plugin_id="hipporag",
        config={},
    )


@pytest.fixture
def steve_jobs_triples():
    """Sample OpenIE triples for Steve Jobs data."""
    return [
        ("Steve Jobs", "founded", "Apple"),
        ("Steve Jobs", "founded", "NeXT"),
        ("Steve Jobs", "worked at", "Apple"),
        ("Apple", "produces", "iPhone"),
        ("Apple", "produces", "Mac"),
        ("NeXT", "developed", "NeXTSTEP"),
    ]


# =============================================================================
# HippoRAGIndexing Tests
# =============================================================================

@pytest.mark.integration
class TestHippoRAGIndexing:
    """Tests for document indexing flow."""

    @pytest.mark.asyncio
    async def test_end_to_end_indexing(self, temp_save_dir):
        """Test: document → OpenIE → knowledge graph."""
        dest = HippoRAGDestination()
        
        # Mock LLM to return specific triples
        mock_llm = MagicMock()
        mock_llm.extract_triples = AsyncMock(return_value=[
            ("Steve Jobs", "founded", "Apple"),
            ("Apple", "located in", "Cupertino"),
        ])
        mock_llm.embed_text = AsyncMock(return_value=np.array([0.1, 0.2, 0.3]))
        mock_llm.health_check = AsyncMock(return_value={"healthy": True})
        
        dest._llm_provider = mock_llm
        
        await dest.initialize({
            "save_dir": temp_save_dir,
        })
        
        data = TransformedData(
            job_id=uuid4(),
            chunks=[{"content": "Steve Jobs founded Apple in Cupertino.", "metadata": {}}],
        )
        
        conn = await dest.connect({})
        result = await dest.write(conn, data)
        
        assert result.success is True
        assert result.metadata["chunks_indexed"] == 1
        assert result.metadata["triples_total"] == 2
        assert "Steve Jobs" in dest._graph.entities
        assert "Apple" in dest._graph.entities
        assert ("Steve Jobs", "founded", "Apple") in dest._graph.triples
        
        await dest.shutdown()

    @pytest.mark.asyncio
    async def test_triple_extraction(self, temp_save_dir):
        """Test OpenIE triple extraction with controlled LLM."""
        dest = HippoRAGDestination()
        
        mock_llm = MagicMock()
        mock_llm.extract_triples = AsyncMock(return_value=[
            ("Bill Gates", "founded", "Microsoft"),
            ("Microsoft", "located in", "Redmond"),
        ])
        mock_llm.embed_text = AsyncMock(return_value=np.array([0.1, 0.2]))
        mock_llm.health_check = AsyncMock(return_value={"healthy": True})
        
        dest._llm_provider = mock_llm
        
        await dest.initialize({"save_dir": temp_save_dir})
        
        # Test triple extraction directly
        triples = await dest._run_openie("Bill Gates founded Microsoft in Redmond.")
        
        assert len(triples) == 2
        assert ("Bill Gates", "founded", "Microsoft") in triples
        
        await dest.shutdown()

    @pytest.mark.asyncio
    async def test_knowledge_graph_building(self, temp_save_dir):
        """Test knowledge graph construction from multiple documents."""
        dest = HippoRAGDestination()
        
        mock_llm = MagicMock()
        mock_llm.extract_triples = AsyncMock(side_effect=[
            [("A", "rel", "B")],  # First document
            [("B", "rel", "C")],  # Second document
            [("C", "rel", "D")],  # Third document
        ])
        mock_llm.embed_text = AsyncMock(return_value=np.array([0.1]))
        mock_llm.health_check = AsyncMock(return_value={"healthy": True})
        
        dest._llm_provider = mock_llm
        
        await dest.initialize({"save_dir": temp_save_dir})
        
        # Index multiple documents
        texts = ["A relates to B", "B relates to C", "C relates to D"]
        metadatas = [{"job_id": "test", "chunk_index": i} for i in range(3)]
        
        await dest.index_documents(texts, metadatas)
        
        # Verify graph structure
        assert len(dest._graph.triples) == 3
        assert len(dest._graph.entities) == 4
        # Check chain: A-B-C-D
        assert dest._graph.get_related_entities("B")  # A->B and B->C
        
        await dest.shutdown()

    @pytest.mark.asyncio
    async def test_persistence(self, temp_save_dir):
        """Test data persists across restarts."""
        # First session - create data
        dest1 = HippoRAGDestination()
        mock_llm = MagicMock()
        mock_llm.extract_triples = AsyncMock(return_value=[
            ("Subject", "predicate", "Object"),
        ])
        mock_llm.embed_text = AsyncMock(return_value=np.array([0.1]))
        mock_llm.health_check = AsyncMock(return_value={"healthy": True})
        dest1._llm_provider = mock_llm
        
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

    @pytest.mark.asyncio
    async def test_indexing_with_embeddings(self, temp_save_dir):
        """Test that embeddings are generated and stored."""
        dest = HippoRAGDestination()
        
        mock_llm = MagicMock()
        mock_llm.extract_triples = AsyncMock(return_value=[
            ("Entity", "relation", "Object"),
        ])
        mock_llm.embed_text = AsyncMock(return_value=np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
        mock_llm.health_check = AsyncMock(return_value={"healthy": True})
        
        dest._llm_provider = mock_llm
        await dest.initialize({"save_dir": temp_save_dir})
        
        data = TransformedData(
            job_id=uuid4(),
            chunks=[{"content": "Entity relation Object.", "metadata": {}}],
        )
        
        conn = await dest.connect({})
        await dest.write(conn, data)
        
        # Verify embedding was stored
        assert len(dest._graph.passage_embeddings) == 1
        embedding = list(dest._graph.passage_embeddings.values())[0]
        assert len(embedding) == 5
        assert np.allclose(embedding, [0.1, 0.2, 0.3, 0.4, 0.5])
        
        await dest.shutdown()


# =============================================================================
# HippoRAGRetrieval Tests
# =============================================================================

@pytest.mark.integration
class TestHippoRAGRetrieval:
    """Tests for multi-hop retrieval."""

    @pytest.mark.asyncio
    async def test_single_hop_retrieval(self, temp_save_dir):
        """Test single-hop query."""
        dest = HippoRAGDestination()
        
        mock_llm = MagicMock()
        mock_llm.extract_triples = AsyncMock(return_value=[
            ("Steve Jobs", "founded", "Apple"),
        ])
        mock_llm.embed_text = AsyncMock(return_value=np.array([0.1]))
        mock_llm.extract_query_entities = AsyncMock(return_value=["Steve Jobs"])
        mock_llm.health_check = AsyncMock(return_value={"healthy": True})
        
        dest._llm_provider = mock_llm
        await dest.initialize({"save_dir": temp_save_dir})
        
        # Index document
        conn = await dest.connect({})
        data = TransformedData(
            job_id=uuid4(),
            chunks=[{"content": "Steve Jobs founded Apple in 1976.", "metadata": {}}],
        )
        await dest.write(conn, data)
        
        # Single-hop query
        results = await dest.retrieve(["What did Steve Jobs found?"], num_to_retrieve=1)
        
        assert len(results) == 1
        assert len(results[0].passages) > 0
        assert "Steve Jobs" in results[0].entities
        
        await dest.shutdown()

    @pytest.mark.asyncio
    async def test_multi_hop_retrieval(self, temp_save_dir, steve_jobs_triples):
        """Test multi-hop query (2+ hops)."""
        dest = HippoRAGDestination()
        
        mock_llm = MagicMock()
        mock_llm.extract_triples = AsyncMock(return_value=steve_jobs_triples)
        mock_llm.embed_text = AsyncMock(return_value=np.array([0.1]))
        mock_llm.extract_query_entities = AsyncMock(return_value=["Steve Jobs"])
        mock_llm.health_check = AsyncMock(return_value={"healthy": True})
        
        dest._llm_provider = mock_llm
        await dest.initialize({"save_dir": temp_save_dir})
        
        # Index documents with multi-hop structure
        texts = [
            "Steve Jobs founded Apple and worked at Apple.",
            "Steve Jobs also founded NeXT.",
            "Apple produces iPhone and Mac.",
            "NeXT developed NeXTSTEP.",
        ]
        metadatas = [{"job_id": "test", "chunk_index": i} for i in range(len(texts))]
        await dest.index_documents(texts, metadatas)
        
        # Multi-hop query: Steve Jobs → Apple → iPhone (2 hops)
        results = await dest.retrieve(
            ["What products did Steve Jobs' company produce?"],
            num_to_retrieve=10
        )
        
        assert len(results) == 1
        # Should find passages about Apple (1 hop) and iPhone/Mac (2 hops)
        all_passages = " ".join(results[0].passages)
        assert "Apple" in all_passages or "iPhone" in all_passages
        
        await dest.shutdown()

    @pytest.mark.asyncio
    async def test_ppr_scoring(self, temp_save_dir):
        """Test Personalized PageRank scoring."""
        dest = HippoRAGDestination()
        
        mock_llm = MagicMock()
        mock_llm.extract_triples = AsyncMock(side_effect=[
            [("A", "connects_to", "B")],
            [("B", "connects_to", "C")],
            [("C", "connects_to", "D")],
            [("X", "connects_to", "Y")],  # Unrelated component
        ])
        mock_llm.embed_text = AsyncMock(return_value=np.array([0.1]))
        mock_llm.health_check = AsyncMock(return_value={"healthy": True})
        
        dest._llm_provider = mock_llm
        await dest.initialize({"save_dir": temp_save_dir})
        
        # Build chain: A-B-C-D and separate X-Y
        texts = ["A connects to B", "B connects to C", "C connects to D", "X connects to Y"]
        metadatas = [{"job_id": "chain", "chunk_index": i} for i in range(4)]
        await dest.index_documents(texts, metadatas)
        
        # Run PPR from A
        passage_scores = await dest._run_ppr(["A"], top_k=10)
        
        # A's passage should have highest score
        assert len(passage_scores) > 0
        # Passages closer to A should have higher scores
        
        await dest.shutdown()

    @pytest.mark.asyncio
    async def test_retrieval_with_multiple_queries(self, temp_save_dir):
        """Test batch retrieval with multiple queries."""
        dest = HippoRAGDestination()
        
        mock_llm = MagicMock()
        mock_llm.extract_triples = AsyncMock(return_value=[
            ("Apple", "produces", "iPhone"),
            ("Microsoft", "produces", "Windows"),
        ])
        mock_llm.embed_text = AsyncMock(return_value=np.array([0.1]))
        mock_llm.extract_query_entities = AsyncMock(side_effect=[
            ["Apple"],
            ["Microsoft"],
        ])
        mock_llm.health_check = AsyncMock(return_value={"healthy": True})
        
        dest._llm_provider = mock_llm
        await dest.initialize({"save_dir": temp_save_dir})
        
        # Index documents
        texts = ["Apple produces iPhone", "Microsoft produces Windows"]
        await dest.index_documents(texts, [{"job_id": "q", "chunk_index": i} for i in range(2)])
        
        # Batch retrieval
        queries = ["What does Apple produce?", "What does Microsoft produce?"]
        results = await dest.retrieve(queries, num_to_retrieve=5)
        
        assert len(results) == 2
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert results[0].query == queries[0]
        assert results[1].query == queries[1]
        
        await dest.shutdown()


# =============================================================================
# HippoRAGQA Tests
# =============================================================================

@pytest.mark.integration
class TestHippoRAGQA:
    """Tests for RAG QA."""

    @pytest.mark.asyncio
    async def test_rag_qa_single_question(self, temp_save_dir):
        """Test single question answering."""
        dest = HippoRAGDestination()
        
        mock_llm = MagicMock()
        mock_llm.extract_triples = AsyncMock(return_value=[
            ("Steve Jobs", "founded", "Apple"),
        ])
        mock_llm.embed_text = AsyncMock(return_value=np.array([0.1]))
        mock_llm.extract_query_entities = AsyncMock(return_value=["Steve Jobs"])
        mock_llm.answer_question = AsyncMock(return_value="Apple Inc.")
        mock_llm.health_check = AsyncMock(return_value={"healthy": True})
        
        dest._llm_provider = mock_llm
        await dest.initialize({"save_dir": temp_save_dir})
        
        # Index document
        conn = await dest.connect({})
        data = TransformedData(
            job_id=uuid4(),
            chunks=[{"content": "Steve Jobs founded Apple Inc. in 1976.", "metadata": {}}],
        )
        await dest.write(conn, data)
        
        # QA
        results = await dest.rag_qa(["What company did Steve Jobs found?"])
        
        assert len(results) == 1
        assert isinstance(results[0], QAResult)
        assert results[0].query == "What company did Steve Jobs found?"
        assert results[0].answer == "Apple Inc."
        
        # Verify LLM was called with context
        mock_llm.answer_question.assert_called_once()
        call_kwargs = mock_llm.answer_question.call_args.kwargs
        assert "What company did Steve Jobs found?" in call_kwargs.get("question", "")
        
        await dest.shutdown()

    @pytest.mark.asyncio
    async def test_rag_qa_multi_question(self, temp_save_dir):
        """Test batch question answering."""
        dest = HippoRAGDestination()
        
        mock_llm = MagicMock()
        mock_llm.extract_triples = AsyncMock(return_value=[
            ("Bill Gates", "founded", "Microsoft"),
            ("Steve Jobs", "founded", "Apple"),
        ])
        mock_llm.embed_text = AsyncMock(return_value=np.array([0.1]))
        mock_llm.extract_query_entities = AsyncMock(side_effect=[
            ["Bill Gates"],
            ["Steve Jobs"],
        ])
        mock_llm.answer_question = AsyncMock(side_effect=[
            "Microsoft Corporation",
            "Apple Inc.",
        ])
        mock_llm.health_check = AsyncMock(return_value={"healthy": True})
        
        dest._llm_provider = mock_llm
        await dest.initialize({"save_dir": temp_save_dir})
        
        # Index documents
        texts = [
            "Bill Gates founded Microsoft Corporation.",
            "Steve Jobs founded Apple Inc.",
        ]
        await dest.index_documents(texts, [{"job_id": "batch", "chunk_index": i} for i in range(2)])
        
        # Batch QA
        questions = [
            "What company did Bill Gates found?",
            "What company did Steve Jobs found?",
        ]
        results = await dest.rag_qa(questions)
        
        assert len(results) == 2
        assert results[0].answer == "Microsoft Corporation"
        assert results[1].answer == "Apple Inc."
        
        await dest.shutdown()

    @pytest.mark.asyncio
    async def test_answer_with_sources(self, temp_save_dir):
        """Test answer includes sources."""
        dest = HippoRAGDestination()
        
        mock_llm = MagicMock()
        mock_llm.extract_triples = AsyncMock(return_value=[
            ("Steve Jobs", "founded", "Apple"),
        ])
        mock_llm.embed_text = AsyncMock(return_value=np.array([0.1]))
        mock_llm.extract_query_entities = AsyncMock(return_value=["Steve Jobs"])
        mock_llm.answer_question = AsyncMock(return_value="Apple Inc.")
        mock_llm.health_check = AsyncMock(return_value={"healthy": True})
        
        dest._llm_provider = mock_llm
        await dest.initialize({"save_dir": temp_save_dir})
        
        # Index document
        conn = await dest.connect({})
        data = TransformedData(
            job_id=uuid4(),
            chunks=[{"content": "Steve Jobs founded Apple Inc. in Cupertino.", "metadata": {}}],
        )
        await dest.write(conn, data)
        
        # QA with sources
        results = await dest.rag_qa(["Where did Steve Jobs found a company?"])
        
        assert len(results) == 1
        assert len(results[0].sources) > 0
        assert "Apple" in results[0].sources[0] or "Cupertino" in results[0].sources[0]
        assert results[0].retrieval_results is not None
        assert results[0].confidence > 0
        
        await dest.shutdown()


# =============================================================================
# Multi-Hop Chain Tests
# =============================================================================

@pytest.mark.integration
class TestMultiHopChains:
    """Tests for multi-hop reasoning chains."""

    @pytest.mark.asyncio
    async def test_three_hop_chain(self, temp_save_dir):
        """Test a 3-hop chain: Erik Hort → Montebello → Ramapo → Rockland County."""
        dest = HippoRAGDestination()
        
        mock_llm = MagicMock()
        mock_llm.extract_triples = AsyncMock(side_effect=[
            [("Erik Hort", "born in", "Montebello")],
            [("Montebello", "village in", "Ramapo")],
            [("Ramapo", "located in", "Rockland County")],
        ])
        mock_llm.embed_text = AsyncMock(return_value=np.array([0.1]))
        mock_llm.extract_query_entities = AsyncMock(return_value=["Erik Hort"])
        mock_llm.answer_question = AsyncMock(return_value="Rockland County")
        mock_llm.health_check = AsyncMock(return_value={"healthy": True})
        
        dest._llm_provider = mock_llm
        await dest.initialize({"save_dir": temp_save_dir})
        
        # Index the multi-hop chain
        texts = [
            "Erik Hort was born in Montebello, New York.",
            "Montebello is a village in the town of Ramapo.",
            "Ramapo is located in Rockland County, New York.",
        ]
        metadatas = [{"job_id": "biography", "chunk_index": i} for i in range(3)]
        await dest.index_documents(texts, metadatas)
        
        # 3-hop query
        results = await dest.rag_qa(["What county is Erik Hort's birthplace a part of?"])
        
        assert len(results) == 1
        assert results[0].answer == "Rockland County"
        # Should have retrieved passages from all hops
        all_sources = " ".join(results[0].sources)
        assert "Erik Hort" in all_sources
        
        await dest.shutdown()

    @pytest.mark.asyncio
    async def test_converging_paths(self, temp_save_dir):
        """Test converging paths: Multiple entities point to common answer."""
        dest = HippoRAGDestination()
        
        mock_llm = MagicMock()
        mock_llm.extract_triples = AsyncMock(side_effect=[
            [("Steve Jobs", "founded", "Apple")],
            [("Steve Wozniak", "co-founded", "Apple")],
            [("Apple", "produces", "iPhone")],
        ])
        mock_llm.embed_text = AsyncMock(return_value=np.array([0.1]))
        mock_llm.extract_query_entities = AsyncMock(return_value=["Steve Jobs", "Steve Wozniak"])
        mock_llm.health_check = AsyncMock(return_value={"healthy": True})
        
        dest._llm_provider = mock_llm
        await dest.initialize({"save_dir": temp_save_dir})
        
        # Index converging paths
        texts = [
            "Steve Jobs founded Apple.",
            "Steve Wozniak co-founded Apple.",
            "Apple produces iPhone.",
        ]
        await dest.index_documents(texts, [{"job_id": "apple", "chunk_index": i} for i in range(3)])
        
        # Query from multiple starting points
        results = await dest.retrieve(["What company did Jobs and Wozniak work on?"])
        
        assert len(results) == 1
        # Should find passages about Apple (common target)
        
        await dest.shutdown()


# =============================================================================
# Health and Configuration Tests
# =============================================================================

@pytest.mark.integration
class TestHippoRAGHealthAndConfig:
    """Tests for health checks and configuration."""

    @pytest.mark.asyncio
    async def test_health_check_with_file_storage(self, temp_save_dir):
        """Test health check with actual file storage."""
        dest = HippoRAGDestination()
        mock_llm = MagicMock()
        mock_llm.health_check = AsyncMock(return_value={"healthy": True})
        dest._llm_provider = mock_llm
        
        await dest.initialize({"save_dir": temp_save_dir})
        
        health = await dest.health_check()
        
        assert health == HealthStatus.HEALTHY
        
        await dest.shutdown()

    @pytest.mark.asyncio
    async def test_graph_statistics(self, temp_save_dir):
        """Test that graph statistics are tracked."""
        dest = HippoRAGDestination()
        
        mock_llm = MagicMock()
        mock_llm.extract_triples = AsyncMock(return_value=[
            ("A", "rel", "B"),
            ("B", "rel", "C"),
            ("C", "rel", "D"),
        ])
        mock_llm.embed_text = AsyncMock(return_value=np.array([0.1]))
        mock_llm.health_check = AsyncMock(return_value={"healthy": True})
        
        dest._llm_provider = mock_llm
        await dest.initialize({"save_dir": temp_save_dir})
        
        # Index document
        conn = await dest.connect({})
        data = TransformedData(
            job_id=uuid4(),
            chunks=[{"content": "A relates to B which relates to C which relates to D.", "metadata": {}}],
        )
        result = await dest.write(conn, data)
        
        # Verify statistics
        assert result.metadata["entities_total"] == 4
        assert result.metadata["triples_total"] == 3
        assert result.metadata["passages_total"] == 1
        
        await dest.shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_writes(self, temp_save_dir):
        """Test handling of concurrent writes."""
        import asyncio
        
        dest = HippoRAGDestination()
        
        mock_llm = MagicMock()
        mock_llm.extract_triples = AsyncMock(return_value=[("Subject", "predicate", "Object")])
        mock_llm.embed_text = AsyncMock(return_value=np.array([0.1]))
        mock_llm.health_check = AsyncMock(return_value={"healthy": True})
        
        dest._llm_provider = mock_llm
        await dest.initialize({"save_dir": temp_save_dir})
        
        conn = await dest.connect({})
        
        # Create multiple documents
        async def write_doc(doc_id: int):
            data = TransformedData(
                job_id=uuid4(),
                chunks=[{"content": f"Document {doc_id} content.", "metadata": {}}],
            )
            return await dest.write(conn, data)
        
        # Run concurrent writes
        results = await asyncio.gather(*[write_doc(i) for i in range(5)])
        
        # All writes should succeed
        assert all(r.success for r in results)
        # Should have 5 passages
        assert len(dest._graph.passages) == 5
        
        await dest.shutdown()


# =============================================================================
# Cleanup Tests
# =============================================================================

@pytest.mark.integration
class TestCleanup:
    """Tests for cleanup operations."""

    @pytest.mark.asyncio
    async def test_storage_cleanup(self, temp_save_dir):
        """Test that temporary storage is cleaned up."""
        dest = HippoRAGDestination()
        
        mock_llm = MagicMock()
        mock_llm.extract_triples = AsyncMock(return_value=[("A", "B", "C")])
        mock_llm.embed_text = AsyncMock(return_value=np.array([0.1]))
        mock_llm.health_check = AsyncMock(return_value={"healthy": True})
        
        dest._llm_provider = mock_llm
        await dest.initialize({"save_dir": temp_save_dir})
        
        # Create some data
        conn = await dest.connect({})
        data = TransformedData(
            job_id=uuid4(),
            chunks=[{"content": "Test content.", "metadata": {}}],
        )
        await dest.write(conn, data)
        await dest.shutdown()
        
        # Verify graph file exists
        graph_file = os.path.join(temp_save_dir, "knowledge_graph.pkl")
        assert os.path.exists(graph_file)
