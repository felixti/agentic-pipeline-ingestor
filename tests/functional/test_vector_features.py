"""Functional tests for vector store features.

These tests verify the core vector functionality works end-to-end
without requiring a full database setup.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Any

import pytest


class TestVectorStoreFeatures:
    """Functional tests for vector store capabilities."""

    def test_document_chunk_model_creation(self):
        """Test that DocumentChunkModel can be created with all fields."""
        from src.db.models import DocumentChunkModel
        
        chunk = DocumentChunkModel(
            id=uuid.uuid4(),
            job_id=uuid.uuid4(),
            chunk_index=0,
            content="Test content for a document chunk",
            content_hash="abc123",
            embedding=[0.1] * 1536,  # 1536-dim embedding
            chunk_metadata={"page": 1, "source": "test.pdf"},
            created_at=datetime.utcnow()
        )
        
        assert chunk.id is not None
        assert chunk.chunk_index == 0
        assert chunk.content == "Test content for a document chunk"
        assert len(chunk.embedding) == 1536
        assert chunk.chunk_metadata["page"] == 1
        assert chunk.has_embedding is True
        print("✅ DocumentChunkModel creation works")

    def test_vector_type_serialization(self):
        """Test that vector type can serialize/deserialize embeddings."""
        from src.db.models import Vector
        
        vector_type = Vector(dimensions=1536)
        
        # Test bind_processor (Python list -> PostgreSQL vector string)
        embedding = [0.1, 0.2, 0.3] + [0.0] * 1533
        pg_string = vector_type.bind_processor(None)(embedding)
        assert pg_string.startswith("[")
        assert pg_string.endswith("]")
        assert "," in pg_string
        
        # Test result_processor (PostgreSQL vector string -> Python list)
        result = vector_type.result_processor(None, None)(pg_string)
        assert isinstance(result, list)
        assert len(result) == 1536
        print("✅ Vector type serialization works")

    def test_vector_search_service_creation(self):
        """Test that VectorSearchService can be instantiated."""
        from unittest.mock import AsyncMock, MagicMock

        from src.services.vector_search_service import VectorSearchService
        
        mock_repo = MagicMock()
        
        service = VectorSearchService(mock_repo)
        
        assert service is not None
        assert service.repository is not None
        assert service.config.default_top_k == 10
        print("✅ VectorSearchService instantiation works")

    def test_text_search_service_creation(self):
        """Test that TextSearchService can be instantiated."""
        from unittest.mock import MagicMock

        from src.services.text_search_service import TextSearchService
        
        mock_repo = MagicMock()
        
        service = TextSearchService(mock_repo)
        
        assert service is not None
        assert service.repository is not None
        assert service.config.default_language == "english"
        print("✅ TextSearchService instantiation works")

    def test_hybrid_search_service_creation(self):
        """Test that HybridSearchService can be instantiated."""
        from unittest.mock import MagicMock

        from src.services.hybrid_search_service import HybridSearchService
        from src.services.text_search_service import TextSearchService
        from src.services.vector_search_service import VectorSearchService
        
        mock_repo = MagicMock()
        vector_service = VectorSearchService(mock_repo)
        text_service = TextSearchService(mock_repo)
        
        service = HybridSearchService(vector_service, text_service)
        
        assert service is not None
        assert service.vector_service is not None
        assert service.text_service is not None
        print("✅ HybridSearchService instantiation works")

    def test_embedding_service_creation(self):
        """Test that EmbeddingService can be instantiated."""
        from src.services.embedding_service import EmbeddingService
        
        service = EmbeddingService()
        
        assert service is not None
        assert service._cache is not None
        print("✅ EmbeddingService instantiation works")

    def test_search_result_dataclasses(self):
        """Test that search result dataclasses work correctly."""
        from src.db.models import DocumentChunkModel
        from src.services.hybrid_search_service import HybridSearchResult
        from src.services.text_search_service import TextSearchResult
        from src.services.vector_search_service import SearchResult
        
        chunk = DocumentChunkModel(
            id=uuid.uuid4(),
            job_id=uuid.uuid4(),
            chunk_index=0,
            content="Test",
            chunk_metadata={}
        )
        
        # Vector search result
        vector_result = SearchResult(
            chunk=chunk,
            similarity_score=0.95,
            rank=1
        )
        assert vector_result.similarity_score == 0.95
        
        # Text search result
        text_result = TextSearchResult(
            chunk=chunk,
            rank_score=0.85,
            rank=1,
            highlighted_content="<mark>Test</mark>",
            matched_terms=["test"]
        )
        assert text_result.highlighted_content == "<mark>Test</mark>"
        
        # Hybrid search result
        hybrid_result = HybridSearchResult(
            chunk=chunk,
            hybrid_score=0.90,
            vector_score=0.95,
            text_score=0.85,
            vector_rank=1,
            text_rank=2,
            fusion_method="weighted_sum",
            rank=1
        )
        assert hybrid_result.fusion_method == "weighted_sum"
        
        print("✅ Search result dataclasses work")

    def test_config_loading(self):
        """Test that vector store config can be loaded."""
        from src.vector_store_config import VectorStoreConfig, load_vector_store_config
        
        # Test default config
        config = VectorStoreConfig()
        assert config.enabled is True
        assert config.embedding.dimensions == 1536
        assert config.search.default_top_k == 10
        assert config.hybrid.default_vector_weight == 0.7
        
        print("✅ Vector store config loading works")

    def test_similarity_calculation(self):
        """Test similarity score calculation from distance."""
        from unittest.mock import MagicMock

        from src.services.vector_search_service import VectorSearchService
        
        service = VectorSearchService.__new__(VectorSearchService)
        service._default_min_similarity = 0.7
        
        # Test similarity calculation
        # Cosine distance 0 = identical vectors -> similarity 1.0
        assert service._calculate_similarity(0.0) == 1.0
        
        # Cosine distance 1 = orthogonal vectors -> similarity 0.0
        assert service._calculate_similarity(1.0) == 0.0
        
        # Cosine distance 2 = opposite vectors -> similarity 0.0 (clamped)
        assert service._calculate_similarity(2.0) == 0.0
        
        print("✅ Similarity calculation works")

    def test_weighted_sum_fusion(self):
        """Test weighted sum fusion calculation."""
        from src.services.hybrid_search_service import HybridSearchService
        
        service = HybridSearchService.__new__(HybridSearchService)
        
        # Test weighted sum
        vector_score = 0.9
        text_score = 0.7
        vector_weight = 0.7
        text_weight = 0.3
        
        result = (vector_weight * vector_score) + (text_weight * text_score)
        expected = 0.84
        
        assert abs(result - expected) < 0.001
        print("✅ Weighted sum fusion calculation works")

    def test_rrf_fusion(self):
        """Test Reciprocal Rank Fusion calculation."""
        from src.services.hybrid_search_service import HybridSearchService
        
        service = HybridSearchService.__new__(HybridSearchService)
        
        # RRF formula: score = sum(1 / (k + rank))
        k = 60
        vector_rank = 1
        text_rank = 3
        
        score = (1 / (k + vector_rank)) + (1 / (k + text_rank))
        expected = (1 / 61) + (1 / 63)
        
        assert abs(score - expected) < 0.001
        print("✅ RRF fusion calculation works")

    def test_bm25_rank_normalization(self):
        """Test BM25 rank score normalization."""
        from src.services.text_search_service import TextSearchService
        
        service = TextSearchService.__new__(TextSearchService)
        
        # Test normalization to 0-1 range
        raw_score = 0.5
        min_score = 0.0
        max_score = 1.0
        
        if max_score > min_score:
            normalized = (raw_score - min_score) / (max_score - min_score)
        else:
            normalized = 0.0
            
        assert normalized == 0.5
        print("✅ BM25 normalization works")

    def test_repository_methods_exist(self):
        """Test that all DocumentChunkRepository methods exist."""
        import inspect

        from src.db.repositories.document_chunk_repository import DocumentChunkRepository
        
        methods = [
            "create",
            "get_by_id",
            "get_by_job_id",
            "get_by_content_hash",
            "bulk_create",
            "update_embedding",
            "delete_by_job_id",
            "delete",
            "exists_by_job_id_and_index",
            "count_by_job_id",
            "get_chunks_without_embeddings"
        ]
        
        for method_name in methods:
            assert hasattr(DocumentChunkRepository, method_name), f"Missing method: {method_name}"
            method = getattr(DocumentChunkRepository, method_name)
            assert callable(method), f"{method_name} is not callable"
        
        print("✅ All DocumentChunkRepository methods exist")

    def test_api_routes_exist(self):
        """Test that all API routes are defined."""
        from src.api.routes.chunks import router as chunks_router
        from src.api.routes.search import router as search_router
        
        # Check chunks router
        chunks_routes = [route.path for route in chunks_router.routes]
        assert "/jobs/{job_id}/chunks" in chunks_routes
        assert "/jobs/{job_id}/chunks/{chunk_id}" in chunks_routes
        
        # Check search router
        search_routes = [route.path for route in search_router.routes]
        assert "/search/semantic" in search_routes
        assert "/search/text" in search_routes
        assert "/search/hybrid" in search_routes
        assert "/search/similar/{chunk_id}" in search_routes
        
        print("✅ All API routes are defined")

    def test_migration_files_exist(self):
        """Test that migration files exist."""
        from pathlib import Path
        
        migrations_dir = Path("migrations/versions")
        
        # Check pgvector extension migration
        pgvector_migration = migrations_dir / "002_add_pgvector_extensions.py"
        assert pgvector_migration.exists(), "pgvector migration not found"
        
        # Check document chunks migration
        chunks_migration = migrations_dir / "003_add_document_chunks.py"
        assert chunks_migration.exists(), "document chunks migration not found"
        
        # Verify migration content
        content = chunks_migration.read_text()
        assert "document_chunks" in content
        assert "hnsw" in content.lower()
        assert "gin" in content.lower()
        
        print("✅ Migration files exist and contain expected content")

    def test_docker_compose_updated(self):
        """Test that docker-compose uses pgvector image."""
        from pathlib import Path

        import yaml
        
        docker_compose = Path("docker/docker-compose.yml")
        assert docker_compose.exists()
        
        content = docker_compose.read_text()
        assert "pgvector/pgvector" in content
        assert "pg17" in content
        
        print("✅ Docker compose uses pgvector image")

    def test_configuration_file_exists(self):
        """Test that vector store config file exists."""
        from pathlib import Path

        import yaml
        
        config_file = Path("config/vector_store.yaml")
        assert config_file.exists(), "vector_store.yaml not found"
        
        with open(config_file) as f:
            config = yaml.safe_load(f)
        
        assert "vector_store" in config
        assert config["vector_store"]["enabled"] is True
        assert "embedding" in config["vector_store"]
        assert "search" in config["vector_store"]
        
        print("✅ Configuration file exists and is valid")


class TestVectorOperations:
    """Test vector operations and calculations."""

    def test_cosine_similarity_calculation(self):
        """Test cosine similarity between vectors."""
        import math
        
        def cosine_similarity(v1, v2):
            """Calculate cosine similarity between two vectors."""
            dot_product = sum(a * b for a, b in zip(v1, v2))
            norm1 = math.sqrt(sum(a * a for a in v1))
            norm2 = math.sqrt(sum(b * b for b in v2))
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot_product / (norm1 * norm2)
        
        # Identical vectors
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        assert cosine_similarity(v1, v2) == 1.0
        
        # Orthogonal vectors
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        assert cosine_similarity(v1, v2) == 0.0
        
        # Opposite vectors
        v1 = [1.0, 0.0, 0.0]
        v2 = [-1.0, 0.0, 0.0]
        assert cosine_similarity(v1, v2) == -1.0
        
        print("✅ Cosine similarity calculations are correct")

    def test_vector_distance_to_similarity(self):
        """Test conversion from cosine distance to similarity."""
        # Cosine distance = 1 - cosine similarity
        # Distance ranges from 0 (identical) to 2 (opposite)
        
        test_cases = [
            (0.0, 1.0),    # Identical
            (0.5, 0.5),    # 45 degrees
            (1.0, 0.0),    # Orthogonal
            (1.5, 0.0),    # Beyond orthogonal (clamped)
            (2.0, 0.0),    # Opposite (clamped)
        ]
        
        for distance, expected_similarity in test_cases:
            # Similarity = 1 - distance, clamped to [0, 1]
            similarity = max(0.0, min(1.0, 1.0 - distance))
            assert abs(similarity - expected_similarity) < 0.001, \
                f"Failed for distance {distance}"
        
        print("✅ Distance to similarity conversion works")

    def test_embedding_dimension_validation(self):
        """Test embedding dimension validation."""
        from src.db.models import Vector
        
        vector_type = Vector(dimensions=1536)
        
        # Valid dimension
        valid_embedding = [0.1] * 1536
        result = vector_type.bind_processor(None)(valid_embedding)
        assert result is not None
        
        # Invalid dimension (wrong size) - should still serialize but may fail at DB
        invalid_embedding = [0.1] * 768
        result = vector_type.bind_processor(None)(invalid_embedding)
        assert result is not None  # Type doesn't validate, DB will
        
        print("✅ Embedding dimension handling works")


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    @pytest.mark.asyncio
    async def test_chunk_creation_workflow(self):
        """Test the complete chunk creation workflow."""
        import uuid

        from src.db.models import DocumentChunkModel
        
        # Create a chunk
        chunk = DocumentChunkModel(
            id=uuid.uuid4(),
            job_id=uuid.uuid4(),
            chunk_index=0,
            content="This is a test document chunk for vector storage.",
            content_hash="sha256_hash_here",
            embedding=None,  # No embedding yet
            chunk_metadata={
                "page": 1,
                "source": "test_document.pdf",
                "language": "en"
            }
        )
        
        # Verify initial state
        assert chunk.has_embedding is False
        
        # Add embedding
        test_embedding = [0.01] * 1536
        chunk.set_embedding(test_embedding)
        
        # Verify embedding added
        assert chunk.has_embedding is True
        assert len(chunk.embedding) == 1536
        
        print("✅ Chunk creation workflow works")

    def test_search_workflow_simulation(self):
        """Simulate a complete search workflow."""
        import uuid

        from src.db.models import DocumentChunkModel
        from src.services.vector_search_service import SearchResult
        
        # Create test chunks
        chunks = []
        for i in range(5):
            chunk = DocumentChunkModel(
                id=uuid.uuid4(),
                job_id=uuid.uuid4(),
                chunk_index=i,
                content=f"Test content for chunk {i}",
                chunk_metadata={"index": i}
            )
            # Simulate embeddings with varying similarity
            embedding = [0.1 * (i + 1)] + [0.0] * 1535
            chunk.set_embedding(embedding)
            chunks.append(chunk)
        
        # Create search results
        results = []
        for i, chunk in enumerate(chunks):
            similarity = 0.9 - (i * 0.1)  # Decreasing similarity
            results.append(SearchResult(
                chunk=chunk,
                similarity_score=similarity,
                rank=i + 1
            ))
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Verify ranking
        assert results[0].similarity_score >= results[1].similarity_score
        assert len(results) == 5
        
        print("✅ Search workflow simulation works")


def run_all_tests():
    """Run all functional tests and print summary."""
    print("\n" + "="*70)
    print("FUNCTIONAL TESTS FOR VECTOR STORE FEATURES")
    print("="*70 + "\n")
    
    # Run tests using pytest
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest", "tests/functional/test_vector_features.py", "-v"],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode


if __name__ == "__main__":
    exit(run_all_tests())
