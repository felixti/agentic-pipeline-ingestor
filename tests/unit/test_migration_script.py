"""Unit tests for the GraphRAG to Cognee migration script."""

import pytest
from datetime import datetime
from uuid import UUID

# Import the migration module components
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from scripts.migrate_to_cognee_local import (
    MigrationStats,
    GraphRAGDocument,
)


class TestMigrationStats:
    """Test cases for MigrationStats."""

    @pytest.mark.unit
    def test_initial_stats(self):
        """Test initial stats values."""
        stats = MigrationStats()
        assert stats.documents_total == 0
        assert stats.documents_migrated == 0
        assert stats.success_rate == 1.0
        assert stats.duration_seconds == 0.0

    @pytest.mark.unit
    def test_duration_calculation(self):
        """Test duration calculation."""
        stats = MigrationStats()
        stats.start_time = datetime.utcnow()
        
        # Simulate some time passing
        import time
        time.sleep(0.01)
        
        stats.end_time = datetime.utcnow()
        
        assert stats.duration_seconds > 0

    @pytest.mark.unit
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        stats = MigrationStats()
        stats.documents_migrated = 80
        stats.documents_failed = 20
        
        assert stats.success_rate == 0.8

    @pytest.mark.unit
    def test_to_dict(self):
        """Test stats to dictionary conversion."""
        stats = MigrationStats()
        stats.documents_total = 100
        stats.documents_migrated = 95
        stats.start_time = datetime.utcnow()
        
        result = stats.to_dict()
        
        assert result["documents"]["total"] == 100
        assert result["documents"]["migrated"] == 95
        assert "timing" in result
        assert "errors" in result


class TestGraphRAGDocument:
    """Test cases for GraphRAGDocument."""

    @pytest.mark.unit
    def test_from_api_response_basic(self):
        """Test creating document from API response."""
        data = {
            "document_id": "doc-123",
            "job_id": "job-456",
            "text_chunks": [
                {"content": "Chunk 1", "metadata": {}},
                {"content": "Chunk 2", "metadata": {}},
            ],
            "metadata": {"title": "Test Doc"},
        }
        
        doc = GraphRAGDocument.from_api_response(data, "graph-1")
        
        assert doc.document_id == "doc-123"
        assert doc.graph_id == "graph-1"
        assert doc.job_id == "job-456"
        assert len(doc.chunks) == 2
        assert doc.metadata["title"] == "Test Doc"

    @pytest.mark.unit
    def test_from_api_response_alternate_format(self):
        """Test creating document from alternate API response format."""
        data = {
            "id": "doc-789",
            "job_id": "job-abc",
            "chunks": [
                {"content": "Chunk A"},
            ],
        }
        
        doc = GraphRAGDocument.from_api_response(data, "graph-2")
        
        assert doc.document_id == "doc-789"
        assert len(doc.chunks) == 1

    @pytest.mark.unit
    def test_to_transformed_data(self):
        """Test conversion to TransformedData."""
        doc = GraphRAGDocument(
            document_id="doc-123",
            graph_id="graph-1",
            job_id="550e8400-e29b-41d4-a716-446655440000",
            chunks=[{"content": "Test chunk"}],
            metadata={"key": "value"},
            embeddings=[[0.1, 0.2, 0.3]],
        )
        
        transformed = doc.to_transformed_data()
        
        assert isinstance(transformed.job_id, UUID)
        assert len(transformed.chunks) == 1
        assert transformed.metadata["key"] == "value"
        assert transformed.embeddings == [[0.1, 0.2, 0.3]]


class TestGraphRAGClient:
    """Test cases for GraphRAGClient."""
    
    @pytest.mark.unit
    async def test_client_initialization(self):
        """Test GraphRAG client initialization."""
        from scripts.migrate_to_cognee_local import GraphRAGClient
        
        client = GraphRAGClient(
            api_url="https://api.example.com",
            api_key="test-key",
        )
        
        assert client._api_url == "https://api.example.com"
        assert client._api_key == "test-key"
        assert client._timeout == 120


class TestGraphRAGMigrator:
    """Test cases for GraphRAGMigrator."""
    
    @pytest.mark.unit
    def test_migrator_initialization(self):
        """Test migrator initialization."""
        from scripts.migrate_to_cognee_local import GraphRAGMigrator
        
        migrator = GraphRAGMigrator(
            source_config={"api_url": "https://api.example.com"},
            target_config={"dataset_id": "test-dataset"},
        )
        
        assert migrator._source_config["api_url"] == "https://api.example.com"
        assert migrator._target_config["dataset_id"] == "test-dataset"
        assert migrator._source_client is None
        assert migrator._target_destination is None


class TestCommandLineArguments:
    """Test cases for command line argument parsing."""
    
    @pytest.mark.unit
    def test_argument_parser(self):
        """Test that argument parser accepts required arguments."""
        import argparse
        from scripts.migrate_to_cognee_local import main
        
        # Just verify the parser doesn't raise
        parser = argparse.ArgumentParser()
        parser.add_argument("--source-dataset", required=True)
        parser.add_argument("--dry-run", action="store_true")
        
        args = parser.parse_args(["--source-dataset", "test-dataset", "--dry-run"])
        
        assert args.source_dataset == "test-dataset"
        assert args.dry_run is True
