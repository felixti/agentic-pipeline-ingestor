#!/usr/bin/env python3
"""Migration script from API GraphRAG to local Cognee.

This script migrates existing graph data from the API-based GraphRAG
destination to the new local Cognee destination with Neo4j.

Usage:
    python scripts/migrate_to_cognee_local.py --source-dataset DATASET_ID --dry-run
    python scripts/migrate_to_cognee_local.py --source-dataset DATASET_ID --batch-size 100
    python scripts/migrate_to_cognee_local.py --source-dataset DATASET_ID --verify

Environment Variables:
    GRAPH_RAG_API_URL: GraphRAG API base URL
    GRAPH_RAG_API_KEY: GraphRAG API key
    NEO4J_URI: Neo4j connection URI (default: bolt://neo4j:7687)
    NEO4J_USER: Neo4j username (default: neo4j)
    NEO4J_PASSWORD: Neo4j password (default: cognee-graph-db)
    DB_URL: PostgreSQL connection URL
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx

# Optional tqdm for progress bars
try:
    from tqdm.asyncio import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    # Fallback progress bar
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, *args, total=None, desc="", unit="", **kwargs):
            self.total = total
            self.n = 0
            self.desc = desc
            if total:
                print(f"{desc}: 0/{total}")
            else:
                print(f"{desc}: started")
        def update(self, n):
            self.n += n
            if self.total and self.n % 10 == 0:
                print(f"{self.desc}: {self.n}/{self.total}")
        def close(self):
            if self.total:
                print(f"{self.desc}: completed {self.n}/{self.total}")
            else:
                print(f"{self.desc}: completed {self.n}")

from src.observability.logging import get_logger, setup_logging
from src.plugins.base import Connection, TransformedData
from src.plugins.destinations.cognee_local import CogneeLocalDestination
from src.infrastructure.neo4j.client import get_neo4j_client, close_neo4j_client

logger = get_logger(__name__)

# Environment configuration
GRAPH_RAG_API_URL = os.getenv("GRAPH_RAG_API_URL", "")
GRAPH_RAG_API_KEY = os.getenv("GRAPH_RAG_API_KEY", "")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "cognee-graph-db")
DB_URL = os.getenv("DB_URL", "")


@dataclass
class MigrationStats:
    """Statistics for migration process.
    
    Attributes:
        documents_total: Total documents to migrate
        documents_migrated: Successfully migrated documents
        documents_failed: Failed document migrations
        documents_skipped: Skipped documents (already exist)
        chunks_total: Total chunks across all documents
        chunks_migrated: Successfully migrated chunks
        entities_extracted: Number of entities extracted during migration
        relationships_created: Number of relationships created
        bytes_transferred: Total bytes transferred
        start_time: Migration start timestamp
        end_time: Migration end timestamp
        errors: List of error messages
    """
    documents_total: int = 0
    documents_migrated: int = 0
    documents_failed: int = 0
    documents_skipped: int = 0
    chunks_total: int = 0
    chunks_migrated: int = 0
    entities_extracted: int = 0
    relationships_created: int = 0
    bytes_transferred: int = 0
    start_time: datetime | None = None
    end_time: datetime | None = None
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """Calculate migration duration in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or datetime.utcnow()
        return (end - self.start_time).total_seconds()

    @property
    def documents_per_second(self) -> float:
        """Calculate documents migrated per second."""
        duration = self.duration_seconds
        if duration > 0:
            return self.documents_migrated / duration
        return 0.0

    @property
    def success_rate(self) -> float:
        """Calculate migration success rate."""
        total = self.documents_migrated + self.documents_failed
        if total > 0:
            return self.documents_migrated / total
        return 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "documents": {
                "total": self.documents_total,
                "migrated": self.documents_migrated,
                "failed": self.documents_failed,
                "skipped": self.documents_skipped,
                "success_rate": round(self.success_rate, 4),
            },
            "chunks": {
                "total": self.chunks_total,
                "migrated": self.chunks_migrated,
            },
            "entities": {
                "extracted": self.entities_extracted,
                "relationships_created": self.relationships_created,
            },
            "transfer": {
                "bytes": self.bytes_transferred,
                "mb": round(self.bytes_transferred / (1024 * 1024), 2),
            },
            "timing": {
                "start": self.start_time.isoformat() if self.start_time else None,
                "end": self.end_time.isoformat() if self.end_time else None,
                "duration_seconds": round(self.duration_seconds, 2),
                "documents_per_second": round(self.documents_per_second, 2),
            },
            "errors": {
                "count": len(self.errors),
                "details": self.errors[:10] if self.errors else [],  # First 10 errors
            },
        }


@dataclass
class GraphRAGDocument:
    """Represents a document from GraphRAG API.
    
    Attributes:
        document_id: Unique document identifier
        graph_id: Parent graph ID
        job_id: Original job ID
        chunks: List of text chunks
        metadata: Document metadata
        lineage: Processing lineage
        original_format: Original document format
        embeddings: Optional embeddings
    """
    document_id: str
    graph_id: str
    job_id: str
    chunks: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    lineage: dict[str, Any] = field(default_factory=dict)
    original_format: str = "json"
    embeddings: list[list[float]] | None = None

    @classmethod
    def from_api_response(cls, data: dict[str, Any], graph_id: str) -> GraphRAGDocument:
        """Create document from GraphRAG API response."""
        # Handle different API response formats
        text_chunks = data.get("text_chunks", data.get("chunks", []))
        
        return cls(
            document_id=data.get("document_id", data.get("id", str(UUID(int=hash(data) % (2**32))))),
            graph_id=graph_id,
            job_id=data.get("job_id", ""),
            chunks=text_chunks,
            metadata=data.get("metadata", {}),
            lineage=data.get("lineage", {}),
            original_format=data.get("original_format", "json"),
            embeddings=data.get("embeddings"),
        )

    def to_transformed_data(self) -> TransformedData:
        """Convert to TransformedData for CogneeLocalDestination."""
        return TransformedData(
            job_id=UUID(self.job_id) if self.job_id else UUID(int=hash(self.document_id) % (2**32)),
            chunks=self.chunks,
            embeddings=self.embeddings,
            metadata=self.metadata,
            lineage=self.lineage,
            original_format=self.original_format,
            output_format="json",
        )


class GraphRAGClient:
    """Client for interacting with GraphRAG API."""

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
        timeout: int = 120,
    ) -> None:
        """Initialize GraphRAG client.
        
        Args:
            api_url: GraphRAG API base URL
            api_key: GraphRAG API key
            timeout: Request timeout in seconds
        """
        self._api_url = (api_url or GRAPH_RAG_API_URL).rstrip("/")
        self._api_key = api_key or GRAPH_RAG_API_KEY
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> GraphRAGClient:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def connect(self) -> None:
        """Create HTTP client connection."""
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        self._client = httpx.AsyncClient(
            base_url=self._api_url,
            headers=headers,
            timeout=self._timeout,
        )
        
        logger.info(
            "graphrag_client_connected",
            api_url=self._api_url,
        )

    async def close(self) -> None:
        """Close HTTP client connection."""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("graphrag_client_closed")

    async def health_check(self) -> dict[str, Any]:
        """Check GraphRAG API health.
        
        Returns:
            Health status dictionary
        """
        if not self._client:
            return {"healthy": False, "error": "Client not connected"}

        try:
            response = await self._client.get("/v1/health", timeout=10.0)
            return {
                "healthy": response.status_code == 200,
                "status_code": response.status_code,
                "response": response.json() if response.status_code == 200 else None,
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
            }

    async def get_documents(
        self,
        graph_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[GraphRAGDocument]:
        """Get documents from GraphRAG.
        
        Args:
            graph_id: Graph ID to query
            limit: Maximum documents to return
            offset: Skip this many documents
            
        Returns:
            List of documents
        """
        if not self._client:
            raise RuntimeError("Client not connected")

        try:
            response = await self._client.get(
                f"/v1/graphs/{graph_id}/documents",
                params={"limit": limit, "offset": offset},
            )
            response.raise_for_status()
            data = response.json()
            
            # Handle different API response formats
            documents_data = data.get("documents", data if isinstance(data, list) else [])
            
            documents = [
                GraphRAGDocument.from_api_response(doc, graph_id)
                for doc in documents_data
            ]
            
            logger.debug(
                "graphrag_documents_fetched",
                graph_id=graph_id,
                count=len(documents),
                limit=limit,
                offset=offset,
            )
            
            return documents
            
        except httpx.HTTPStatusError as e:
            logger.error(
                "graphrag_documents_fetch_failed",
                graph_id=graph_id,
                status_code=e.response.status_code,
                error=e.response.text,
            )
            raise
        except Exception as e:
            logger.error(
                "graphrag_documents_fetch_error",
                graph_id=graph_id,
                error=str(e),
            )
            raise

    async def get_document_count(self, graph_id: str) -> int:
        """Get total document count for a graph.
        
        Args:
            graph_id: Graph ID to query
            
        Returns:
            Total number of documents
        """
        if not self._client:
            raise RuntimeError("Client not connected")

        try:
            response = await self._client.get(
                f"/v1/graphs/{graph_id}/documents/count",
            )
            response.raise_for_status()
            data = response.json()
            return data.get("count", 0)
        except Exception:
            # Fallback: estimate from first batch
            docs = await self.get_documents(graph_id, limit=1)
            return len(docs)

    async def get_entities(
        self,
        graph_id: str,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Get entities from GraphRAG.
        
        Args:
            graph_id: Graph ID to query
            limit: Maximum entities to return
            
        Returns:
            List of entities
        """
        if not self._client:
            raise RuntimeError("Client not connected")

        try:
            response = await self._client.get(
                f"/v1/graphs/{graph_id}/entities",
                params={"limit": limit},
            )
            response.raise_for_status()
            data = response.json()
            return data.get("entities", [])
        except Exception as e:
            logger.warning(
                "graphrag_entities_fetch_failed",
                graph_id=graph_id,
                error=str(e),
            )
            return []

    async def get_relationships(
        self,
        graph_id: str,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Get relationships from GraphRAG.
        
        Args:
            graph_id: Graph ID to query
            limit: Maximum relationships to return
            
        Returns:
            List of relationships
        """
        if not self._client:
            raise RuntimeError("Client not connected")

        try:
            response = await self._client.get(
                f"/v1/graphs/{graph_id}/relationships",
                params={"limit": limit},
            )
            response.raise_for_status()
            data = response.json()
            return data.get("relationships", [])
        except Exception as e:
            logger.warning(
                "graphrag_relationships_fetch_failed",
                graph_id=graph_id,
                error=str(e),
            )
            return []


class GraphRAGMigrator:
    """Migrates data from API GraphRAG to local Cognee.
    
    This class handles the complete migration process:
    - Connecting to source and target
    - Exporting documents from GraphRAG
    - Importing to CogneeLocalDestination
    - Verification and rollback support
    """

    def __init__(
        self,
        source_config: dict[str, Any] | None = None,
        target_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize migrator.
        
        Args:
            source_config: GraphRAG API configuration
            target_config: CogneeLocalDestination configuration
        """
        self._source_config = source_config or {}
        self._target_config = target_config or {}
        self._source_client: GraphRAGClient | None = None
        self._target_destination: CogneeLocalDestination | None = None
        self._target_connection: Connection | None = None
        self._migrated_document_ids: list[str] = []
        self._stats = MigrationStats()

    async def connect(self) -> None:
        """Connect to source and target."""
        logger.info("migrator_connecting")

        # Connect to source (GraphRAG)
        self._source_client = GraphRAGClient(
            api_url=self._source_config.get("api_url"),
            api_key=self._source_config.get("api_key"),
            timeout=self._source_config.get("timeout", 120),
        )
        await self._source_client.connect()

        # Verify source health
        source_health = await self._source_client.health_check()
        if not source_health.get("healthy"):
            logger.warning(
                "graphrag_health_check_failed",
                health=source_health,
            )
        else:
            logger.info("graphrag_source_healthy")

        # Connect to target (Cognee)
        self._target_destination = CogneeLocalDestination()
        await self._target_destination.initialize(self._target_config)
        
        logger.info("migrator_connected")

    async def migrate_documents(
        self,
        dataset_id: str,
        batch_size: int = 100,
        target_dataset_id: str | None = None,
        dry_run: bool = False,
        skip_existing: bool = False,
        progress_callback: callable | None = None,
    ) -> dict[str, Any]:
        """Migrate documents from GraphRAG to Cognee.
        
        Args:
            dataset_id: Source dataset/graph ID
            batch_size: Number of documents per batch
            target_dataset_id: Target dataset ID (defaults to source)
            dry_run: If True, show what would be migrated without making changes
            skip_existing: Skip documents that already exist in target
            progress_callback: Optional callback for progress updates
            
        Returns:
            Migration statistics
        """
        self._stats.start_time = datetime.utcnow()
        target_dataset = target_dataset_id or dataset_id
        
        logger.info(
            "migration_started",
            source_dataset=dataset_id,
            target_dataset=target_dataset,
            batch_size=batch_size,
            dry_run=dry_run,
        )

        if not self._source_client or not self._target_destination:
            raise RuntimeError("Migrator not connected. Call connect() first.")

        # Create target connection
        self._target_connection = await self._target_destination.connect({
            "dataset_id": target_dataset,
            "extract_entities": True,
            "extract_relationships": True,
            "store_vectors": True,
        })

        # Get total document count
        try:
            total_docs = await self._source_client.get_document_count(dataset_id)
            self._stats.documents_total = total_docs
            logger.info(
                "graphrag_document_count",
                count=total_docs,
            )
        except Exception as e:
            logger.warning(
                "failed_to_get_document_count",
                error=str(e),
            )
            total_docs = None

        if dry_run:
            logger.info(
                "dry_run_mode",
                message="Would migrate documents without making changes",
                estimated_documents=total_docs or "unknown",
            )
            # Fetch first batch to show sample
            sample_docs = await self._source_client.get_documents(dataset_id, limit=5)
            logger.info(
                "dry_run_sample",
                sample_documents=[d.document_id for d in sample_docs],
            )
            self._stats.end_time = datetime.utcnow()
            return self._stats.to_dict()

        # Migration loop with pagination
        offset = 0
        has_more = True
        
        pbar = tqdm(
            total=total_docs,
            desc="Migrating documents",
            unit="docs",
        ) if total_docs else tqdm(desc="Migrating documents", unit="docs")

        try:
            while has_more:
                # Fetch batch from source
                try:
                    documents = await self._source_client.get_documents(
                        dataset_id,
                        limit=batch_size,
                        offset=offset,
                    )
                except Exception as e:
                    logger.error(
                        "batch_fetch_failed",
                        offset=offset,
                        batch_size=batch_size,
                        error=str(e),
                    )
                    self._stats.errors.append({
                        "phase": "fetch",
                        "offset": offset,
                        "error": str(e),
                    })
                    offset += batch_size
                    continue

                if not documents:
                    has_more = False
                    break

                # Process each document
                for doc in documents:
                    try:
                        # Check if document already exists (if skip_existing)
                        if skip_existing and await self._document_exists(
                            target_dataset, doc.document_id
                        ):
                            self._stats.documents_skipped += 1
                            logger.debug(
                                "document_skipped",
                                document_id=doc.document_id,
                            )
                            pbar.update(1)
                            continue

                        # Convert and write to target
                        transformed_data = doc.to_transformed_data()
                        
                        result = await self._target_destination.write(
                            self._target_connection,
                            transformed_data,
                        )

                        if result.success:
                            self._stats.documents_migrated += 1
                            self._stats.chunks_total += len(doc.chunks)
                            self._stats.chunks_migrated += result.records_written - 1  # Exclude document
                            self._stats.bytes_transferred += result.bytes_written
                            self._stats.entities_extracted += result.metadata.get("entities_extracted", 0)
                            self._stats.relationships_created += result.metadata.get("relationships_created", 0)
                            self._migrated_document_ids.append(doc.document_id)
                            
                            logger.debug(
                                "document_migrated",
                                document_id=doc.document_id,
                                chunks=len(doc.chunks),
                            )
                        else:
                            self._stats.documents_failed += 1
                            self._stats.errors.append({
                                "phase": "write",
                                "document_id": doc.document_id,
                                "error": result.error or "Unknown error",
                            })
                            logger.error(
                                "document_migration_failed",
                                document_id=doc.document_id,
                                error=result.error,
                            )

                    except Exception as e:
                        self._stats.documents_failed += 1
                        self._stats.errors.append({
                            "phase": "process",
                            "document_id": doc.document_id,
                            "error": str(e),
                        })
                        logger.error(
                            "document_migration_error",
                            document_id=doc.document_id,
                            error=str(e),
                        )

                    pbar.update(1)
                    if progress_callback:
                        progress_callback(self._stats)

                offset += len(documents)
                
                # If we got fewer documents than batch_size, we've reached the end
                if len(documents) < batch_size:
                    has_more = False

        finally:
            pbar.close()

        self._stats.end_time = datetime.utcnow()
        
        logger.info(
            "migration_completed",
            stats=self._stats.to_dict(),
        )

        return self._stats.to_dict()

    async def _document_exists(self, dataset_id: str, document_id: str) -> bool:
        """Check if a document already exists in the target.
        
        Args:
            dataset_id: Target dataset ID
            document_id: Document ID to check
            
        Returns:
            True if document exists
        """
        try:
            neo4j_client = await get_neo4j_client()
            result = await neo4j_client.execute_query(
                """
                MATCH (d:Document {id: $doc_id, dataset_id: $dataset_id})
                RETURN count(d) as count
                """,
                {"doc_id": document_id, "dataset_id": dataset_id},
            )
            return result[0].get("count", 0) > 0 if result else False
        except Exception as e:
            logger.warning(
                "document_existence_check_failed",
                document_id=document_id,
                error=str(e),
            )
            return False

    async def verify_migration(
        self,
        dataset_id: str,
        target_dataset_id: str | None = None,
    ) -> dict[str, Any]:
        """Verify migrated data.
        
        Args:
            dataset_id: Source dataset ID
            target_dataset_id: Target dataset ID (defaults to source)
            
        Returns:
            Verification results
        """
        target_dataset = target_dataset_id or dataset_id
        
        logger.info(
            "migration_verification_started",
            source_dataset=dataset_id,
            target_dataset=target_dataset,
        )

        results = {
            "verified_at": datetime.utcnow().isoformat(),
            "source_dataset": dataset_id,
            "target_dataset": target_dataset,
            "checks": {},
            "overall_verified": False,
        }

        if not self._source_client:
            results["checks"]["source_connection"] = {
                "passed": False,
                "error": "Source not connected",
            }
            return results

        # Get source counts
        try:
            source_entities = await self._source_client.get_entities(dataset_id, limit=1000)
            results["checks"]["source_entities_count"] = {
                "count": len(source_entities),
                "passed": True,
            }
        except Exception as e:
            results["checks"]["source_entities"] = {
                "passed": False,
                "error": str(e),
            }
            source_entities = []

        # Get target counts from Neo4j
        try:
            neo4j_client = await get_neo4j_client()
            
            # Count documents
            doc_result = await neo4j_client.execute_query(
                """
                MATCH (d:Document {dataset_id: $dataset_id})
                RETURN count(d) as count
                """,
                {"dataset_id": target_dataset},
            )
            target_doc_count = doc_result[0].get("count", 0) if doc_result else 0
            
            # Count chunks
            chunk_result = await neo4j_client.execute_query(
                """
                MATCH (c:Chunk {dataset_id: $dataset_id})
                RETURN count(c) as count
                """,
                {"dataset_id": target_dataset},
            )
            target_chunk_count = chunk_result[0].get("count", 0) if chunk_result else 0
            
            # Count entities
            entity_result = await neo4j_client.execute_query(
                """
                MATCH (e:Entity {dataset_id: $dataset_id})
                RETURN count(e) as count
                """,
                {"dataset_id": target_dataset},
            )
            target_entity_count = entity_result[0].get("count", 0) if entity_result else 0
            
            results["checks"]["target_counts"] = {
                "passed": True,
                "documents": target_doc_count,
                "chunks": target_chunk_count,
                "entities": target_entity_count,
            }

            # Compare with migration stats
            if hasattr(self, '_stats') and self._stats.documents_migrated > 0:
                doc_match = target_doc_count >= self._stats.documents_migrated
                results["checks"]["document_count_match"] = {
                    "passed": doc_match,
                    "expected": self._stats.documents_migrated,
                    "actual": target_doc_count,
                }

        except Exception as e:
            results["checks"]["target_counts"] = {
                "passed": False,
                "error": str(e),
            }

        # Sample verification
        try:
            if self._migrated_document_ids:
                sample_size = min(5, len(self._migrated_document_ids))
                sample_ids = self._migrated_document_ids[:sample_size]
                
                verified_samples = 0
                for doc_id in sample_ids:
                    # Check if document exists in target
                    exists = await self._document_exists(target_dataset, doc_id)
                    if exists:
                        verified_samples += 1

                results["checks"]["sample_verification"] = {
                    "passed": verified_samples == sample_size,
                    "verified": verified_samples,
                    "total": sample_size,
                }
        except Exception as e:
            results["checks"]["sample_verification"] = {
                "passed": False,
                "error": str(e),
            }

        # Overall verification
        all_passed = all(
            check.get("passed", False)
            for check in results["checks"].values()
        )
        results["overall_verified"] = all_passed

        logger.info(
            "migration_verification_completed",
            verified=all_passed,
            results=results,
        )

        return results

    async def rollback(self, dataset_id: str) -> dict[str, Any]:
        """Rollback migration if needed.
        
        Removes all documents and related data that were migrated.
        
        Args:
            dataset_id: Dataset to rollback
            
        Returns:
            Rollback results
        """
        logger.info(
            "rollback_started",
            dataset_id=dataset_id,
            documents_to_rollback=len(self._migrated_document_ids),
        )

        results = {
            "started_at": datetime.utcnow().isoformat(),
            "dataset_id": dataset_id,
            "documents_removed": 0,
            "chunks_removed": 0,
            "entities_removed": 0,
            "errors": [],
        }

        try:
            neo4j_client = await get_neo4j_client()

            # Delete chunks
            chunk_result = await neo4j_client.execute_write(
                """
                MATCH (c:Chunk {dataset_id: $dataset_id})
                WITH c LIMIT 1000
                DETACH DELETE c
                RETURN count(*) as deleted
                """,
                {"dataset_id": dataset_id},
            )
            results["chunks_removed"] = chunk_result[0].get("deleted", 0) if chunk_result else 0

            # Delete entities
            entity_result = await neo4j_client.execute_write(
                """
                MATCH (e:Entity {dataset_id: $dataset_id})
                WITH e LIMIT 1000
                DETACH DELETE e
                RETURN count(*) as deleted
                """,
                {"dataset_id": dataset_id},
            )
            results["entities_removed"] = entity_result[0].get("deleted", 0) if entity_result else 0

            # Delete documents
            doc_result = await neo4j_client.execute_write(
                """
                MATCH (d:Document {dataset_id: $dataset_id})
                WITH d LIMIT 1000
                DETACH DELETE d
                RETURN count(*) as deleted
                """,
                {"dataset_id": dataset_id},
            )
            results["documents_removed"] = doc_result[0].get("deleted", 0) if doc_result else 0

            # Delete dataset node
            await neo4j_client.execute_write(
                """
                MATCH (ds:Dataset {id: $dataset_id})
                DELETE ds
                """,
                {"dataset_id": dataset_id},
            )

            logger.info(
                "rollback_completed",
                documents_removed=results["documents_removed"],
                chunks_removed=results["chunks_removed"],
                entities_removed=results["entities_removed"],
            )

        except Exception as e:
            logger.error(
                "rollback_error",
                error=str(e),
            )
            results["errors"].append(str(e))

        results["completed_at"] = datetime.utcnow().isoformat()
        return results

    async def close(self) -> None:
        """Close connections."""
        logger.info("migrator_closing")

        if self._source_client:
            await self._source_client.close()
            self._source_client = None

        if self._target_destination:
            await self._target_destination.shutdown()
            self._target_destination = None

        await close_neo4j_client()
        
        logger.info("migrator_closed")


async def main() -> int:
    """Main entry point.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Migrate from API GraphRAG to local Cognee",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dry run to see what would be migrated
    python scripts/migrate_to_cognee_local.py --source-dataset my-dataset --dry-run

    # Migrate with default settings
    python scripts/migrate_to_cognee_local.py --source-dataset my-dataset

    # Migrate with custom batch size
    python scripts/migrate_to_cognee_local.py --source-dataset my-dataset --batch-size 50

    # Migrate to a different target dataset
    python scripts/migrate_to_cognee_local.py --source-dataset old-dataset --target-dataset new-dataset

    # Migrate and verify
    python scripts/migrate_to_cognee_local.py --source-dataset my-dataset --verify

    # Skip existing documents
    python scripts/migrate_to_cognee_local.py --source-dataset my-dataset --skip-existing
        """,
    )
    parser.add_argument(
        "--source-dataset",
        required=True,
        help="Source GraphRAG dataset/graph ID",
    )
    parser.add_argument(
        "--target-dataset",
        help="Target Cognee dataset ID (defaults to source)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of documents per batch (default: 100)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify migration after completion",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip documents that already exist in target",
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback a previous migration",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file for migration report (JSON)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--graphrag-url",
        help="GraphRAG API URL (overrides env var)",
    )
    parser.add_argument(
        "--graphrag-key",
        help="GraphRAG API key (overrides env var)",
    )
    parser.add_argument(
        "--neo4j-uri",
        help="Neo4j URI (overrides env var)",
    )
    parser.add_argument(
        "--neo4j-password",
        help="Neo4j password (overrides env var)",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(json_format=False, log_level=log_level)

    # Validate environment
    if not args.rollback and not args.dry_run:
        if not GRAPH_RAG_API_URL and not args.graphrag_url:
            logger.error(
                "missing_graphrag_url",
                message="Set GRAPH_RAG_API_URL environment variable or use --graphrag-url",
            )
            return 1

    # Build configuration
    source_config = {
        "api_url": args.graphrag_url or GRAPH_RAG_API_URL,
        "api_key": args.graphrag_key or GRAPH_RAG_API_KEY,
        "timeout": 120,
    }

    target_config = {
        "dataset_id": args.target_dataset or args.source_dataset,
        "graph_name": args.target_dataset or args.source_dataset,
        "neo4j_uri": args.neo4j_uri or NEO4J_URI,
        "neo4j_user": NEO4J_USER,
        "neo4j_password": args.neo4j_password or NEO4J_PASSWORD,
    }

    # Run migration
    migrator = GraphRAGMigrator(
        source_config=source_config,
        target_config=target_config,
    )

    try:
        await migrator.connect()

        if args.rollback:
            # Rollback mode
            target_dataset = args.target_dataset or args.source_dataset
            results = await migrator.rollback(target_dataset)
            print("\n" + "=" * 60)
            print("ROLLBACK RESULTS")
            print("=" * 60)
            print(json.dumps(results, indent=2))
        else:
            # Migration mode
            stats = await migrator.migrate_documents(
                dataset_id=args.source_dataset,
                batch_size=args.batch_size,
                target_dataset_id=args.target_dataset,
                dry_run=args.dry_run,
                skip_existing=args.skip_existing,
            )

            print("\n" + "=" * 60)
            print("MIGRATION RESULTS")
            print("=" * 60)
            print(json.dumps(stats, indent=2))

            # Verification
            if args.verify and not args.dry_run:
                print("\n" + "=" * 60)
                print("VERIFICATION RESULTS")
                print("=" * 60)
                
                verification = await migrator.verify_migration(
                    dataset_id=args.source_dataset,
                    target_dataset_id=args.target_dataset,
                )
                print(json.dumps(verification, indent=2))

                if verification.get("overall_verified"):
                    print("\n✓ Migration verified successfully!")
                else:
                    print("\n⚠ Migration verification found issues")
                    return 1

            # Success criteria
            if not args.dry_run and stats.get("documents", {}).get("success_rate", 0) < 0.95:
                print("\n⚠ Warning: Success rate below 95%")
                return 1

        # Save report to file if requested
        if args.output:
            report = {
                "timestamp": datetime.utcnow().isoformat(),
                "source_dataset": args.source_dataset,
                "target_dataset": args.target_dataset or args.source_dataset,
                "dry_run": args.dry_run,
                "results": stats if not args.rollback else results,
            }
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\nReport saved to: {args.output}")

        return 0

    except KeyboardInterrupt:
        print("\n\nMigration interrupted by user")
        return 130
    except Exception as e:
        logger.error("migration_failed", error=str(e), exc_info=True)
        print(f"\n✗ Migration failed: {e}")
        return 1
    finally:
        await migrator.close()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
