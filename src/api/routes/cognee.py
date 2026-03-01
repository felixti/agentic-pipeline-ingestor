"""Cognee GraphRAG API routes.

This module provides endpoints for:
- Searching the Cognee knowledge graph (vector, graph, hybrid search)
- Extracting entities and relationships from text
- Retrieving graph statistics

All endpoints integrate with the CogneeLocalDestination plugin for
graph-based document storage and retrieval.
"""

import time
import uuid
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from src.api.models.cognee import (
    CogneeEntity,
    CogneeExtractEntitiesRequest,
    CogneeExtractEntitiesResponse,
    CogneeRelationship,
    CogneeSearchRequest,
    CogneeSearchResponse,
    CogneeSearchResult,
    CogneeStatsResponse,
)
from src.observability.logging import get_logger
from src.plugins.base import Connection
from src.plugins.destinations.cognee_local import CogneeLocalDestination

router = APIRouter(prefix="/cognee", tags=["Cognee GraphRAG"])
logger = get_logger(__name__)

# ============================================================================
# Singleton instance for Cognee destination (initialized on first use)
# ============================================================================

_cognee_destination: CogneeLocalDestination | None = None
_cognee_destination_lock = Lock()


def _get_cognee_destination() -> CogneeLocalDestination:
    """Get or initialize the CogneeLocalDestination singleton.

    Returns:
        CogneeLocalDestination instance
    """
    global _cognee_destination

    existing = _cognee_destination
    if existing is not None:
        return existing

    with _cognee_destination_lock:
        if _cognee_destination is None:
            logger.info("Initializing CogneeLocalDestination")

            _cognee_destination = CogneeLocalDestination()

            # Initialize with default configuration
            try:
                import asyncio

                asyncio.run(_cognee_destination.initialize({}))
                logger.info("CogneeLocalDestination initialized successfully")
            except Exception as e:
                logger.warning(
                    "CogneeLocalDestination initialization failed",
                    error=str(e),
                )
                # Don't raise - let endpoints handle unavailability

        return _cognee_destination


async def _get_cognee_destination_async() -> CogneeLocalDestination:
    """Get or initialize the CogneeLocalDestination singleton (async version).

    Returns:
        CogneeLocalDestination instance

    Raises:
        HTTPException: If destination is not available
    """
    global _cognee_destination

    if _cognee_destination is None:
        with _cognee_destination_lock:
            if _cognee_destination is None:
                logger.info("Initializing CogneeLocalDestination")

                destination = CogneeLocalDestination()

                try:
                    await destination.initialize({})
                    _cognee_destination = destination
                    logger.info("CogneeLocalDestination initialized successfully")
                except Exception as e:
                    logger.error(
                        "CogneeLocalDestination initialization failed",
                        error=str(e),
                    )
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail={
                            "error": {
                                "code": "COGNEE_UNAVAILABLE",
                                "message": f"Cognee service is not available: {e!s}",
                            }
                        },
                    ) from e

    assert _cognee_destination is not None
    return _cognee_destination


async def _get_connection(dataset_id: str) -> Connection:
    """Get a connection to the specified dataset.

    Args:
        dataset_id: Dataset identifier

    Returns:
        Connection handle
    """
    destination = await _get_cognee_destination_async()
    return await destination.connect({"dataset_id": dataset_id})


def _convert_search_results(
    cognee_results: list[Any],
) -> list[CogneeSearchResult]:
    """Convert Cognee search results to API CogneeSearchResult models.

    Args:
        cognee_results: Raw results from Cognee search

    Returns:
        List of CogneeSearchResult models
    """
    results = []

    for result in cognee_results:
        if hasattr(result, "chunk_id"):
            # SearchResult dataclass
            results.append(
                CogneeSearchResult(
                    chunk_id=result.chunk_id,
                    content=result.content,
                    score=result.score,
                    source_document=result.source_document,
                    entities=result.entities if hasattr(result, "entities") else [],
                )
            )
        elif isinstance(result, dict):
            # Dictionary result
            results.append(
                CogneeSearchResult(
                    chunk_id=result.get("chunk_id", result.get("id", "unknown")),
                    content=result.get("content", result.get("text", "")),
                    score=result.get("score", 0.0),
                    source_document=result.get("source_document", result.get("document_id", "unknown")),
                    entities=result.get("entities", []),
                )
            )
        else:
            # String or other type
            results.append(
                CogneeSearchResult(
                    chunk_id=f"result_{len(results)}",
                    content=str(result),
                    score=1.0 - (len(results) * 0.1),
                    source_document="unknown",
                    entities=[],
                )
            )

    return results


# ============================================================================
# Routes
# ============================================================================


@router.post(
    "/search",
    response_model=CogneeSearchResponse,
    summary="Search the Cognee knowledge graph",
    description="""
    Search the Cognee knowledge graph using vector, graph, or hybrid search.

    The endpoint supports three search types:
    - **vector**: Vector similarity search using embeddings
    - **graph**: Graph traversal search using relationships
    - **hybrid**: Combined vector and graph search (default)

    Results include the matched chunks, relevance scores, source documents,
    and any entities found in the content.
    """,
)
async def cognee_search(
    request: Request,
    search_request: CogneeSearchRequest,
) -> CogneeSearchResponse:
    """Search the Cognee knowledge graph.

    Args:
        request: FastAPI request object
        search_request: Search parameters

    Returns:
        Search response with results and metadata

    Raises:
        HTTPException: If search fails or service unavailable
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    start_time = time.perf_counter()

    logger.info(
        "cognee_search_request",
        request_id=request_id,
        query=search_request.query[:100],
        search_type=search_request.search_type,
        dataset_id=search_request.dataset_id,
        top_k=search_request.top_k,
    )

    try:
        # Get connection to dataset
        conn = await _get_connection(search_request.dataset_id)

        # Get Cognee destination
        destination = await _get_cognee_destination_async()

        # Execute search
        cognee_results = await destination.search(
            conn=conn,
            query=search_request.query,
            search_type=search_request.search_type,
            top_k=search_request.top_k,
        )

        # Convert results
        results = _convert_search_results(cognee_results)

        # Calculate query time
        query_time_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "cognee_search_completed",
            request_id=request_id,
            result_count=len(results),
            query_time_ms=round(query_time_ms, 2),
        )

        return CogneeSearchResponse(
            results=results,
            search_type=search_request.search_type,
            dataset_id=search_request.dataset_id,
            query_time_ms=round(query_time_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        query_time_ms = (time.perf_counter() - start_time) * 1000

        logger.error(
            "cognee_search_failed",
            request_id=request_id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "SEARCH_FAILED",
                    "message": f"Failed to execute search: {e!s}",
                },
                "query_time_ms": round(query_time_ms, 2),
            },
        ) from e


@router.post(
    "/extract-entities",
    response_model=CogneeExtractEntitiesResponse,
    summary="Extract entities and relationships from text",
    description="""
    Extract entities and relationships from the provided text using Cognee's
    entity extraction capabilities.

    The endpoint identifies:
    - **Entities**: Named entities (people, organizations, locations, etc.)
    - **Relationships**: Connections between entities

    This is useful for:
    - Analyzing unstructured text
    - Preparing content for knowledge graph ingestion
    - Understanding entity relationships in documents
    """,
)
async def cognee_extract_entities(
    request: Request,
    extract_request: CogneeExtractEntitiesRequest,
) -> CogneeExtractEntitiesResponse:
    """Extract entities and relationships from text.

    Args:
        request: FastAPI request object
        extract_request: Extraction parameters

    Returns:
        Extraction response with entities and relationships

    Raises:
        HTTPException: If extraction fails or service unavailable
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    start_time = time.perf_counter()

    logger.info(
        "cognee_extract_entities_request",
        request_id=request_id,
        text_length=len(extract_request.text),
        dataset_id=extract_request.dataset_id,
    )

    try:
        # Get connection to dataset
        conn = await _get_connection(extract_request.dataset_id)

        # Get Cognee destination
        destination = await _get_cognee_destination_async()

        # Execute extraction
        result = await destination.extract_entities(
            conn=conn,
            text=extract_request.text,
        )

        # Convert to API models
        entities = [
            CogneeEntity(
                name=e.get("name", ""),
                type=e.get("type", "UNKNOWN"),
                description=e.get("description", ""),
            )
            for e in result.get("entities", [])
        ]

        relationships = [
            CogneeRelationship(
                source=r.get("source", ""),
                target=r.get("target", ""),
                type=r.get("type", "UNKNOWN"),
            )
            for r in result.get("relationships", [])
        ]

        query_time_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "cognee_extract_entities_completed",
            request_id=request_id,
            entity_count=len(entities),
            relationship_count=len(relationships),
            query_time_ms=round(query_time_ms, 2),
        )

        return CogneeExtractEntitiesResponse(
            entities=entities,
            relationships=relationships,
        )

    except HTTPException:
        raise
    except Exception as e:
        query_time_ms = (time.perf_counter() - start_time) * 1000

        logger.error(
            "cognee_extract_entities_failed",
            request_id=request_id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "EXTRACTION_FAILED",
                    "message": f"Failed to extract entities: {e!s}",
                }
            },
        ) from e


@router.get(
    "/stats",
    response_model=CogneeStatsResponse,
    summary="Get Cognee graph statistics",
    description="""
    Retrieve statistics about the Cognee knowledge graph for a dataset.

    Returns:
    - **document_count**: Total number of documents
    - **chunk_count**: Total number of chunks
    - **entity_count**: Total entities in the knowledge graph
    - **relationship_count**: Total relationships in the graph
    - **graph_density**: Graph density metric (0-1)
    - **last_updated**: Timestamp of last update
    """,
)
async def cognee_stats(
    request: Request,
    dataset_id: str = "default",
) -> CogneeStatsResponse:
    """Get Cognee graph statistics.

    Args:
        request: FastAPI request object
        dataset_id: Dataset identifier (default: "default")

    Returns:
        Statistics response with graph metrics

    Raises:
        HTTPException: If stats retrieval fails or service unavailable
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    start_time = time.perf_counter()

    logger.info(
        "cognee_stats_request",
        request_id=request_id,
        dataset_id=dataset_id,
    )

    try:
        # Get connection to dataset
        conn = await _get_connection(dataset_id)

        # Get Cognee destination
        destination = await _get_cognee_destination_async()

        # Get stats
        stats = await destination.get_stats(conn)

        query_time_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "cognee_stats_completed",
            request_id=request_id,
            dataset_id=dataset_id,
            document_count=stats.get("document_count", 0),
            entity_count=stats.get("entity_count", 0),
            query_time_ms=round(query_time_ms, 2),
        )

        return CogneeStatsResponse(
            dataset_id=stats.get("dataset_id", dataset_id),
            document_count=stats.get("document_count", 0),
            chunk_count=stats.get("chunk_count", 0),
            entity_count=stats.get("entity_count", 0),
            relationship_count=stats.get("relationship_count", 0),
            graph_density=stats.get("graph_density", 0.0),
            last_updated=stats.get("last_updated", datetime.now(timezone.utc)),
        )

    except HTTPException:
        raise
    except Exception as e:
        query_time_ms = (time.perf_counter() - start_time) * 1000

        logger.error(
            "cognee_stats_failed",
            request_id=request_id,
            dataset_id=dataset_id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "STATS_FAILED",
                    "message": f"Failed to retrieve statistics: {e!s}",
                }
            },
        ) from e
