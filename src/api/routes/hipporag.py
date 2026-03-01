"""HippoRAG Multi-Hop RAG API routes.

This module provides endpoints for:
- Multi-hop retrieval using HippoRAG Personalized PageRank algorithm
- Complete RAG pipeline with retrieval and answer generation
- Triple extraction from text using OpenIE

All endpoints integrate with the HippoRAGDestination plugin for
graph-based multi-hop reasoning and knowledge graph operations.
"""

import time
import uuid
from threading import Lock
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.models.hipporag import (
    HippoRAGExtractTriplesRequest,
    HippoRAGExtractTriplesResponse,
    HippoRAGQARequest,
    HippoRAGQAResponse,
    HippoRAGQAResult,
    HippoRAGRetrievalResult,
    HippoRAGRetrieveRequest,
    HippoRAGRetrieveResponse,
    HippoRAGTriple,
)
from src.auth.dependencies import get_current_user
from src.observability.logging import get_logger
from src.plugins.destinations.hipporag import HippoRAGDestination, QAResult, RetrievalResult

router = APIRouter(prefix="/hipporag", tags=["HippoRAG Multi-Hop"])
logger = get_logger(__name__)

# ============================================================================
# Singleton instance for HippoRAG destination (initialized on first use)
# ============================================================================

_hipporag_destination: HippoRAGDestination | None = None
_hipporag_destination_lock = Lock()


def _get_hipporag_destination() -> HippoRAGDestination:
    """Get or initialize the HippoRAGDestination singleton.

    Returns:
        HippoRAGDestination instance
    """
    global _hipporag_destination

    existing = _hipporag_destination
    if existing is not None:
        return existing

    with _hipporag_destination_lock:
        if _hipporag_destination is None:
            logger.info("Initializing HippoRAGDestination")

            _hipporag_destination = HippoRAGDestination()

            # Initialize with default configuration
            try:
                import asyncio

                asyncio.run(_hipporag_destination.initialize({}))
                logger.info("HippoRAGDestination initialized successfully")
            except Exception as e:
                logger.warning(
                    "HippoRAGDestination initialization failed",
                    error=str(e),
                )
                # Don't raise - let endpoints handle unavailability

        return _hipporag_destination


async def _get_hipporag_destination_async() -> HippoRAGDestination:
    """Get or initialize the HippoRAGDestination singleton (async version).

    Returns:
        HippoRAGDestination instance

    Raises:
        HTTPException: If destination is not available
    """
    global _hipporag_destination

    if _hipporag_destination is None:
        with _hipporag_destination_lock:
            if _hipporag_destination is None:
                logger.info("Initializing HippoRAGDestination")

                destination = HippoRAGDestination()

                try:
                    await destination.initialize({})
                    _hipporag_destination = destination
                    logger.info("HippoRAGDestination initialized successfully")
                except Exception as e:
                    logger.error(
                        "HippoRAGDestination initialization failed",
                        error=str(e),
                    )
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail={
                            "error": {
                                "code": "HIPPO_RAG_UNAVAILABLE",
                                "message": f"HippoRAG service is not available: {e!s}",
                            }
                        },
                    ) from e

    assert _hipporag_destination is not None
    return _hipporag_destination


def _convert_retrieval_result(result: RetrievalResult) -> HippoRAGRetrievalResult:
    """Convert internal RetrievalResult to API HippoRAGRetrievalResult model.

    Args:
        result: Internal RetrievalResult dataclass

    Returns:
        API HippoRAGRetrievalResult model
    """
    return HippoRAGRetrievalResult(
        query=result.query,
        passages=result.passages,
        scores=result.scores,
        source_documents=result.source_documents,
        entities=result.entities,
    )


def _convert_qa_result(result: QAResult) -> HippoRAGQAResult:
    """Convert internal QAResult to API HippoRAGQAResult model.

    Args:
        result: Internal QAResult dataclass

    Returns:
        API HippoRAGQAResult model
    """
    retrieval_results = (
        _convert_retrieval_result(result.retrieval_results)
        if result.retrieval_results
        else HippoRAGRetrievalResult(query=result.query)
    )

    return HippoRAGQAResult(
        query=result.query,
        answer=result.answer,
        sources=result.sources,
        confidence=result.confidence,
        retrieval_results=retrieval_results,
    )


# ============================================================================
# Routes
# ============================================================================


@router.post(
    "/retrieve",
    response_model=HippoRAGRetrieveResponse,
    summary="Multi-hop retrieval using HippoRAG",
    description="""
    Perform multi-hop retrieval using HippoRAG's Personalized PageRank algorithm.
    
    The endpoint:
    1. Extracts entities from each query
    2. Runs Personalized PageRank on the knowledge graph
    3. Retrieves relevant passages across multiple hops
    4. Returns passages with relevance scores and source documents
    
    This is useful for:
    - Finding connected information across multiple documents
    - Multi-hop reasoning without explicit chain-of-thought
    - Discovering implicit relationships in the knowledge graph

    **Authentication Required**
    """,
)
async def hipporag_retrieve(
    request: Request,
    retrieve_request: HippoRAGRetrieveRequest,
    current_user: dict = Depends(get_current_user),
) -> HippoRAGRetrieveResponse:
    """Perform multi-hop retrieval using HippoRAG.

    Args:
        request: FastAPI request object
        retrieve_request: Retrieval parameters

    Returns:
        Retrieval response with results and query time

    Raises:
        HTTPException: If retrieval fails or service unavailable
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    start_time = time.perf_counter()

    logger.info(
        "hipporag_retrieve_request",
        request_id=request_id,
        query_count=len(retrieve_request.queries),
        num_to_retrieve=retrieve_request.num_to_retrieve,
    )

    try:
        # Get HippoRAG destination
        destination = await _get_hipporag_destination_async()

        # Execute retrieval
        retrieval_results = await destination.retrieve(
            queries=retrieve_request.queries,
            num_to_retrieve=retrieve_request.num_to_retrieve,
        )

        # Convert results to API models
        results = [_convert_retrieval_result(r) for r in retrieval_results]

        # Calculate query time
        query_time_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "hipporag_retrieve_completed",
            request_id=request_id,
            result_count=len(results),
            query_time_ms=round(query_time_ms, 2),
        )

        return HippoRAGRetrieveResponse(
            results=results,
            query_time_ms=round(query_time_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        query_time_ms = (time.perf_counter() - start_time) * 1000

        logger.error(
            "hipporag_retrieve_failed",
            request_id=request_id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "RETRIEVAL_FAILED",
                    "message": f"Failed to execute retrieval: {e!s}",
                },
                "query_time_ms": round(query_time_ms, 2),
            },
        ) from e


@router.post(
    "/qa",
    response_model=HippoRAGQAResponse,
    summary="RAG QA with HippoRAG multi-hop retrieval",
    description="""
    Complete RAG pipeline with HippoRAG multi-hop retrieval and answer generation.
    
    The endpoint:
    1. Performs multi-hop retrieval for each question
    2. Generates answers using the retrieved context
    3. Returns answers with confidence scores and sources
    
    This combines the power of HippoRAG's knowledge graph traversal
    with LLM-based answer generation for comprehensive question answering.

    **Authentication Required**
    """,
)
async def hipporag_qa(
    request: Request,
    qa_request: HippoRAGQARequest,
    current_user: dict = Depends(get_current_user),
) -> HippoRAGQAResponse:
    """Execute RAG QA with HippoRAG multi-hop retrieval.

    Args:
        request: FastAPI request object
        qa_request: QA parameters

    Returns:
        QA response with answers, sources, and query time

    Raises:
        HTTPException: If QA fails or service unavailable
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    start_time = time.perf_counter()

    logger.info(
        "hipporag_qa_request",
        request_id=request_id,
        query_count=len(qa_request.queries),
        num_to_retrieve=qa_request.num_to_retrieve,
    )

    try:
        # Get HippoRAG destination
        destination = await _get_hipporag_destination_async()

        # Execute RAG QA
        qa_results = await destination.rag_qa(
            queries=qa_request.queries,
            num_to_retrieve=qa_request.num_to_retrieve,
        )

        # Convert results to API models
        results = [_convert_qa_result(r) for r in qa_results]

        # Estimate total tokens (rough approximation)
        total_tokens = sum(
            len(r.answer.split()) + len(r.query.split())
            for r in qa_results
        )

        # Calculate query time
        query_time_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "hipporag_qa_completed",
            request_id=request_id,
            result_count=len(results),
            total_tokens=total_tokens,
            query_time_ms=round(query_time_ms, 2),
        )

        return HippoRAGQAResponse(
            results=results,
            total_tokens=total_tokens,
            query_time_ms=round(query_time_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        query_time_ms = (time.perf_counter() - start_time) * 1000

        logger.error(
            "hipporag_qa_failed",
            request_id=request_id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "QA_FAILED",
                    "message": f"Failed to execute QA: {e!s}",
                },
                "query_time_ms": round(query_time_ms, 2),
            },
        ) from e


@router.post(
    "/extract-triples",
    response_model=HippoRAGExtractTriplesResponse,
    summary="Extract knowledge graph triples from text",
    description="""
    Extract subject-predicate-object triples from text using OpenIE.
    
    The endpoint uses an LLM to identify entities and relationships
    in the provided text, returning structured triples that can be
    used for knowledge graph construction.
    
    Example triples:
    - ("Steve Jobs", "founded", "Apple Inc.")
    - ("Apple Inc.", "headquartered in", "Cupertino")
    
    This is useful for:
    - Building knowledge graphs from unstructured text
    - Understanding entity relationships
    - Preparing content for HippoRAG indexing

    **Authentication Required**
    """,
)
async def hipporag_extract_triples(
    request: Request,
    extract_request: HippoRAGExtractTriplesRequest,
    current_user: dict = Depends(get_current_user),
) -> HippoRAGExtractTriplesResponse:
    """Extract knowledge graph triples from text.

    Args:
        request: FastAPI request object
        extract_request: Extraction parameters

    Returns:
        Extraction response with triples

    Raises:
        HTTPException: If extraction fails or service unavailable
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    start_time = time.perf_counter()

    logger.info(
        "hipporag_extract_triples_request",
        request_id=request_id,
        text_length=len(extract_request.text),
    )

    try:
        # Get HippoRAG destination
        destination = await _get_hipporag_destination_async()

        # Check if LLM provider is available
        if not destination._llm_provider:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": {
                        "code": "LLM_UNAVAILABLE",
                        "message": "LLM provider not available for triple extraction",
                    }
                },
            )

        # Execute triple extraction
        triples_tuples = await destination._llm_provider.extract_triples(
            extract_request.text
        )

        # Convert to API models
        triples = [
            HippoRAGTriple(
                subject=subject,
                predicate=predicate,
                object=obj,
            )
            for subject, predicate, obj in triples_tuples
        ]

        # Calculate query time
        query_time_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "hipporag_extract_triples_completed",
            request_id=request_id,
            triple_count=len(triples),
            query_time_ms=round(query_time_ms, 2),
        )

        return HippoRAGExtractTriplesResponse(
            triples=triples,
        )

    except HTTPException:
        raise
    except Exception as e:
        query_time_ms = (time.perf_counter() - start_time) * 1000

        logger.error(
            "hipporag_extract_triples_failed",
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
                    "message": f"Failed to extract triples: {e!s}",
                }
            },
        ) from e
