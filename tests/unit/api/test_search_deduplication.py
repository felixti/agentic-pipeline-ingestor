from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.api.routes.search import (
    _deduplicate_results_by_hash,
    _merge_rerank_results,
    _to_reranker_chunks,
)
from src.rag.models import RankedChunk
from src.services.hybrid_search_service import HybridSearchResult
from src.services.text_search_service import TextSearchResult
from src.services.vector_search_service import SearchResult


@pytest.mark.unit
def test_deduplicate_vector_results_keeps_highest_score() -> None:
    results = [
        SearchResult(
            chunk=SimpleNamespace(content_hash="same", id="a", job_id="j1", chunk_index=0),
            similarity_score=0.80,
            rank=1,
        ),
        SearchResult(
            chunk=SimpleNamespace(content_hash="same", id="b", job_id="j2", chunk_index=1),
            similarity_score=0.95,
            rank=2,
        ),
    ]

    deduped = _deduplicate_results_by_hash(results)

    assert len(deduped) == 1
    assert deduped[0].chunk.id == "b"
    assert deduped[0].rank == 1


@pytest.mark.unit
def test_deduplicate_text_and_hybrid_results_respects_content_hash() -> None:
    text_results = [
        TextSearchResult(
            chunk=SimpleNamespace(content_hash="t", id="t1", job_id="j1", chunk_index=0),
            rank_score=0.6,
            rank=1,
        ),
        TextSearchResult(
            chunk=SimpleNamespace(content_hash="t", id="t2", job_id="j2", chunk_index=1),
            rank_score=0.4,
            rank=2,
        ),
    ]
    hybrid_results = [
        HybridSearchResult(
            chunk=SimpleNamespace(content_hash="h", id="h1", job_id="j1", chunk_index=0),
            hybrid_score=0.7,
            rank=1,
        ),
        HybridSearchResult(
            chunk=SimpleNamespace(content_hash="h", id="h2", job_id="j2", chunk_index=1),
            hybrid_score=0.9,
            rank=2,
        ),
    ]

    deduped_text = _deduplicate_results_by_hash(text_results)
    deduped_hybrid = _deduplicate_results_by_hash(hybrid_results)

    assert len(deduped_text) == 1
    assert deduped_text[0].chunk.id == "t1"
    assert deduped_text[0].rank == 1
    assert len(deduped_hybrid) == 1
    assert deduped_hybrid[0].chunk.id == "h2"
    assert deduped_hybrid[0].rank == 1


@pytest.mark.unit
def test_deduplicate_keeps_results_without_content_hash() -> None:
    results = [
        SearchResult(
            chunk=SimpleNamespace(content_hash=None, id="a", job_id="j1", chunk_index=0),
            similarity_score=0.9,
            rank=1,
        ),
        SearchResult(
            chunk=SimpleNamespace(content_hash=None, id="b", job_id="j2", chunk_index=1),
            similarity_score=0.8,
            rank=2,
        ),
    ]

    deduped = _deduplicate_results_by_hash(results)

    assert len(deduped) == 2
    assert [r.rank for r in deduped] == [1, 2]


@pytest.mark.unit
def test_to_reranker_chunks_uses_chunk_metadata_or_metadata() -> None:
    first_id = uuid4()
    second_id = uuid4()

    results = [
        SearchResult(
            chunk=SimpleNamespace(
                id=first_id,
                content="chunk one",
                chunk_metadata={"page": 1},
                metadata={"ignored": True},
            ),
            similarity_score=0.9,
            rank=1,
        ),
        HybridSearchResult(
            chunk=SimpleNamespace(
                id=second_id,
                content="chunk two",
                chunk_metadata=None,
                metadata={"source": "doc"},
            ),
            hybrid_score=0.8,
            rank=2,
        ),
    ]

    chunks = _to_reranker_chunks(results)

    assert len(chunks) == 2
    assert chunks[0].chunk_id == str(first_id)
    assert chunks[0].metadata == {"page": 1}
    assert chunks[1].chunk_id == str(second_id)
    assert chunks[1].metadata == {"source": "doc"}


@pytest.mark.unit
def test_merge_rerank_results_applies_order_rank_and_score() -> None:
    first_id = uuid4()
    second_id = uuid4()

    candidate_results = [
        SearchResult(
            chunk=SimpleNamespace(id=first_id, content_hash="a", job_id="j1", chunk_index=0),
            similarity_score=0.95,
            rank=1,
        ),
        SearchResult(
            chunk=SimpleNamespace(id=second_id, content_hash="b", job_id="j2", chunk_index=1),
            similarity_score=0.9,
            rank=2,
        ),
    ]

    reranked = [
        RankedChunk(
            chunk_id=str(second_id),
            content="chunk two",
            score=0.99,
            rank=1,
            metadata={},
        ),
        RankedChunk(
            chunk_id=str(first_id),
            content="chunk one",
            score=0.93,
            rank=2,
            metadata={},
        ),
    ]

    merged = _merge_rerank_results(candidate_results, reranked)

    assert [str(item.chunk.id) for item in merged] == [str(second_id), str(first_id)]
    assert merged[0].rank == 1
    assert merged[1].rank == 2
    assert getattr(merged[0], "rerank_score") == 0.99
    assert getattr(merged[1], "rerank_score") == 0.93
