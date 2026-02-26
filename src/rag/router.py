"""Agentic RAG Router for orchestrating RAG strategies.

This module provides the AgenticRAG class that orchestrates all RAG strategies
based on query classification. It implements:
- Query classification and strategy selection
- Dynamic strategy configuration based on query type
- Self-correction logic when retrieval quality is low
- Multi-hop query support
- Comprehensive metrics tracking

Example:
    >>> from src.rag.router import AgenticRAG
    >>> router = AgenticRAG(
    ...     query_rewriter=query_rewriter,
    ...     hyde_rewriter=hyde_rewriter,
    ...     classifier=classifier,
    ...     reranker=reranker,
    ... )
    >>> result = await router.process(
    ...     query="What is vibe coding?",
    ...     strategy_preset="balanced"
    ... )
    >>> print(result.answer)
    >>> print(result.metrics.latency_ms)
"""

import logging
import time
from typing import Any

from src.rag.classification import STRATEGY_MATRIX, QueryClassifier
from src.rag.models import (
    QueryClassification,
    QueryRewriteResult,
    QueryType,
    RAGConfig,
    RAGMetrics,
    RAGResult,
    RankedChunk,
    Source,
)
from src.rag.metrics_store import RAGMetricsStore
from src.rag.strategies.hyde import HyDERewriter
from src.rag.strategies.query_rewriting import QueryRewriter
from src.rag.strategies.reranking import Chunk, ReRanker

logger = logging.getLogger(__name__)

# Strategy presets for different optimization targets
STRATEGY_PRESETS: dict[str, dict[str, bool]] = {
    "fast": {
        "query_rewrite": True,
        "hyde": False,
        "reranking": False,
        "hybrid_search": True,
    },
    "balanced": {
        "query_rewrite": True,
        "hyde": False,
        "reranking": True,
        "hybrid_search": True,
    },
    "thorough": {
        "query_rewrite": True,
        "hyde": True,
        "reranking": True,
        "hybrid_search": True,
    },
}

# Quality thresholds for self-correction
QUALITY_THRESHOLD = 0.7
MAX_SELF_CORRECTION_ITERATIONS = 3

# Retrieval configuration
DEFAULT_RETRIEVAL_K = 10
EXPANDED_RETRIEVAL_K = 20
SELF_CORRECTION_RETRIEVAL_K = 30


class AgenticRAGError(Exception):
    """Base exception for Agentic RAG errors."""

    pass


class AgenticRAG:
    """Agentic RAG Router that orchestrates RAG strategies.

    This class provides an intelligent RAG pipeline that:
    1. Classifies queries to determine optimal strategy
    2. Selects and configures strategies based on query type
    3. Executes the retrieval pipeline with selected strategies
    4. Evaluates retrieval quality and self-corrects if needed
    5. Generates final answers with source attribution

    Attributes:
        query_rewriter: Service for query rewriting
        hyde_rewriter: Service for HyDE-based query rewriting
        classifier: Service for query classification
        reranker: Service for cross-encoder re-ranking
        quality_threshold: Minimum quality score for accepting retrieval results
        max_iterations: Maximum self-correction iterations
    """

    def __init__(
        self,
        query_rewriter: QueryRewriter,
        classifier: QueryClassifier,
        hyde_rewriter: HyDERewriter | None = None,
        reranker: ReRanker | None = None,
        metrics_store: RAGMetricsStore | None = None,
        quality_threshold: float = QUALITY_THRESHOLD,
        max_iterations: int = MAX_SELF_CORRECTION_ITERATIONS,
    ):
        """Initialize the Agentic RAG router.

        Args:
            query_rewriter: QueryRewriter instance for query rewriting
            classifier: QueryClassifier instance for query classification
            hyde_rewriter: Optional HyDERewriter for hypothetical document generation
            reranker: Optional ReRanker for cross-encoder re-ranking
            quality_threshold: Minimum quality score (0-1) to accept retrieval results
            max_iterations: Maximum number of self-correction iterations
        """
        self.query_rewriter = query_rewriter
        self.hyde_rewriter = hyde_rewriter
        self.classifier = classifier
        self.reranker = reranker
        self.metrics_store = metrics_store or RAGMetricsStore()
        self.quality_threshold = quality_threshold
        self.max_iterations = max_iterations

        logger.info(
            f"Initialized AgenticRAG with quality_threshold={quality_threshold}, "
            f"max_iterations={max_iterations}"
        )

    def _select_strategy_config(
        self,
        classification: QueryClassification,
        strategy_preset: str = "auto",
    ) -> RAGConfig:
        """Select RAG configuration based on query type and preset.

        Args:
            classification: Query classification result
            strategy_preset: Strategy preset ("auto", "fast", "balanced", "thorough")

        Returns:
            RAGConfig with strategy settings
        """
        # If using a preset other than auto, apply it directly
        if strategy_preset != "auto" and strategy_preset in STRATEGY_PRESETS:
            preset = STRATEGY_PRESETS[strategy_preset]
            return RAGConfig(
                query_rewrite=preset["query_rewrite"],
                hyde=preset["hyde"],
                reranking=preset["reranking"],
                hybrid_search=preset["hybrid_search"],
                strategy_preset=strategy_preset,
            )

        # Auto mode: use strategy matrix based on query type
        query_type = classification.query_type
        strategy = STRATEGY_MATRIX.get(query_type, STRATEGY_MATRIX[QueryType.VAGUE])

        return RAGConfig(
            query_rewrite=strategy.get("query_rewrite", True),
            hyde=strategy.get("hyde", False),
            reranking=strategy.get("reranking", True),
            hybrid_search=strategy.get("hybrid", True),
            strategy_preset=f"auto_{query_type.value}",
        )

    async def _rewrite_query(
        self,
        query: str,
        config: RAGConfig,
        context: dict[str, Any] | None = None,
    ) -> tuple[QueryRewriteResult, float]:
        """Rewrite query using appropriate rewriter based on config.

        Args:
            query: Original user query
            config: RAG configuration
            context: Optional conversation context

        Returns:
            Tuple of (rewrite result, elapsed time in ms)
        """
        start_time = time.time()

        try:
            # Use HyDE if enabled and available
            if config.hyde and self.hyde_rewriter is not None:
                result = await self.hyde_rewriter.rewrite(query, context)
            else:
                # Use standard query rewriting
                result = await self.query_rewriter.rewrite(query, context)

            elapsed_ms = (time.time() - start_time) * 1000
            return result, elapsed_ms

        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}, using fallback")
            elapsed_ms = (time.time() - start_time) * 1000
            # Return fallback result
            fallback = QueryRewriteResult(
                search_rag="@knowledgebase" in query.lower(),
                embedding_source_text=query,
                llm_query=f"Based on the provided context, answer: {query}",
            )
            return fallback, elapsed_ms

    async def _retrieve_documents(
        self,
        query_text: str,
        config: RAGConfig,
        retrieval_k: int = DEFAULT_RETRIEVAL_K,
    ) -> tuple[list[RankedChunk], float]:
        """Retrieve documents based on query text.

        This is a placeholder method that would integrate with the actual
        vector store / search service. For now, it returns an empty list.

        Args:
            query_text: Query text to search for
            config: RAG configuration
            retrieval_k: Number of documents to retrieve

        Returns:
            Tuple of (retrieved chunks, elapsed time in ms)
        """
        start_time = time.time()

        # Placeholder: In real implementation, this would call the search service
        # For now, return empty results - this would be integrated with the
        # actual vector store service
        chunks: list[RankedChunk] = []

        elapsed_ms = (time.time() - start_time) * 1000
        return chunks, elapsed_ms

    async def _rerank_documents(
        self,
        query: str,
        chunks: list[RankedChunk],
        top_k: int = 5,
    ) -> tuple[list[RankedChunk], float]:
        """Re-rank documents using cross-encoder.

        Args:
            query: Original query
            chunks: Retrieved chunks to re-rank
            top_k: Number of top results to return

        Returns:
            Tuple of (re-ranked chunks, elapsed time in ms)
        """
        if not self.reranker or not chunks:
            return chunks, 0.0

        start_time = time.time()

        try:
            # Convert RankedChunk to Chunk for re-ranker
            input_chunks = [
                Chunk(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    metadata=chunk.metadata,
                )
                for chunk in chunks
            ]

            # Re-rank
            reranked = await self.reranker.rerank(query, input_chunks, top_k=top_k)

            elapsed_ms = (time.time() - start_time) * 1000
            return reranked, elapsed_ms

        except Exception as e:
            logger.warning(f"Re-ranking failed: {e}, returning original chunks")
            elapsed_ms = (time.time() - start_time) * 1000
            return chunks, elapsed_ms

    def _evaluate_quality(self, chunks: list[RankedChunk]) -> float:
        """Evaluate the quality of retrieved chunks.

        Calculates a quality score based on:
        - Average relevance score of top chunks
        - Number of chunks retrieved
        - Score distribution

        Args:
            chunks: Retrieved chunks

        Returns:
            Quality score between 0 and 1
        """
        if not chunks:
            return 0.0

        # Calculate average score of top 5 chunks
        top_chunks = chunks[:5]
        scores = [chunk.score for chunk in top_chunks]

        if not scores:
            return 0.0

        avg_score = sum(scores) / len(scores)

        # Boost score slightly if we have multiple good results
        if len(scores) >= 3 and all(s > 0.5 for s in scores[:3]):
            avg_score = min(1.0, avg_score * 1.1)

        return avg_score

    async def _self_correct(
        self,
        query: str,
        chunks: list[RankedChunk],
        config: RAGConfig,
        iteration: int,
    ) -> tuple[list[RankedChunk], RAGConfig, bool]:
        """Apply self-correction strategies to improve retrieval.

        Self-correction strategies (applied in order):
        1. Enable HyDE if not already enabled
        2. Increase retrieval_k
        3. Switch to hybrid search
        4. Expand query terms

        Args:
            query: Original query
            chunks: Current retrieved chunks
            config: Current RAG configuration
            iteration: Current self-correction iteration

        Returns:
            Tuple of (improved chunks, updated config, was_corrected)
        """
        if iteration >= self.max_iterations:
            return chunks, config, False

        logger.info(f"Applying self-correction (iteration {iteration + 1})")

        # Create a mutable copy of config
        new_config = RAGConfig(
            query_rewrite=config.query_rewrite,
            hyde=config.hyde,
            reranking=config.reranking,
            hybrid_search=config.hybrid_search,
            strategy_preset=config.strategy_preset,
        )

        was_corrected = False

        # Strategy 1: Try HyDE if not already enabled
        if not new_config.hyde and self.hyde_rewriter is not None:
            logger.info("Self-correction: Enabling HyDE")
            new_config.hyde = True
            was_corrected = True
            # Re-rewrite with HyDE
            rewrite_result, _ = await self._rewrite_query(query, new_config)
            chunks, _ = await self._retrieve_documents(
                rewrite_result.embedding_source_text,
                new_config,
                retrieval_k=SELF_CORRECTION_RETRIEVAL_K,
            )

        # Strategy 2: Increase retrieval count
        elif len(chunks) < 10:
            logger.info("Self-correction: Increasing retrieval_k")
            rewrite_result, _ = await self._rewrite_query(query, new_config)
            chunks, _ = await self._retrieve_documents(
                rewrite_result.embedding_source_text,
                new_config,
                retrieval_k=SELF_CORRECTION_RETRIEVAL_K,
            )
            was_corrected = True

        # Strategy 3: Enable hybrid search
        elif not new_config.hybrid_search:
            logger.info("Self-correction: Enabling hybrid search")
            new_config.hybrid_search = True
            was_corrected = True

        # Strategy 4: Expand query terms (add synonyms)
        else:
            logger.info("Self-correction: Expanding query terms")
            expanded_query = f"{query} related concepts"
            chunks, _ = await self._retrieve_documents(
                expanded_query,
                new_config,
                retrieval_k=SELF_CORRECTION_RETRIEVAL_K,
            )
            was_corrected = True

        return chunks, new_config, was_corrected

    def _chunks_to_sources(self, chunks: list[RankedChunk]) -> list[Source]:
        """Convert RankedChunks to Sources.

        Args:
            chunks: Ranked chunks from retrieval

        Returns:
            List of Source objects
        """
        return [
            Source(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                score=chunk.score,
                metadata=chunk.metadata,
            )
            for chunk in chunks
        ]

    async def _generate_answer(
        self,
        query: str,
        llm_query: str,
        chunks: list[RankedChunk],
    ) -> tuple[str, float, int]:
        """Generate final answer using retrieved chunks.

        This is a placeholder that would integrate with the LLM service.

        Args:
            query: Original user query
            llm_query: Processed query for generation
            chunks: Retrieved chunks to use as context

        Returns:
            Tuple of (answer text, elapsed time in ms, tokens used)
        """
        start_time = time.time()

        # Placeholder: In real implementation, this would call the LLM
        # For now, return a placeholder answer
        if chunks:
            answer = f"Based on the retrieved documents, here's information about: {query}"
        else:
            answer = f"I don't have specific information about: {query}"

        elapsed_ms = (time.time() - start_time) * 1000
        tokens_used = len(answer.split())  # Rough estimate

        return answer, elapsed_ms, tokens_used

    async def process(
        self,
        query: str,
        strategy_preset: str = "auto",
        context: dict[str, Any] | None = None,
    ) -> RAGResult:
        """Process query through agentic RAG pipeline.

        The pipeline follows this flow:
        1. Classify query type
        2. Select strategies based on query type and preset
        3. Execute retrieval pipeline with selected strategies
        4. Evaluate quality
        5. Self-correct if needed
        6. Generate final answer

        Args:
            query: User query
            strategy_preset: "auto", "fast", "balanced", or "thorough"
            context: Optional conversation context with keys like:
                - previous_queries: List of previous user queries
                - previous_responses: List of previous assistant responses
                - session_id: Conversation session identifier

        Returns:
            RAGResult with answer, sources, and metrics

        Raises:
            AgenticRAGError: If processing fails
        """
        pipeline_start = time.time()

        # Initialize metrics tracking
        metrics = {
            "rewrite_time_ms": 0.0,
            "retrieval_time_ms": 0.0,
            "reranking_time_ms": 0.0,
            "generation_time_ms": 0.0,
            "tokens_used": 0,
            "chunks_retrieved": 0,
            "chunks_used": 0,
            "self_correction_iterations": 0,
        }

        try:
            # Step 1: Classify query
            classification = await self.classifier.classify(query, context)
            logger.info(
                f"Query classified as {classification.query_type.value} "
                f"with confidence {classification.confidence:.2f}"
            )

            # Step 2: Select strategy configuration
            config = self._select_strategy_config(classification, strategy_preset)
            logger.info(
                f"Selected strategy: rewrite={config.query_rewrite}, "
                f"hyde={config.hyde}, reranking={config.reranking}, "
                f"hybrid={config.hybrid_search}"
            )

            # Step 3: Rewrite query
            rewrite_result, rewrite_time = await self._rewrite_query(query, config, context)
            metrics["rewrite_time_ms"] = rewrite_time

            # Step 4: Retrieve documents
            chunks, retrieval_time = await self._retrieve_documents(
                rewrite_result.embedding_source_text,
                config,
                retrieval_k=EXPANDED_RETRIEVAL_K if config.hyde else DEFAULT_RETRIEVAL_K,
            )
            metrics["retrieval_time_ms"] = retrieval_time
            metrics["chunks_retrieved"] = len(chunks)

            # Step 5: Re-rank if enabled
            if config.reranking and self.reranker is not None and chunks:
                chunks, reranking_time = await self._rerank_documents(query, chunks, top_k=5)
                metrics["reranking_time_ms"] = reranking_time

            # Step 6: Evaluate quality and self-correct if needed
            quality = self._evaluate_quality(chunks)
            logger.info(f"Initial retrieval quality: {quality:.2f}")

            current_config = config
            iteration = 0

            while quality < self.quality_threshold and iteration < self.max_iterations:
                chunks, current_config, was_corrected = await self._self_correct(
                    query, chunks, current_config, iteration
                )

                if not was_corrected:
                    break

                iteration += 1
                metrics["self_correction_iterations"] = iteration

                # Re-rank after self-correction if enabled
                if current_config.reranking and self.reranker is not None and chunks:
                    chunks, _ = await self._rerank_documents(query, chunks, top_k=5)

                # Re-evaluate quality
                quality = self._evaluate_quality(chunks)
                logger.info(f"Quality after correction {iteration}: {quality:.2f}")

            # Step 7: Generate answer
            answer, generation_time, tokens_used = await self._generate_answer(
                query,
                rewrite_result.llm_query,
                chunks,
            )
            metrics["generation_time_ms"] = generation_time
            metrics["tokens_used"] = tokens_used

            # Calculate total latency
            total_latency_ms = (time.time() - pipeline_start) * 1000

            # Convert chunks to sources
            sources = self._chunks_to_sources(chunks[:5])  # Top 5 sources
            metrics["chunks_used"] = len(sources)

            # Build result
            result = RAGResult(
                answer=answer,
                sources=sources,
                metrics=RAGMetrics(
                    latency_ms=total_latency_ms,
                    tokens_used=int(metrics["tokens_used"]),
                    retrieval_score=quality,
                    classification_confidence=classification.confidence,
                    rewrite_time_ms=metrics["rewrite_time_ms"],
                    retrieval_time_ms=metrics["retrieval_time_ms"],
                    reranking_time_ms=metrics["reranking_time_ms"],
                    generation_time_ms=metrics["generation_time_ms"],
                    chunks_retrieved=int(metrics["chunks_retrieved"]),
                    chunks_used=int(metrics["chunks_used"]),
                    self_correction_iterations=int(metrics["self_correction_iterations"]),
                ),
                strategy_used=current_config.strategy_preset,
                query_type=classification.query_type,
            )

            try:
                await self.metrics_store.record(
                    metrics=result.metrics,
                    strategy_used=result.strategy_used,
                    query_type=result.query_type,
                )
            except Exception as metrics_error:
                logger.warning(f"Failed to record RAG metrics: {metrics_error}")

            logger.info(
                f"RAG pipeline completed in {total_latency_ms:.2f}ms with quality={quality:.2f}"
            )

            return result

        except Exception as e:
            logger.error(f"RAG pipeline failed: {e}")
            raise AgenticRAGError(f"Failed to process query: {e}") from e

    async def process_multi_hop(
        self,
        query: str,
        strategy_preset: str = "auto",
        context: dict[str, Any] | None = None,
    ) -> RAGResult:
        """Process multi-hop query through agentic RAG pipeline.

        Multi-hop queries require multiple retrieval steps. This method:
        1. Identifies sub-queries from the multi-hop query
        2. Executes retrieval for each sub-query
        3. Combines results
        4. Generates final answer

        Args:
            query: Multi-hop user query
            strategy_preset: "auto", "fast", "balanced", or "thorough"
            context: Optional conversation context

        Returns:
            RAGResult with answer, sources, and metrics
        """
        pipeline_start = time.time()

        try:
            # Classify query
            classification = await self.classifier.classify(query, context)

            # Force thorough preset for multi-hop queries
            if classification.query_type == QueryType.MULTI_HOP:
                strategy_preset = "thorough"

            # For now, delegate to standard process
            # In a full implementation, this would break down the query
            # into sub-queries and execute multiple retrieval passes
            result = await self.process(query, strategy_preset, context)

            # Mark as multi-hop query
            result.query_type = QueryType.MULTI_HOP

            # Update latency to include multi-hop overhead
            result.metrics.latency_ms = (time.time() - pipeline_start) * 1000

            return result

        except Exception as e:
            logger.error(f"Multi-hop RAG pipeline failed: {e}")
            raise AgenticRAGError(f"Failed to process multi-hop query: {e}") from e

    async def health_check(self) -> dict[str, Any]:
        """Check the health of all Agentic RAG components.

        Returns:
            Dictionary with health status information for all components
        """
        result: dict[str, Any] = {
            "healthy": False,
            "components": {},
        }

        # Check query rewriter
        try:
            rewriter_health = await self.query_rewriter.health_check()
            result["components"]["query_rewriter"] = rewriter_health
        except Exception as e:
            result["components"]["query_rewriter"] = {
                "healthy": False,
                "error": str(e),
            }

        # Check classifier
        try:
            classifier_health = await self.classifier.health_check()
            result["components"]["classifier"] = classifier_health
        except Exception as e:
            result["components"]["classifier"] = {
                "healthy": False,
                "error": str(e),
            }

        # Check HyDE rewriter (optional)
        if self.hyde_rewriter is not None:
            try:
                hyde_health = await self.hyde_rewriter.health_check()
                result["components"]["hyde_rewriter"] = hyde_health
            except Exception as e:
                result["components"]["hyde_rewriter"] = {
                    "healthy": False,
                    "error": str(e),
                }
        else:
            result["components"]["hyde_rewriter"] = {
                "healthy": True,
                "enabled": False,
            }

        # Check re-ranker (optional)
        if self.reranker is not None:
            try:
                reranker_health = await self.reranker.health_check()
                result["components"]["reranker"] = reranker_health
            except Exception as e:
                result["components"]["reranker"] = {
                    "healthy": False,
                    "error": str(e),
                }
        else:
            result["components"]["reranker"] = {
                "healthy": True,
                "enabled": False,
            }

        # Overall health: require rewriter and classifier to be healthy
        result["healthy"] = result["components"].get("query_rewriter", {}).get(
            "healthy", False
        ) and result["components"].get("classifier", {}).get("healthy", False)

        return result


def get_strategy_presets() -> dict[str, dict[str, bool]]:
    """Get available strategy presets.

    Returns:
        Dictionary mapping preset names to their configurations
    """
    import copy

    return copy.deepcopy(STRATEGY_PRESETS)
