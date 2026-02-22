"""Re-ranking Service using cross-encoders for improved RAG relevance.

This module provides a ReRanker class that uses cross-encoder models to
re-rank retrieved chunks for significantly improved relevance. Cross-encoders
encode query and document together, capturing finer relevance signals than
bi-encoders used in initial retrieval.

Example:
    >>> from src.rag.strategies.reranking import ReRanker
    >>> reranker = ReRanker()
    >>> ranked = await reranker.rerank(
    ...     query="What is vibe coding?",
    ...     chunks=[chunk1, chunk2, chunk3],
    ...     top_k=5
    ... )
    >>> for item in ranked:
    ...     print(f"Rank {item.rank}: {item.score:.3f} - {item.content[:50]}...")
"""

import asyncio
import logging
import time
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from src.config import get_settings
from src.rag.models import RankedChunk

logger = logging.getLogger(__name__)

# Available cross-encoder models
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
HIGH_PRECISION_MODEL = "cross-encoder/ms-marco-electra-base"
FAST_MODEL = "cross-encoder/ms-marco-TinyBERT-L-2"
ALTERNATIVE_MODEL = "BAAI/bge-reranker-base"

SUPPORTED_MODELS = {
    "default": DEFAULT_MODEL,
    "high_precision": HIGH_PRECISION_MODEL,
    "fast": FAST_MODEL,
    "alternative": ALTERNATIVE_MODEL,
}


class Chunk(BaseModel):
    """Input chunk for re-ranking.
    
    This is a simplified chunk representation used as input to the re-ranker.
    It contains the essential information needed for cross-encoder scoring.
    
    Attributes:
        chunk_id: Unique identifier of the chunk
        content: Text content of the chunk
        metadata: Optional metadata about the chunk
    """
    
    chunk_id: str = Field(description="Unique identifier of the chunk")
    content: str = Field(description="Text content of the chunk")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata about the chunk"
    )


class ReRankingError(Exception):
    """Base exception for re-ranking errors."""
    pass


class ReRankingTimeoutError(ReRankingError):
    """Exception raised when re-ranking times out."""
    pass


class ReRankingModelError(ReRankingError):
    """Exception raised when the model fails to load or predict."""
    pass


class ReRanker:
    """Re-ranking service using cross-encoder models.
    
    This class uses cross-encoder models from the sentence-transformers library
    to score and re-rank retrieved chunks by relevance to a query. Cross-encoders
    encode query and document together, capturing finer relevance signals than
    bi-encoders used in initial vector retrieval.
    
    Supported models:
        - cross-encoder/ms-marco-MiniLM-L-6-v2 (default): Balanced speed/quality
        - cross-encoder/ms-marco-electra-base: Higher precision, slower
        - cross-encoder/ms-marco-TinyBERT-L-2: Fast, lower precision
        - BAAI/bge-reranker-base: Alternative architecture
    
    Attributes:
        model_name: Name of the cross-encoder model to use
        model: The loaded cross-encoder model instance
        batch_size: Batch size for processing
        timeout_ms: Timeout for re-ranking operations in milliseconds
    
    Example:
        >>> reranker = ReRanker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        >>> ranked = await reranker.rerank(
        ...     query="What is vibe coding?",
        ...     chunks=retrieved_chunks,
        ...     top_k=5
        ... )
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 8,
        timeout_ms: int = 500,
        device: str | None = None,
    ):
        """Initialize the re-ranker with a cross-encoder model.
        
        Args:
            model_name: Name of the cross-encoder model to load.
                Can be a full HuggingFace model name or a preset key
                ("default", "high_precision", "fast", "alternative").
            batch_size: Number of query-document pairs to process at once.
                Larger batches are more efficient but use more memory.
            timeout_ms: Maximum time allowed for re-ranking in milliseconds.
            device: Device to run the model on ("cpu", "cuda", "cuda:0", etc.).
                If None, automatically selects the best available device.
        
        Raises:
            ReRankingModelError: If the model fails to load.
        """
        # Resolve preset model names
        self.model_name = SUPPORTED_MODELS.get(model_name, model_name)
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms
        self.device = device
        
        # Model will be loaded lazily on first use
        self._model: Any | None = None
        self._model_loading = False
        
        logger.info(
            f"Initialized ReRanker with model: {self.model_name}, "
            f"batch_size: {batch_size}, timeout: {timeout_ms}ms"
        )
    
    def _load_model(self) -> Any:
        """Load the cross-encoder model.
        
        Returns:
            Loaded CrossEncoder model instance.
        
        Raises:
            ReRankingModelError: If the model fails to load.
        """
        if self._model is not None:
            return self._model
        
        if self._model_loading:
            # Another thread is loading the model, wait briefly
            import time as time_module
            for _ in range(50):  # Wait up to 5 seconds
                if self._model is not None:
                    return self._model  # type: ignore[unreachable]
                time_module.sleep(0.1)
            raise ReRankingModelError("Timeout waiting for model to load")
        
        self._model_loading = True
        
        try:
            # Import here to avoid dependency issues if sentence-transformers
            # is not installed
            try:
                from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]
            except ImportError as e:
                raise ReRankingModelError(
                    "sentence-transformers is required for re-ranking. "
                    "Install with: pip install sentence-transformers"
                ) from e
            
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            start_time = time.time()
            
            # Load the model
            model_kwargs = {}
            if self.device:
                model_kwargs["device"] = self.device
            
            self._model = CrossEncoder(self.model_name, **model_kwargs)
            
            load_time = (time.time() - start_time) * 1000
            logger.info(f"Model loaded in {load_time:.2f}ms")
            
            return self._model
            
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")
            raise ReRankingModelError(f"Failed to load model: {e}") from e
        finally:
            self._model_loading = False
    
    def _ensure_model_loaded(self) -> Any:
        """Ensure the model is loaded, loading it if necessary.
        
        Returns:
            Loaded CrossEncoder model instance.
        """
        if self._model is None:
            return self._load_model()
        return self._model
    
    def _normalize_score(self, score: float) -> float:
        """Normalize a raw score to 0-1 range.
        
        Different models output scores in different ranges. This method
        normalizes them to a consistent 0-1 range.
        
        Args:
            score: Raw score from the model.
        
        Returns:
            Normalized score in [0, 1] range.
        """
        # MS-MARCO models typically output logits that can be negative
        # or positive. Use sigmoid-like normalization.
        import math
        
        # If score is already in [0, 1], return as-is
        if 0.0 <= score <= 1.0:
            return float(score)
        
        # Apply sigmoid normalization for unbounded scores
        # This maps (-inf, +inf) to (0, 1)
        try:
            normalized = 1.0 / (1.0 + math.exp(-score))
            return float(normalized)
        except OverflowError:
            # Handle extreme values
            return 1.0 if score > 0 else 0.0
    
    def _score_pairs_sync(
        self,
        query: str,
        chunks: Sequence[Chunk],
    ) -> list[float]:
        """Score query-chunk pairs synchronously.
        
        This is the synchronous version that actually calls the model.
        It should be run in a thread pool for async compatibility.
        
        Args:
            query: User query string.
            chunks: List of chunks to score.
        
        Returns:
            List of scores for each chunk.
        """
        model = self._ensure_model_loaded()
        
        # Create query-document pairs
        pairs = [(query, chunk.content) for chunk in chunks]
        
        if not pairs:
            return []
        
        # Score in batches for memory efficiency
        all_scores: list[float] = []
        
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            batch_scores = model.predict(batch)
            
            # Normalize scores to [0, 1]
            for score in batch_scores:
                all_scores.append(self._normalize_score(float(score)))
        
        return all_scores
    
    async def rerank(
        self,
        query: str,
        chunks: Sequence[Chunk],
        top_k: int = 5,
    ) -> list[RankedChunk]:
        """Re-rank chunks by relevance to query using cross-encoder.
        
        This method scores each chunk using the cross-encoder model and
        returns them sorted by relevance score in descending order.
        
        Args:
            query: User query string.
            chunks: Initial retrieval results to re-rank.
            top_k: Number of top results to return.
        
        Returns:
            List of RankedChunk objects sorted by score (highest first).
            Each RankedChunk contains the original chunk, relevance score,
            and rank position.
        
        Raises:
            ReRankingTimeoutError: If re-ranking exceeds the timeout.
            ReRankingError: If re-ranking fails for other reasons.
        
        Example:
            >>> chunks = [
            ...     Chunk(chunk_id="1", content="Python is a programming language..."),
            ...     Chunk(chunk_id="2", content="JavaScript runs in browsers..."),
            ... ]
            >>> ranked = await reranker.rerank("What is Python?", chunks, top_k=2)
            >>> print(ranked[0].score)  # Highest score
            0.95
        """
        start_time = time.time()
        
        # Handle edge cases
        if not query or not query.strip():
            logger.warning("Empty query provided for re-ranking")
            return []
        
        if not chunks:
            logger.debug("No chunks provided for re-ranking")
            return []
        
        if top_k <= 0:
            logger.warning(f"Invalid top_k value: {top_k}, using default of 5")
            top_k = 5
        
        try:
            # Run the scoring in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            # Create a task for the scoring with timeout
            scoring_task = loop.run_in_executor(
                None,  # Default executor
                self._score_pairs_sync,
                query,
                chunks
            )
            
            # Wait for completion with timeout
            timeout_seconds = self.timeout_ms / 1000.0
            scores = await asyncio.wait_for(scoring_task, timeout=timeout_seconds)
            
            # Create ranked chunks with initial data
            ranked_data: list[dict[str, Any]] = [
                {
                    "chunk_id": str(chunk.chunk_id),
                    "content": chunk.content,
                    "score": float(score),
                    "metadata": dict(chunk.metadata),
                }
                for chunk, score in zip(chunks, scores)
            ]
            
            # Sort by score descending
            ranked_data.sort(key=lambda x: float(x["score"]), reverse=True)
            
            # Create RankedChunk objects with proper ranks (1-indexed)
            ranked: list[RankedChunk] = []
            for i, data in enumerate(ranked_data):
                ranked.append(
                    RankedChunk(
                        chunk_id=str(data["chunk_id"]),
                        content=str(data["content"]),
                        score=float(data["score"]),
                        rank=i + 1,  # 1-indexed rank
                        metadata=dict(data["metadata"]),
                    )
                )
            
            # Take top_k
            result = ranked[:top_k]
            
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(
                "Re-ranking completed",
                extra={
                    "query_length": len(query),
                    "num_chunks": len(chunks),
                    "top_k": top_k,
                    "elapsed_ms": elapsed_ms,
                    "top_score": result[0].score if result else None,
                }
            )
            
            return result
            
        except TimeoutError as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(
                f"Re-ranking timed out after {elapsed_ms:.2f}ms (limit: {self.timeout_ms}ms)"
            )
            raise ReRankingTimeoutError(
                f"Re-ranking timed out after {elapsed_ms:.2f}ms"
            ) from e
            
        except ReRankingModelError:
            raise
            
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(f"Re-ranking failed after {elapsed_ms:.2f}ms: {e}")
            raise ReRankingError(f"Re-ranking failed: {e}") from e
    
    async def rerank_batch(
        self,
        queries: Sequence[str],
        chunks_per_query: Sequence[Sequence[Chunk]],
        top_k: int = 5,
    ) -> list[list[RankedChunk]]:
        """Re-rank chunks for multiple queries in batch.
        
        This method efficiently processes multiple queries, reusing the
        loaded model across all re-ranking operations.
        
        Args:
            queries: List of user query strings.
            chunks_per_query: List of chunk lists, one per query.
            top_k: Number of top results to return per query.
        
        Returns:
            List of RankedChunk lists, one per query.
        
        Example:
            >>> queries = ["What is Python?", "What is JavaScript?"]
            >>> chunks = [[python_chunks], [js_chunks]]
            >>> results = await reranker.rerank_batch(queries, chunks, top_k=3)
        """
        if len(queries) != len(chunks_per_query):
            raise ValueError(
                f"Number of queries ({len(queries)}) must match "
                f"number of chunk lists ({len(chunks_per_query)})"
            )
        
        # Ensure model is loaded once for all operations
        self._ensure_model_loaded()
        
        results: list[list[RankedChunk]] = []
        
        for query, chunks in zip(queries, chunks_per_query):
            try:
                ranked = await self.rerank(query, chunks, top_k)
                results.append(ranked)
            except ReRankingError as e:
                logger.warning(f"Failed to re-rank query '{query[:50]}...': {e}")
                # Return empty list for failed queries
                results.append([])
        
        return results
    
    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Dictionary with model information.
        """
        info: dict[str, Any] = {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "timeout_ms": self.timeout_ms,
            "device": self.device,
            "loaded": self._model is not None,
        }
        
        if self._model is not None:
            try:
                info["max_length"] = getattr(self._model, "max_length", None)
                info["num_labels"] = getattr(self._model, "num_labels", None)
            except Exception:
                pass
        
        return info
    
    async def health_check(self) -> dict[str, Any]:
        """Check the health of the re-ranker.
        
        Returns:
            Dictionary with health status information.
        """
        result: dict[str, Any] = {
            "healthy": False,
            "components": {},
        }
        
        # Check if sentence-transformers is available
        try:
            from sentence_transformers import CrossEncoder
            result["components"]["sentence_transformers"] = {
                "healthy": True,
                "available": True,
            }
        except ImportError:
            result["components"]["sentence_transformers"] = {
                "healthy": False,
                "available": False,
                "error": "sentence-transformers not installed",
            }
            return result
        
        # Check if model can be loaded
        try:
            model = self._ensure_model_loaded()
            result["components"]["model"] = {
                "healthy": True,
                "loaded": True,
                "name": self.model_name,
            }
            result["healthy"] = True
        except Exception as e:
            result["components"]["model"] = {
                "healthy": False,
                "loaded": False,
                "error": str(e),
            }
        
        return result
    
    @classmethod
    def from_settings(
        cls,
        settings: Any | None = None,
        model_preset: str = "default",
    ) -> "ReRanker":
        """Create a ReRanker instance from application settings.
        
        Args:
            settings: Application settings. If None, loads from get_settings().
            model_preset: Model preset to use ("default", "high_precision", "fast", "alternative").
        
        Returns:
            Configured ReRanker instance.
        """
        if settings is None:
            settings = get_settings()
        
        reranking_settings = getattr(settings, "reranking", None)
        
        if reranking_settings:
            # Get model name from preset or settings
            model_name = SUPPORTED_MODELS.get(model_preset, model_preset)
            
            # Check if there's a model override in settings
            if hasattr(reranking_settings, "models"):
                models_config = reranking_settings.models
                if isinstance(models_config, dict) and model_preset in models_config:
                    model_name = models_config[model_preset]
            
            batch_size = getattr(reranking_settings, "batch_size", 8)
            timeout_ms = getattr(reranking_settings, "timeout_ms", 500)
        else:
            # Use defaults if no settings available
            model_name = SUPPORTED_MODELS.get(model_preset, DEFAULT_MODEL)
            batch_size = 8
            timeout_ms = 500
        
        return cls(
            model_name=model_name,
            batch_size=batch_size,
            timeout_ms=timeout_ms,
        )
