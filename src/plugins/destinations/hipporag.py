"""HippoRAG destination plugin for multi-hop reasoning.

Uses neurobiological memory model with Personalized PageRank
for single-step multi-hop retrieval.

This module provides the HippoRAGDestination which uses the HippoRAG
Python library for graph-based document storage with:
- File-based storage (no separate database required)
- OpenIE for triple extraction from documents
- Knowledge graph construction from triples
- Personalized PageRank for multi-hop retrieval
- litellm for LLM integration

Example:
    >>> destination = HippoRAGDestination()
    >>> await destination.initialize({
    ...     "save_dir": "/data/hipporag",
    ...     "llm_model": "azure/gpt-4.1",
    ...     "embedding_model": "azure/text-embedding-3-small",
    ... })
    >>> conn = await destination.connect({})
    >>> result = await destination.write(conn, transformed_data)
    >>> retrieval_results = await destination.retrieve(
    ...     queries=["What company did Steve Jobs found?"],
    ...     num_to_retrieve=10
    ... )

Environment Variables:
    HIPPO_SAVE_DIR: Storage directory path (default: /data/hipporag)
    HIPPO_LLM_MODEL: LLM model for OpenIE and QA (default: azure/gpt-4.1)
    HIPPO_EMBEDDING_MODEL: Embedding model (default: azure/text-embedding-3-small)
    HIPPO_RETRIEVAL_K: Default retrieval count (default: 10)
    AZURE_OPENAI_API_BASE: Azure OpenAI endpoint
    AZURE_OPENAI_API_KEY: Azure OpenAI API key
    OPENROUTER_API_KEY: OpenRouter API key (fallback)
"""

from __future__ import annotations

import asyncio
import json
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import numpy as np

from src.observability.logging import get_logger
from src.plugins.base import (
    Connection,
    DestinationPlugin,
    HealthStatus,
    PluginMetadata,
    PluginType,
    TransformedData,
    ValidationResult,
    WriteResult,
)
from src.plugins.destinations.hipporag_llm import HippoRAGLLMProvider

logger = get_logger(__name__)

# Default configuration from environment variables
HIPPO_SAVE_DIR = os.getenv("HIPPO_SAVE_DIR", "/data/hipporag")
HIPPO_LLM_MODEL = os.getenv("HIPPO_LLM_MODEL", "azure/gpt-4.1")
HIPPO_EMBEDDING_MODEL = os.getenv("HIPPO_EMBEDDING_MODEL", "azure/text-embedding-3-small")
HIPPO_RETRIEVAL_K = int(os.getenv("HIPPO_RETRIEVAL_K", "10"))


@dataclass
class RetrievalResult:
    """Result from multi-hop retrieval.
    
    Attributes:
        query: The original query string
        passages: List of retrieved passage texts
        scores: Relevance scores for each passage
        source_documents: IDs of source documents for passages
        entities: Entities extracted from the query
    """
    query: str
    passages: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    source_documents: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)


@dataclass
class QAResult:
    """Result from RAG QA.
    
    Attributes:
        query: The original question
        answer: Generated answer text
        sources: Source passages used for the answer
        retrieval_results: Full retrieval result with metadata
        confidence: Confidence score for the answer
    """
    query: str
    answer: str = ""
    sources: list[str] = field(default_factory=list)
    retrieval_results: RetrievalResult | None = None
    confidence: float = 0.0


@dataclass
class KnowledgeGraph:
    """In-memory knowledge graph for HippoRAG.
    
    Stores entities as nodes and relationships as edges
    with associated passages and embeddings.
    
    Attributes:
        entities: Dict mapping entity names to entity data
        triples: List of (subject, predicate, object) tuples
        passages: Dict mapping passage IDs to passage text
        passage_embeddings: Dict mapping passage IDs to embedding vectors
        entity_passages: Dict mapping entity names to passage IDs
    """
    entities: dict[str, dict[str, Any]] = field(default_factory=dict)
    triples: list[tuple[str, str, str]] = field(default_factory=list)
    passages: dict[str, str] = field(default_factory=dict)
    passage_embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    entity_passages: dict[str, list[str]] = field(default_factory=dict)
    
    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        passage_id: str,
        passage_text: str,
    ) -> None:
        """Add a triple to the knowledge graph.
        
        Args:
            subject: Subject entity
            predicate: Relationship predicate
            obj: Object entity
            passage_id: ID of the source passage
            passage_text: Text of the source passage
        """
        triple = (subject, predicate, obj)
        if triple not in self.triples:
            self.triples.append(triple)
        
        # Add entities
        if subject not in self.entities:
            self.entities[subject] = {"name": subject, "type": "entity"}
        if obj not in self.entities:
            self.entities[obj] = {"name": obj, "type": "entity"}
        
        # Add passage
        self.passages[passage_id] = passage_text
        
        # Link entities to passage
        if subject not in self.entity_passages:
            self.entity_passages[subject] = []
        if passage_id not in self.entity_passages[subject]:
            self.entity_passages[subject].append(passage_id)
            
        if obj not in self.entity_passages:
            self.entity_passages[obj] = []
        if passage_id not in self.entity_passages[obj]:
            self.entity_passages[obj].append(passage_id)
    
    def get_entity_passages(self, entity_name: str) -> list[str]:
        """Get passages associated with an entity.
        
        Args:
            entity_name: Entity name
            
        Returns:
            List of passage texts
        """
        passage_ids = self.entity_passages.get(entity_name, [])
        return [self.passages[pid] for pid in passage_ids if pid in self.passages]
    
    def get_related_entities(self, entity_name: str) -> list[tuple[str, str, str]]:
        """Get triples related to an entity.
        
        Args:
            entity_name: Entity name
            
        Returns:
            List of (subject, predicate, object) tuples
        """
        related = []
        for s, p, o in self.triples:
            if s == entity_name or o == entity_name:
                related.append((s, p, o))
        return related


class HippoRAGDestination(DestinationPlugin):
    """HippoRAG destination for multi-hop reasoning.
    
    Stores documents in file-based knowledge graph and provides
    single-step multi-hop retrieval using Personalized PageRank.
    
    This implementation uses:
    - OpenIE (via LLM) for triple extraction from documents
    - In-memory knowledge graph for entity-relationship storage
    - PPR (Personalized PageRank) for multi-hop retrieval
    - File-based persistence (pickle/JSON)
    
    Example:
        >>> destination = HippoRAGDestination()
        >>> await destination.initialize({})
        >>> conn = await destination.connect({})
        >>> result = await destination.write(conn, data)
        >>> qa_results = await destination.rag_qa(["What did Steve Jobs found?"])
    """
    
    def __init__(self) -> None:
        """Initialize HippoRAG destination."""
        self._config: dict[str, Any] = {}
        self._save_dir: str = HIPPO_SAVE_DIR
        self._llm_model: str = HIPPO_LLM_MODEL
        self._embedding_model: str = HIPPO_EMBEDDING_MODEL
        self._retrieval_k: int = HIPPO_RETRIEVAL_K
        self._llm_provider: HippoRAGLLMProvider | None = None
        self._graph: KnowledgeGraph = KnowledgeGraph()
        self._is_initialized = False
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Ensure save directory exists
        os.makedirs(self._save_dir, exist_ok=True)
        
    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            id="hipporag",
            name="HippoRAG Multi-Hop Reasoning",
            version="1.0.0",
            type=PluginType.DESTINATION,
            description="Multi-hop reasoning using neurobiological memory model with PPR",
            author="Pipeline Team",
            supported_formats=["json", "text", "markdown"],
            requires_auth=False,
            config_schema={
                "type": "object",
                "properties": {
                    "save_dir": {
                        "type": "string",
                        "description": "Directory for file-based storage",
                        "default": "/data/hipporag",
                    },
                    "llm_model": {
                        "type": "string",
                        "description": "LLM model for OpenIE and QA",
                        "default": "azure/gpt-4.1",
                    },
                    "embedding_model": {
                        "type": "string",
                        "description": "Embedding model for passages",
                        "default": "azure/text-embedding-3-small",
                    },
                    "retrieval_k": {
                        "type": "integer",
                        "description": "Default number of passages to retrieve",
                        "default": 10,
                    },
                },
            },
        )
    
    async def initialize(self, config: dict[str, Any]) -> None:
        """Initialize HippoRAG with file-based storage.
        
        Args:
            config: Configuration dict with optional keys:
                - save_dir: Storage directory path
                - llm_model: LLM model name
                - embedding_model: Embedding model name
                - retrieval_k: Default retrieval count
        """
        import_start = time.time()
        
        # Check if HippoRAG library is available (optional)
        try:
            import hipporag
            logger.info(
                "hipporag_library_available",
                version=getattr(hipporag, "__version__", "unknown"),
            )
        except ImportError:
            logger.warning(
                "hipporag_library_not_installed",
                message="HippoRAG library not found. Using local implementation.",
            )
        
        self._config = config
        self._save_dir = config.get("save_dir", HIPPO_SAVE_DIR)
        self._llm_model = config.get("llm_model", HIPPO_LLM_MODEL)
        self._embedding_model = config.get("embedding_model", HIPPO_EMBEDDING_MODEL)
        self._retrieval_k = config.get("retrieval_k", HIPPO_RETRIEVAL_K)
        
        # Ensure save directory exists
        os.makedirs(self._save_dir, exist_ok=True)
        
        # Initialize LLM provider
        self._llm_provider = HippoRAGLLMProvider(
            llm_model=self._llm_model,
            embedding_model=self._embedding_model,
        )
        
        # Try to load existing graph
        await self._load_graph()
        
        self._is_initialized = True
        
        init_duration = time.time() - import_start
        logger.info(
            "hipporag_initialized",
            save_dir=self._save_dir,
            llm_model=self._llm_model,
            embedding_model=self._embedding_model,
            entities_loaded=len(self._graph.entities),
            triples_loaded=len(self._graph.triples),
            duration_ms=round(init_duration * 1000, 2),
        )
    
    async def _load_graph(self) -> None:
        """Load knowledge graph from file storage."""
        graph_path = os.path.join(self._save_dir, "knowledge_graph.pkl")
        
        if os.path.exists(graph_path):
            try:
                # Run in thread pool for sync file I/O
                loop = asyncio.get_event_loop()
                with open(graph_path, "rb") as f:
                    self._graph = await loop.run_in_executor(
                        self._executor, pickle.load, f
                    )
                logger.info(
                    "hipporag_graph_loaded",
                    entities=len(self._graph.entities),
                    triples=len(self._graph.triples),
                    passages=len(self._graph.passages),
                )
            except Exception as e:
                logger.warning(
                    "hipporag_graph_load_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                self._graph = KnowledgeGraph()
    
    async def _save_graph(self) -> None:
        """Save knowledge graph to file storage."""
        graph_path = os.path.join(self._save_dir, "knowledge_graph.pkl")
        
        try:
            # Run in thread pool for sync file I/O
            loop = asyncio.get_event_loop()
            
            def _save():
                with open(graph_path, "wb") as f:
                    pickle.dump(self._graph, f)
            
            await loop.run_in_executor(self._executor, _save)
            
            logger.debug(
                "hipporag_graph_saved",
                entities=len(self._graph.entities),
                triples=len(self._graph.triples),
                passages=len(self._graph.passages),
            )
        except Exception as e:
            logger.error(
                "hipporag_graph_save_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
    
    async def connect(self, config: dict[str, Any]) -> Connection:
        """Connect to HippoRAG storage.
        
        Args:
            config: Connection configuration (not used for file-based storage)
            
        Returns:
            Connection handle
            
        Raises:
            ConnectionError: If not initialized
        """
        if not self._is_initialized:
            raise ConnectionError("HippoRAGDestination not initialized. Call initialize() first.")
        
        logger.debug("hipporag_connected")
        
        return Connection(
            id=UUID(int=hash("hipporag") % (2**32)),
            plugin_id="hipporag",
            config={
                "save_dir": self._save_dir,
                "llm_model": self._llm_model,
                "embedding_model": self._embedding_model,
                "retrieval_k": self._retrieval_k,
            },
        )
    
    async def write(
        self,
        conn: Connection,
        data: TransformedData,
    ) -> WriteResult:
        """Write documents to HippoRAG.
        
        Process:
        1. Extract text from chunks
        2. Run OpenIE to extract triples
        3. Build knowledge graph
        4. Generate embeddings for passages
        5. Store in file-based storage
        
        Args:
            conn: Connection handle from connect()
            data: Transformed data to write
            
        Returns:
            WriteResult with operation status
        """
        start_time = time.time()
        
        if not self._is_initialized:
            return WriteResult(
                success=False,
                error="HippoRAGDestination not initialized",
            )
        
        if not self._llm_provider:
            return WriteResult(
                success=False,
                error="LLM provider not initialized",
            )
        
        try:
            # Extract texts and metadatas from chunks
            texts: list[str] = []
            metadatas: list[dict[str, Any]] = []
            
            for i, chunk in enumerate(data.chunks):
                content = chunk.get("content", "")
                if content:
                    texts.append(content)
                    metadatas.append({
                        "chunk_index": i,
                        "job_id": str(data.job_id),
                        "original_format": data.original_format,
                        **chunk.get("metadata", {}),
                    })
            
            if not texts:
                return WriteResult(
                    success=True,
                    destination_id="hipporag",
                    records_written=0,
                    metadata={"message": "No text content to index"},
                )
            
            # Index documents
            await self.index_documents(texts, metadatas)
            
            # Save updated graph
            await self._save_graph()
            
            processing_time = int((time.time() - start_time) * 1000)
            
            logger.info(
                "hipporag_write_completed",
                job_id=str(data.job_id),
                chunks_indexed=len(texts),
                entities_total=len(self._graph.entities),
                triples_total=len(self._graph.triples),
                duration_ms=processing_time,
            )
            
            return WriteResult(
                success=True,
                destination_id="hipporag",
                destination_uri=f"hipporag://{self._save_dir}",
                records_written=len(texts),
                bytes_written=len(json.dumps(texts)),
                processing_time_ms=processing_time,
                metadata={
                    "chunks_indexed": len(texts),
                    "entities_total": len(self._graph.entities),
                    "triples_total": len(self._graph.triples),
                    "passages_total": len(self._graph.passages),
                },
            )
            
        except Exception as e:
            logger.error(
                "hipporag_write_failed",
                error=str(e),
                error_type=type(e).__name__,
                job_id=str(data.job_id),
            )
            return WriteResult(
                success=False,
                error=f"Write failed: {e!s}",
            )
    
    async def index_documents(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Index documents into HippoRAG.
        
        Steps:
        1. Run OpenIE to extract triples from each document
        2. Build knowledge graph from triples
        3. Generate embeddings for passages
        4. Store in file-based storage
        
        Args:
            texts: List of document texts
            metadatas: List of metadata dicts for each document
        """
        if not self._llm_provider:
            raise RuntimeError("LLM provider not initialized")
        
        for i, (text, metadata) in enumerate(zip(texts, metadatas)):
            passage_id = f"passage_{metadata.get('job_id', 'unknown')}_{metadata.get('chunk_index', i)}"
            
            try:
                # Run OpenIE to extract triples
                triples = await self._run_openie(text)
                
                if triples:
                    # Add triples to knowledge graph
                    for subject, predicate, obj in triples:
                        self._graph.add_triple(
                            subject=subject,
                            predicate=predicate,
                            obj=obj,
                            passage_id=passage_id,
                            passage_text=text,
                        )
                    
                    logger.debug(
                        "hipporag_triples_extracted",
                        passage_id=passage_id,
                        triple_count=len(triples),
                    )
                
                # Generate and store embedding
                embedding = await self._llm_provider.embed_text(text)
                if embedding is not None:
                    self._graph.passage_embeddings[passage_id] = embedding
                
            except Exception as e:
                logger.warning(
                    "hipporag_index_document_warning",
                    passage_id=passage_id,
                    error=str(e),
                )
    
    async def _run_openie(self, text: str) -> list[tuple[str, str, str]]:
        """Run OpenIE to extract subject-predicate-object triples.
        
        Uses LLM to extract triples from text.
        
        Args:
            text: Text to extract triples from
            
        Returns:
            List of (subject, predicate, object) tuples
        """
        if not self._llm_provider:
            return []
        
        try:
            triples = await self._llm_provider.extract_triples(text)
            return triples
        except Exception as e:
            logger.warning(
                "hipporag_openie_extraction_failed",
                error=str(e),
                text_length=len(text),
            )
            return []
    
    async def retrieve(
        self,
        queries: list[str],
        num_to_retrieve: int = 10,
    ) -> list[RetrievalResult]:
        """Multi-hop retrieval using PPR.
        
        Single-step retrieval that can traverse multiple hops
        in the knowledge graph using Personalized PageRank.
        
        Args:
            queries: List of query strings
            num_to_retrieve: Number of passages to retrieve
            
        Returns:
            List of RetrievalResult with passages and scores
        """
        if not self._is_initialized:
            raise RuntimeError("HippoRAGDestination not initialized")
        
        if not self._llm_provider:
            raise RuntimeError("LLM provider not initialized")
        
        results: list[RetrievalResult] = []
        
        for query in queries:
            try:
                # Extract entities from query
                query_entities = await self._extract_query_entities(query)
                
                # Run PPR to get passage scores
                passage_scores = await self._run_ppr(query_entities, num_to_retrieve)
                
                # Build result
                passages = []
                scores = []
                source_docs = []
                
                for passage_id, score in passage_scores:
                    if passage_id in self._graph.passages:
                        passages.append(self._graph.passages[passage_id])
                        scores.append(score)
                        # Extract source document from passage_id
                        parts = passage_id.split("_")
                        if len(parts) >= 2:
                            source_docs.append(parts[1])
                        else:
                            source_docs.append("unknown")
                
                result = RetrievalResult(
                    query=query,
                    passages=passages,
                    scores=scores,
                    source_documents=source_docs,
                    entities=query_entities,
                )
                results.append(result)
                
                logger.debug(
                    "hipporag_retrieval_completed",
                    query=query[:100],
                    entities=query_entities,
                    passages_retrieved=len(passages),
                )
                
            except Exception as e:
                logger.error(
                    "hipporag_retrieval_failed",
                    query=query[:100],
                    error=str(e),
                )
                # Return empty result for this query
                results.append(RetrievalResult(query=query))
        
        return results
    
    async def _extract_query_entities(self, query: str) -> list[str]:
        """Extract entities from query string.
        
        Args:
            query: Query string
            
        Returns:
            List of entity names
        """
        if not self._llm_provider:
            return []
        
        try:
            # Use LLM to extract entities
            entities = await self._llm_provider.extract_query_entities(query)
            
            # Filter to only entities that exist in the graph
            graph_entities = [
                e for e in entities
                if e in self._graph.entities
            ]
            
            # If no graph entities found, try fuzzy matching
            if not graph_entities:
                query_lower = query.lower()
                for entity_name in self._graph.entities:
                    if entity_name.lower() in query_lower:
                        graph_entities.append(entity_name)
            
            return graph_entities[:5]  # Limit to top 5
            
        except Exception as e:
            logger.warning(
                "hipporag_query_entity_extraction_failed",
                query=query[:100],
                error=str(e),
            )
            # Fallback: return empty list
            return []
    
    async def _run_ppr(
        self,
        query_entities: list[str],
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Run Personalized PageRank for multi-hop retrieval.
        
        PPR spreads probability from query entities through the
        knowledge graph to find relevant passages.
        
        Args:
            query_entities: Starting entities for PPR
            top_k: Number of results to return
            
        Returns:
            List of (passage_id, score) tuples sorted by score
        """
        if not query_entities or not self._graph.triples:
            # No entities or empty graph - return recent passages
            passages = list(self._graph.passages.keys())
            return [(pid, 1.0) for pid in passages[-top_k:]]
        
        # Build adjacency list from triples
        adjacency: dict[str, list[str]] = {}
        for subject, predicate, obj in self._graph.triples:
            if subject not in adjacency:
                adjacency[subject] = []
            if obj not in adjacency:
                adjacency[obj] = []
            adjacency[subject].append(obj)
            adjacency[obj].append(subject)  # Undirected for PPR
        
        # PPR parameters
        alpha = 0.15  # Teleport probability
        max_iterations = 50
        convergence_threshold = 1e-6
        
        # Initialize scores
        all_nodes = list(self._graph.entities.keys())
        scores: dict[str, float] = {node: 0.0 for node in all_nodes}
        
        # Set personalization vector (query entities)
        personalization: dict[str, float] = {node: 0.0 for node in all_nodes}
        for entity in query_entities:
            if entity in personalization:
                personalization[entity] = 1.0 / len(query_entities)
        
        # Power iteration
        for _ in range(max_iterations):
            new_scores: dict[str, float] = {}
            max_diff = 0.0
            
            for node in all_nodes:
                # Teleport component
                score = alpha * personalization.get(node, 0.0)
                
                # Random walk component
                neighbors = adjacency.get(node, [])
                if neighbors:
                    neighbor_sum = sum(
                        scores.get(neighbor, 0.0) / len(adjacency.get(neighbor, [1]))
                        for neighbor in neighbors
                    )
                    score += (1 - alpha) * neighbor_sum
                
                new_scores[node] = score
                max_diff = max(max_diff, abs(score - scores.get(node, 0.0)))
            
            scores = new_scores
            
            if max_diff < convergence_threshold:
                break
        
        # Score passages based on entity scores
        passage_scores: dict[str, float] = {}
        for passage_id in self._graph.passages:
            score = 0.0
            # Find entities mentioned in this passage
            for entity_name, entity_passages in self._graph.entity_passages.items():
                if passage_id in entity_passages:
                    score += scores.get(entity_name, 0.0)
            
            if score > 0:
                passage_scores[passage_id] = score
        
        # Sort by score and return top_k
        sorted_passages = sorted(
            passage_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        return sorted_passages[:top_k]
    
    async def rag_qa(
        self,
        queries: list[str],
        num_to_retrieve: int = 10,
    ) -> list[QAResult]:
        """Full RAG pipeline: retrieve + generate answer.
        
        Args:
            queries: List of questions
            num_to_retrieve: Number of passages for context
            
        Returns:
            List of QAResult with answers and sources
        """
        if not self._is_initialized or not self._llm_provider:
            return [
                QAResult(query=q, answer="System not initialized")
                for q in queries
            ]
        
        # Step 1: Retrieve relevant passages
        retrieval_results = await self.retrieve(queries, num_to_retrieve)
        
        # Step 2: Generate answers
        qa_results: list[QAResult] = []
        
        for query, retrieval in zip(queries, retrieval_results):
            try:
                if not retrieval.passages:
                    qa_results.append(QAResult(
                        query=query,
                        answer="I couldn't find relevant information to answer this question.",
                        retrieval_results=retrieval,
                    ))
                    continue
                
                # Generate answer using LLM
                answer = await self._llm_provider.answer_question(
                    question=query,
                    context=retrieval.passages,
                )
                
                # Calculate confidence based on retrieval scores
                confidence = sum(retrieval.scores) / len(retrieval.scores) if retrieval.scores else 0.0
                
                qa_results.append(QAResult(
                    query=query,
                    answer=answer,
                    sources=retrieval.passages,
                    retrieval_results=retrieval,
                    confidence=min(1.0, confidence),
                ))
                
                logger.debug(
                    "hipporag_qa_completed",
                    query=query[:100],
                    answer_length=len(answer),
                    confidence=confidence,
                )
                
            except Exception as e:
                logger.error(
                    "hipporag_qa_failed",
                    query=query[:100],
                    error=str(e),
                )
                qa_results.append(QAResult(
                    query=query,
                    answer=f"Error generating answer: {e}",
                    retrieval_results=retrieval,
                ))
        
        return qa_results
    
    async def health_check(self, config: dict[str, Any] | None = None) -> HealthStatus:
        """Check the health of the HippoRAG destination.
        
        Args:
            config: Optional configuration for health check
            
        Returns:
            HealthStatus indicating plugin health
        """
        if not self._is_initialized:
            return HealthStatus.UNHEALTHY
        
        # Check save directory is writable
        try:
            test_file = os.path.join(self._save_dir, ".health_check")
            with open(test_file, "w") as f:
                f.write("healthy")
            os.remove(test_file)
        except Exception as e:
            logger.error(
                "hipporag_health_check_failed",
                error=str(e),
                save_dir=self._save_dir,
            )
            return HealthStatus.UNHEALTHY
        
        # Check LLM provider
        if self._llm_provider:
            llm_health = await self._llm_provider.health_check()
            if not llm_health.get("healthy", False):
                return HealthStatus.DEGRADED
        
        return HealthStatus.HEALTHY
    
    async def shutdown(self) -> None:
        """Shutdown the plugin and cleanup resources."""
        # Save final state
        if self._is_initialized:
            await self._save_graph()
        
        # Shutdown executor
        self._executor.shutdown(wait=True)
        
        self._is_initialized = False
        logger.info("hipporag_shutdown_complete")


class HippoRAGMockDestination(DestinationPlugin):
    """Mock HippoRAG destination for testing.
    
    Provides the same interface as HippoRAGDestination but with
    in-memory storage and no external dependencies.
    """
    
    def __init__(self) -> None:
        """Initialize mock HippoRAG destination."""
        self._config: dict[str, Any] = {}
        self._documents: list[dict[str, Any]] = []
        self._is_initialized = False
    
    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            id="hipporag_mock",
            name="HippoRAG Mock (Testing)",
            version="1.0.0",
            type=PluginType.DESTINATION,
            description="Mock HippoRAG destination for testing",
            author="Pipeline Team",
            supported_formats=["json", "text"],
            requires_auth=False,
        )
    
    async def initialize(self, config: dict[str, Any]) -> None:
        """Initialize mock HippoRAG."""
        self._config = config
        self._documents = []
        self._is_initialized = True
        logger.info("hipporag_mock_initialized")
    
    async def connect(self, config: dict[str, Any]) -> Connection:
        """Connect to mock storage."""
        if not self._is_initialized:
            raise ConnectionError("HippoRAGMockDestination not initialized")
        
        return Connection(
            id=UUID(int=hash("hipporag_mock") % (2**32)),
            plugin_id="hipporag_mock",
            config={},
        )
    
    async def write(
        self,
        conn: Connection,
        data: TransformedData,
    ) -> WriteResult:
        """Write data to mock storage."""
        if not self._is_initialized:
            return WriteResult(
                success=False,
                error="HippoRAGMockDestination not initialized",
            )
        
        # Store documents in memory
        for i, chunk in enumerate(data.chunks):
            self._documents.append({
                "job_id": str(data.job_id),
                "chunk_index": i,
                "content": chunk.get("content", ""),
                "metadata": chunk.get("metadata", {}),
            })
        
        return WriteResult(
            success=True,
            destination_id="hipporag_mock",
            records_written=len(data.chunks),
            metadata={"total_documents": len(self._documents)},
        )
    
    async def retrieve(
        self,
        queries: list[str],
        num_to_retrieve: int = 10,
    ) -> list[RetrievalResult]:
        """Mock retrieval - returns random documents."""
        results: list[RetrievalResult] = []
        
        for query in queries:
            # Simple keyword matching
            passages = []
            for doc in self._documents[:num_to_retrieve]:
                content = doc.get("content", "")
                if any(word.lower() in content.lower() for word in query.split()[:3]):
                    passages.append(content)
            
            results.append(RetrievalResult(
                query=query,
                passages=passages or ["Mock passage 1", "Mock passage 2"],
                scores=[0.9, 0.8][:len(passages) or 2],
                source_documents=["mock_1", "mock_2"][:len(passages) or 2],
            ))
        
        return results
    
    async def rag_qa(
        self,
        queries: list[str],
        num_to_retrieve: int = 10,
    ) -> list[QAResult]:
        """Mock RAG QA - returns placeholder answers."""
        retrieval_results = await self.retrieve(queries, num_to_retrieve)
        
        return [
            QAResult(
                query=query,
                answer=f"Mock answer for: {query[:50]}...",
                sources=retrieval.passages if retrieval else [],
                retrieval_results=retrieval,
                confidence=0.85,
            )
            for query, retrieval in zip(queries, retrieval_results)
        ]
    
    async def health_check(self, config: dict[str, Any] | None = None) -> HealthStatus:
        """Mock health check."""
        return HealthStatus.HEALTHY if self._is_initialized else HealthStatus.UNHEALTHY
    
    async def shutdown(self) -> None:
        """Shutdown mock destination."""
        self._is_initialized = False
        self._documents = []
