# Design: add-vector-store-pgvector

## Overview

This design document outlines the implementation of a native vector storage and search system using PostgreSQL with the pgvector extension. This enables the pipeline to store chunk embeddings and perform vector similarity search without external dependencies.

## Architecture

### High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API Layer                                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │  Chunk Routes   │  │  Search Routes  │  │      Health Routes          │  │
│  │                 │  │                 │  │                             │  │
│  │ GET/POST/DELETE │  │ POST /semantic  │  │  GET /health/vector_store   │  │
│  │ /jobs/{id}/...  │  │ POST /text      │  │                             │  │
│  │                 │  │ POST /hybrid    │  │                             │  │
│  │                 │  │ GET /similar    │  │                             │  │
│  └────────┬────────┘  └────────┬────────┘  └─────────────────────────────┘  │
└───────────┼────────────────────┼────────────────────────────────────────────┘
            │                    │
            ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Service Layer                                      │
│                                                                              │
│  ┌────────────────────────┐  ┌────────────────────────┐  ┌────────────────┐ │
│  │  VectorSearchService   │  │   TextSearchService    │  │ ChunkService   │ │
│  │                        │  │                        │  │                │ │
│  │ - search_by_vector()   │  │ - search_by_text()     │  │ - get_chunk()  │ │
│  │ - find_similar()       │  │ - search_bm25()        │  │ - list_chunks()│ │
│  │ - batch_search()       │  │ - fuzzy_search()       │  │ - create_chunk()││
│  └───────────┬────────────┘  └───────────┬────────────┘  └───────┬────────┘ │
│              │                           │                       │          │
│              └───────────────┬───────────┘                       │          │
│                              ▼                                   │          │
│                   ┌─────────────────────┐                        │          │
│                   │ HybridSearchService │◄───────────────────────┘          │
│                   │                     │                                    │
│                   │ - hybrid_search()   │  • Weighted Sum fusion             │
│                   │ - rrf_fusion()      │  • Reciprocal Rank Fusion (RRF)    │
│                   │ - combine_results() │                                    │
│                   └──────────┬──────────┘                                    │
└──────────────────────────────┼──────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Repository Layer                                    │
│                                                                              │
│  ┌────────────────────────┐  ┌────────────────────────┐  ┌────────────────┐ │
│  │ DocumentChunkRepository│  │   SearchRepository     │  │ JobRepository  │ │
│  │                        │  │                        │  │                │ │
│  │ - get_by_id()          │  │ - vector_search()      │  │ - get_by_id()  │ │
│  │ - get_by_job()         │  │ - text_search()        │  │ - exists()     │ │
│  │ - create()             │  │ - hybrid_search()      │  │ - list()       │ │
│  │ - batch_create()       │  │ - count_results()      │  │                │ │
│  └───────────┬────────────┘  └───────────┬────────────┘  └───────┬────────┘ │
└──────────────┼───────────────────────────┼───────────────────────┼──────────┘
               │                           │                       │
               ▼                           ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Database Layer (PostgreSQL)                           │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                    Extensions & Types                                   │  │
│  │  • pgvector (VECTOR type, <=>, <#>, <-> operators)                      │  │
│  │  • pg_trgm (trigram matching for fuzzy search)                          │  │
│  │  • Full-text search (tsvector, tsquery)                                 │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                    document_chunks Table                                │  │
│  │  ┌─────────────┬──────────┬─────────────────────────────────────────┐  │  │
│  │  │ id          │ UUID     │ PRIMARY KEY                             │  │  │
│  │  │ job_id      │ UUID     │ FK → jobs(id), indexed                  │  │  │
│  │  │ chunk_index │ INTEGER  │ NOT NULL, unique per job                │  │  │
│  │  │ content     │ TEXT     │ NOT NULL, full-text indexed             │  │  │
│  │  │ embedding   │ VECTOR   │ 1536 dimensions (configurable)          │  │  │
│  │  │ metadata    │ JSONB    │ GIN indexed for filtering               │  │  │
│  │  │ created_at  │ TIMESTAMPTZ │ Auto-populated                       │  │  │
│  │  └─────────────┴──────────┴─────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                    Indexes                                              │  │
│  │  • idx_document_chunks_embedding_hnsw (HNSW on embedding)              │  │
│  │  • idx_document_chunks_content_search (GIN on to_tsvector)             │  │
│  │  • idx_document_chunks_content_trgm (GIN trigram)                      │  │
│  │  • idx_document_chunks_job_id (B-tree)                                 │  │
│  │  • idx_document_chunks_metadata (GIN jsonb_path_ops)                   │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Document   │────▶│   Chunker    │────▶│  Embedding   │────▶│   Storage    │
│   Ingestion  │     │   Pipeline   │     │   Service    │     │   (DB)       │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                              │
                                                              ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐  ┌──────────────┐
│   Response   │◀────│   Fusion     │◀────│   Search     │◀─│   Query      │
│   Ranking    │     │   (Hybrid)   │     │   Services   │  │   Embedding  │
└──────────────┘     └──────────────┘     └──────────────┘  └──────────────┘
```

### Pipeline Integration

The vector store integrates at the transformation stage of the existing pipeline:

1. **Document Processing**: Documents are chunked by the existing chunking service
2. **Embedding Generation**: Each chunk is passed to the LLM adapter for embedding generation
3. **Storage**: Chunks with embeddings are persisted to `document_chunks` table
4. **Indexing**: HNSW and GIN indexes are maintained automatically by PostgreSQL

## Component Design

### Database Layer

#### pgvector Extension Setup

```sql
-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Verify installation
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
```

#### Document Chunks Table

```sql
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(1536),  -- Configurable dimensions
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT unique_job_chunk UNIQUE (job_id, chunk_index),
    CONSTRAINT positive_chunk_index CHECK (chunk_index >= 0)
);

-- Indexes
CREATE INDEX idx_document_chunks_job_id ON document_chunks(job_id);
CREATE INDEX idx_document_chunks_created_at ON document_chunks USING BRIN(created_at);
CREATE INDEX idx_document_chunks_metadata ON document_chunks USING GIN(metadata jsonb_path_ops);

-- HNSW index for vector similarity search
CREATE INDEX idx_document_chunks_embedding_hnsw
ON document_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Full-text search index
CREATE INDEX idx_document_chunks_content_search
ON document_chunks
USING gin (to_tsvector('english', content));

-- Trigram index for fuzzy matching
CREATE INDEX idx_document_chunks_content_trgm
ON document_chunks
USING gin (content gin_trgm_ops);
```

#### SQLAlchemy Model

```python
from pgvector.sqlalchemy import Vector
from sqlalchemy import Index, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(1536), nullable=True
    )
    metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default="{}"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        default=func.now(), server_default=func.now()
    )
    
    # Relationships
    job: Mapped["Job"] = relationship("Job", back_populates="chunks")
    
    __table_args__ = (
        Index('idx_document_chunks_content_search',
              text("to_tsvector('english', content)"),
              postgresql_using='gin'),
        Index('idx_document_chunks_content_trgm', 'content',
              postgresql_using='gin', postgresql_ops={'content': 'gin_trgm_ops'}),
    )
```

### Service Layer

#### VectorSearchService

```python
class VectorSearchService:
    """Service for vector similarity search operations."""
    
    def __init__(
        self,
        db_session: AsyncSession,
        config: VectorStoreConfig,
    ):
        self.db = db_session
        self.config = config
        self.logger = get_logger(__name__)
    
    async def search_by_vector(
        self,
        query_vector: list[float],
        top_k: int = 10,
        job_id: uuid.UUID | None = None,
        similarity_threshold: float = 0.0,
        metadata_filters: list[MetadataFilter] | None = None,
    ) -> list[SearchResult]:
        """
        Search for similar chunks by vector similarity.
        
        Uses HNSW index for approximate nearest neighbor search.
        Returns results ordered by similarity (highest first).
        """
        # Validate dimensions
        if len(query_vector) != self.config.dimensions:
            raise InvalidVectorError(
                f"Query vector dimension {len(query_vector)} "
                f"does not match expected {self.config.dimensions}"
            )
        
        # Set ef_search for query accuracy
        await self.db.execute(
            text(f"SET LOCAL hnsw.ef_search = {self.config.hnsw.ef_search}")
        )
        
        # Build query with filters
        query = select(
            DocumentChunk,
            (1 - DocumentChunk.embedding.op("<=>")(query_vector)).label("similarity")
        )
        
        if job_id:
            query = query.where(DocumentChunk.job_id == job_id)
        
        if similarity_threshold > 0:
            query = query.where(
                DocumentChunk.embedding.op("<=>")(query_vector) <= (1 - similarity_threshold)
            )
        
        # Apply metadata filters
        if metadata_filters:
            query = self._apply_metadata_filters(query, metadata_filters)
        
        query = query.order_by(text("similarity DESC")).limit(top_k)
        
        result = await self.db.execute(query)
        return [self._to_search_result(row) for row in result]
    
    async def find_similar_chunks(
        self,
        chunk_id: uuid.UUID,
        top_k: int = 10,
        exclude_same_job: bool = False,
    ) -> list[SearchResult]:
        """Find chunks similar to a given chunk by ID."""
        # Get reference chunk embedding
        chunk = await self._get_chunk(chunk_id)
        if not chunk or not chunk.embedding:
            raise ChunkNotFoundError(chunk_id)
        
        # Search using reference embedding
        results = await self.search_by_vector(
            query_vector=chunk.embedding,
            top_k=top_k + 1,  # +1 to account for self-match
        )
        
        # Exclude the reference chunk
        results = [r for r in results if r.chunk.id != chunk_id]
        
        if exclude_same_job:
            results = [r for r in results if r.chunk.job_id != chunk.job_id]
        
        return results[:top_k]
```

#### TextSearchService

```python
class TextSearchService:
    """Service for full-text and fuzzy search operations."""
    
    BM25_WEIGHTS = '{0.1, 0.2, 0.4, 1.0}'
    NORMALIZATION = 32  # Combination of 1 + 2 + 4 (BM25-like)
    
    async def search_bm25(
        self,
        query: str,
        top_k: int = 10,
        language: str = "english",
        job_id: uuid.UUID | None = None,
        highlight: bool = True,
    ) -> list[TextSearchResult]:
        """
        Search using PostgreSQL full-text search with BM25-like ranking.
        
        Uses ts_rank_cd with custom weights for relevance scoring.
        """
        # Build tsvector and tsquery
        tsvector = func.to_tsvector(language, DocumentChunk.content)
        tsquery = func.plainto_tsquery(language, query)
        
        # Calculate rank using ts_rank_cd (cover density)
        rank = func.ts_rank_cd(
            self.BM25_WEIGHTS,
            tsvector,
            tsquery,
            self.NORMALIZATION
        )
        
        # Build query
        stmt = select(
            DocumentChunk,
            rank.label("relevance_score"),
            func.ts_headline(
                language, DocumentChunk.content, tsquery,
                'StartSel=<mark>, StopSel=</mark>, MaxWords=50, MinWords=10'
            ).label("highlighted_content") if highlight else None
        ).where(tsvector.op("@@")(tsquery))
        
        if job_id:
            stmt = stmt.where(DocumentChunk.job_id == job_id)
        
        stmt = stmt.order_by(desc(rank)).limit(top_k)
        
        result = await self.db.execute(stmt)
        return [self._to_text_result(row) for row in result]
    
    async def fuzzy_search(
        self,
        query: str,
        similarity_threshold: float = 0.3,
        top_k: int = 10,
    ) -> list[TextSearchResult]:
        """
        Fuzzy search using pg_trgm trigram similarity.
        
        Handles typos and spelling variations.
        """
        if len(query) < 3:
            raise ValidationError("Query must be at least 3 characters for fuzzy search")
        
        # Set similarity threshold
        await self.db.execute(
            text(f"SET LOCAL pg_trgm.similarity_threshold = {similarity_threshold}")
        )
        
        # Use % operator for similarity
        similarity = func.similarity(DocumentChunk.content, query)
        
        stmt = select(
            DocumentChunk,
            similarity.label("similarity_score")
        ).where(
            DocumentChunk.content.op("%")(query)
        ).order_by(desc(similarity)).limit(top_k)
        
        result = await self.db.execute(stmt)
        return [self._to_text_result(row) for row in result]
```

#### HybridSearchService

```python
class HybridSearchService:
    """
    Service for combining vector and text search results.
    
    Supports two fusion methods:
    1. Weighted Sum: Linear combination of normalized scores
    2. RRF (Reciprocal Rank Fusion): Rank-based combination
    """
    
    def __init__(
        self,
        vector_service: VectorSearchService,
        text_service: TextSearchService,
        config: HybridSearchConfig,
    ):
        self.vector_service = vector_service
        self.text_service = text_service
        self.config = config
    
    async def hybrid_search(
        self,
        query_text: str,
        top_k: int = 10,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        fusion_method: str = "weighted_sum",
        rrf_k: int = 60,
        similarity_threshold: float = 0.5,
    ) -> HybridSearchResult:
        """
        Perform hybrid search combining vector and text results.
        
        Executes both searches in parallel and fuses results.
        """
        # Generate query embedding from text
        query_vector = await self._generate_embedding(query_text)
        
        # Execute searches concurrently
        vector_task = self.vector_service.search_by_vector(
            query_vector=query_vector,
            top_k=top_k * 2,  # Get more for fusion
            similarity_threshold=similarity_threshold,
        )
        text_task = self.text_service.search_bm25(
            query=query_text,
            top_k=top_k * 2,
        )
        
        vector_results, text_results = await asyncio.gather(
            vector_task, text_task, return_exceptions=True
        )
        
        # Handle failures gracefully
        if isinstance(vector_results, Exception):
            self.logger.warning("vector_search_failed", error=str(vector_results))
            vector_results = []
        if isinstance(text_results, Exception):
            self.logger.warning("text_search_failed", error=str(text_results))
            text_results = []
        
        # Fallback if one search returns no results
        if not vector_results and text_results:
            return self._to_hybrid_result(text_results, fusion_method="text_only")
        if not text_results and vector_results:
            return self._to_hybrid_result(vector_results, fusion_method="vector_only")
        
        # Fuse results
        if fusion_method == "rrf":
            fused = self._rrf_fusion(vector_results, text_results, rrf_k, top_k)
        else:
            fused = self._weighted_fusion(
                vector_results, text_results, vector_weight, text_weight, top_k
            )
        
        return fused
    
    def _rrf_fusion(
        self,
        vector_results: list[SearchResult],
        text_results: list[TextSearchResult],
        k: int,
        top_k: int,
    ) -> list[HybridResult]:
        """
        Reciprocal Rank Fusion.
        
        RRF_score(d) = Σ(1 / (k + rank_i(d)))
        """
        # Create rank dictionaries
        vector_ranks = {r.chunk.id: i + 1 for i, r in enumerate(vector_results)}
        text_ranks = {r.chunk.id: i + 1 for i, r in enumerate(text_results)}
        
        # Calculate RRF scores
        all_ids = set(vector_ranks.keys()) | set(text_ranks.keys())
        rrf_scores = {}
        
        for doc_id in all_ids:
            score = 0.0
            if doc_id in vector_ranks:
                score += 1.0 / (k + vector_ranks[doc_id])
            if doc_id in text_ranks:
                score += 1.0 / (k + text_ranks[doc_id])
            rrf_scores[doc_id] = score
        
        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Build results
        chunk_map = {r.chunk.id: r for r in vector_results + text_results}
        return [
            HybridResult(
                chunk=chunk_map[doc_id].chunk,
                combined_score=rrf_scores[doc_id],
                vector_rank=vector_ranks.get(doc_id),
                text_rank=text_ranks.get(doc_id),
            )
            for doc_id in sorted_ids[:top_k]
        ]
    
    def _weighted_fusion(
        self,
        vector_results: list[SearchResult],
        text_results: list[TextSearchResult],
        vector_weight: float,
        text_weight: float,
        top_k: int,
    ) -> list[HybridResult]:
        """
        Weighted sum fusion.
        
        combined_score = (vector_weight × vector_score) + (text_weight × text_score)
        """
        # Normalize weights
        total = vector_weight + text_weight
        vector_weight = vector_weight / total
        text_weight = text_weight / total
        
        # Build score dictionaries
        vector_scores = {r.chunk.id: r.similarity for r in vector_results}
        text_scores = {r.chunk.id: r.relevance_score for r in text_results}
        
        # Get all document IDs
        all_ids = set(vector_scores.keys()) | set(text_scores.keys())
        
        # Calculate combined scores
        combined = {}
        for doc_id in all_ids:
            v_score = vector_scores.get(doc_id, 0.0)
            t_score = text_scores.get(doc_id, 0.0)
            combined[doc_id] = (vector_weight * v_score) + (text_weight * t_score)
        
        # Sort by combined score
        sorted_ids = sorted(combined.keys(), key=lambda x: combined[x], reverse=True)
        
        # Build results
        chunk_map = {r.chunk.id: r for r in vector_results + text_results}
        vector_ranks = {r.chunk.id: i + 1 for i, r in enumerate(vector_results)}
        text_ranks = {r.chunk.id: i + 1 for i, r in enumerate(text_results)}
        
        return [
            HybridResult(
                chunk=chunk_map[doc_id].chunk,
                combined_score=combined[doc_id],
                vector_score=vector_scores.get(doc_id, 0.0),
                text_score=text_scores.get(doc_id, 0.0),
                vector_rank=vector_ranks.get(doc_id),
                text_rank=text_ranks.get(doc_id),
            )
            for doc_id in sorted_ids[:top_k]
        ]
```

### API Layer

#### Search Endpoints

```python
router = APIRouter(prefix="/api/v1/search", tags=["search"])

@router.post("/semantic", response_model=SemanticSearchResponse)
async def semantic_search(
    request: SemanticSearchRequest,
    service: VectorSearchService = Depends(get_vector_search_service),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
):
    """
    Semantic similarity search using vector embeddings.
    
    - Accepts pre-computed query vector or generates from text
    - Returns chunks ordered by cosine similarity
    - Supports metadata filtering and job scoping
    """
    await rate_limiter.check_limit("search_semantic", request)
    
    results = await service.search_by_vector(
        query_vector=request.query_vector,
        top_k=request.top_k,
        job_id=request.job_id,
        similarity_threshold=request.similarity_threshold,
        metadata_filters=request.metadata_filter,
    )
    
    return SemanticSearchResponse(
        success=True,
        data={"results": results, "total": len(results)},
        error=None,
    )

@router.post("/text", response_model=TextSearchResponse)
async def text_search(
    request: TextSearchRequest,
    service: TextSearchService = Depends(get_text_search_service),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
):
    """
    Full-text search with BM25-like ranking.
    
    - Uses PostgreSQL tsvector/tsquery with ts_rank_cd
    - Supports highlighting and fuzzy matching
    """
    await rate_limiter.check_limit("search_text", request)
    
    results = await service.search_bm25(
        query=request.query,
        top_k=request.top_k,
        language=request.language,
        job_id=request.job_id,
        highlight=request.highlight,
    )
    
    return TextSearchResponse(
        success=True,
        data={"results": results, "total": len(results)},
        error=None,
    )

@router.post("/hybrid", response_model=HybridSearchResponse)
async def hybrid_search(
    request: HybridSearchRequest,
    service: HybridSearchService = Depends(get_hybrid_search_service),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
):
    """
    Hybrid search combining vector similarity and text relevance.
    
    - Supports weighted_sum and RRF fusion methods
    - Generates embedding from query text
    - Returns combined scoring with individual ranks
    """
    await rate_limiter.check_limit("search_hybrid", request)
    
    results = await service.hybrid_search(
        query_text=request.query_text,
        top_k=request.top_k,
        vector_weight=request.vector_weight,
        text_weight=request.text_weight,
        fusion_method=request.fusion_method,
        rrf_k=request.rrf_k,
        similarity_threshold=request.similarity_threshold,
    )
    
    return HybridSearchResponse(
        success=True,
        data=results,
        error=None,
    )

@router.get("/similar/{chunk_id}", response_model=SimilarChunksResponse)
async def find_similar_chunks(
    chunk_id: uuid.UUID,
    top_k: int = Query(10, ge=1, le=100),
    exclude_same_job: bool = Query(False),
    service: VectorSearchService = Depends(get_vector_search_service),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
):
    """Find chunks similar to a given chunk by ID."""
    await rate_limiter.check_limit("search_similar", chunk_id)
    
    results = await service.find_similar_chunks(
        chunk_id=chunk_id,
        top_k=top_k,
        exclude_same_job=exclude_same_job,
    )
    
    return SimilarChunksResponse(
        success=True,
        data={"results": results, "reference_chunk_id": chunk_id},
        error=None,
    )
```

#### Chunk Retrieval Endpoints

```python
router = APIRouter(prefix="/api/v1/jobs/{job_id}/chunks", tags=["chunks"])

@router.get("", response_model=ChunkListResponse)
async def list_job_chunks(
    job_id: uuid.UUID,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    sort_by: str = Query("chunk_index", regex="^(created_at|chunk_index)$"),
    sort_order: str = Query("asc", regex="^(asc|desc)$"),
    service: ChunkService = Depends(get_chunk_service),
):
    """List all chunks for a specific job with pagination."""
    chunks, total = await service.list_chunks(
        job_id=job_id,
        limit=limit,
        offset=offset,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    
    return ChunkListResponse(
        success=True,
        data={
            "chunks": chunks,
            "pagination": {
                "total": total,
                "page": (offset // limit) + 1,
                "per_page": limit,
                "total_pages": (total + limit - 1) // limit,
            }
        },
        error=None,
    )

@router.get("/{chunk_id}", response_model=ChunkResponse)
async def get_chunk(
    job_id: uuid.UUID,
    chunk_id: uuid.UUID,
    service: ChunkService = Depends(get_chunk_service),
):
    """Get a specific chunk by ID."""
    chunk = await service.get_chunk(chunk_id, job_id=job_id)
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")
    
    return ChunkResponse(success=True, data=chunk, error=None)
```

### Configuration

#### YAML Configuration Structure

```yaml
# config/vector_store.yaml
vector_store:
  enabled: true
  
  # Embedding configuration
  embedding:
    model: "text-embedding-3-small"  # or "text-embedding-3-large", "text-embedding-ada-002"
    dimensions: 1536  # 384, 768, 1536, or 3072
    batch_size: 100
    max_retries: 3
    timeout_seconds: 30
  
  # Search configuration
  search:
    default_top_k: 10
    max_top_k: 100
    similarity_threshold: 0.7
    
  # HNSW index configuration
  hnsw:
    enabled: true
    m: 16                    # Number of bi-directional links (2-100)
    ef_construction: 64      # Build-time candidate list size (4-1000)
    ef_search: 32           # Query-time candidate list size (1-1000)
    
  # Hybrid search configuration
  hybrid_search:
    default_vector_weight: 0.7
    default_text_weight: 0.3
    default_fusion_method: "weighted_sum"  # or "rrf"
    rrf_k: 60  # RRF constant
    
  # Full-text search configuration
  text_search:
    default_language: "english"
    bm25_weights: [0.1, 0.2, 0.4, 1.0]  # A, B, C, D weights
    normalization: 32  # ts_rank_cd normalization option
    fuzzy_threshold: 0.3  # pg_trgm similarity threshold
    
  # Rate limiting
  rate_limiting:
    enabled: true
    redis_url: "${REDIS_URL:-redis://localhost:6379/0}"
    endpoints:
      search_semantic:
        limit: 30
        window: 60
        burst: 5
      search_text:
        limit: 60
        window: 60
        burst: 10
      search_hybrid:
        limit: 20
        window: 60
        burst: 3
      list_chunks:
        limit: 100
        window: 60
        burst: 10
      get_chunk:
        limit: 200
        window: 60
        burst: 20
      search_similar:
        limit: 40
        window: 60
        burst: 5
```

## Key Technical Decisions

### Why pgvector Over External Vector Stores

| Factor | pgvector | Pinecone/Weaviate |
|--------|----------|-------------------|
| **Infrastructure** | Single PostgreSQL instance | Additional service to manage |
| **Cost** | Included with PostgreSQL | Per-query or per-dimension pricing |
| **Latency** | In-database (no network hop) | Network round-trip |
| **Transactions** | ACID with rest of data | Separate transaction boundary |
| **Backup/DR** | Unified with PostgreSQL | Separate backup strategy |
| **Scaling** | Vertical + read replicas | Purpose-built horizontal scaling |
| **Max Vectors** | Millions (sufficient for most use cases) | Billions |

**Decision**: pgvector is the right choice for this use case because:
1. Target scale is < 10M chunks per deployment
2. Existing PostgreSQL infrastructure can be leveraged
3. Unified backup and recovery simplifies operations
4. No additional vendor lock-in or costs

### HNSW Index Configuration

The HNSW (Hierarchical Navigable Small World) index provides approximate nearest neighbor search with configurable accuracy/speed tradeoffs.

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `m` | 16 | 2-100 | Number of connections per node. Higher = more accurate, slower build, more memory |
| `ef_construction` | 64 | 4-1000 | Candidate list size during build. Higher = better index quality, slower build |
| `ef_search` | 32 | 1-1000 | Candidate list size during search. Higher = better recall, slower query |

**Recommended Configurations**:

```yaml
# Development
hnsw:
  m: 8
  ef_construction: 32
  ef_search: 16

# Production (balanced)
hnsw:
  m: 16
  ef_construction: 64
  ef_search: 32

# High-accuracy (search-heavy)
hnsw:
  m: 32
  ef_construction: 128
  ef_search: 64
```

### Embedding Dimension Strategy

| Model | Dimensions | Use Case |
|-------|------------|----------|
| text-embedding-3-small | 1536 | Default, good balance |
| text-embedding-3-large | 3072 | Maximum accuracy |
| text-embedding-ada-002 | 1536 | Legacy compatibility |
| Open source (e.g., all-MiniLM) | 384 | Resource-constrained |

**Decision**: Default to 1536 dimensions (OpenAI text-embedding-3-small) with support for:
- Runtime dimension validation
- Multiple dimension support (separate tables or validation)
- Configuration-driven dimension selection

### Hybrid Scoring Approach

Two fusion methods are supported:

#### 1. Weighted Sum (Default)
```
combined_score = (vector_weight × vector_score) + (text_weight × text_score)
```

**Best for**: Balanced relevance scenarios where score magnitude matters.

**Pros**:
- Preserves relative score differences
- Intuitive weight interpretation
- Fast to compute

**Cons**:
- Requires score normalization
- Sensitive to score distribution

#### 2. Reciprocal Rank Fusion (RRF)
```
RRF_score(d) = Σ(1 / (k + rank_i(d)))
```

**Best for**: When vector and text rankings differ significantly.

**Pros**:
- No score normalization needed
- Robust to outliers
- Single parameter (k)

**Cons**:
- Rank-only (ignores score magnitude)
- Slightly more complex to explain

**Decision**: Support both methods, default to weighted_sum for simplicity, allow RRF for advanced use cases.

### BM25 Emulation Using ts_rank_cd

PostgreSQL's `ts_rank_cd` (cover density ranking) provides BM25-like scoring:

```sql
SELECT ts_rank_cd(
    '{0.1, 0.2, 0.4, 1.0}',  -- A, B, C, D weights (BM25-like)
    to_tsvector('english', content),
    plainto_tsquery('english', 'search terms'),
    32  -- Normalization: length + proximity + unique words
) AS rank
```

**Normalization Options**:
- `0`: No normalization
- `1`: Divide by 1 + log(document length)
- `2`: Divide by document length
- `32` (recommended): Combination of 1 + 2 + 4 (BM25-like behavior)

**Why ts_rank_cd over ts_rank**:
- `ts_rank_cd` considers proximity of terms (cover density)
- Better for multi-term queries
- More similar to BM25 behavior

## Integration Points

### Pipeline Integration

The vector store integrates at the transformation stage:

```python
# Pipeline stage integration
class EmbeddingTransformStage:
    """Pipeline stage that generates and stores embeddings."""
    
    async def process(self, job: Job, chunks: list[Chunk]) -> list[DocumentChunk]:
        # Generate embeddings in batches
        embeddings = await self.embedding_service.embed_batch(
            [c.content for c in chunks]
        )
        
        # Create document chunk records
        document_chunks = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            doc_chunk = DocumentChunk(
                job_id=job.id,
                chunk_index=i,
                content=chunk.content,
                embedding=embedding,
                metadata={
                    "source_file": chunk.source_file,
                    "page_number": chunk.page_number,
                }
            )
            document_chunks.append(doc_chunk)
        
        # Persist to database
        await self.chunk_repository.batch_create(document_chunks)
        
        return document_chunks
```

### Existing Job/Result Models Relationship

```
┌─────────────┐       ┌──────────────────┐       ┌─────────────────┐
│    jobs     │◄──────┤ document_chunks  │       │  job_results    │
├─────────────┤       ├──────────────────┤       ├─────────────────┤
│ id (PK)     │       │ id (PK)          │       │ id (PK)         │
│ status      │       │ job_id (FK)      │──────►│ job_id (FK)     │
│ config      │       │ chunk_index      │       │ result_type     │
│ created_at  │       │ content          │       │ data            │
└─────────────┘       │ embedding        │       └─────────────────┘
                      │ metadata         │
                      │ created_at       │
                      └──────────────────┘
```

- `document_chunks.job_id` references `jobs.id` with CASCADE delete
- Job deletion automatically removes associated chunks
- `job_results` remains for high-level results
- `document_chunks` provides granular access

### Configuration from Existing llm.yaml

The vector store configuration extends the existing LLM configuration:

```yaml
# config/llm.yaml (existing)
llm:
  provider: "azure_openai"
  embedding_model: "text-embedding-3-small"
  
# config/vector_store.yaml (new)
vector_store:
  embedding:
    # Inherits from llm.yaml if not specified
    model: "${LLM_EMBEDDING_MODEL:-text-embedding-3-small}"
    dimensions: 1536
```

## Performance Considerations

### Index Tuning for Different Data Sizes

| Data Size | HNSW m | HNSW ef_construction | ef_search | Notes |
|-----------|--------|---------------------|-----------|-------|
| < 10K | 8 | 32 | 16 | Fast builds, acceptable recall |
| 10K - 100K | 16 | 64 | 32 | Balanced default |
| 100K - 1M | 16 | 64 | 32 | May need increased work_mem |
| 1M - 10M | 24 | 96 | 48 | Consider partitioning by job_id |
| > 10M | 32 | 128 | 64 | May need dedicated resources |

### Query Optimization Strategies

1. **Filter Pushdown**: Apply metadata filters before vector search when possible
   ```sql
   -- Good: Filter first
   SELECT * FROM document_chunks 
   WHERE job_id = 'uuid'  -- Uses index
   ORDER BY embedding <=> query 
   LIMIT 10;
   ```

2. **Selective Column Retrieval**: Don't fetch embedding vectors for list views
   ```python
   # Exclude embedding from list queries
   query = select(DocumentChunk.id, DocumentChunk.content)
   ```

3. **Connection Pooling**: Size pool for concurrent search load
   ```python
   create_async_engine(
       DATABASE_URL,
       pool_size=10,
       max_overflow=20,
       pool_timeout=30,
   )
   ```

4. **Query-Time ef_search**: Adjust per-query based on accuracy requirements
   ```sql
   SET LOCAL hnsw.ef_search = 64;  -- Higher accuracy for critical queries
   ```

### Caching Layer

```python
# Redis-based result caching for frequent searches
class SearchCache:
    def __init__(self, redis: Redis):
        self.redis = redis
        self.ttl = 300  # 5 minutes
    
    async def get_cached_search(
        self, 
        query_hash: str,
        filters_hash: str
    ) -> list[SearchResult] | None:
        key = f"search:{query_hash}:{filters_hash}"
        cached = await self.redis.get(key)
        if cached:
            return json.loads(cached)
        return None
    
    async def cache_search(
        self,
        query_hash: str,
        filters_hash: str,
        results: list[SearchResult]
    ) -> None:
        key = f"search:{query_hash}:{filters_hash}"
        await self.redis.setex(
            key,
            self.ttl,
            json.dumps([r.to_dict() for r in results])
        )
```

## Security & Validation

### Input Sanitization Approach

```python
from pydantic import BaseModel, Field, validator
import re

class SearchRequest(BaseModel):
    query: str = Field(..., max_length=4096)
    top_k: int = Field(default=10, ge=1, le=100)
    
    @validator('query')
    def sanitize_query(cls, v: str) -> str:
        # Strip HTML
        v = re.sub(r'<[^>]+>', '', v)
        # Block SQL injection patterns
        dangerous = [';', '--', '/*', '*/', 'DROP', 'DELETE', 'INSERT']
        for pattern in dangerous:
            if pattern.lower() in v.lower():
                raise ValueError(f"Invalid characters in query")
        return v.strip()
```

### SQL Injection Prevention

1. **Parameterized Queries**: All SQL uses SQLAlchemy parameterized statements
   ```python
   # Safe: Parameters bound separately
   query = select(DocumentChunk).where(DocumentChunk.job_id == job_id)
   ```

2. **Input Validation**: Pydantic models validate all inputs before database access

3. **Allowlist Approach**: Only allow expected characters for each field type
   ```python
   UUID_PATTERN = re.compile(
       r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
       re.IGNORECASE
   )
   ```

### Rate Limiting Strategy

| Endpoint | Limit | Window | Burst | Key |
|----------|-------|--------|-------|-----|
| POST /search/semantic | 30 | 60s | 5 | search_semantic:{user_id} |
| POST /search/text | 60 | 60s | 10 | search_text:{user_id} |
| POST /search/hybrid | 20 | 60s | 3 | search_hybrid:{user_id} |
| GET /jobs/{id}/chunks | 100 | 60s | 10 | list_chunks:{user_id} |
| GET /chunks/{id} | 200 | 60s | 20 | get_chunk:{user_id} |
| GET /search/similar | 40 | 60s | 5 | search_similar:{user_id} |

**Implementation**: Redis-based sliding window with fail-open behavior on Redis failure.

## Migration Strategy

### Database Migration Plan

#### Migration 1: Enable Extensions
```python
# migrations/versions/001_enable_pgvector.py
"""Enable pgvector and pg_trgm extensions."""

revision = '001'
down_revision = None

def upgrade():
    # Check PostgreSQL version
    conn = op.get_bind()
    pg_version = conn.execute(text("SELECT current_setting('server_version')")).scalar()
    major = int(pg_version.split('.')[0])
    if major < 14:
        raise RuntimeError(f"PostgreSQL 14+ required, got {pg_version}")
    
    # Check extension availability
    available = conn.execute(text(
        "SELECT 1 FROM pg_available_extensions WHERE name = 'vector'"
    )).fetchone()
    if not available:
        raise RuntimeError("pgvector extension not available")
    
    # Create extensions
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    
    # Verify
    version = conn.execute(text(
        "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
    )).scalar()
    print(f"pgvector enabled: version {version}")

def downgrade():
    op.execute("DROP EXTENSION IF EXISTS pg_trgm")
    op.execute("DROP EXTENSION IF EXISTS vector")
```

#### Migration 2: Create Document Chunks Table
```python
# migrations/versions/002_create_document_chunks.py
"""Create document_chunks table with indexes."""

revision = '002'
down_revision = '001'

def upgrade():
    # Create table
    op.create_table(
        'document_chunks',
        sa.Column('id', postgresql.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('job_id', postgresql.UUID(), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('embedding', sa.nullable=True),  # Vector type via custom DDL
        sa.Column('metadata', postgresql.JSONB(), server_default='{}', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.ForeignKeyConstraint(['job_id'], ['jobs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('job_id', 'chunk_index', name='unique_job_chunk'),
        sa.CheckConstraint('chunk_index >= 0', name='positive_chunk_index'),
    )
    
    # Create indexes
    op.create_index('idx_document_chunks_job_id', 'document_chunks', ['job_id'])
    op.create_index('idx_document_chunks_created_at', 'document_chunks', ['created_at'], postgresql_using='BRIN')
    op.create_index('idx_document_chunks_metadata', 'document_chunks', ['metadata'], postgresql_using='GIN', postgresql_ops={'metadata': 'jsonb_path_ops'})
    
    # Create HNSW index (custom DDL)
    op.execute("""
        CREATE INDEX idx_document_chunks_embedding_hnsw 
        ON document_chunks 
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)
    
    # Create full-text search index
    op.execute("""
        CREATE INDEX idx_document_chunks_content_search 
        ON document_chunks 
        USING gin (to_tsvector('english', content))
    """)
    
    # Create trigram index
    op.execute("""
        CREATE INDEX idx_document_chunks_content_trgm 
        ON document_chunks 
        USING gin (content gin_trgm_ops)
    """)

def downgrade():
    op.drop_table('document_chunks')
```

### Backward Compatibility

This is an **additive feature** with no breaking changes:

1. **New tables only**: No modifications to existing schema
2. **New endpoints only**: Existing API routes unchanged
3. **Opt-in configuration**: Vector store can remain disabled
4. **Optional pipeline stage**: Embedding generation is configurable

### Rollback Procedures

#### Rollback Scenario 1: Performance Issues
```bash
# 1. Disable vector store via configuration
export VECTOR_STORE_ENABLED=false

# 2. Restart application (routes will 503 if accessed)

# 3. Optional: Drop indexes to reclaim resources
psql -c "DROP INDEX idx_document_chunks_embedding_hnsw;"
```

#### Rollback Scenario 2: Migration Failure
```bash
# Downgrade to previous version
alembic downgrade -1

# Or specific revision
alembic downgrade 001

# In case of partial failure, manual cleanup may be needed
psql -c "DROP TABLE IF EXISTS document_chunks CASCADE;"
psql -c "DROP EXTENSION IF EXISTS vector CASCADE;"
```

#### Rollback Scenario 3: Data Corruption
```bash
# Restore from backup (document_chunks is part of PostgreSQL backup)
pg_restore --clean --if-exists backup.dump

# Or selective table restore
pg_restore --table=document_chunks backup.dump
```

## Appendix: API Contract Summary

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/v1/jobs/{job_id}/chunks | List chunks for a job |
| GET | /api/v1/jobs/{job_id}/chunks/{chunk_id} | Get specific chunk |
| POST | /api/v1/search/semantic | Vector similarity search |
| POST | /api/v1/search/text | Full-text search |
| POST | /api/v1/search/hybrid | Combined vector + text search |
| GET | /api/v1/search/similar/{chunk_id} | Find similar chunks |

### Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Validation error |
| 404 | Resource not found |
| 429 | Rate limit exceeded |
| 500 | Internal server error |
| 503 | Vector store disabled |

### Dependencies

- PostgreSQL 14+
- pgvector extension 0.5.0+
- pg_trgm extension
- Redis (for rate limiting)
- OpenAI/Azure OpenAI or compatible embedding endpoint
