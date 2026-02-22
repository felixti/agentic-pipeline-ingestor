# Spec: Intelligent Caching

## Overview
Implement multi-layer caching for embeddings, query results, and LLM responses to reduce API costs and improve response times.

## Requirements

### Functional Requirements
1. Embedding cache to avoid recomputing embeddings
2. Query result cache for repeated queries
3. LLM response cache for common questions
4. Semantic cache for similar queries
5. Cache invalidation strategies

### Cache Layers

```
┌─────────────────────────────────────────────────────────┐
│                    CACHE HIERARCHY                       │
├─────────────────────────────────────────────────────────┤
│  L1: In-Memory (Redis)                                   │
│      - Hot embeddings                                    │
│      - Recent query results                              │
│      - TTL: 1 hour                                       │
├─────────────────────────────────────────────────────────┤
│  L2: Persistent (PostgreSQL)                             │
│      - All embeddings                                    │
│      - Query history                                     │
│      - LLM responses                                     │
│      - TTL: 24 hours                                     │
├─────────────────────────────────────────────────────────┤
│  L3: Semantic (Vector Similarity)                        │
│      - Similar query matching                            │
│      - Threshold-based retrieval                         │
│      - TTL: 7 days                                       │
└─────────────────────────────────────────────────────────┘
```

## API Design

```python
class MultiLayerCache:
    def __init__(self):
        self.l1_cache = RedisCache(ttl=3600)
        self.l2_cache = PostgresCache(ttl=86400)
        self.l3_cache = SemanticCache(ttl=604800)
    
    async def get_embedding(self, text: str, model: str) -> Optional[list[float]]:
        """Get embedding from cache hierarchy."""
        cache_key = f"emb:{model}:{hash(text)}"
        
        # L1: Redis
        if embedding := await self.l1_cache.get(cache_key):
            return embedding
        
        # L2: PostgreSQL
        if embedding := await self.l2_cache.get_embedding(text, model):
            await self.l1_cache.set(cache_key, embedding)
            return embedding
        
        return None
    
    async def set_embedding(self, text: str, model: str, embedding: list[float]):
        """Store embedding in all cache layers."""
        cache_key = f"emb:{model}:{hash(text)}"
        
        await asyncio.gather(
            self.l1_cache.set(cache_key, embedding),
            self.l2_cache.set_embedding(text, model, embedding)
        )
    
    async def get_query_result(
        self,
        query: str,
        similarity_threshold: float = 0.95
    ) -> Optional[QueryResult]:
        """Get cached result, including semantic matches."""
        # Exact match
        cache_key = f"query:{hash(query)}"
        if result := await self.l1_cache.get(cache_key):
            return result
        
        # Semantic match
        if result := await self.l3_cache.find_similar(query, similarity_threshold):
            return result
        
        return None

class SemanticCache:
    """Cache based on semantic similarity."""
    
    def __init__(self, embedding_service: EmbeddingService):
        self.embeddings = embedding_service
        self.vector_store = VectorStore()
    
    async def find_similar(
        self,
        query: str,
        threshold: float = 0.95
    ) -> Optional[QueryResult]:
        """Find cached result for semantically similar query."""
        query_emb = await self.embeddings.embed(query)
        
        # Search cache vector store
        results = await self.vector_store.similarity_search(
            query_emb,
            top_k=1,
            threshold=threshold
        )
        
        if results:
            return await self.load_result(results[0].id)
        
        return None
    
    async def store(self, query: str, result: QueryResult):
        """Store query-result pair in semantic cache."""
        query_emb = await self.embeddings.embed(query)
        
        await self.vector_store.store(
            id=result.id,
            embedding=query_emb,
            metadata={
                "query": query,
                "result_id": result.id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
```

## Configuration
```yaml
caching:
  enabled: true
  
  layers:
    l1_redis:
      enabled: true
      ttl: 3600  # 1 hour
      max_size: 10000  # entries
      
    l2_postgres:
      enabled: true
      ttl: 86400  # 24 hours
      
    l3_semantic:
      enabled: true
      ttl: 604800  # 7 days
      similarity_threshold: 0.95
  
  # What to cache
  cache_targets:
    embeddings:
      enabled: true
      ttl: 86400
    
    query_results:
      enabled: true
      ttl: 3600
      min_query_length: 10
    
    llm_responses:
      enabled: true
      ttl: 7200
      cacheable_patterns:
        - "what is"
        - "how to"
        - "explain"
    
    reranking_scores:
      enabled: true
      ttl: 1800
  
  # Invalidation
  invalidation:
    strategies:
      - ttl_based
      - manual_flush
      - document_update
    
    # Flush cache when documents updated
    flush_on_document_update: true
  
  # Monitoring
  monitoring:
    track_hit_rates: true
    alert_on_low_hit_rate:
      threshold: 0.5
      window: 1h
```

## Database Schema
```sql
-- Embedding cache
CREATE TABLE embedding_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    text_hash VARCHAR(64) UNIQUE,
    text_preview VARCHAR(200),
    model VARCHAR(100),
    embedding VECTOR(1536),
    created_at TIMESTAMP DEFAULT NOW(),
    accessed_at TIMESTAMP DEFAULT NOW(),
    access_count INTEGER DEFAULT 1
);

-- Query result cache
CREATE TABLE query_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_hash VARCHAR(64) UNIQUE,
    query_text TEXT,
    query_embedding VECTOR(1536),
    result_json JSONB,
    strategy_config JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    ttl_seconds INTEGER DEFAULT 3600
);

-- LLM response cache
CREATE TABLE llm_response_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prompt_hash VARCHAR(64),
    prompt_preview TEXT,
    model VARCHAR(100),
    response TEXT,
    tokens_used INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_emb_cache_model ON embedding_cache(model);
CREATE INDEX idx_emb_cache_hash ON embedding_cache(text_hash);
CREATE INDEX idx_query_cache_hash ON query_cache(query_hash);
CREATE INDEX idx_llm_cache_hash ON llm_response_cache(prompt_hash);
```

## Acceptance Criteria
- [ ] All 3 cache layers implemented
- [ ] Semantic cache finds similar queries
- [ ] Cache invalidation works correctly
- [ ] Hit rates tracked and reported
- [ ] Latency reduction >50% for cached queries

## Performance Expectations
| Cache Layer | Latency | Hit Rate Target |
|-------------|---------|-----------------|
| L1 (Redis)  | <5ms    | 30%             |
| L2 (Postgres)| <50ms  | 50%             |
| L3 (Semantic)| <100ms | 20%             |
| **Overall** | **<5ms avg** | **70%**     |

## Cost Savings
| Component | Cost Reduction |
|-----------|----------------|
| Embeddings | 60-80% |
| LLM calls | 40-60% |
| Total API costs | 50-70% |

## Dependencies
- Redis (L1 cache)
- PostgreSQL (L2 cache)
- pgvector (L3 semantic cache)
- Embedding service
