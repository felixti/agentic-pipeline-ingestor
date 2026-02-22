# Spec: Hybrid Search Enhancement

## Overview
Enhance existing hybrid search with fusion ranking and metadata filtering to combine semantic and lexical matching effectively.

## Requirements

### Functional Requirements
1. Combine vector similarity scores with full-text BM25 scores
2. Apply Reciprocal Rank Fusion (RRF) for score combination
3. Support metadata filtering (date, source, document type)
4. Weight tuning for vector vs. text components
5. Query expansion for better lexical matching

### Score Fusion Formula (RRF)
```
RRF_score(d) = Σ 1 / (k + rank_i(d))

Where:
- k = constant (typically 60)
- rank_i(d) = rank of document d in result list i
```

## API Design

```python
class HybridSearchService:
    async def search(
        self,
        query: str,
        embedding: list[float],
        filters: dict = None,
        limit: int = 10,
        vector_weight: float = 0.7,
        text_weight: float = 0.3
    ) -> list[SearchResult]:
        """
        Perform hybrid search with fusion ranking.
        
        Args:
            query: Text query for full-text search
            embedding: Vector embedding for similarity search
            filters: Metadata filters
            limit: Number of results to return
            vector_weight: Weight for vector scores
            text_weight: Weight for text scores
            
        Returns:
            Fused and ranked search results
        """
        # Get vector results
        vector_results = await self.vector_search(embedding, filters, limit * 2)
        
        # Get text results
        text_results = await self.text_search(query, filters, limit * 2)
        
        # Fuse with RRF
        fused = self.reciprocal_rank_fusion(
            vector_results, 
            text_results,
            vector_weight,
            text_weight
        )
        
        return fused[:limit]
    
    def reciprocal_rank_fusion(
        self,
        vector_results: list[SearchResult],
        text_results: list[SearchResult],
        vector_weight: float,
        text_weight: float,
        k: int = 60
    ) -> list[SearchResult]:
        """Apply RRF to combine result lists."""
        scores = defaultdict(float)
        
        # Score from vector results
        for rank, result in enumerate(vector_results):
            scores[result.id] += vector_weight / (k + rank)
        
        # Score from text results
        for rank, result in enumerate(text_results):
            scores[result.id] += text_weight / (k + rank)
        
        # Sort by fused score
        return sorted(
            [(id, score) for id, score in scores.items()],
            key=lambda x: x[1],
            reverse=True
        )
```

## Configuration
```yaml
hybrid_search:
  enabled: true
  
  # Fusion parameters
  rrf_k: 60
  
  # Default weights (can be overridden per query)
  default_weights:
    vector: 0.7
    text: 0.3
  
  # Weight presets
  presets:
    semantic_focus:
      vector: 0.9
      text: 0.1
    balanced:
      vector: 0.7
      text: 0.3
    lexical_focus:
      vector: 0.3
      text: 0.7
  
  # Metadata filtering
  filterable_fields:
    - source_type
    - document_type
    - created_date
    - author
    - tags
  
  # Query expansion
  query_expansion:
    enabled: true
    synonym_expansion: true
    max_expanded_terms: 5
```

## Acceptance Criteria
- [ ] Fusion ranking combines scores correctly
- [ ] Metadata filtering works for all defined fields
- [ ] Weight presets affect ranking appropriately
- [ ] Query expansion improves lexical recall
- [ ] Performance: <100ms for combined search

## Performance Expectations
| Metric | Target |
|--------|--------|
| Recall@10 | >0.85 |
| Precision@5 | >0.75 |
| Latency (p95) | <150ms |
| Throughput | 200 req/s |

## Dependencies
- pgvector extension
- PostgreSQL full-text search
- Existing VectorSearchService
- Existing TextSearchService
