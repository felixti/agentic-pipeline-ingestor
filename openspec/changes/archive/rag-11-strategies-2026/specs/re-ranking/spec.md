# Spec: Re-ranking

## Overview
Implement a secondary scoring model using cross-encoders to re-rank retrieved chunks for significantly improved relevance.

## Requirements

### Functional Requirements
1. Accept initial retrieval results (top-k from vector search)
2. Score each chunk using cross-encoder model
3. Reorder results by cross-encoder relevance score
4. Return top-n most relevant chunks
5. Support different re-ranking models per use case

### Why Re-ranking Matters
- **Bi-encoders** (used in initial retrieval) encode query and document separately
- **Cross-encoders** encode query + document together, capturing finer relevance signals
- Can improve relevance by 15-30% over vector-only retrieval

## API Design

```python
class ReRanker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
    
    async def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int = 5
    ) -> list[RankedChunk]:
        """
        Re-rank chunks by relevance to query.
        
        Args:
            query: User query
            chunks: Initial retrieval results
            top_k: Number of results to return
            
        Returns:
            Chunks reordered by cross-encoder score
        """
        pairs = [(query, chunk.content) for chunk in chunks]
        scores = self.model.predict(pairs)
        
        ranked = [
            RankedChunk(chunk=chunk, score=float(score))
            for chunk, score in zip(chunks, scores)
        ]
        ranked.sort(key=lambda x: x.score, reverse=True)
        
        return ranked[:top_k]
```

## Configuration
```yaml
reranking:
  enabled: true
  
  models:
    default: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    high_precision: "cross-encoder/ms-marco-electra-base"
    fast: "cross-encoder/ms-marco-TinyBERT-L-2"
  
  # Performance tuning
  initial_retrieval_k: 20  # Get more for re-ranking
  final_k: 5  # Return top 5 after re-ranking
  batch_size: 8
  
  # Async processing for latency
  async_mode: true
  timeout_ms: 500
```

## Supported Models
1. `cross-encoder/ms-marco-MiniLM-L-6-v2` - Balanced speed/quality
2. `cross-encoder/ms-marco-electra-base` - Higher precision, slower
3. `cross-encoder/ms-marco-TinyBERT-L-2` - Fast, lower precision
4. `BAAI/bge-reranker-base` - Alternative architecture

## Acceptance Criteria
- [ ] Re-ranking improves relevance scores by >15%
- [ ] Processing time < 200ms for 20 chunks
- [ ] Supports multiple re-ranking models
- [ ] Graceful degradation if model unavailable
- [ ] Cache support for common query-chunk pairs

## Performance Expectations
| Metric | Target |
|--------|--------|
| NDCG@5 improvement | +15% |
| Latency (p95) | <200ms |
| Memory usage | <500MB |
| Throughput | 100 req/s |

## Dependencies
- sentence-transformers library
- Cross-encoder models (downloadable)
- Initial retrieval service
