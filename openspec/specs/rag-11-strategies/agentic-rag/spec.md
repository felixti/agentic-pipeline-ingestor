# Spec: Agentic RAG

## Overview
Enhance the existing agentic processing with AI-driven decision making for routing queries, selecting retrieval strategies, and optimizing the RAG pipeline dynamically.

## Requirements

### Functional Requirements
1. Classify queries by type (factual, analytical, comparative, vague)
2. Select optimal RAG strategy combination per query
3. Route to specialized retrieval paths
4. Decide when to use HyDE, re-ranking, or hybrid search
5. Self-correct based on retrieval quality metrics
6. Multi-step reasoning for complex queries

### Query Classification
```python
class QueryType(Enum):
    FACTUAL = "factual"          # Simple lookup
    ANALYTICAL = "analytical"    # Requires synthesis
    COMPARATIVE = "comparative"  # Compare multiple items
    VAGUE = "vague"             # Needs clarification
    MULTI_HOP = "multi_hop"     # Multiple retrieval steps
```

### Strategy Selection Matrix
| Query Type | HyDE | Re-ranking | Hybrid | Query Rewrite |
|------------|------|------------|--------|---------------|
| Factual    | No   | Yes        | Yes    | Yes           |
| Analytical | Yes  | Yes        | Yes    | Yes           |
| Comparative| No   | Yes        | Yes    | Yes           |
| Vague      | Yes  | Yes        | Yes    | Yes           |
| Multi-hop  | Yes  | Yes        | Yes    | Yes           |

## API Design

```python
class AgenticRAG:
    async def process(self, query: str, context: dict = None) -> RAGResult:
        """
        Process query through agentic RAG pipeline.
        
        Flow:
        1. Classify query type
        2. Select strategies
        3. Execute retrieval pipeline
        4. Evaluate quality
        5. Self-correct if needed
        6. Generate final answer
        """
        # Step 1: Classify
        query_type = await self.classify_query(query)
        
        # Step 2: Select strategies
        config = self.select_strategy(query_type)
        
        # Step 3: Execute with selected strategies
        chunks = await self.retrieve(query, config)
        
        # Step 4: Evaluate
        quality = await self.evaluate_quality(query, chunks)
        
        # Step 5: Self-correct if needed
        if quality.score < 0.7:
            chunks = await self.self_correct(query, chunks, quality)
        
        # Step 6: Generate answer
        return await self.generate(query, chunks)
    
    async def classify_query(self, query: str) -> QueryType:
        """Classify query to determine optimal strategy."""
        pass
    
    async def select_strategy(self, query_type: QueryType) -> RAGConfig:
        """Select RAG configuration based on query type."""
        pass
```

## Configuration
```yaml
agentic_rag:
  enabled: true
  
  # Classification model
  classifier_model: "gpt-4o-mini"
  
  # Quality thresholds
  quality_threshold: 0.7
  max_iterations: 3
  
  # Strategy presets
  strategies:
    fast:
      hyde: false
      reranking: false
      hybrid: true
      query_rewrite: true
    
    balanced:
      hyde: false
      reranking: true
      hybrid: true
      query_rewrite: true
    
    thorough:
      hyde: true
      reranking: true
      hybrid: true
      query_rewrite: true
    
    auto:
      # Dynamically selected per query
      selection_model: "gpt-4o-mini"
  
  # Self-correction
  self_correction:
    enabled: true
    strategies:
      - expand_query
      - use_hyde
      - increase_retrieval_k
      - switch_to_hybrid
```

## Acceptance Criteria
- [ ] Correctly classifies >90% of query types
- [ ] Strategy selection improves overall accuracy by >10%
- [ ] Self-correction triggers appropriately (<30% of queries)
- [ ] Multi-hop queries handled correctly
- [ ] Latency overhead < 100ms

## Performance Expectations
| Query Type | Avg Latency | Accuracy Target |
|------------|-------------|-----------------|
| Factual    | 200ms       | 95%             |
| Analytical | 500ms       | 90%             |
| Vague      | 600ms       | 85%             |
| Multi-hop  | 800ms       | 85%             |

## Dependencies
- Query Rewriting service
- HyDE service
- Re-ranking service
- Hybrid Search service
- LLM for classification and self-correction
