# Spec: HyDE (Hypothetical Document Embeddings)

## Overview
Implement Hypothetical Document Embeddings to generate synthetic documents for semantic search, improving retrieval quality beyond keyword matching.

## Requirements

### Functional Requirements
1. Generate hypothetical documents that answer the user's query
2. Use hypothetical document embedding for vector search instead of raw query
3. Find conceptually related documents that don't contain exact keywords
4. Support configurable toggle per query type
5. Cache hypothetical documents for reuse

### How HyDE Works
1. User asks: `"What is vibe coding?"`
2. LLM generates hypothetical answer:
   `"Vibe coding is a programming approach emphasizing intuition and flow state..."`
3. Embed the hypothetical answer
4. Search for documents similar to the hypothetical answer
5. Return results to LLM for final generation

### Example
```json
{
  "search_rag": true,
  "embedding_source_text": "Vibe coding is a programming approach that emphasizes writing code based on intuition, flow state, and personal rhythm rather than strict methodologies. This coding style prioritizes developer comfort and creativity.",
  "llm_query": "Based on the provided context, explain what vibe coding is, including its pros and cons."
}
```

## API Design

```python
class HyDERewriter(QueryRewriter):
    async def generate_hypothetical_document(
        self, 
        query: str,
        context: dict = None
    ) -> str:
        """Generate a hypothetical document that answers the query."""
        pass
    
    async def rewrite(self, query: str, context: dict = None) -> QueryRewriteResult:
        """Override to use HyDE strategy."""
        hypothetical = await self.generate_hypothetical_document(query, context)
        return QueryRewriteResult(
            search_rag=True,
            embedding_source_text=hypothetical,
            llm_query=self._create_llm_prompt(query)
        )
```

## Configuration
```yaml
hyde:
  enabled: true
  model: "gpt-4o"  # Needs strong generative capability
  max_hypothetical_length: 512
  
  system_prompt: |
    Generate a focused 2-4 sentence hypothetical document that answers the 
    user's question. This will be used for semantic search, so include key 
    concepts and terminology that would appear in relevant documents.
  
  # Memory optimization
  enable_for_query_types:
    - complex_questions
    - vague_queries
    - out_of_domain
  
  cache:
    enabled: true
    ttl: 7200
```

## Acceptance Criteria
- [ ] Hypothetical documents are semantically rich and relevant
- [ ] Retrieval scores improve by >20% over keyword matching
- [ ] Memory usage optimized (32x efficiency claim validated)
- [ ] Processing time < 500ms including generation
- [ ] Graceful fallback to standard query if HyDE fails

## Performance Expectations
- Vector similarity scores significantly higher than keyword approach
- Better retrieval for vague or out-of-domain queries
- Improved document relevance ranking

## Dependencies
- Query Rewriting service
- LLM with strong generative capabilities
- Vector embedding service
