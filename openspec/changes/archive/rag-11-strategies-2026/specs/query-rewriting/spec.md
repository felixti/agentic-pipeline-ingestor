# Spec: Query Rewriting

## Overview
Implement LLM-based query rewriting to separate user intent into optimized prompts for vector search and LLM generation.

## Requirements

### Functional Requirements
1. Extract core search topics from verbose user queries
2. Generate structured JSON output with separated fields
3. Remove noise words ("search", "summarize", "list") from embedding queries
4. Create clear instruction prompts for LLM generation
5. Support conversational context and follow-up queries

### Input/Output Format
```json
{
  "search_rag": true,
  "embedding_source_text": "Core topic keywords only",
  "llm_query": "Clear instruction for the LLM"
}
```

### Example Transformations

**Input**: `"@knowledgebase search on vibe coding, then summarize, list pros and cons"`

**Output**:
```json
{
  "search_rag": true,
  "embedding_source_text": "vibe coding programming approach",
  "llm_query": "Based on the provided context, explain what vibe coding is, including its pros and cons, and cite sources."
}
```

## API Design

```python
class QueryRewriter:
    async def rewrite(self, query: str, context: dict = None) -> QueryRewriteResult:
        """
        Rewrite user query for optimal retrieval and generation.
        
        Args:
            query: Original user query
            context: Optional conversation context
            
        Returns:
            QueryRewriteResult with search and generation components
        """
        pass
```

## Configuration
```yaml
query_rewriting:
  enabled: true
  model: "gpt-4o-mini"  # Fast, cost-effective
  system_prompt: |
    You must respond with a JSON object containing exactly these fields:
    - "search_rag": boolean - True if query contains "@knowledgebase"
    - "embedding_source_text": string - Core topic keywords only
    - "llm_query": string - Clear instruction for the LLM
  
  # Performance
  cache_ttl: 3600
  max_retries: 2
  timeout_ms: 2000
```

## Acceptance Criteria
- [ ] Successfully transforms verbose queries into clean search terms
- [ ] JSON output is valid and schema-compliant
- [ ] Processing time < 100ms for simple queries
- [ ] Handles edge cases (empty queries, very long queries)
- [ ] Cache hit rate > 60% for repeated queries

## Dependencies
- LLM adapter (existing)
- JSON schema validation
- Caching layer
