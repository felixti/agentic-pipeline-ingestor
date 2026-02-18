# Spec: Configurable Weight Parameters Per Query

## Purpose
Allow API consumers to override default hybrid search weights on a per-query basis, enabling dynamic tuning based on query type, user preferences, or contextual requirements.

## Interface

### API Request Body
```json
{
  "query": "machine learning algorithms",
  "weights": {
    "vector": 0.8,
    "text": 0.2
  },
  "fusion_method": "weighted_sum",
  "top_k": 10,
  "filters": {
    "job_id": "uuid"
  }
}
```

### Alternative Weight Specification
```json
{
  "query": "machine learning algorithms",
  "vector_weight": 0.8,
  "text_weight": 0.2,
  "fusion_method": "weighted_sum",
  "top_k": 10
}
```

### Configuration Schema
```yaml
# config/vector_store.yaml
hybrid_search:
  # Default weights (used when not specified in query)
  default_weights:
    vector: 0.7
    text: 0.3
  
  # Allowable range for per-query weights
  weight_constraints:
    min: 0.0
    max: 1.0
    sum_tolerance: 0.001  # Allowed deviation from 1.0
  
  # Predefined weight profiles for common use cases
  profiles:
    semantic_focus:
      name: "Semantic Focus"
      description: "Prioritizes vector similarity for conceptual queries"
      weights:
        vector: 0.9
        text: 0.1
    
    keyword_focus:
      name: "Keyword Focus"
      description: "Prioritizes text matching for precise term queries"
      weights:
        vector: 0.2
        text: 0.8
    
    balanced:
      name: "Balanced"
      description: "Equal weighting for general search"
      weights:
        vector: 0.5
        text: 0.5
```

## Behavior

### Weight Resolution Priority
1. **Explicit weights in request** - Highest priority
2. **Named profile in request** - `{"profile": "semantic_focus"}`
3. **Default weights from config** - Fallback

### Weight Validation Rules
```python
def validate_weights(weights: dict) -> tuple[bool, str]:
    """Validate weight configuration"""
    
    # Check range
    for name, value in weights.items():
        if not 0.0 <= value <= 1.0:
            return False, f"Weight '{name}' must be between 0.0 and 1.0"
    
    # Check sum (with tolerance)
    total = sum(weights.values())
    tolerance = config.hybrid_search.weight_constraints.sum_tolerance
    
    if abs(total - 1.0) > tolerance:
        return False, f"Weights must sum to 1.0 (±{tolerance}), got {total}"
    
    return True, None
```

### Profile Selection
```json
{
  "query": "neural network architecture",
  "profile": "semantic_focus",
  "fusion_method": "weighted_sum"
}
```

Profiles are resolved server-side and expanded to explicit weights in the response.

### Auto-Weight Suggestion (Future Enhancement)
```json
{
  "query": "Python 3.9 release date",
  "auto_weights": true
}
```

System analyzes query characteristics:
- Presence of version numbers, dates → increase text weight
- Abstract/conceptual terms → increase vector weight
- Named entities → balanced approach

### Response Format
```json
{
  "success": true,
  "data": {
    "results": [...],
    "weights_applied": {
      "vector": 0.8,
      "text": 0.2,
      "source": "explicit",  // or "profile", "default"
      "profile": null        // or "semantic_focus", etc.
    }
  },
  "meta": {
    "total_results": 45,
    "query_time_ms": 45.2
  }
}
```

## Error Handling

| Error Code | Condition | Response |
|------------|-----------|----------|
| 400 | Weight sum != 1.0 (outside tolerance) | `{"error": "Weights must sum to 1.0", "sum": 0.95, "tolerance": 0.001}` |
| 400 | Negative weight | `{"error": "Weight 'vector' cannot be negative"}` |
| 400 | Weight > 1.0 | `{"error": "Weight 'text' cannot exceed 1.0"}` |
| 400 | Unknown profile name | `{"error": "Unknown weight profile 'semantic'", "available_profiles": [...]}` |
| 400 | Both weights and profile specified | `{"error": "Specify either 'weights' object or 'profile', not both"}` |
| 400 | Invalid weight key | `{"error": "Invalid weight key 'semantic'", "valid_keys": ["vector", "text"]}` |

## Performance Considerations

### Validation Overhead
- Weight validation: < 1ms
- Profile lookup: O(1) via dictionary
- No impact on search performance

### Caching
- Cache profile configurations in memory
- Reload on config file change (hot reload)
- No database queries for weight resolution

### Query Plan Stability
- PostgreSQL query plan is independent of weight values
- Same query structure regardless of weights
- Plan caching remains effective

### Rate Limiting Considerations
- Weight changes don't affect rate limiting
- Same endpoint, same cost calculation
- Consider separate limits for complex auto-weight queries

## Usage Examples

### Example 1: Semantic Research Query
```bash
curl -X POST /api/v1/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query": "transformer attention mechanisms",
    "weights": {"vector": 0.85, "text": 0.15},
    "top_k": 20
  }'
```

### Example 2: Precise Technical Lookup
```bash
curl -X POST /api/v1/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query": "PostgreSQL 14 pgvector installation",
    "weights": {"vector": 0.2, "text": 0.8},
    "top_k": 10
  }'
```

### Example 3: Using Profile
```bash
curl -X POST /api/v1/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query": "cloud computing best practices",
    "profile": "balanced",
    "top_k": 15
  }'
```

### Example 4: Fallback to Defaults
```bash
curl -X POST /api/v1/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query": "default search without weight specification"
  }'
# Uses default weights from config (vector: 0.7, text: 0.3)
```
