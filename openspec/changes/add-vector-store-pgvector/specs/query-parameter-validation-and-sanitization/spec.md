# Spec: Query Parameter Validation and Sanitization

## Purpose
Validate and sanitize all user inputs to prevent SQL injection, XSS attacks, and ensure type safety across all search API endpoints.

## Interface
- **Implementation**: Pydantic validators and custom sanitization functions
- **Integration**: Applied to all `/api/v1/search/*` and `/api/v1/jobs/*/chunks/*` endpoints
- **Middleware**: Global request validation layer

## Request Schema
N/A - This is a cross-cutting capability applied to all endpoints.

### Validation Rules by Parameter Type

#### UUID Parameters (`job_id`, `chunk_id`)
```python
pattern: r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
behavior: Case-insensitive validation, strict format checking
```

#### Integer Parameters (`limit`, `offset`, `top_k`)
```python
limit: min=1, max=100, default=20
top_k: min=1, max=100, default=10
offset: min=0, max=100000, default=0
```

#### String Parameters (`query`, `query_text`)
```python
max_length: 4096 (semantic search), 1024 (text search)
strip_whitespace: true
sanitize_html: true  # Remove/escape HTML tags
prevent_sql_injection: true  # Block SQL keywords/patterns
```

#### Number Parameters (`similarity_threshold`, `vector_weight`, `text_weight`)
```python
similarity_threshold: min=0.0, max=1.0, default=0.7
vector_weight: min=0.0, max=1.0, default=0.7
text_weight: min=0.0, max=1.0, default=0.3
```

#### Enum Parameters (`fusion_method`, `language`, `sort_by`)
```python
fusion_method: Literal['weighted_sum', 'rrf']
language: Literal['english', 'spanish', 'french', 'german', 'portuguese', 'simple']
sort_by: Literal['created_at', 'chunk_index', 'updated_at']
```

#### Metadata Filter Object
```python
job_id: Optional[UUID] - validated as UUID
source_file: Optional[str] - max 512 chars, sanitized
date_from/date_to: Optional[datetime] - ISO 8601 format
custom_fields: Optional[dict] - max 10 keys, key names sanitized
```

## Response Schema
**400 Bad Request - Validation Error**
```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": [
      {
        "field": "query_text",
        "error": "Field required",
        "type": "missing"
      },
      {
        "field": "limit",
        "error": "Must be less than or equal to 100",
        "type": "range_error",
        "provided": 500,
        "maximum": 100
      },
      {
        "field": "job_id",
        "error": "Invalid UUID format",
        "type": "format_error"
      }
    ],
    "request_id": "uuid-for-debugging"
  }
}
```

## Behavior

### 1. Type Validation
- Use Pydantic v2 models for strict type checking
- Coerce types where safe (e.g., string "10" → integer 10 for limit)
- Reject invalid types with descriptive error messages

### 2. SQL Injection Prevention
- **Parameterized Queries**: All database queries use SQLAlchemy parameterized statements
- **Input Sanitization**: Block dangerous patterns:
  - SQL keywords: `DROP`, `DELETE`, `UPDATE`, `INSERT`, `EXEC`, `UNION`
  - Comment sequences: `--`, `/*`, `*/`
  - Semicolons in suspicious contexts
- **Allowlist Approach**: Only allow expected characters for each field type

### 3. XSS Prevention
- **HTML Escaping**: Escape `<`, `>`, `&`, `"`, `'` characters in string inputs
- **Content Security Policy**: API returns JSON only, no HTML rendering
- **Output Encoding**: All response strings are JSON-encoded

### 4. Metadata Filter Sanitization
- **Key Validation**: Metadata filter keys must match allowlist pattern: `^[a-zA-Z_][a-zA-Z0-9_]*$`
- **Value Types**: Restrict to string, number, boolean, null
- **Nesting Depth**: Maximum 3 levels of nesting in custom_fields
- **Size Limits**: Maximum 10KB for entire metadata_filter object

### 5. Logging
- Log validation failures at WARNING level
- Include sanitized field names (not values) in logs
- Track repeated validation failures for security monitoring

## Error Handling

### Validation Error Response Format
All validation errors return HTTP 400 with standardized error structure:
- `code`: Machine-readable error code
- `message`: Human-readable summary
- `details`: Array of specific field errors
- `request_id`: Unique ID for debugging

### Security-Related Blocks
When SQL injection or XSS patterns detected:
- Return 400 error (not 403 to avoid revealing security logic)
- Log incident with client IP and request details
- Increment security metric counter

### Edge Cases
- **Empty strings**: Treated as missing for required fields
- **Unicode**: Full Unicode support with NFKC normalization
- **Null/None**: Explicitly handled based on field requirements
- **Array inputs**: Validated element-by-element

## Rate Limiting
- **Validation failures**: Tracked separately from successful requests
- **Threshold**: 50 validation failures per minute triggers temporary IP block
- **Purpose**: Prevent automated probing attacks

## Implementation Notes

### Pydantic Model Structure
```python
from pydantic import BaseModel, Field, validator, constr

class SemanticSearchRequest(BaseModel):
    query_text: constr(min_length=1, max_length=4096, strip_whitespace=True)
    top_k: int = Field(default=10, ge=1, le=100)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    metadata_filter: Optional[MetadataFilter] = None
    
    @validator('query_text')
    def sanitize_query(cls, v):
        # Remove HTML tags
        # Check for SQL injection patterns
        return sanitize_input(v)
```

### Custom Validators
- UUID validator with strict format checking
- Datetime validator accepting ISO 8601 format only
- Nested object validator with depth limiting
