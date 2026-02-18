# Spec: Rate Limiting Integration

## Purpose
Protect search API endpoints from abuse, ensure fair resource usage, and maintain system stability under load.

## Interface
- **Implementation**: FastAPI middleware + Redis backend
- **Integration**: Applied globally to all `/api/v1/search/*` and `/api/v1/jobs/*/chunks/*` endpoints
- **Configuration**: Per-endpoint limits via configuration file

## Request Schema
N/A - Rate limiting is transparent to API consumers.

## Response Schema

### Normal Response (Within Limit)
HTTP headers included in every response:
```
X-RateLimit-Limit: 30
X-RateLimit-Remaining: 25
X-RateLimit-Reset: 1706745600
X-RateLimit-Retry-After: 0
```

### Rate Limit Exceeded (429 Too Many Requests)
```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Please retry after 45 seconds.",
    "details": {
      "limit": 30,
      "remaining": 0,
      "reset_at": "2024-02-01T00:00:45Z",
      "retry_after_seconds": 45,
      "endpoint": "search_semantic"
    },
    "request_id": "uuid-for-debugging"
  }
}
```

With HTTP headers:
```
Retry-After: 45
X-RateLimit-Limit: 30
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1706745600
```

## Behavior

### Rate Limiting Strategy
1. **Sliding Window**: Uses Redis sorted sets for accurate sliding window counting
2. **Token Bucket**: Alternative mode for burst-heavy workloads
3. **Fixed Window**: Simpler mode for basic protection

### Limit Identification
Rate limit key construction (in priority order):
1. **Authenticated User**: `{endpoint}:{user_id}`
2. **API Key**: `{endpoint}:{api_key_hash}`
3. **IP Address**: `{endpoint}:{client_ip}`
4. **Global**: `{endpoint}:global` (fallback)

### Endpoint-Specific Limits

| Endpoint | Limit | Window | Burst | Key Prefix |
|----------|-------|--------|-------|------------|
| GET /jobs/{id}/chunks | 100/min | 60s | 10 | list_chunks |
| GET /jobs/{id}/chunks/{id} | 200/min | 60s | 20 | get_chunk |
| POST /search/semantic | 30/min | 60s | 5 | search_semantic |
| POST /search/text | 60/min | 60s | 10 | search_text |
| POST /search/hybrid | 20/min | 60s | 3 | search_hybrid |
| GET /search/similar/{id} | 40/min | 60s | 5 | search_similar |

### Burst Handling
- **Burst Allowance**: Immediate capacity for burst requests
- **Refill Rate**: Steady-state rate after burst exhausted
- **Example**: 30/min with burst 5 = 5 immediate, then 1 per 2 seconds

### Whitelisting
Bypass rate limiting for:
- Internal service-to-service calls (via service token)
- Admin users with `search:unlimited` permission
- Health check endpoints

### Header Injection
All responses include rate limit headers:
- `X-RateLimit-Limit`: Maximum requests allowed in window
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Unix timestamp when limit resets
- `X-RateLimit-Retry-After`: Seconds until next request allowed (0 if not limited)

## Error Handling

### 429 Too Many Requests
- Returned when limit exceeded
- Includes `Retry-After` header with seconds to wait
- JSON body with detailed limit information
- Logged at INFO level for monitoring

### Redis Unavailable
- **Behavior**: Fail open (allow request) to prevent cascade failure
- **Logging**: ERROR level log for Redis connection issues
- **Alerting**: Metric incremented for operator visibility
- **Fallback**: In-memory rate limiting (per-process, less accurate)

### Key Collision
- Use Redis key hashing for long identifiers
- Partition keys across Redis clusters if needed

## Configuration

### YAML Configuration
```yaml
rate_limiting:
  enabled: true
  backend: redis  # redis, memory, or none
  redis:
    url: redis://localhost:6379/0
    key_prefix: "rl:"
    
  endpoints:
    list_chunks:
      limit: 100
      window: 60
      burst: 10
    get_chunk:
      limit: 200
      window: 60
      burst: 20
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
    search_similar:
      limit: 40
      window: 60
      burst: 5
  
  # Global limits across all endpoints per user
  global:
    enabled: true
    limit: 500
    window: 60
  
  # IP-based limits for unauthenticated requests
  ip:
    enabled: true
    limit: 100
    window: 60
  
  whitelist:
    ips: ["127.0.0.1", "10.0.0.0/8"]
    api_keys: ["${INTERNAL_API_KEY}"]
```

### Environment Variables
```bash
RATE_LIMITING_ENABLED=true
RATE_LIMITING_REDIS_URL=redis://localhost:6379/0
RATE_LIMITING_SEARCH_SEMANTIC_LIMIT=30
```

## Implementation Notes

### Middleware Structure
```python
class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Identify client (user_id, api_key, or IP)
        # Check Redis for current count
        # If within limit: increment, add headers, proceed
        # If exceeded: return 429 with retry info
```

### Redis Data Structure
- **Key**: `rl:{endpoint}:{identifier}`
- **Type**: Sorted set (sliding window)
- **Score**: Request timestamp (milliseconds)
- **Member**: Unique request ID
- **TTL**: Window duration + buffer

### Monitoring Metrics
- `rate_limit_hits_total`: Counter of 429 responses by endpoint
- `rate_limit_current`: Gauge of current request rate by endpoint
- `rate_limit_redis_errors`: Counter of Redis failures

## Testing

### Load Testing Thresholds
- Verify limits enforced under 1000 req/s sustained load
- Confirm accurate sliding window (no over-limit edge cases)
- Test Redis failover behavior

### Security Testing
- Confirm IP spoofing not possible (use X-Forwarded-For correctly)
- Verify key collision handling
- Test bypass prevention (no client-side rate limit manipulation)
