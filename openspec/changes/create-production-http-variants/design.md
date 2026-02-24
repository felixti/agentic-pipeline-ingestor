# Design: Production HTTP File Variants

## Overview

Create production-ready HTTP client files that mirror the local development versions but target the production deployment.

## Directory Structure

```
http/
├── all-searches.http          # Local development
├── hybrid.http                # Local development
├── semantic.http              # Local development
├── similar.http               # Local development
├── text.http                  # Local development
└── production/                # NEW: Production variants
    ├── all-searches.http
    ├── hybrid.http
    ├── semantic.http
    ├── similar.http
    └── text.http
```

## Configuration Changes

### Local Development (`http/*.http`)
```http
@baseUrl = http://localhost:8000/api/v1
@apiKey = your-api-key
```

### Production (`http/production/*.http`)
```http
@baseUrl = https://ag-dt-ppl-api.felixtek.cloud/api/v1
@apiKey = production-api-key
```

## File Modifications

Each production HTTP file should:

1. **Update header comment** to indicate production context
2. **Change baseUrl** to production HTTPS endpoint
3. **Update API key placeholder** to indicate production key
4. **Add warning comments** about production usage
5. **Keep all request examples identical** to local versions

## Example: Production Header

```http
### RAG Search API - All Search Types (PRODUCTION)
# ⚠️  WARNING: These requests target the PRODUCTION environment
# Production URL: https://ag-dt-ppl-api.felixtek.cloud/api/v1
# Use with caution - modifications affect live data

@baseUrl = https://ag-dt-ppl-api.felixtek.cloud/api/v1
@apiKey = your-production-api-key
```

## Security Considerations

- Production files should not contain actual API keys
- Use descriptive placeholders like `your-production-api-key`
- Include warning comments about production usage
- Document that HTTPS is required for production
