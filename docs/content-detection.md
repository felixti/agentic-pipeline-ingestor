# Content Detection API Documentation

## Overview

The Content Detection API automatically analyzes PDF documents to determine their content type:

- **text_based**: Documents with extractable text layers (optimal for Docling)
- **scanned**: Image-based documents requiring OCR (optimal for Azure OCR)
- **mixed**: Documents with both text and significant image content

## Use Cases

### 1. Single File Detection

Detect content type of a single PDF file:

```bash
curl -X POST http://localhost:8000/api/v1/detect \
  -H "X-API-Key: your-api-key" \
  -F "file=@document.pdf" \
  -F "detailed=true"
```

**Response:**
```json
{
  "data": {
    "content_type": "text_based",
    "confidence": 0.985,
    "recommended_parser": "docling",
    "alternative_parsers": ["azure_ocr"],
    "text_statistics": {
      "total_pages": 10,
      "total_characters": 45000,
      "has_text_layer": true,
      "font_count": 3
    },
    "image_statistics": {
      "total_images": 2,
      "image_area_ratio": 0.02
    },
    "processing_time_ms": 245
  },
  "meta": {
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "timestamp": "2026-02-17T14:30:00Z",
    "api_version": "v1"
  }
}
```

### 2. URL-Based Detection

Detect content of a PDF hosted at a URL:

```bash
curl -X POST http://localhost:8000/api/v1/detect/url \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "url": "https://example.com/document.pdf",
    "headers": {
      "Authorization": "Bearer token"
    },
    "detailed": true
  }'
```

### 3. Batch Detection

Process multiple files efficiently (up to 10):

```bash
curl -X POST http://localhost:8000/api/v1/detect/batch \
  -H "X-API-Key: your-api-key" \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.pdf" \
  -F "files=@doc3.pdf"
```

**Response:**
```json
{
  "data": {
    "results": [
      {
        "filename": "doc1.pdf",
        "content_type": "text_based",
        "confidence": 0.98,
        "recommended_parser": "docling",
        "processing_time_ms": 120
      },
      {
        "filename": "doc2.pdf",
        "content_type": "scanned",
        "confidence": 0.95,
        "recommended_parser": "azure_ocr",
        "processing_time_ms": 450
      }
    ],
    "summary": {
      "total": 2,
      "successful": 2,
      "errors": 0,
      "text_based": 1,
      "scanned": 1,
      "mixed": 0
    }
  },
  "meta": {
    "request_id": "...",
    "timestamp": "...",
    "api_version": "v1"
  }
}
```

## Content Types

### Text-Based Documents

Documents with embedded text layers:
- Research papers
- Reports
- Books
- Digital documents

**Characteristics:**
- Text ratio > 95%
- Has extractable text layer
- Fast processing with Docling

### Scanned Documents

Image-based documents requiring OCR:
- Scanned invoices
- Signed contracts
- Historical documents
- Image-only PDFs

**Characteristics:**
- Text ratio < 5%
- Image ratio > 90%
- Requires Azure OCR

### Mixed Content

Documents with both text and images:
- Brochures
- Presentations
- Textbooks with diagrams
- Marketing materials

**Characteristics:**
- Mixed text/image ratio
- May require both parsers
- Conservative fallback strategy

## Parser Selection Strategy

### Automatic Selection

The system automatically selects the optimal parser:

| Content Type | Primary Parser | Fallback Parser |
|--------------|----------------|-----------------|
| text_based | Docling | Azure OCR |
| scanned | Azure OCR | Docling |
| mixed | Docling | Azure OCR |

### Low Confidence Handling

When detection confidence < 0.70:
- Uses conservative strategy (both parsers)
- Prioritizes quality over speed
- Logs for manual review

### Explicit Override

Override automatic selection via pipeline config:

```json
{
  "parser": {
    "primary": "azure_ocr",
    "force_ocr": true
  }
}
```

## Performance

### Response Times

| Operation | Typical | Maximum |
|-----------|---------|---------|
| Single detection | 200-500ms | 2s |
| URL detection | 1-3s | 10s |
| Batch (10 files) | 1-3s | 10s |

### Rate Limits

- **60 requests/minute** per API key
- Batch counts as 1 request
- Returns `429 Too Many Requests` when exceeded

### Caching

Results are cached for 30 days based on file hash (SHA-256):
- Cache hit: < 10ms response
- No re-analysis for identical files
- Automatic cache expiration

## Error Handling

### Common Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `INVALID_FILE_FORMAT` | Not a valid PDF | 400 |
| `CORRUPTED_PDF` | PDF structure error | 400 |
| `PROTECTED_PDF` | Password protected | 400 |
| `RATE_LIMIT_EXCEEDED` | Too many requests | 429 |
| `ANALYSIS_ERROR` | Internal error | 500 |

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_FILE_FORMAT",
    "message": "File is not a valid PDF",
    "details": [
      {
        "field": "file",
        "issue": "missing_pdf_header"
      }
    ]
  },
  "meta": {
    "request_id": "...",
    "timestamp": "..."
  }
}
```

## Best Practices

### For Batch Processing

1. **Group similar files**: Batch text-based docs together
2. **Limit batch size**: Max 10 files per request
3. **Use detailed=false**: Unless you need per-page analysis
4. **Handle errors gracefully**: Some files may fail

### For URL Downloads

1. **Set reasonable timeouts**: Download may take time
2. **Include auth headers**: For protected URLs
3. **Verify URL accessibility**: Test before production
4. **Handle redirects**: API follows redirects automatically

### For Caching

1. **Reuse file hashes**: Check cache before re-upload
2. **Don't skip_cache unnecessarily**: Unless file changed
3. **Monitor cache hit ratio**: Should be > 90%

## Integration with Pipeline

Content detection is integrated into the 7-stage pipeline:

```
Stage 1: Ingest → File validation
Stage 2: Detect → Content detection (NEW)
Stage 3: Parse → Parser selection based on detection
Stage 4: Enrich → Metadata extraction
Stage 5: Quality → Validation
Stage 6: Transform → Format conversion
Stage 7: Output → Destination routing
```

### Pipeline Configuration

```python
config = {
    "parser": {
        "primary": "docling",      # Or "azure_ocr"
        "fallback": "azure_ocr",   # Optional
        "force_ocr": False         # Override detection
    }
}
```

## Monitoring

### Metrics

Available at `/metrics` endpoint:

- `content_detection_total` - Total detections by content type
- `content_detection_duration_seconds` - Detection latency
- `content_detection_cache_hit_ratio` - Cache efficiency
- `content_detection_errors_total` - Error counts

### Dashboards

Grafana dashboard available at:
- `config/grafana/dashboards/content-detection.json`

Shows:
- Detection volume
- Content type distribution
- Cache performance
- Error rates

## SDK Example (Python)

```python
import requests

def detect_pdf(file_path: str) -> dict:
    """Detect content type of PDF."""
    with open(file_path, 'rb') as f:
        response = requests.post(
            'http://localhost:8000/api/v1/detect',
            headers={'X-API-Key': 'your-key'},
            files={'file': f},
            data={'detailed': 'false'}
        )
    
    response.raise_for_status()
    return response.json()['data']

# Usage
result = detect_pdf('document.pdf')
print(f"Content type: {result['content_type']}")
print(f"Confidence: {result['confidence']}")
print(f"Recommended parser: {result['recommended_parser']}")
```

## Support

For issues or questions:
- API documentation: `/docs` endpoint
- OpenAPI spec: `/api/v1/openapi.yaml`
- Metrics: `/metrics` endpoint
