# Production HTTP Client Files

This folder contains HTTP client files for testing the **production** deployment of the Agentic Data Pipeline Ingestor API.

## ⚠️  Warning

These files target the **PRODUCTION** environment:
- **URL**: `https://ag-dt-ppl-api.felixtek.cloud/api/v1`
- **Impact**: Requests modify live production data
- **Caution**: Use with care - test thoroughly before running destructive operations

## Files

| File | Description |
|------|-------------|
| `all-searches.http` | Complete collection of all search endpoints (semantic, text, hybrid, similar) |
| `hybrid.http` | Hybrid search examples with weighted sum and RRF fusion |
| `jobs.http` | **Create and manage ingestion jobs** (PDF, OCR, Office docs, sync/async modes) |
| `semantic.http` | Semantic/vector similarity search examples |
| `similar.http` | Find semantically similar chunks |
| `text.http` | Text/BM25 search with fuzzy matching and highlighting |

## Usage

1. **Install REST Client extension** in VS Code:
   - Search for "REST Client" by Huachao Mao in the VS Code marketplace

2. **Configure your API key**:
   - Open any `.http` file
   - Replace `your-api-key` with your actual production API key:
   ```http
   @apiKey = your-actual-production-api-key
   ```

3. **Send requests**:
   - Click "Send Request" link above any request
   - Or use keyboard shortcut `Ctrl+Alt+R` (Windows/Linux) or `Cmd+Alt+R` (Mac)

## Environment Variables

Each file defines these variables at the top:

```http
@baseUrl = https://ag-dt-ppl-api.felixtek.cloud/api/v1
@apiKey = your-api-key  # Replace with production API key
```

## Local Development

For local development testing, use the files in the parent `http/` directory which target `http://localhost:8000/api/v1`.

## API Documentation

- OpenAPI Spec: `https://ag-dt-ppl-api.felixtek.cloud/api/v1/openapi.yaml`
- Swagger UI: `https://ag-dt-ppl-api.felixtek.cloud/docs`
- ReDoc: `https://ag-dt-ppl-api.felixtek.cloud/redoc`
