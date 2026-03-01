# GraphRAG to Cognee Local Migration Script

## Overview

The `migrate_to_cognee_local.py` script provides a complete migration solution for moving data from the API-based GraphRAG destination to the local Cognee destination with Neo4j.

## Features

### Core Migration Features

1. **Document Export from GraphRAG:**
   - Reads documents from GraphRAG API
   - Handles pagination for large datasets
   - Preserves document metadata, chunks, and embeddings

2. **Document Import to Cognee:**
   - Uses CogneeLocalDestination for import
   - Re-extracts entities using LLM
   - Rebuilds graph in Neo4j
   - Stores vectors in PostgreSQL/pgvector

3. **Batch Processing:**
   - Configurable batch sizes (default: 100)
   - Memory-efficient streaming
   - Resumable after errors

4. **Verification:**
   - Compares document counts
   - Compares entity counts
   - Sample verification of content
   - Overall migration health check

5. **Rollback Support:**
   - Tracks migrated documents
   - Allows selective rollback
   - Cleans up Neo4j nodes and relationships

6. **Progress Reporting:**
   - Progress bar with tqdm
   - ETA calculation
   - Detailed logging
   - JSON report generation

7. **Dry-Run Mode:**
   - Shows what would be migrated
   - No actual data changes
   - Useful for planning

## Usage

### Prerequisites

```bash
# Set environment variables
export GRAPH_RAG_API_URL="https://your-graphrag-api.com"
export GRAPH_RAG_API_KEY="your-api-key"
export NEO4J_URI="bolt://neo4j:7687"
export NEO4J_PASSWORD="cognee-graph-db"
export DB_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/pipeline"
```

### Basic Commands

```bash
# Dry run to see what would be migrated
python scripts/migrate_to_cognee_local.py --source-dataset my-dataset --dry-run

# Migrate with default settings
python scripts/migrate_to_cognee_local.py --source-dataset my-dataset

# Migrate with custom batch size
python scripts/migrate_to_cognee_local.py --source-dataset my-dataset --batch-size 50

# Migrate to a different target dataset
python scripts/migrate_to_cognee_local.py --source-dataset old-dataset --target-dataset new-dataset

# Migrate and verify
python scripts/migrate_to_cognee_local.py --source-dataset my-dataset --verify

# Skip existing documents (for resuming)
python scripts/migrate_to_cognee_local.py --source-dataset my-dataset --skip-existing

# Save report to file
python scripts/migrate_to_cognee_local.py --source-dataset my-dataset --output migration-report.json

# Rollback a migration
python scripts/migrate_to_cognee_local.py --source-dataset my-dataset --rollback
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--source-dataset` | Source GraphRAG dataset ID (required) | - |
| `--target-dataset` | Target Cognee dataset ID | Same as source |
| `--batch-size` | Documents per batch | 100 |
| `--dry-run` | Preview without changes | False |
| `--verify` | Verify after migration | False |
| `--skip-existing` | Skip existing documents | False |
| `--rollback` | Rollback migration | False |
| `--output` | Save report to file | - |
| `--verbose` | Enable debug logging | False |
| `--graphrag-url` | Override GRAPH_RAG_API_URL | - |
| `--graphrag-key` | Override GRAPH_RAG_API_KEY | - |
| `--neo4j-uri` | Override NEO4J_URI | - |
| `--neo4j-password` | Override NEO4J_PASSWORD | - |

## Migration Time Estimates

Based on typical performance:

| Documents | Estimated Time | Batch Size |
|-----------|----------------|------------|
| 100 | 2-3 minutes | 100 |
| 1,000 | 15-25 minutes | 100 |
| 10,000 | 2-4 hours | 100 |
| 100,000 | 15-25 hours | 500 |

Factors affecting migration time:
- Document size and chunk count
- LLM entity extraction (slowest part)
- Network latency to GraphRAG API
- Neo4j write performance
- Batch size

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Migration Script                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────┐        ┌──────────────────────────┐    │
│  │   GraphRAGClient  │───────►│   GraphRAG API           │    │
│  │   (HTTP/httpx)    │        │   (Source)               │    │
│  └───────────────────┘        └──────────────────────────┘    │
│           │                                                      │
│           ▼                                                      │
│  ┌───────────────────┐        ┌──────────────────────────┐    │
│  │  GraphRAGMigrator │───────►│   CogneeLocalDestination │    │
│  │   (Orchestrator)  │        │   (Target)               │    │
│  └───────────────────┘        └──────────────────────────┘    │
│                                          │                       │
│                                          ▼                       │
│                            ┌────────────────────────┐          │
│                            │  Neo4j (Graph)         │          │
│                            │  PostgreSQL (Vectors)  │          │
│                            └────────────────────────┘          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Migration Flow

1. **Connect Phase:**
   - Connect to GraphRAG API
   - Initialize CogneeLocalDestination
   - Validate both connections

2. **Discovery Phase:**
   - Count documents in source
   - Check target for existing data
   - Calculate estimated time

3. **Migration Phase:**
   - Fetch documents in batches from GraphRAG
   - Convert to TransformedData
   - Write to CogneeLocalDestination
   - Track statistics

4. **Verification Phase:**
   - Compare document counts
   - Verify entity counts
   - Sample document verification

5. **Report Phase:**
   - Generate statistics
   - Save report (if requested)
   - Display summary

## Error Handling

The script handles various error scenarios:

| Error | Handling |
|-------|----------|
| GraphRAG API unavailable | Logs error, retry with backoff |
| Document fetch failed | Logs error, continues with next batch |
| Document write failed | Logs error, continues with next document |
| Neo4j connection lost | Fails fast with clear error |
| LLM extraction timeout | Logs warning, continues without entities |
| Duplicate document | Skips if `--skip-existing` enabled |

## Output Format

### Console Output Example

```
Migrating documents: 45%|████▌     | 450/1000 [15:30<18:45, 1.63s/doc]

============================================================
MIGRATION RESULTS
============================================================
{
  "documents": {
    "total": 1000,
    "migrated": 995,
    "failed": 5,
    "skipped": 0,
    "success_rate": 0.995
  },
  "chunks": {
    "total": 4500,
    "migrated": 4485
  },
  "entities": {
    "extracted": 12500,
    "relationships_created": 8750
  },
  "transfer": {
    "bytes": 52428800,
    "mb": 50.0
  },
  "timing": {
    "duration_seconds": 1860,
    "documents_per_second": 0.53
  }
}
```

### JSON Report Format

```json
{
  "timestamp": "2026-02-28T20:30:00.000000",
  "source_dataset": "my-dataset",
  "target_dataset": "my-dataset",
  "dry_run": false,
  "results": {
    "documents": { ... },
    "chunks": { ... },
    "entities": { ... },
    "timing": { ... }
  }
}
```

## Testing

Run unit tests:

```bash
pytest tests/unit/test_migration_script.py -v
```

Run integration tests (requires GraphRAG and Cognee):

```bash
pytest tests/integration/test_migration.py -v
```

## Limitations

1. **Entity Re-extraction:** Entities are re-extracted using LLM during migration, not transferred from GraphRAG. This may result in different entities.

2. **Vector Embeddings:** Embeddings are preserved if available in GraphRAG response, but may need regeneration.

3. **Graph Structure:** Community detection results from GraphRAG are not migrated; communities are rebuilt in Cognee.

4. **Large Datasets:** Very large datasets (>1M documents) may require manual chunking or multiple migration runs.

5. **API Rate Limits:** GraphRAG API rate limits may slow migration.

## Troubleshooting

### Common Issues

**Issue:** Migration fails with "Neo4j authentication failed"
```
Solution: Verify NEO4J_PASSWORD environment variable
```

**Issue:** "GraphRAG API health check failed"
```
Solution: Check GRAPH_RAG_API_URL and GRAPH_RAG_API_KEY
```

**Issue:** Slow migration speed
```
Solution: 
- Increase batch size (--batch-size 500)
- Check network latency
- Verify Neo4j is properly indexed
```

**Issue:** Memory errors
```
Solution: Reduce batch size (--batch-size 50)
```

## Best Practices

1. **Always do a dry-run first:**
   ```bash
   python scripts/migrate_to_cognee_local.py --source-dataset DATASET --dry-run
   ```

2. **Start with a small batch:**
   ```bash
   python scripts/migrate_to_cognee_local.py --source-dataset DATASET --batch-size 10
   ```

3. **Save reports for auditing:**
   ```bash
   python scripts/migrate_to_cognee_local.py --source-dataset DATASET --output report.json
   ```

4. **Verify after migration:**
   ```bash
   python scripts/migrate_to_cognee_local.py --source-dataset DATASET --verify
   ```

5. **Monitor logs:**
   ```bash
   python scripts/migrate_to_cognee_local.py --source-dataset DATASET --verbose 2>&1 | tee migration.log
   ```

## Support

For issues or questions:
- Check logs with `--verbose`
- Review `MIGRATION_README.md`
- Contact the backend team
