# CogneeLocalDestination Rewrite Summary

## Overview

The `CogneeLocalDestination` class has been completely rewritten to use the actual Cognee library APIs instead of directly querying Neo4j.

## Key Changes

### 1. Core API Usage

The new implementation uses the actual Cognee library methods:

| Operation | Old Implementation | New Implementation |
|-----------|-------------------|-------------------|
| Add documents | Direct Neo4j Cypher queries | `cognee.add(text, dataset_name)` |
| Build knowledge graph | Custom entity extraction + Neo4j | `cognee.cognify(datasets)` |
| Search | Direct Neo4j Cypher queries | `cognee.search(query_text, query_type)` |

### 2. Configuration Changes

**Removed options:**
- `extract_entities` - Now handled internally by Cognee
- `extract_relationships` - Now handled internally by Cognee  
- `store_vectors` - Now handled internally by Cognee

**New options:**
- `auto_cognify` - Automatically call `cognify()` after each write (default: False)

### 3. Environment Variables

Cognee reads configuration from environment variables:

```bash
# Neo4j (used internally by Cognee)
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=cognee-graph-db

# PostgreSQL/pgvector (parsed from DB_URL or set explicitly)
PGVECTOR_HOST=postgres
PGVECTOR_PORT=5432
PGVECTOR_USER=postgres
PGVECTOR_PASSWORD=postgres
PGVECTOR_DB=pipeline

# LLM Configuration for Cognee's litellm adapter
LLM_API_KEY=your-openai-key
LLM_MODEL=gpt-4.1
EMBEDDING_MODEL=text-embedding-3-small
```

### 4. New Methods

- `process_dataset(conn)` - Explicitly process dataset using `cognee.cognify()`
- `get_dataset_stats(conn)` - Get dataset statistics
- `_setup_cognee_environment(config)` - Set up environment variables for Cognee
- `_get_search_type(search_type)` - Map search type strings to Cognee SearchType
- `_convert_cognee_results(results, top_k)` - Convert Cognee results to SearchResult

### 5. Search Types

The search method now supports Cognee's native search types:

```python
search_type_map = {
    "graph": SearchType.GRAPH_COMPLETION,
    "vector": SearchType.RAG_COMPLETION,
    "hybrid": SearchType.FEELING_LUCKY,
    "summary": SearchType.SUMMARIES,
    "chunks": SearchType.CHUNKS,
}
```

## Usage Flow

### Basic Usage

```python
from src.plugins.destinations.cognee_local import CogneeLocalDestination

# Initialize
destination = CogneeLocalDestination()
await destination.initialize({
    "dataset_id": "my-dataset",
    "graph_name": "knowledge-graph",
})

# Connect
conn = await destination.connect({"dataset_id": "my-dataset"})

# Add documents
result = await destination.write(conn, transformed_data)

# Process into knowledge graph (required!)
await destination.process_dataset(conn)

# Search
results = await destination.search(
    conn, 
    "your query",
    search_type="hybrid",
    top_k=10
)
```

### Auto-Processing Usage

```python
# Enable auto_cognify to process after each write
await destination.initialize({
    "dataset_id": "my-dataset",
    "auto_cognify": True,  # Automatically calls cognify() after each write
})

conn = await destination.connect({"dataset_id": "my-dataset"})
await destination.write(conn, data)  # Automatically processes dataset
```

## Files Modified

1. `src/plugins/destinations/cognee_local.py` - Complete rewrite
2. `tests/unit/test_plugins_destinations_cognee_local.py` - Updated tests
3. `.env.example` - Added Cognee environment variables

## Backward Compatibility

⚠️ **Breaking Changes:**
- The `write()` method no longer returns `entities_extracted` and `relationships_created` in metadata
- The `write()` method now returns `chunks_added` instead of `chunks_created`
- Direct Neo4j entity/relationship methods removed (now handled by Cognee)
- Config options `extract_entities`, `extract_relationships`, `store_vectors` removed

## Testing

Run unit tests:
```bash
pytest tests/unit/test_plugins_destinations_cognee_local.py -v
```

Run integration tests (requires Neo4j and PostgreSQL):
```bash
pytest tests/integration/test_cognee_integration.py -v
```

## Dependencies

Required:
```bash
pip install cognee
```

Cognee will internally use:
- Neo4j for graph storage
- PostgreSQL/pgvector for vector storage
- litellm for LLM integration
