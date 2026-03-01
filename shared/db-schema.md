# Database Schema

## Overview

This document describes the database schema for the pipeline ingestion system with Cognee GraphRAG support.

## Entities

### cognee_vectors

Stores vector embeddings for document chunks using pgvector extension.

| Column | Type | Constraints | Index |
|--------|------|-------------|-------|
| id | UUID | PK, DEFAULT gen_random_uuid() | - |
| chunk_id | VARCHAR(255) | NOT NULL | B-tree |
| document_id | VARCHAR(255) | NOT NULL | B-tree |
| dataset_id | VARCHAR(255) | NOT NULL, DEFAULT 'default' | B-tree |
| embedding | VECTOR(1536) | - | IVFFlat (cosine) |
| metadata | JSONB | NULL | - |
| created_at | TIMESTAMP WITH TZ | DEFAULT NOW() | - |
| updated_at | TIMESTAMP WITH TZ | DEFAULT NOW() | - |

**Indexes:**
- `idx_cognee_vectors_chunk_id` (chunk_id) - B-tree for fast chunk lookups
- `idx_cognee_vectors_document_id` (document_id) - B-tree for document filtering
- `idx_cognee_vectors_dataset_id` (dataset_id) - B-tree for dataset filtering
- `idx_cognee_vectors_embedding` (embedding) - IVFFlat for ANN search with cosine similarity

**Purpose:** Store chunk embeddings for semantic search with Cognee integration.

### cognee_documents

Document metadata storage for hybrid queries with Neo4j graph data.

| Column | Type | Constraints | Index |
|--------|------|-------------|-------|
| id | UUID | PK, DEFAULT gen_random_uuid() | - |
| document_id | VARCHAR(255) | UNIQUE, NOT NULL | B-tree |
| dataset_id | VARCHAR(255) | NOT NULL, DEFAULT 'default' | B-tree |
| title | VARCHAR(500) | NULL | - |
| source_path | TEXT | NULL | - |
| content_hash | VARCHAR(64) | NULL | - |
| metadata | JSONB | NULL | - |
| chunk_count | INTEGER | DEFAULT 0 | - |
| entity_count | INTEGER | DEFAULT 0 | - |
| created_at | TIMESTAMP WITH TZ | DEFAULT NOW() | - |
| updated_at | TIMESTAMP WITH TZ | DEFAULT NOW() | - |

**Indexes:**
- `idx_cognee_documents_dataset_id` (dataset_id) - B-tree for dataset filtering
- `idx_cognee_documents_document_id` (document_id) - B-tree for document lookups

**Relationships:**
- One-to-many with cognee_vectors (via document_id)
- One-to-many with cognee_entities (via document_id)

### cognee_entities

Entity mirror from Neo4j for hybrid graph + vector queries.

| Column | Type | Constraints | Index |
|--------|------|-------------|-------|
| id | UUID | PK, DEFAULT gen_random_uuid() | - |
| entity_id | VARCHAR(255) | UNIQUE, NOT NULL | B-tree |
| name | VARCHAR(500) | NOT NULL | B-tree |
| type | VARCHAR(100) | NULL | B-tree |
| description | TEXT | NULL | - |
| document_id | VARCHAR(255) | NULL | B-tree |
| dataset_id | VARCHAR(255) | NOT NULL, DEFAULT 'default' | - |
| metadata | JSONB | NULL | - |
| created_at | TIMESTAMP WITH TZ | DEFAULT NOW() | - |

**Indexes:**
- `idx_cognee_entities_name` (name) - B-tree for entity name search
- `idx_cognee_entities_type` (type) - B-tree for entity type filtering
- `idx_cognee_entities_document_id` (document_id) - B-tree for document filtering

**Relationships:**
- Many-to-one with cognee_documents (via document_id)

## Query Patterns

### Semantic Search (Vector Similarity)

```sql
-- Find similar chunks using cosine similarity
SELECT 
    cv.chunk_id,
    cv.document_id,
    cv.metadata,
    1 - (cv.embedding <=> query_embedding) AS similarity
FROM cognee_vectors cv
WHERE cv.dataset_id = 'default'
ORDER BY cv.embedding <=> query_embedding
LIMIT 10;
```

### Hybrid Search (Vector + Metadata)

```sql
-- Find similar chunks with document metadata
SELECT 
    cv.chunk_id,
    cv.document_id,
    cd.title,
    cd.source_path,
    1 - (cv.embedding <=> query_embedding) AS similarity
FROM cognee_vectors cv
JOIN cognee_documents cd ON cd.document_id = cv.document_id
WHERE cv.dataset_id = 'default'
  AND cd.chunk_count > 0
ORDER BY cv.embedding <=> query_embedding
LIMIT 10;
```

### Entity-Based Search

```sql
-- Find entities by name or type
SELECT 
    ce.entity_id,
    ce.name,
    ce.type,
    ce.description,
    cd.title AS document_title
FROM cognee_entities ce
LEFT JOIN cognee_documents cd ON cd.document_id = ce.document_id
WHERE ce.name ILIKE '%search_term%'
   OR ce.type = 'Person'
ORDER BY ce.name;
```

## Migrations

| Version | Description | Breaking Change |
|---------|-------------|-----------------|
| 015 | Add Cognee pgvector schema (vectors, documents, entities) | No |

## pgvector Configuration

### Vector Dimensions

- **1536 dimensions**: OpenAI text-embedding-3-small model
- **Future support**: Configurable dimensions via COGNEE_EMBEDDING_DIMENSIONS

### Index Types

#### IVFFlat (Used for cognee_vectors)

- **Algorithm**: Inverted file with flat compression
- **Build params**: `lists = 100`
- **Query operator**: `vector_cosine_ops` for cosine similarity
- **Best for**: 10k-1M vectors, fast build, moderate memory

#### HNSW (Alternative)

- Used in existing tables (document_chunks)
- Better for higher recall at larger scales
- Higher memory usage than IVFFlat

## Performance Considerations

1. **IVFFlat index** requires periodic reindexing for optimal performance as data grows
2. **JSONB columns** support GIN indexes for complex metadata queries
3. **Composite indexes** on (dataset_id, chunk_id) recommended for high-volume filtering
4. **Partitioning** by dataset_id recommended for multi-tenant scenarios

## Data Flow

### Ingestion Flow

```
Document → Chunks → Embeddings → cognee_vectors
       ↓
   Entities → Neo4j (Graph) → cognee_entities (Mirror)
       ↓
   Metadata → cognee_documents
```

### Query Flow

```
Query → Vector Search (cognee_vectors)
     → Graph Traversal (Neo4j)
     → Metadata Lookup (cognee_documents)
     → Result Fusion → Response
```
