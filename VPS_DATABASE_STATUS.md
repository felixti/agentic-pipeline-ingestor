# VPS Database Structure Status

## Summary

The VPS database structure has been verified and updated. Here's the current status:

### Current Migration Status

**Latest Migration**: `015` - Add Cognee pgvector schema  
**Total Migrations**: 16 (000-015)

### Migration Chain (Complete)

```
000 → 001 → 002 → 003 → 004 → 005 → 006 → 007 → 008 → 009 ─┬─→ 010 ─┬─→ 013 ─┐
                                                              │        │        │
                                                              └─→ 011 ─┼─→ 012 ─┤
                                                                       │        │
                                                              └─→ 014 ─┘
                                                                         ↓
                                                                        015 (head)
```

| Revision | Description | Cognee Related |
|----------|-------------|----------------|
| 000 | Create core tables (pipelines, jobs) | |
| 001 | Add content detection tables | |
| 002 | Add pgvector extensions | |
| 003 | Add document_chunks table with indexes | |
| 004 | Add contextual retrieval tables | |
| 005 | Add cache tables | |
| 006 | Add job_results table | |
| 007 | Add chunk unique constraint | |
| 008 | Add audit logs, API keys, webhook tables | |
| 009 | Add search_vector column to document_chunks | |
| 010 | Add chunk_quality_score column | |
| 011 | Add document_entities table | |
| 012 | Add search_analytics tables | |
| 013 | Add sparse_vectors column | |
| 014 | Add chunk_embeddings table | |
| **015** | **Add Cognee pgvector schema** | **✅ Yes** |

## Cognee-Specific Tables (Migration 015)

Since Cognee uses the same PostgreSQL instance (with Neo4j for graph storage), these tables support Cognee's vector operations:

### `cognee_vectors`
Stores chunk embeddings for Cognee's vector search operations.

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `chunk_id` | VARCHAR(255) | Reference to document chunk |
| `document_id` | VARCHAR(255) | Reference to document |
| `dataset_id` | VARCHAR(255) | Cognee dataset identifier |
| `embedding` | VECTOR(1536) | OpenAI embedding vector |
| `metadata` | JSONB | Additional metadata |
| `created_at` | TIMESTAMP | Creation timestamp |
| `updated_at` | TIMESTAMP | Last update timestamp |

**Indexes**:
- `idx_cognee_vectors_embedding` - IVFFlat index for ANN search
- `idx_cognee_vectors_chunk_id` - B-tree for chunk lookups
- `idx_cognee_vectors_document_id` - B-tree for document lookups
- `idx_cognee_vectors_dataset_id` - B-tree for dataset filtering

### `cognee_documents`
Mirrors document metadata from Neo4j for hybrid queries.

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `document_id` | VARCHAR(255) | Unique document identifier |
| `dataset_id` | VARCHAR(255) | Cognee dataset identifier |
| `title` | VARCHAR(500) | Document title |
| `source_path` | TEXT | Original file path |
| `content_hash` | VARCHAR(64) | Content hash for deduplication |
| `metadata` | JSONB | Additional metadata |
| `chunk_count` | INTEGER | Number of chunks |
| `entity_count` | INTEGER | Number of extracted entities |
| `created_at` | TIMESTAMP | Creation timestamp |
| `updated_at` | TIMESTAMP | Last update timestamp |

### `cognee_entities`
Mirrors entities from Neo4j for hybrid graph + vector queries.

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `entity_id` | VARCHAR(255) | Unique entity identifier |
| `name` | VARCHAR(500) | Entity name |
| `type` | VARCHAR(100) | Entity type (e.g., PERSON, ORG) |
| `description` | TEXT | Entity description |
| `document_id` | VARCHAR(255) | Source document reference |
| `dataset_id` | VARCHAR(255) | Cognee dataset identifier |
| `metadata` | JSONB | Additional metadata |
| `created_at` | TIMESTAMP | Creation timestamp |

## VPS Database Setup Steps

### Option 1: Using Alembic Migrations (Recommended)

```bash
# 1. Set your VPS database URL
export DB_URL="postgresql+asyncpg://postgres:SENHA@seu-vps:5432/pipeline"

# 2. Verify current status
python3 scripts/verify_vps_database_complete.py --db-url "$DB_URL"

# 3. Run all migrations to head (015)
DB_URL="$DB_URL" alembic upgrade head

# 4. Verify after migration
python3 scripts/verify_vps_database_complete.py --db-url "$DB_URL"
```

### Option 2: Direct SQL for Cognee Schema Only

If you already have migrations 000-014 applied and just need the Cognee schema:

```bash
# Connect to VPS PostgreSQL and run Cognee schema setup
psql "postgresql://postgres:SENHA@seu-vps:5432/pipeline" << 'EOF'
-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Create cognee_vectors table
CREATE TABLE IF NOT EXISTS cognee_vectors (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    chunk_id VARCHAR(255) NOT NULL,
    document_id VARCHAR(255) NOT NULL,
    dataset_id VARCHAR(255) NOT NULL DEFAULT 'default',
    embedding VECTOR(1536),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes
CREATE INDEX idx_cognee_vectors_chunk_id ON cognee_vectors(chunk_id);
CREATE INDEX idx_cognee_vectors_document_id ON cognee_vectors(document_id);
CREATE INDEX idx_cognee_vectors_dataset_id ON cognee_vectors(dataset_id);
CREATE INDEX idx_cognee_vectors_embedding ON cognee_vectors 
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Create cognee_documents table
CREATE TABLE IF NOT EXISTS cognee_documents (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    document_id VARCHAR(255) NOT NULL UNIQUE,
    dataset_id VARCHAR(255) NOT NULL DEFAULT 'default',
    title VARCHAR(500),
    source_path TEXT,
    content_hash VARCHAR(64),
    metadata JSONB,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    entity_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_cognee_documents_document_id ON cognee_documents(document_id);
CREATE INDEX idx_cognee_documents_dataset_id ON cognee_documents(dataset_id);

-- Create cognee_entities table
CREATE TABLE IF NOT EXISTS cognee_entities (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    entity_id VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(500) NOT NULL,
    type VARCHAR(100),
    description TEXT,
    document_id VARCHAR(255),
    dataset_id VARCHAR(255) NOT NULL DEFAULT 'default',
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_cognee_entities_name ON cognee_entities(name);
CREATE INDEX idx_cognee_entities_type ON cognee_entities(type);
CREATE INDEX idx_cognee_entities_document_id ON cognee_entities(document_id);

-- Update alembic version
INSERT INTO alembic_version (version_num) VALUES ('015')
ON CONFLICT (version_num) DO UPDATE SET version_num = '015';
EOF
```

## Required Database Extensions

The VPS database must have these PostgreSQL extensions:

```sql
CREATE EXTENSION IF NOT EXISTS "vector";     -- pgvector for embeddings (Cognee + main app)
CREATE EXTENSION IF NOT EXISTS "pg_trgm";    -- fuzzy text search
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";  -- UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";   -- gen_random_uuid()
```

## All Tables

### Core Application Tables
- `jobs` - Pipeline processing jobs with `file_name` and `source_uri`
- `document_chunks` - Text chunks with vector embeddings
- `pipelines` - Pipeline configurations

### Content Detection Tables
- `content_detection_results` - Content type detection
- `job_detection_results` - Link table

### Contextual Retrieval Tables
- `document_hierarchy` - Document structure

### Results Tables
- `job_results` - Job processing results

### Supporting Tables
- `audit_logs` - Audit trail
- `api_keys` - API authentication
- `webhook_subscriptions` - Webhook configurations
- `webhook_deliveries` - Webhook delivery attempts

### Cache Tables
- `embedding_cache` - Embedding cache
- `query_cache` - Query result cache
- `llm_response_cache` - LLM response cache

### Analytics Tables
- `search_analytics` - Search analytics events
- `search_queries` - Search query log

### Cognee GraphRAG Tables
- `cognee_vectors` - Chunk embeddings for Cognee
- `cognee_documents` - Document metadata mirror
- `cognee_entities` - Entity mirror from Neo4j

## Critical Indexes

### Main Application Indexes

| Index | Purpose |
|-------|---------|
| `idx_document_chunks_embedding_hnsw` | Vector similarity search (HNSW) |
| `idx_document_chunks_content_tsvector` | Full-text search (GIN) |
| `idx_document_chunks_content_trgm` | Fuzzy search (GIN) |
| `idx_document_chunks_job_id` | Job filtering |
| `idx_jobs_status` | Job status filtering |

### Cognee Indexes

| Index | Purpose |
|-------|---------|
| `idx_cognee_vectors_embedding` | IVFFlat ANN search (cosine similarity) |
| `idx_cognee_vectors_chunk_id` | Chunk lookups |
| `idx_cognee_vectors_document_id` | Document lookups |
| `idx_cognee_vectors_dataset_id` | Dataset filtering |
| `idx_cognee_documents_document_id` | Document ID lookups |
| `idx_cognee_documents_dataset_id` | Dataset filtering |
| `idx_cognee_entities_name` | Entity name search |
| `idx_cognee_entities_type` | Entity type filtering |
| `idx_cognee_entities_document_id` | Document entity lookups |

## Verification

After running migrations, verify with:

```bash
python3 scripts/verify_vps_database_complete.py \
    --db-url "postgresql+asyncpg://postgres:SENHA@seu-vps:5432/pipeline"
```

Expected output: All checks should show ✅, especially:
- ✅ Alembic Version: 015
- ✅ All Cognee tables (cognee_vectors, cognee_documents, cognee_entities)
- ✅ IVFFlat index on cognee_vectors.embedding

## Troubleshooting

### "cognee_vectors table does not exist" Error

Migration 015 hasn't been applied. Run:

```bash
DB_URL="postgresql+asyncpg://postgres:SENHA@seu-vps:5432/pipeline" alembic upgrade 015
```

### "vector extension not found" Error

Install pgvector:

```bash
psql "postgresql://postgres:SENHA@seu-vps:5432/pipeline" -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Migration Version Mismatch

Check current version and upgrade:

```bash
# Check current version
psql "postgresql://postgres:SENHA@seu-vps:5432/pipeline" -c "SELECT * FROM alembic_version;"

# Upgrade to head (015)
DB_URL="postgresql+asyncpg://postgres:SENHA@seu-vps:5432/pipeline" alembic upgrade head
```

### IVFFlat Index Not Created

The IVFFlat index is critical for Cognee performance. Verify it exists:

```sql
SELECT indexname, indexdef 
FROM pg_indexes 
WHERE schemaname = 'public' 
AND indexname = 'idx_cognee_vectors_embedding';
```

If missing, recreate:

```sql
CREATE INDEX idx_cognee_vectors_embedding 
ON cognee_vectors 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);
```

## Docker Compose Note

If using Docker Compose on VPS, ensure the PostgreSQL image includes pgvector:

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg17
    environment:
      POSTGRES_DB: pipeline
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: sua_senha_segura
    volumes:
      - postgres-data:/var/lib/postgresql/data
      
  neo4j:
    image: neo4j:5.15-community
    environment:
      NEO4J_AUTH: neo4j/cognee-graph-db
    volumes:
      - neo4j-data:/data
```

## Architecture Notes

### Shared PostgreSQL Instance

Both the main application and Cognee share the same PostgreSQL instance:

- **Main app**: Uses `document_chunks` table for standard RAG
- **Cognee**: Uses `cognee_vectors` table for GraphRAG
- **Shared**: `pgvector` extension, connection pool

### Neo4j Separation

Cognee uses Neo4j separately for graph storage:
- Neo4j stores: Entity nodes, relationships, graph structure
- PostgreSQL stores: Vector embeddings, document metadata, entity mirror

This hybrid approach allows:
- Fast graph traversals in Neo4j
- Efficient vector similarity in PostgreSQL/pgvector
- Hybrid queries combining both stores

## Status

✅ **Migration 015 (Cognee pgvector schema) is ready for VPS deployment**  
✅ **Database structure supports both main app and Cognee GraphRAG**
