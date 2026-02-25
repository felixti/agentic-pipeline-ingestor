# VPS Database Setup and Management

This directory contains scripts to verify and manage the VPS database structure for the Agentic Pipeline Ingestor.

## Quick Start

### 1. Verify Database Structure

Check if your VPS database has all required tables, indexes, and extensions:

```bash
# Set your VPS database URL
export DB_URL="postgresql+asyncpg://postgres:SENHA@seu-vps:5432/pipeline"

# Run verification
python3 scripts/verify_vps_database.py
```

### 2. Run Migrations

Apply all pending migrations to the VPS database:

```bash
# Option 1: Using the migration runner script
./scripts/run_vps_migrations.sh "postgresql+asyncpg://postgres:SENHA@seu-vps:5432/pipeline"

# Option 2: Using environment variable
export DB_URL="postgresql+asyncpg://postgres:SENHA@seu-vps:5432/pipeline"
./scripts/run_vps_migrations.sh

# Option 3: Direct alembic command
DB_URL="postgresql+asyncpg://postgres:SENHA@seu-vps:5432/pipeline" alembic upgrade head
```

### 3. Fresh Database Setup

If setting up a fresh VPS database, run the SQL setup script first:

```bash
# Connect to your VPS PostgreSQL and run the setup script
psql "postgresql://postgres:SENHA@seu-vps:5432/pipeline" -f scripts/vps_database_setup.sql
```

Or run the commands directly:

```bash
psql "postgresql://postgres:SENHA@seu-vps:5432/pipeline" << 'EOF'
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
EOF
```

## Database Structure

### Required Extensions

| Extension | Purpose |
|-----------|---------|
| `vector` | pgvector for embedding storage and similarity search |
| `pg_trgm` | Trigram matching for fuzzy text search |
| `uuid-ossp` | UUID generation |

### Core Tables

| Table | Purpose |
|-------|---------|
| `jobs` | Pipeline processing jobs |
| `document_chunks` | Text chunks with vector embeddings |
| `job_results` | Job processing results |
| `pipelines` | Pipeline configurations |
| `content_detection_results` | Content type detection results |
| `job_detection_results` | Link table for jobs and detection results |
| `audit_logs` | Audit log entries |
| `api_keys` | API key management |
| `webhook_subscriptions` | Webhook subscriptions |
| `webhook_deliveries` | Webhook delivery attempts |

### Critical Indexes

| Index | Type | Purpose |
|-------|------|---------|
| `idx_document_chunks_embedding_hnsw` | HNSW | Vector similarity search |
| `idx_document_chunks_content_tsvector` | GIN | Full-text search |
| `idx_document_chunks_content_trgm` | GIN | Fuzzy text search |
| `idx_document_chunks_job_id` | B-tree | Job filtering |

## Troubleshooting

### Connection Issues

```bash
# Test connection
psql "postgresql://postgres:SENHA@seu-vps:5432/pipeline" -c "SELECT 1;"
```

### Missing Extensions

```bash
# Install pgvector and pg_trgm
psql "postgresql://postgres:SENHA@seu-vps:5432/pipeline" << 'EOF'
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
\dx
EOF
```

### Migration Failures

If migrations fail, check the current version:

```sql
SELECT * FROM alembic_version;
```

To reset and re-run migrations (WARNING: Destructive):

```sql
-- Drop all tables (DESTRUCTIVE - DATA LOSS)
DROP TABLE IF EXISTS job_detection_results CASCADE;
DROP TABLE IF EXISTS content_detection_results CASCADE;
DROP TABLE IF EXISTS document_chunks CASCADE;
DROP TABLE IF EXISTS job_results CASCADE;
DROP TABLE IF EXISTS webhook_deliveries CASCADE;
DROP TABLE IF EXISTS webhook_subscriptions CASCADE;
DROP TABLE IF EXISTS audit_logs CASCADE;
DROP TABLE IF EXISTS api_keys CASCADE;
DROP TABLE IF EXISTS jobs CASCADE;
DROP TABLE IF EXISTS pipelines CASCADE;
DROP TABLE IF EXISTS alembic_version CASCADE;

-- Re-run migrations
-- alembic upgrade head
```

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `DB_URL` | Database connection URL | `postgresql+asyncpg://postgres:pass@host:5432/pipeline` |
| `DATABASE_URL` | Alternative to DB_URL | Same as above |

## Docker Compose VPS Setup

If using Docker Compose on VPS:

```yaml
version: '3.8'
services:
  postgres:
    image: pgvector/pgvector:pg17
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: sua_senha_segura
      POSTGRES_DB: pipeline
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/vps_database_setup.sql:/docker-entrypoint-initdb.d/01-setup.sql
    ports:
      - "5432:5432"

  api:
    build: .
    environment:
      DB_URL: postgresql+asyncpg://postgres:sua_senha_segura@postgres:5432/pipeline
    depends_on:
      - postgres
    command: >
      sh -c "alembic upgrade head && uvicorn src.main:app --host 0.0.0.0 --port 8000"

volumes:
  postgres_data:
```

## Checking Search Functionality

After setting up the database, verify search works:

```sql
-- Check if chunks have embeddings
SELECT COUNT(*) as total_chunks, 
       COUNT(embedding) as chunks_with_embeddings
FROM document_chunks;

-- Check if indexes are being used
EXPLAIN ANALYZE
SELECT * FROM document_chunks 
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector 
LIMIT 10;

-- Check text search
EXPLAIN ANALYZE
SELECT * FROM document_chunks 
WHERE to_tsvector('english', content) @@ to_tsquery('search_term');
```

## Maintenance

### Vacuum and Analyze

```sql
-- Run after large data imports
VACUUM ANALYZE document_chunks;
```

### Reindex (if needed)

```sql
REINDEX INDEX idx_document_chunks_embedding_hnsw;
REINDEX INDEX idx_document_chunks_content_tsvector;
```

## Support

For issues with database setup:
1. Check the migration files in `migrations/versions/`
2. Review the model definitions in `src/db/models.py`
3. Check application logs for specific errors
