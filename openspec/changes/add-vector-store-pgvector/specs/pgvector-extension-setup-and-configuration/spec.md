# Spec: pgvector Extension Setup and Configuration

## Purpose
Enable the pgvector extension in PostgreSQL to support vector storage and similarity search operations for document embeddings.

## Interface

### SQL Commands
```sql
-- Check if extension is available
SELECT * FROM pg_available_extensions WHERE name = 'vector';

-- Create extension with specific version
CREATE EXTENSION IF NOT EXISTS vector WITH VERSION '0.5.1';

-- Verify extension is installed
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';

-- Get extension control information
SELECT * FROM pg_available_extension_versions WHERE name = 'vector';
```

### Configuration Parameters
```yaml
# config/vector_store.yaml
vector_store:
  extension:
    name: "vector"
    min_version: "0.5.0"
    required: true
  dimensions:
    default: 1536
    supported: [384, 768, 1536, 3072]
```

### Python Interface
```python
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

async def setup_pgvector_extension(session: AsyncSession) -> dict:
    """
    Setup pgvector extension in the database.
    
    Returns:
        dict: {
            "installed": bool,
            "version": str,
            "available_versions": list[str]
        }
    """
    pass

async def check_pgvector_available(session: AsyncSession) -> bool:
    """Check if pgvector extension is available for installation."""
    pass
```

## Behavior

1. **Extension Availability Check**
   - Query `pg_available_extensions` to verify pgvector is installed on the PostgreSQL server
   - Return clear error if extension is not available (requires PostgreSQL 14+)
   - List all available versions for debugging purposes

2. **Idempotent Creation**
   - Use `CREATE EXTENSION IF NOT EXISTS` to avoid errors on re-runs
   - Skip creation if extension is already enabled in current database
   - Log informational message when extension is newly created vs. already exists

3. **Version Verification**
   - Verify installed version meets minimum requirements (≥ 0.5.0)
   - Log warning if version is outdated but still functional
   - Support version upgrades via `ALTER EXTENSION vector UPDATE`

4. **Permission Validation**
   - Verify current database user has CREATE privilege on the database
   - Verify user can execute `CREATE EXTENSION` (superuser or extension owner)
   - Provide clear error message if permissions are insufficient

5. **Schema Placement**
   - Create extension in default schema (public) unless configured otherwise
   - Support custom schema via `CREATE EXTENSION ... SCHEMA custom_schema`
   - Ensure vector type is accessible from application schema

6. **Dimension Configuration**
   - Validate configured dimensions are supported (384, 768, 1536, 3072)
   - Log configured default dimensions on successful setup
   - Support runtime dimension validation for embedding models

## Error Handling

| Error Case | Response | Action |
|------------|----------|--------|
| Extension not available | `RuntimeError` | Log: "pgvector extension not found. Install with: `apt install postgresql-14-pgvector` or use pgvector Docker image" |
| Permission denied | `PermissionError` | Log: "Database user lacks CREATE EXTENSION privilege. Required: superuser or explicit permission" |
| Version too old | `Warning` | Log: "pgvector version X.Y.Z is below recommended 0.5.0. Some features may not work." |
| Invalid dimensions | `ValueError` | Log: "Configured dimensions {N} not in supported list: [384, 768, 1536, 3072]" |
| Schema conflict | `RuntimeError` | Log: "Extension already exists in different schema. Manual intervention required." |
| Connection failure | `ConnectionError` | Log: "Failed to connect to database for extension setup" |

## Dependencies

- PostgreSQL 14+ (pgvector 0.5.0+ requires PG14)
- Database user with CREATE EXTENSION privilege
- `pgvector` extension files installed on PostgreSQL server

## Success Criteria

- [ ] Extension `vector` appears in `pg_extension` system catalog
- [ ] `vector` data type is usable in SQL/ORM queries
- [ ] Version check passes (≥ 0.5.0)
- [ ] No errors on idempotent re-runs
- [ ] Clear error messages for all failure cases
