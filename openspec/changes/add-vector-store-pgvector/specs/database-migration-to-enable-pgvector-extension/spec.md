# Spec: Database Migration to Enable pgvector Extension

## Purpose
Alembic migration script to enable pgvector extension with idempotent operations, version checking, and rollback support.

## Interface

### Migration File Structure
```python
# migrations/versions/xxxx_enable_pgvector_extension.py

revision = 'xxxx'
down_revision = 'yyyy'
branch_labels = None
depends_on = None

MIN_PGVVECTOR_VERSION = '0.5.0'
REQUIRED_DIMENSIONS = [384, 768, 1536, 3072]

def upgrade():
    """Enable pgvector extension and verify requirements."""
    pass

def downgrade():
    """Disable pgvector extension (optional)."""
    pass
```

### Migration Operations
```python
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

def upgrade():
    # 1. Check extension availability
    # 2. Create extension if not exists
    # 3. Verify version
    # 4. Log success
    pass

def downgrade():
    # 1. Drop extension if safe
    # 2. Log downgrade
    pass
```

### Alembic CLI Commands
```bash
# Generate migration (if using autogenerate)
alembic revision --autogenerate -m "Enable pgvector extension"

# Run migration
alembic upgrade head

# Rollback
alembic downgrade -1

# Check current version
alembic current

# Show migration history
alembic history --verbose
```

## Behavior

1. **Pre-Migration Checks**
   - Query `pg_available_extensions` to confirm pgvector is installed on server
   - Check current PostgreSQL version (must be ≥ 14.0)
   - Verify no conflicting extensions or types exist
   - Validate database connection and permissions

2. **Idempotent Extension Creation**
   - Execute: `CREATE EXTENSION IF NOT EXISTS vector`
   - Do not fail if extension already exists
   - Capture and log the actual version installed
   - Support custom schema via `SCHEMA` clause if configured

3. **Version Validation**
   - Query installed version via `pg_extension`
   - Compare against `MIN_PGVVECTOR_VERSION` ('0.5.0')
   - Raise `RuntimeError` if version requirement not met
   - Log installed version for audit trail

4. **Post-Creation Verification**
   - Verify `vector` type is available: `SELECT 'vector'::regtype`
   - Test basic vector operations: `SELECT '[1,2,3]'::vector`
   - Verify operators are registered (`<=>`, `<#>`, `<->`)
   - Log success message with version info

5. **Downgrade Behavior**
   - Drop extension only if no tables use `vector` type
   - Check for dependent objects before dropping
   - Execute: `DROP EXTENSION IF EXISTS vector`
   - Log warning if extension cannot be dropped (dependencies exist)

6. **Migration Metadata**
   - Record extension version in migration log
   - Store dimension configuration reference
   - Document any manual steps required for cloud providers

## Error Handling

| Error Case | Migration Action | Log Output |
|------------|------------------|------------|
| pgvector not available | Abort with `RuntimeError` | "pgvector extension not available. Ensure postgresql-XX-pgvector is installed" |
| PostgreSQL version < 14 | Abort with `RuntimeError` | "PostgreSQL 14+ required for pgvector 0.5.0+. Current: {version}" |
| Version below minimum | Abort with `RuntimeError` | "pgvector version {installed} < required {minimum}" |
| Permission denied | Abort with `RuntimeError` | "Insufficient privileges to CREATE EXTENSION" |
| Extension type conflict | Abort with `RuntimeError` | "Type 'vector' already exists from different source" |
| Downgrade blocked | Warn and skip | "Cannot drop pgvector: tables using vector type exist" |
| Verification failure | Abort with `RuntimeError` | "Extension verification failed: vector type not functional" |

## Migration Script Template

```python
"""Enable pgvector extension

Revision ID: xxxx
Revises: yyyy
Create Date: 2024-XX-XX

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text
from alembic.runtime import migration

# revision identifiers, used by Alembic.
revision = 'xxxx'
down_revision = 'yyyy'
branch_labels = None
depends_on = None

MIN_PGVVECTOR_VERSION = '0.5.0'

def upgrade() -> None:
    conn = op.get_bind()
    
    # Check PostgreSQL version
    pg_version = conn.execute(text("SELECT current_setting('server_version')")).scalar()
    major_version = int(pg_version.split('.')[0])
    if major_version < 14:
        raise RuntimeError(f"PostgreSQL 14+ required. Current: {pg_version}")
    
    # Check extension availability
    available = conn.execute(text(
        "SELECT 1 FROM pg_available_extensions WHERE name = 'vector'"
    )).fetchone()
    if not available:
        raise RuntimeError(
            "pgvector extension not available. "
            "Install: apt install postgresql-14-pgvector "
            "or use pgvector/pgvector Docker image"
        )
    
    # Create extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    
    # Verify version
    version = conn.execute(text(
        "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
    )).scalar()
    
    # Compare versions (simplified - use packaging.version in production)
    if version < MIN_PGVVECTOR_VERSION:
        raise RuntimeError(
            f"pgvector version {version} < required {MIN_PGVVECTOR_VERSION}"
        )
    
    # Verify functionality
    conn.execute(text("SELECT '[1,2,3]'::vector"))
    
    print(f"pgvector extension enabled: version {version}")

def downgrade() -> None:
    conn = op.get_bind()
    
    # Check for dependencies
    dependencies = conn.execute(text("""
        SELECT COUNT(*) FROM pg_depend d
        JOIN pg_class c ON d.objid = c.oid
        JOIN pg_type t ON c.reltype = t.oid
        WHERE t.typname = 'vector'
    """)).scalar()
    
    if dependencies > 0:
        print("WARNING: Cannot drop pgvector extension - tables using vector type exist")
        return
    
    op.execute("DROP EXTENSION IF EXISTS vector")
    print("pgvector extension disabled")
```

## Cloud Provider Notes

| Provider | Setup Instructions |
|----------|-------------------|
| AWS RDS | Enable via parameter group: `shared_preload_libraries = 'pgvector'` |
| Azure PostgreSQL | Available in Flexible Server; enable via portal or CLI |
| GCP Cloud SQL | Requires PostgreSQL 15+; enable via `gcloud sql instances patch` |
| Supabase | Pre-enabled on all projects |
| Neon | Pre-enabled on all projects |
| Local Docker | Use `pgvector/pgvector:pg14` image |

## Dependencies

- Alembic migration environment configured
- Database connection with CREATE EXTENSION privilege
- PostgreSQL 14+ server with pgvector extension files

## Success Criteria

- [ ] Migration runs successfully on fresh database
- [ ] Migration is idempotent (re-runnable without errors)
- [ ] Version check enforces minimum requirements
- [ ] Downgrade gracefully handles dependencies
- [ ] Clear error messages for all failure scenarios
- [ ] Migration works on AWS RDS, Azure, GCP Cloud SQL
