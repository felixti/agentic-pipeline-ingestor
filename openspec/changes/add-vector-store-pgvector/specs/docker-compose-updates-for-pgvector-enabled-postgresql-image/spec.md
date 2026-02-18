# Spec: Docker Compose Updates for pgvector-Enabled PostgreSQL Image

## Purpose
Update Docker Compose configuration to use the pgvector-enabled PostgreSQL image instead of the standard postgres image for local development and testing.

## Interface

### Docker Compose Configuration
```yaml
# docker-compose.yml
version: '3.8'

services:
  db:
    # OLD: image: postgres:14-alpine
    image: pgvector/pgvector:pg14
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-app}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-app}
      POSTGRES_DB: ${POSTGRES_DB:-app}
    ports:
      - "${POSTGRES_PORT:-5432}:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      # Optional: initialization scripts
      - ./docker/postgres/init:/docker-entrypoint-initdb.d
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-app}"]
      interval: 5s
      timeout: 5s
      retries: 5
    networks:
      - app_network

volumes:
  postgres_data:

networks:
  app_network:
    driver: bridge
```

### Available Image Tags
```yaml
# Production/stable tags
pgvector/pgvector:pg14          # PostgreSQL 14 with latest pgvector
pgvector/pgvector:pg15          # PostgreSQL 15 with latest pgvector
pgvector/pgvector:pg16          # PostgreSQL 16 with latest pgvector

# Specific version tags (recommended for reproducibility)
pgvector/pgvector:pg14-v0.5.1   # PostgreSQL 14 + pgvector 0.5.1
pgvector/pgvector:pg15-v0.5.1   # PostgreSQL 15 + pgvector 0.5.1
pgvector/pgvector:pg16-v0.5.1   # PostgreSQL 16 + pgvector 0.5.1

# Alpine variants (smaller)
pgvector/pgvector:pg14-alpine
pgvector/pgvector:pg15-alpine
```

### Environment Configuration
```yaml
# .env.example
# PostgreSQL Configuration
POSTGRES_USER=app
POSTGRES_PASSWORD=app
POSTGRES_DB=app
POSTGRES_PORT=5432
POSTGRES_IMAGE=pgvector/pgvector:pg14

# pgvector-specific settings
PGVECTOR_EXTENSION_VERSION=0.5.1
PGVECTOR_MIN_VERSION=0.5.0
```

## Behavior

1. **Image Selection**
   - Replace `postgres:14-alpine` with `pgvector/pgvector:pg14` (or pg15/pg16)
   - Use specific version tag for reproducible builds: `pg14-v0.5.1`
   - Support environment variable override for image selection

2. **Backward Compatibility**
   - Maintain all existing environment variables (POSTGRES_USER, etc.)
   - Preserve existing volume mounts and data persistence
   - Keep same port mapping (5432:5432)
   - Support existing healthcheck configuration

3. **Extension Pre-installation**
   - pgvector extension files are pre-installed in the image
   - Extension is available but not auto-enabled (requires `CREATE EXTENSION`)
   - pg_trgm extension files also included for text search

4. **Health Check Integration**
   - Verify PostgreSQL is ready via `pg_isready`
   - Optional: Add pgvector-specific health check
   ```yaml
   healthcheck:
     test: >
       pg_isready -U ${POSTGRES_USER:-app} && 
       psql -U ${POSTGRES_USER:-app} -c "SELECT 1 FROM pg_available_extensions WHERE name = 'vector'"
   ```

5. **Volume Persistence**
   - Named volume `postgres_data` persists across container restarts
   - Migration from standard postgres: data is preserved but pgvector must be enabled
   - No data migration required (same PostgreSQL base image)

6. **Initialization Scripts**
   - Support `/docker-entrypoint-initdb.d` for automatic extension setup
   - Optional init script to enable extension on first run:
   ```sql
   -- docker/postgres/init/01-enable-pgvector.sql
   CREATE EXTENSION IF NOT EXISTS vector;
   CREATE EXTENSION IF NOT EXISTS pg_trgm;
   ```

## Error Handling

| Error Case | Detection | Resolution |
|------------|-----------|------------|
| Image pull failure | Docker compose up fails | Verify network connectivity; check image tag exists on Docker Hub |
| Port conflict | Container fails to start | Change `POSTGRES_PORT` env var or stop conflicting service |
| Volume permission issues | Permission denied on startup | Fix ownership: `sudo chown -R 999:999 postgres_data/` |
| Extension not found | Health check fails | Verify correct image tag; rebuild with `--no-cache` |
| Version mismatch | Application logs warning | Update `POSTGRES_IMAGE` to required version |

## Migration Path

### From Standard PostgreSQL
```bash
# 1. Stop existing containers
docker-compose down

# 2. Update docker-compose.yml (change image)
# 3. Pull new image
docker-compose pull db

# 4. Start with new image
docker-compose up -d db

# 5. Enable extension (one-time)
docker-compose exec db psql -U app -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 6. Verify
docker-compose exec db psql -U app -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

### Fresh Start
```bash
# Remove old data for clean slate
docker-compose down -v
docker-compose up -d db

# Verify pgvector is available
docker-compose exec db psql -U app -c "SELECT * FROM pg_available_extensions WHERE name = 'vector';"
```

## Configuration Files

### docker-compose.override.yml (for local development)
```yaml
# docker-compose.override.yml
version: '3.8'

services:
  db:
    image: pgvector/pgvector:pg14-v0.5.1
    ports:
      - "5432:5432"
    # Add pgadmin or other dev tools
    
  pgadmin:
    image: dpage/pgadmin4:latest
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@localhost
      PGADMIN_DEFAULT_PASSWORD: admin
    ports:
      - "5050:80"
    depends_on:
      - db
```

### Makefile Targets
```makefile
# Makefile
.PHONY: db-up db-down db-reset db-logs

db-up:
	docker-compose up -d db

db-down:
	docker-compose down

db-reset:
	docker-compose down -v
	docker-compose up -d db
	docker-compose exec db psql -U app -c "CREATE EXTENSION IF NOT EXISTS vector;"

db-logs:
	docker-compose logs -f db

db-shell:
	docker-compose exec db psql -U app

db-verify-pgvector:
	docker-compose exec db psql -U app -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"
```

## Multi-Environment Support

| Environment | Image Tag | Notes |
|-------------|-----------|-------|
| Development | `pgvector/pgvector:pg14` | Latest pgvector for development |
| Testing | `pgvector/pgvector:pg14-v0.5.1` | Pinned version for CI reproducibility |
| Staging | `pgvector/pgvector:pg14-v0.5.1` | Match production version |
| Production | N/A | Use managed service (RDS, Azure, etc.) |

## Dependencies

- Docker Engine 20.10+
- Docker Compose 2.0+
- Existing `docker-compose.yml` to modify
- Network access to Docker Hub

## Success Criteria

- [ ] `docker-compose up -d db` starts pgvector-enabled PostgreSQL
- [ ] pgvector extension appears in `pg_available_extensions`
- [ ] Existing environment variables continue to work
- [ ] Data persistence works across restarts
- [ ] Health check passes
- [ ] Migration from standard postgres is documented
- [ ] Works on Linux, macOS, and Windows (WSL2)
