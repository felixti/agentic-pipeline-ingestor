# Capability: Neo4j Infrastructure

## Overview

Neo4j graph database service for Cognee GraphRAG storage.

## Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| NEO-001 | Neo4j Community Edition in Docker Compose | Must |
| NEO-002 | Persistent volume for graph data | Must |
| NEO-003 | Health check endpoint | Must |
| NEO-004 | Memory limit configuration (max 2GB) | Must |
| NEO-005 | Bolt protocol on port 7687 | Must |
| NEO-006 | Browser UI on port 7474 | Should |
| NEO-007 | Authentication enabled | Must |

## Docker Compose Service

```yaml
neo4j:
  image: neo4j:5.15-community
  container_name: pipeline-neo4j
  environment:
    - NEO4J_AUTH=neo4j/cognee-graph-db
    - NEO4J_PLUGINS=["apoc", "gds"]  # Graph Data Science
    - NEO4J_dbms_memory_heap_max__size=2G
    - NEO4J_dbms_memory_pagecache_size=1G
  volumes:
    - neo4j-data:/data
    - neo4j-logs:/logs
  ports:
    - "7687:7687"  # Bolt
    - "7474:7474"  # HTTP/Browser
  networks:
    - pipeline-network
  restart: unless-stopped
  healthcheck:
    test: ["CMD", "cypher-shell", "-u", "neo4j", "-p", "cognee-graph-db", "RETURN 1"]
    interval: 30s
    timeout: 10s
    retries: 5
```

## Configuration

| Setting | Value | Description |
|---------|-------|-------------|
| Image | `neo4j:5.15-community` | Community Edition |
| Memory | 2GB max | JVM heap limit |
| Page Cache | 1GB | Graph data caching |
| Auth | Enabled | neo4j/cognee-graph-db |
| Plugins | APOC, GDS | Graph algorithms |

## Volumes

| Volume | Mount | Purpose |
|--------|-------|---------|
| neo4j-data | /data | Graph database files |
| neo4j-logs | /logs | Query logs |

## Network

- Joins `pipeline-network`
- Accessible as `neo4j` hostname
- Bolt: port 7687
- HTTP: port 7474

## Security

- Authentication required
- Internal network only
- No external ports exposed (internal use only)

## Health Check

```cypher
RETURN 1
```

## Backup Strategy

```bash
# Create backup
docker exec pipeline-neo4j neo4j-admin database dump neo4j --to-path=/backups

# Restore from backup
docker exec pipeline-neo4j neo4j-admin database load neo4j --from-path=/backups
```

## Monitoring

| Metric | Source |
|--------|--------|
| Node count | `MATCH (n) RETURN count(n)` |
| Relationship count | `MATCH ()-[r]->() RETURN count(r)` |
| Memory usage | Neo4j metrics endpoint |
| Query performance | Query log analysis |
