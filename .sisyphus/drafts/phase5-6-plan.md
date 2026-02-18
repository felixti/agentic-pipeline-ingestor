# Draft: Phases 5 & 6 — Observability, Scale & Advanced Features

## Requirements (confirmed)
- User wants to plan Phases 5 and 6 of the spec together as one work plan
- Phases 1-4 already complete (Foundation, Core Pipeline, Agentic Features, Enterprise)

## What the Spec Says

### Phase 5: Observability & Scale (Weeks 17-20)
- OpenTelemetry enhancement
- Azure deployment
- Performance optimization
- 20GB/day throughput target

### Phase 6: Advanced Features (Weeks 21-24)
- Additional parsers (CSV, JSON, Email)
- More destinations (GraphRAG, Neo4j)
- Entity extraction
- Bulk operations

## What Already Exists (from codebase exploration)
- Observability module exists: tracing.py, metrics.py, logging.py, middleware.py, genai_spans.py
- GenAI spans following OpenTelemetry semantic conventions
- Prometheus metrics with MetricsManager
- OTEL collector config and Prometheus config files exist
- Docker compose with services but unclear if Grafana/Jaeger included
- AKS deployment YAML exists
- CSV, JSON, XML, Email parser files exist — COMPLETENESS TBD
- GraphRAG, Neo4j, Pinecone, Weaviate destination files exist — COMPLETENESS TBD
- Entity extraction module exists — COMPLETENESS TBD
- Bulk operations route exists — COMPLETENESS TBD
- GraphRAG core (knowledge_graph, community_detection) exists — COMPLETENESS TBD

## Known Production Gaps (from PHASE4_IMPLEMENTATION_SUMMARY)
1. Database migrations not created (Alembic)
2. In-memory stores for audit/lineage (need PostgreSQL/OpenSearch backends)
3. No Azure Key Vault or HashiCorp Vault integration
4. No API key rate limiting implementation
5. No JWT token blacklist/revocation
6. Missing audit log export metrics and lineage tracking metrics
7. Missing unit/integration tests for auth components
8. Coverage threshold set at 1% (target is 80%)
9. Job storage is in-memory (OrchestrationEngine._active_jobs) — no DB persistence

## Technical Decisions (CONFIRMED)
- **Test strategy**: TDD (test-first) — RED → GREEN → REFACTOR cycle
- **Production gaps**: Include CRITICAL only (Alembic migrations, DB-backed job store). Defer rate limiting, vault, JWT revocation to Phase 7.
- **Phase 6 completeness**: Functional with graceful fallbacks — real LLM calls where possible, keep regex/statistical fallbacks as acceptable alternatives
- **Load testing**: YES — include tasks for 20GB/day throughput validation
- **Scope**: No exclusions — all spec items included

## Research Findings

### Phase 5 Assessment (COMPLETE — bg_87974c1d)

| Component | Status | Notes |
|-----------|--------|-------|
| OTEL Tracing | PRODUCTION-READY | Jaeger/OTLP exporters, pipeline stage spans |
| Prometheus Metrics | PRODUCTION-READY | 50+ metrics, MetricsManager, /metrics endpoint |
| Structured Logging | PRODUCTION-READY | structlog JSON with trace context |
| Request Middleware | PRODUCTION-READY | Span context, correlation IDs |
| GenAI Spans | PRODUCTION-READY | OpenTelemetry semantic conventions |
| OTEL Collector Config | PRODUCTION-READY | Jaeger + Prometheus exporters |
| Prometheus Config | PRODUCTION-READY | Scrape jobs configured |
| Docker Compose | PRODUCTION-READY | Prometheus, Grafana, Jaeger, OTEL Collector |
| AKS Deployment | PRODUCTION-READY | HPA, PDB, NetworkPolicy, Prometheus annotations |
| Grafana Dashboard | PRODUCTION-READY | Basic pipeline dashboard (563 lines) |
| **DB Migrations** | **NEEDS-WORK** | **No Alembic directory — CRITICAL GAP** |

**Phase 5 ~80% done. Remaining: 6-8 days.**

Gaps:
- Alembic migrations (CRITICAL, 2-3 days)
- Prometheus alerting rules (1 day)
- Enhanced Grafana dashboards (2 days)
- GenAI span wiring in LLM provider (1 day)
- Metrics refinement (0.5 day)
- Load testing & verification (1 day)

### Phase 6 Assessment (COMPLETE — bg_e30564f5)

| File | Status | Lines | Summary |
|------|--------|-------|---------|
| csv_parser.py | PARTIAL | 584 | Real CSV parsing with pandas; conditional JSON/schema validation mocked |
| json_parser.py | PARTIAL | 705 | Real JSON parsing with schema; advanced validation placeholder LLM |
| xml_parser.py | PARTIAL | 568 | Real XML parsing with XPath/XSD; XSLT transformation stub only |
| email_parser.py | PARTIAL | 806 | Real email parsing (headers, multipart); attachment OCR placeholder |
| graphrag.py | PARTIAL | 640 | Real graph building; community detection async placeholders |
| neo4j.py | PARTIAL | 720 | Real Neo4j driver calls; custom relationship mapping incomplete |
| pinecone.py | PARTIAL | 553 | Real Pinecone API calls; metadata enrichment stubbed |
| weaviate.py | PARTIAL | 656 | Real schema validation; batch operations mock responses |
| entity_extraction.py | PARTIAL | 555 | Real entity extraction; LLM falls back to regex patterns |
| advanced.py | COMPLETE | 741 | Full enrichment (summarization, sentiment, topics, keywords) |
| knowledge_graph.py | COMPLETE | 612 | Complete KG construction; pure data structures, no external calls |
| community_detection.py | PARTIAL | 544 | Louvain/label propagation real; spectral clustering placeholder |
| optimizations.py | PARTIAL | 522 | Real caching/batching; performance profiling mock |
| healing.py | PARTIAL | 693 | Real retry/circuit breaker; anomaly detection placeholder ML |
| learning.py | PARTIAL | 576 | Real feedback loop; model retraining NotImplementedError |
| bulk.py | PARTIAL | 567 | Real bulk job submission; retry logic simplified |
| dlq.py | PARTIAL | 419 | Real DLQ management; failure analysis basic heuristics |

**Result: 2 COMPLETE, 15 PARTIAL, 0 STUB. All files have real logic, but external service integration and advanced features have gaps.**

Key gap patterns:
1. External service mocking (Neo4j/Weaviate/Pinecone batch ops)
2. LLM fallback (entity extraction, email classification, healing)
3. Algorithm placeholders (spectral clustering, ML anomaly detection)
4. Simplified heuristics (DLQ failure classification, bulk retry)

## Open Questions — ALL RESOLVED
1. Production gaps: Include CRITICAL only ✅
2. Priority: Foundation first (API stubs, DB) → Phase 5 → Phase 6 ✅
3. Load testing: YES ✅
4. Azure: Config/templates already production-ready ✅
5. Test strategy: TDD ✅
6. Scope: Everything ✅

## Metis Corrections (IMPORTANT)
1. Parsers (CSV/JSON/XML/Email) are 100% COMPLETE, zero LLM dependency — SKIP in plan
2. learning.py has NO retrain_model() NotImplementedError — it's a working statistical system
3. Destination batch ops are REAL — mock classes are separate test helpers
4. GraphRAG community detection is 75% real — only Spectral Clustering needs work
5. ALL 12 API routes in main.py are TODO STUBS — must wire before any Phase 5-6 work
6. Only health.router is wired via include_router() — 5 other route files are dead code
7. Middleware is duplicated (inline in main.py + class in middleware.py) — must consolidate
8. Embedding generation is None placeholder in TransformStage — pipeline output incomplete
9. create_all() in startup will conflict with Alembic — must guard

## Scope Boundaries
- INCLUDE: All Phase 5-6 spec items, Alembic migrations, DB-backed job store, load testing
- EXCLUDE: Azure Key Vault, rate limiting, JWT revocation, secret vault, test coverage target increase (defer to Phase 7 hardening)
