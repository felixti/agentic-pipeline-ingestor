# Product Requirements Document: Job Destinations API
## GraphRAG Integration Guide for Developers

**Version:** 1.0  
**Date:** 2026-03-01  
**Status:** Implemented  
**API Version:** v1  

---

## 1. Executive Summary

The **Job Destinations API** enables developers to route processed documents to specialized GraphRAG systems during job creation. Instead of just storing documents in the standard vector store, you can now simultaneously send processed content to:

- **Cognee**: Knowledge graph construction with Neo4j (entity extraction, relationship mapping)
- **HippoRAG**: Multi-hop retrieval indexing with Personalized PageRank

This document provides complete integration guidance for developers building on the Agentic Pipeline Ingestor API.

---

## 2. Overview

### 2.1 What Are Destinations?

Destinations are optional routing configurations specified during job creation. When a document is processed through the pipeline (parse → chunk → embed), the resulting data is sent to all configured destinations.

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────┐
│   Upload    │───▶│   Parse     │───▶│   Embed     │───▶│        DESTINATIONS         │
│  Document   │    │   & Chunk   │    │   Chunks    │    ├─────────────┬───────────────┤
└─────────────┘    └─────────────┘    └─────────────┘    │   Cognee    │   HippoRAG    │
                                                         │   (Neo4j)   │   (PPR Index) │
                                                         └─────────────┴───────────────┘
```

### 2.2 Destination Types

| Destination | Type | Use Case | Query Endpoint |
|-------------|------|----------|----------------|
| **cognee_local** | Knowledge Graph | Entity extraction, relationship analysis, graph traversal | `POST /cognee/search` |
| **hipporag** | Multi-hop Index | Complex reasoning, multi-hop questions, PPR retrieval | `POST /hipporag/retrieve` `POST /hipporag/qa` |

### 2.3 Benefits

- **Single Pipeline**: Process once, route to multiple systems
- **Flexibility**: Mix and match destinations per job
- **No Extra Work**: Destinations are transparent - standard job flow continues
- **Powerful Queries**: Access GraphRAG capabilities for complex queries

---

## 3. API Specification

### 3.1 Job Creation with Destinations

**Endpoint:** `POST /api/v1/jobs`

**Authentication:** Required (`X-API-Key` header)

**Content-Type:** `application/json`

#### Request Body Schema

```json
{
  "source_type": "upload",
  "source_uri": "/uploads/document.pdf",
  "file_name": "document.pdf",
  "mime_type": "application/pdf",
  "mode": "async",
  "destinations": [
    {
      "type": "cognee_local",
      "config": {
        "dataset_id": "default",
        "graph_name": "default",
        "extract_entities": true,
        "extract_relationships": true
      },
      "filters": {},
      "enabled": true
    }
  ],
  "metadata": {}
}
```

#### Destinations Array Structure

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | ✅ | Destination type: `"cognee_local"` or `"hipporag"` |
| `config` | object | ✅ | Destination-specific configuration |
| `filters` | object | ❌ | Optional filters for conditional routing |
| `enabled` | boolean | ❌ | Enable/disable this destination (default: `true`) |

---

## 4. Destination Configurations

### 4.1 Cognee (Knowledge Graph)

**Type:** `cognee_local`

Cognee constructs a knowledge graph from your documents using Neo4j. It extracts entities and relationships, enabling graph-based search and traversal.

#### Config Schema

```json
{
  "type": "cognee_local",
  "config": {
    "dataset_id": "string",           // Dataset identifier (default: "default")
    "graph_name": "string",           // Graph name (default: "default")
    "extract_entities": true,         // Extract named entities
    "extract_relationships": true     // Extract entity relationships
  }
}
```

#### Config Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_id` | string | `"default"` | Logical grouping for related documents |
| `graph_name` | string | `"default"` | Named graph within the dataset |
| `extract_entities` | boolean | `true` | Enable named entity extraction |
| `extract_relationships` | boolean | `true` | Enable relationship extraction |

#### Querying Cognee Data

After processing, query the knowledge graph:

```bash
curl -X POST "https://api.example.com/api/v1/cognee/search" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "query": "What are the relationships between Kubernetes and Docker?",
    "search_type": "hybrid",
    "dataset_id": "default",
    "top_k": 10
  }'
```

#### Search Types

| Type | Description | Best For |
|------|-------------|----------|
| `hybrid` | Vector + Graph traversal | General queries (recommended) |
| `vector` | Pure vector similarity | Semantic similarity |
| `graph` | Graph traversal only | Finding relationships |
| `summary` | Synthesized answer | High-level questions |
| `chunks` | Raw chunks | Direct content retrieval |

---

### 4.2 HippoRAG (Multi-hop Retrieval)

**Type:** `hipporag`

HippoRAG indexes documents for multi-hop reasoning using Personalized PageRank (PPR). It excels at complex questions requiring multiple inference steps.

#### Config Schema

```json
{
  "type": "hipporag",
  "config": {
    "dataset_id": "string"            // Dataset identifier (default: "default")
  }
}
```

#### Config Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_id` | string | `"default"` | Logical grouping for related documents |

#### Querying HippoRAG Data

**Multi-hop Retrieval:**
```bash
curl -X POST "https://api.example.com/api/v1/hipporag/retrieve" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "queries": ["How does Docker relate to Kubernetes orchestration?"],
    "dataset_id": "default",
    "num_to_retrieve": 10
  }'
```

**Full RAG with Answer Generation:**
```bash
curl -X POST "https://api.example.com/api/v1/hipporag/qa" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "queries": ["Explain the relationship between containers and microservices"],
    "dataset_id": "default",
    "num_to_retrieve": 15,
    "generate_answer": true
  }'
```

---

## 5. Integration Examples

### 5.1 Basic Cognee Integration

Route a technical document to Cognee for knowledge graph construction:

```bash
curl -X POST "https://api.example.com/api/v1/jobs" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -H "Accept: application/json" \
  -d '{
    "source_type": "upload",
    "source_uri": "/uploads/architecture-doc.pdf",
    "file_name": "architecture-doc.pdf",
    "mime_type": "application/pdf",
    "mode": "async",
    "destinations": [
      {
        "type": "cognee_local",
        "config": {
          "dataset_id": "tech-docs",
          "graph_name": "architecture",
          "extract_entities": true,
          "extract_relationships": true
        }
      }
    ],
    "metadata": {
      "project": "platform-architecture",
      "team": "engineering"
    }
  }'
```

**Response:**
```json
{
  "id": "job-uuid-123",
  "status": "pending",
  "message": "Job created successfully",
  "destinations": ["cognee_local"]
}
```

---

### 5.2 Basic HippoRAG Integration

Route a research paper to HippoRAG for multi-hop indexing:

```bash
curl -X POST "https://api.example.com/api/v1/jobs" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -H "Accept: application/json" \
  -d '{
    "source_type": "upload",
    "source_uri": "/uploads/research-paper.pdf",
    "file_name": "research-paper.pdf",
    "mime_type": "application/pdf",
    "mode": "async",
    "destinations": [
      {
        "type": "hipporag",
        "config": {
          "dataset_id": "research-papers"
        }
      }
    ]
  }'
```

---

### 5.3 Multi-Destination Integration

Route a comprehensive document to **both** Cognee and HippoRAG:

```bash
curl -X POST "https://api.example.com/api/v1/jobs" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -H "Accept: application/json" \
  -d '{
    "source_type": "upload",
    "source_uri": "/uploads/comprehensive-guide.pdf",
    "file_name": "comprehensive-guide.pdf",
    "mime_type": "application/pdf",
    "mode": "async",
    "destinations": [
      {
        "type": "cognee_local",
        "config": {
          "dataset_id": "knowledge-base",
          "graph_name": "main",
          "extract_entities": true,
          "extract_relationships": true
        }
      },
      {
        "type": "hipporag",
        "config": {
          "dataset_id": "knowledge-base"
        }
      }
    ],
    "metadata": {
      "type": "comprehensive-guide",
      "priority": "high"
    }
  }'
```

**Why Both?**
- **Cognee**: Entity relationships, graph visualization, structured queries
- **HippoRAG**: Complex reasoning, multi-hop questions, PPR-based retrieval

---

### 5.4 Conditional Routing with Filters

Route only specific content types to destinations:

```bash
curl -X POST "https://api.example.com/api/v1/jobs" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -H "Accept: application/json" \
  -d '{
    "source_type": "upload",
    "source_uri": "/uploads/technical-spec.pdf",
    "file_name": "technical-spec.pdf",
    "mime_type": "application/pdf",
    "mode": "async",
    "destinations": [
      {
        "type": "cognee_local",
        "config": {
          "dataset_id": "tech-specs",
          "graph_name": "specifications"
        },
        "filters": {
          "mime_type": "application/pdf"
        },
        "enabled": true
      }
    ]
  }'
```

---

## 6. Use Cases

### 6.1 Technical Documentation Platform

**Scenario:** Build a knowledge base from technical documentation

**Implementation:**
```json
{
  "destinations": [
    {
      "type": "cognee_local",
      "config": {
        "dataset_id": "docs",
        "extract_entities": true,
        "extract_relationships": true
      }
    }
  ]
}
```

**Query Pattern:**
```bash
# Find all components related to "authentication"
POST /cognee/search
{
  "query": "What components are involved in authentication?",
  "search_type": "graph"
}
```

---

### 6.2 Research Paper Analysis

**Scenario:** Index research papers for complex literature review

**Implementation:**
```json
{
  "destinations": [
    {
      "type": "hipporag",
      "config": {
        "dataset_id": "papers-2024"
      }
    }
  ]
}
```

**Query Pattern:**
```bash
# Multi-hop reasoning across papers
POST /hipporag/qa
{
  "queries": ["How does method X in paper A relate to results in paper B?"],
  "generate_answer": true
}
```

---

### 6.3 Enterprise Knowledge Management

**Scenario:** Comprehensive knowledge management with both graph and multi-hop

**Implementation:**
```json
{
  "destinations": [
    {
      "type": "cognee_local",
      "config": {
        "dataset_id": "enterprise",
        "graph_name": "org-knowledge"
      }
    },
    {
      "type": "hipporag",
      "config": {
        "dataset_id": "enterprise"
      }
    }
  ]
}
```

**Query Patterns:**
- Use **Cognee** for: "Show me the org chart structure"
- Use **HippoRAG** for: "How does the marketing strategy impact sales in Q3?"

---

### 6.4 Customer Support Knowledge Base

**Scenario:** Build searchable support documentation

**Implementation:**
```json
{
  "destinations": [
    {
      "type": "hipporag",
      "config": {
        "dataset_id": "support-kb"
      }
    }
  ]
}
```

**Query Pattern:**
```bash
# Complex troubleshooting questions
POST /hipporag/qa
{
  "queries": ["Why am I getting error X when doing Y with configuration Z?"],
  "generate_answer": true
}
```

---

## 7. Best Practices

### 7.1 Dataset Organization

**Use meaningful dataset IDs:**
```json
// Good
"dataset_id": "engineering-docs-2024"

// Avoid
"dataset_id": "default"
```

**Organize by project/domain:**
```json
"dataset_id": "product-user-guides"
"dataset_id": "api-documentation"
"dataset_id": "research-papers-ml"
```

---

### 7.2 Job Mode Selection

| Mode | Use Case | Destinations Support |
|------|----------|---------------------|
| `async` | Production workloads | ✅ Yes |
| `sync` | Testing/development | ✅ Yes (but slower) |

**Recommendation:** Always use `async` for production with destinations.

---

### 7.3 Error Handling

Destinations run asynchronously after job completion. Check job status to verify destination processing:

```bash
# Check job status
GET /api/v1/jobs/{job_id}

# Response includes destination_results
{
  "id": "job-uuid",
  "status": "completed",
  "destination_results": {
    "cognee_local": {
      "status": "success",
      "entities_extracted": 45,
      "relationships_extracted": 78
    },
    "hipporag": {
      "status": "success",
      "triples_indexed": 120
    }
  }
}
```

---

### 7.4 Performance Considerations

| Factor | Impact | Recommendation |
|--------|--------|----------------|
| Document size | Larger = slower | Use chunking options |
| Multiple destinations | 2x processing | Use async mode |
| Entity extraction | CPU intensive | Disable if not needed |
| Graph complexity | Memory intensive | Monitor Neo4j resources |

---

### 7.5 Security

- **API Key Required** for job creation (including destinations)
- **No Auth Required** for Cognee/HippoRAG query endpoints
- **Dataset Isolation** - datasets are logically separated
- **Graph Names** - use to separate different knowledge domains

---

## 8. Migration Guide

### 8.1 From Standard Jobs

**Before (Standard Vector Store Only):**
```json
{
  "source_type": "upload",
  "source_uri": "/uploads/doc.pdf",
  "mode": "async"
  // No destinations - only vector store
}
```

**After (With Cognee):**
```json
{
  "source_type": "upload",
  "source_uri": "/uploads/doc.pdf",
  "mode": "async",
  "destinations": [
    {
      "type": "cognee_local",
      "config": {
        "dataset_id": "my-dataset",
        "extract_entities": true
      }
    }
  ]
}
```

**Behavior:** Document goes to both vector store (default) AND Cognee.

---

### 8.2 Backward Compatibility

Jobs without `destinations` field continue to work exactly as before (vector store only).

---

## 9. Troubleshooting

### 9.1 Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Destination not triggered | Invalid type | Use `"cognee_local"` or `"hipporag"` |
| Empty graph results | Wrong dataset_id | Verify dataset_id matches |
| Slow processing | Large documents | Use async mode with chunking |
| Entity extraction fails | Memory limits | Reduce document size |

### 9.2 Debugging

Check job destination results:
```bash
GET /api/v1/jobs/{job_id}
```

Look for:
```json
{
  "destination_results": {
    "cognee_local": {
      "status": "success|failed|pending"
    }
  }
}
```

---

## 10. HTTP Client Examples

Complete HTTP client files are available in the repository:

| File | Description |
|------|-------------|
| `http/jobs.http` | Local development examples |
| `http/production/jobs.http` | Production examples |
| `http/cognee.graphrag.http` | Cognee query examples |
| `http/hipporag.graphrag.http` | HippoRAG query examples |
| `http/all-searches.http` | Combined search examples |

---

## 11. API Reference

### 11.1 Related Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/jobs` | POST | Create job with destinations |
| `/api/v1/jobs/{id}` | GET | Get job status with destination results |
| `/api/v1/cognee/search` | POST | Query Cognee knowledge graph |
| `/api/v1/cognee/extract-entities` | POST | Extract entities from text |
| `/api/v1/cognee/stats` | GET | Get graph statistics |
| `/api/v1/hipporag/retrieve` | POST | Multi-hop retrieval |
| `/api/v1/hipporag/qa` | POST | RAG with answer generation |
| `/api/v1/hipporag/extract-triples` | POST | Extract OpenIE triples |

### 11.2 OpenAPI Specification

See `api/openapi.yaml` for complete schema definitions:
- `JobCreateRequest`
- `DestinationConfig`
- `CogneeSearchRequest`
- `HippoRAGRetrieveRequest`

---

## 12. Glossary

| Term | Definition |
|------|------------|
| **GraphRAG** | Retrieval-Augmented Generation using knowledge graphs |
| **Knowledge Graph** | Graph structure representing entities and relationships |
| **Multi-hop** | Reasoning across multiple document chunks/steps |
| **PPR** | Personalized PageRank - algorithm for importance scoring |
| **OpenIE** | Open Information Extraction - extracting triples from text |
| **Dataset** | Logical grouping of documents in a destination |
| **Destination** | Target system for processed document data |

---

## 13. Support

For issues or questions:
1. Check this PRD and HTTP examples
2. Review OpenAPI spec: `api/openapi.yaml`
3. Check QA report: `shared/qa-reports/`

---

**End of Document**
