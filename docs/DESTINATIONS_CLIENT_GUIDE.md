# Client Guide: Sending Documents to GraphRAG Destinations

**Quick Reference for API Consumers**

---

## ⚠️ Common Mistake

### Wrong (No Destinations)
```json
POST /api/v1/jobs
{
  "source_type": "upload",
  "source_uri": "/uploads/doc.pdf",
  "file_name": "doc.pdf",
  "mime_type": "application/pdf"
  // ❌ Missing destinations - only goes to vector store!
}
```
**Result:** Document processed but NOT available in Cognee or HippoRAG search.

### Right (With Destinations)
```json
POST /api/v1/jobs
{
  "source_type": "upload",
  "source_uri": "/uploads/doc.pdf",
  "file_name": "doc.pdf",
  "mime_type": "application/pdf",
  "destinations": [
    {
      "type": "cognee_local",
      "config": { "dataset_id": "default" }
    }
  ]
  // ✅ Document routed to Cognee knowledge graph!
}
```

---

## 📋 Complete Examples

### 1. Send to Cognee (Knowledge Graph)

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
          "dataset_id": "default",
          "graph_name": "default",
          "extract_entities": true,
          "extract_relationships": true
        }
      }
    ],
    "metadata": {
      "project": "tech-docs",
      "category": "specification"
    }
  }'
```

**Query after processing:**
```bash
POST /api/v1/cognee/search
{
  "query": "What are the system requirements?",
  "search_type": "hybrid",
  "dataset_id": "default",
  "top_k": 10
}
```

---

### 2. Send to HippoRAG (Multi-hop Retrieval)

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
          "dataset_id": "default"
        }
      }
    ]
  }'
```

**Query after processing:**
```bash
POST /api/v1/hipporag/qa
{
  "queries": ["How does this method compare to previous approaches?"],
  "dataset_id": "default",
  "num_to_retrieve": 10,
  "generate_answer": true
}
```

---

### 3. Send to Both (Cognee + HippoRAG)

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
    ]
  }'
```

**Benefits:**
- **Cognee**: Entity relationships, graph visualization
- **HippoRAG**: Complex reasoning, multi-hop questions

---

## ✅ Verification Steps

### Step 1: Check Job Status
```bash
GET /api/v1/jobs/{job_id}
```

**Expected response:**
```json
{
  "data": {
    "id": "job-uuid",
    "status": "completed"
    // ...
  }
}
```

### Step 2: Check Cognee Stats
```bash
GET /api/v1/cognee/stats?dataset_id=default
```

**Expected response (after processing):**
```json
{
  "dataset_id": "default",
  "document_count": 1,      // ✅ Should be > 0
  "chunk_count": 15,
  "entity_count": 45,       // ✅ Entities extracted
  "relationship_count": 78, // ✅ Relationships mapped
  "graph_density": 0.34
}
```

**If counts are 0:** Document wasn't sent with Cognee destination.

### Step 3: Test Search
```bash
POST /api/v1/cognee/search
{
  "query": "test query",
  "search_type": "hybrid",
  "dataset_id": "default",
  "top_k": 5
}
```

**Expected:** Results array with content (not empty).

---

## 🎯 Destination Configuration Reference

### Cognee Config Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_id` | string | `"default"` | Logical grouping for documents |
| `graph_name` | string | `"default"` | Named graph within dataset |
| `extract_entities` | boolean | `true` | Extract named entities |
| `extract_relationships` | boolean | `true` | Extract entity relationships |

### HippoRAG Config Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_id` | string | `"default"` | Logical grouping for documents |

---

## 🐛 Troubleshooting

### Problem: "No results found" in Cognee search

**Cause:** Document processed without `cognee_local` destination.

**Fix:** Resubmit with destinations:
```json
{
  "destinations": [
    {
      "type": "cognee_local",
      "config": { "dataset_id": "default" }
    }
  ]
}
```

---

### Problem: Cognee stats show all zeros

```json
{
  "document_count": 0,
  "entity_count": 0,
  "relationship_count": 0
}
```

**Cause:** Knowledge graph is empty.

**Fix:** 
1. Check job was created WITH destinations
2. Check job status is "completed"
3. Verify dataset_id matches between job and query

---

### Problem: "The knowledge graph may be empty" message

**Cause:** Searching before any documents processed with Cognee.

**Fix:** Process at least one document with Cognee destination before searching.

---

## 📊 Quick Decision Tree

```
Want to use Cognee Graph Search?
├── YES → Include destination:
│         {
│           "type": "cognee_local",
│           "config": { "dataset_id": "your-dataset" }
│         }
│
└── NO  → Standard vector search works without destinations

Want to use HippoRAG Multi-hop?
├── YES → Include destination:
│         {
│           "type": "hipporag",
│           "config": { "dataset_id": "your-dataset" }
│         }
│
└── NO  → Standard RAG works without destinations

Want both capabilities?
└── YES → Include BOTH destinations in array
```

---

## 🔗 Related Resources

- **Full PRD:** `docs/JOB_DESTINATIONS_PRD.md`
- **HTTP Examples:** `http/jobs.http` and `http/production/jobs.http`
- **OpenAPI Spec:** `api/openapi.yaml` (schemas: `JobCreateRequest`, `DestinationConfig`)

---

**Remember:** The `destinations` array is **optional but REQUIRED** for GraphRAG. Without it, documents only go to the standard vector store.
