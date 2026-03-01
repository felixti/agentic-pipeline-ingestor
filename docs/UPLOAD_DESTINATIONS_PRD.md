# PRD: Upload with Destinations Support

**Version:** 1.0  
**Date:** 2026-03-01  
**Status:** Implemented  

---

## Overview

The `/upload` endpoint now supports **destinations** directly. This eliminates the need for a two-step process (upload then create job with destinations).

**Before (Two-Step):**
1. POST `/upload` → Get job_id
2. POST `/jobs` with destinations → Route to Cognee/HippoRAG

**After (Single Step):**
1. POST `/upload` with destinations → Job created with destinations

---

## API Changes

### Endpoint
```
POST /api/v1/upload
Content-Type: multipart/form-data
```

### New Form Data Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `files` | File | ✅ | File(s) to upload |
| `destinations` | JSON string | ❌ | Array of destination configurations |
| `priority` | string | ❌ | Job priority: `low`, `normal`, `high` (default: `normal`) |
| `mode` | string | ❌ | Processing mode: `sync`, `async` (default: `async`) |

### Destinations JSON Format

```json
[
  {
    "type": "cognee_local",
    "config": {
      "dataset_id": "default",
      "graph_name": "default",
      "extract_entities": true,
      "extract_relationships": true
    }
  },
  {
    "type": "hipporag",
    "config": {
      "dataset_id": "default"
    }
  }
]
```

---

## Usage Examples

### 1. Upload with Cognee Destination

```bash
curl -X POST "https://api.example.com/api/v1/upload" \
  -H "X-API-Key: your-api-key" \
  -F "files=@document.pdf" \
  -F 'destinations=[{"type":"cognee_local","config":{"dataset_id":"default","extract_entities":true,"extract_relationships":true}}]'
```

**Response:**
```json
{
  "data": {
    "message": "File uploaded successfully",
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "file_name": "document.pdf",
    "file_size": 10196204,
    "destinations": [
      {
        "type": "cognee_local",
        "config": {
          "dataset_id": "default",
          "extract_entities": true,
          "extract_relationships": true
        }
      }
    ]
  },
  "meta": {
    "request_id": "...",
    "timestamp": "2026-03-01T..."
  }
}
```

---

### 2. Upload with HippoRAG Destination

```bash
curl -X POST "https://api.example.com/api/v1/upload" \
  -H "X-API-Key: your-api-key" \
  -F "files=@research-paper.pdf" \
  -F 'destinations=[{"type":"hipporag","config":{"dataset_id":"default"}}]'
```

---

### 3. Upload with Multiple Destinations

```bash
curl -X POST "https://api.example.com/api/v1/upload" \
  -H "X-API-Key: your-api-key" \
  -F "files=@comprehensive-guide.pdf" \
  -F 'destinations=[{"type":"cognee_local","config":{"dataset_id":"kb"}},{"type":"hipporag","config":{"dataset_id":"kb"}}]'
```

---

### 4. Upload with Priority and Mode

```bash
curl -X POST "https://api.example.com/api/v1/upload" \
  -H "X-API-Key: your-api-key" \
  -F "files=@urgent-doc.pdf" \
  -F 'destinations=[{"type":"cognee_local","config":{"dataset_id":"default"}}]' \
  -F "priority=high" \
  -F "mode=async"
```

---

## JavaScript/TypeScript Example

```typescript
async function uploadWithDestinations(
  file: File,
  destinations: DestinationConfig[]
): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('files', file);
  formData.append('destinations', JSON.stringify(destinations));
  formData.append('priority', 'normal');
  formData.append('mode', 'async');

  const response = await fetch('https://api.example.com/api/v1/upload', {
    method: 'POST',
    headers: {
      'X-API-Key': 'your-api-key',
    },
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Upload failed: ${response.statusText}`);
  }

  return response.json();
}

// Usage
const destinations = [
  {
    type: 'cognee_local',
    config: {
      dataset_id: 'default',
      graph_name: 'default',
      extract_entities: true,
      extract_relationships: true,
    },
  },
];

const result = await uploadWithDestinations(file, destinations);
console.log('Job ID:', result.data.job_id);
console.log('Destinations:', result.data.destinations);
```

---

## Destination Types Reference

### Cognee (Knowledge Graph)

```json
{
  "type": "cognee_local",
  "config": {
    "dataset_id": "default",
    "graph_name": "default",
    "extract_entities": true,
    "extract_relationships": true
  }
}
```

| Config Parameter | Type | Default | Description |
|-----------------|------|---------|-------------|
| `dataset_id` | string | `default` | Logical grouping for documents |
| `graph_name` | string | `default` | Named graph within dataset |
| `extract_entities` | boolean | `true` | Extract named entities |
| `extract_relationships` | boolean | `true` | Extract entity relationships |

---

### HippoRAG (Multi-hop Retrieval)

```json
{
  "type": "hipporag",
  "config": {
    "dataset_id": "default"
  }
}
```

| Config Parameter | Type | Default | Description |
|-----------------|------|---------|-------------|
| `dataset_id` | string | `default` | Logical grouping for documents |

---

## Error Handling

### Invalid Destinations JSON

```json
{
  "detail": {
    "error": {
      "code": "INVALID_DESTINATIONS",
      "message": "destinations must be valid JSON: ..."
    }
  }
}
```

### Destinations Not an Array

```json
{
  "detail": {
    "error": {
      "code": "INVALID_DESTINATIONS",
      "message": "destinations must be a JSON array"
    }
  }
}
```

---

## Migration Guide

### Before (Two-Step)

```javascript
// Step 1: Upload file
const uploadForm = new FormData();
uploadForm.append('files', file);
const uploadRes = await fetch('/api/v1/upload', {
  method: 'POST',
  body: uploadForm,
});
const uploadData = await uploadRes.json();

// Step 2: Get job details to find source_uri
const jobRes = await fetch(`/api/v1/jobs/${uploadData.data.job_id}`);
const jobData = await jobRes.json();

// Step 3: Create new job with destinations
const jobForm = {
  source_type: 'upload',
  source_uri: jobData.data.source_uri,
  file_name: file.name,
  mime_type: file.type,
  destinations: [...],
};
const newJobRes = await fetch('/api/v1/jobs', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(jobForm),
});
```

### After (Single Step)

```javascript
// Single step: Upload with destinations
const formData = new FormData();
formData.append('files', file);
formData.append('destinations', JSON.stringify([...]));

const response = await fetch('/api/v1/upload', {
  method: 'POST',
  body: formData,
});
const data = await response.json();
// Job is already created with destinations!
```

---

## Backward Compatibility

The `/upload` endpoint is **fully backward compatible**. Existing clients that don't send `destinations` will continue to work exactly as before (files go to standard vector store only).

---

## Summary

| Feature | Status |
|---------|--------|
| Single-step upload with destinations | ✅ Implemented |
| Backward compatibility | ✅ Maintained |
| Multiple destinations support | ✅ Supported |
| Priority and mode parameters | ✅ Supported |
| Error handling for invalid JSON | ✅ Implemented |

---

**End of Document**
