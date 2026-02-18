# Spec: Content Detection API

## Overview

REST API endpoint for on-demand content detection of PDF files. Supports both standalone detection and integration with job creation flow.

### Requirements

#### REQ-1: On-Demand Detection Endpoint
**Given** a PDF file uploaded by a user  
**When** POST /api/v1/detect is called with the file  
**Then** the system returns content detection results within 2 seconds

#### REQ-2: Detection from URL
**Given** a PDF accessible via URL  
**When** POST /api/v1/detect/url is called with the URL  
**Then** the system downloads, analyzes, and returns detection results

#### REQ-3: Synchronous Response
**Given** a detection request  
**When** processing completes  
**Then** the response includes: content_type, confidence, recommended_parser, and processing_time_ms

#### REQ-4: Batch Detection
**Given** multiple PDF files (up to 10)  
**When** POST /api/v1/detect/batch is called  
**Then** the system returns aggregated results for all files

#### REQ-5: Error Handling
**Given** an invalid file (corrupted PDF, wrong format)  
**When** detection is attempted  
**Then** the API returns 400 Bad Request with specific error code

#### REQ-6: Rate Limiting
**Given** a user making detection requests  
**When** the rate exceeds 60 requests/minute  
**Then** the API returns 429 Too Many Requests

### API Specification

#### POST /api/v1/detect

**Request:**
```http
POST /api/v1/detect
Content-Type: multipart/form-data
X-API-Key: <api_key>

file: <binary_pdf_data>
detailed: true (optional, default: false)
```

**Response (200 OK):**
```json
{
  "data": {
    "content_type": "text_based",
    "confidence": 0.985,
    "recommended_parser": "docling",
    "alternative_parsers": ["azure_ocr"],
    "text_statistics": {
      "total_pages": 10,
      "total_characters": 45000,
      "has_text_layer": true,
      "font_count": 3
    },
    "image_statistics": {
      "total_images": 2,
      "image_area_ratio": 0.02
    },
    "page_results": [
      {
        "page_number": 1,
        "content_type": "text_based",
        "confidence": 0.99,
        "text_ratio": 0.95
      }
    ],
    "processing_time_ms": 245
  },
  "meta": {
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "timestamp": "2026-02-17T14:30:00Z"
  }
}
```

**Response (400 Bad Request):**
```json
{
  "error": {
    "code": "INVALID_FILE_FORMAT",
    "message": "File is not a valid PDF",
    "details": [{"field": "file", "issue": "corrupted_pdf_structure"}]
  },
  "meta": {
    "request_id": "uuid",
    "timestamp": "2026-02-17T14:30:00Z"
  }
}
```

#### POST /api/v1/detect/url

**Request:**
```http
POST /api/v1/detect/url
Content-Type: application/json

{
  "url": "https://example.com/document.pdf",
  "headers": {
    "Authorization": "Bearer token"
  }
}
```

**Response:** Same as /detect endpoint

#### POST /api/v1/detect/batch

**Request:**
```http
POST /api/v1/detect/batch
Content-Type: multipart/form-data

files: [file1.pdf, file2.pdf, ...]  // Max 10 files
```

**Response:**
```json
{
  "data": {
    "results": [
      {
        "filename": "doc1.pdf",
        "content_type": "text_based",
        "confidence": 0.98
      },
      {
        "filename": "doc2.pdf",
        "content_type": "scanned",
        "confidence": 0.95
      }
    ],
    "summary": {
      "total": 2,
      "text_based": 1,
      "scanned": 1,
      "mixed": 0
    }
  }
}
```

### Scenarios

#### SC-1: Single File Detection
User uploads a PDF via web UI. System returns detection in 245ms indicating "text_based" with 98.5% confidence. UI shows "✓ Text-based document detected" before processing begins.

#### SC-2: URL-Based Detection
User provides a SharePoint URL to a PDF. System downloads (2MB), analyzes, returns result in 1.2s. Recommended parser: azure_ocr (scanned document detected).

#### SC-3: Batch Upload
User uploads 5 PDFs from project folder. System processes all in parallel, returns results in 800ms. 3 text-based, 2 scanned - batch processing routes each to appropriate parser.

#### SC-4: Invalid File Handling
User uploads a corrupted PDF. System returns 400 error immediately with code "CORRUPTED_PDF", avoiding unnecessary processing attempts.
