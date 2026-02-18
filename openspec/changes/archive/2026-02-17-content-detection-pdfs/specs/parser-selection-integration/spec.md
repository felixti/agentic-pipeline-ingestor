# Spec: Parser Selection Integration

## Overview

Integration of content detection results with the parser selection logic in the 7-stage processing pipeline. Enables automatic routing to optimal parser based on detected content type.

### Requirements

#### REQ-1: Automatic Parser Selection
**Given** a job with content detection result  
**When** the parser selection stage executes  
**Then** the system automatically selects primary and fallback parsers

#### REQ-2: Selection Strategy Matrix
**Given** detected content type  
**When** applying selection logic  
**Then** the following matrix is used:
- TEXT_BASED → Primary: Docling, Fallback: Azure OCR
- SCANNED → Primary: Azure OCR, Fallback: Docling
- MIXED → Primary: Docling, Fallback: Azure OCR (OCR mode)

#### REQ-3: Override Capability
**Given** a job with explicit parser configuration  
**When** parser selection runs  
**Then** the explicit configuration takes precedence over automatic selection

#### REQ-4: Detection-First Pipeline Flow
**Given** a job created without explicit parser config  
**When** the pipeline starts  
**Then** detection stage runs before parser selection

#### REQ-5: Confidence Threshold
**Given** detection confidence below 0.70  
**When** parser selection occurs  
**Then** the system uses conservative fallback strategy (always include OCR)

#### REQ-6: Parser Selection API
**Given** a detection result  
**When** calling the selection service  
**Then** it returns: primary_parser, fallback_parser, rationale

### Parser Selection Logic

```python
class ParserSelector:
    def select_parser(
        self, 
        detection_result: ContentDetectionResult,
        explicit_config: Optional[ParserConfig] = None
    ) -> ParserSelection:
        
        # Priority 1: Explicit configuration override
        if explicit_config:
            return ParserSelection(
                primary=explicit_config.primary_parser,
                fallback=explicit_config.fallback_parser,
                rationale="User-specified configuration"
            )
        
        # Priority 2: Low confidence - use conservative approach
        if detection_result.confidence < 0.70:
            return ParserSelection(
                primary="docling",
                fallback="azure_ocr",
                rationale=f"Low detection confidence ({detection_result.confidence:.2f}), using conservative strategy"
            )
        
        # Priority 3: Content-based selection
        if detection_result.content_type == ContentType.TEXT_BASED:
            return ParserSelection(
                primary="docling",
                fallback="azure_ocr",
                rationale=f"Text-based PDF detected (confidence: {detection_result.confidence:.2f})"
            )
        elif detection_result.content_type == ContentType.SCANNED:
            return ParserSelection(
                primary="azure_ocr",
                fallback="docling",
                rationale=f"Scanned PDF detected (confidence: {detection_result.confidence:.2f})"
            )
        else:  # MIXED
            return ParserSelection(
                primary="docling",
                fallback="azure_ocr",
                rationale=f"Mixed content detected (confidence: {detection_result.confidence:.2f}), using Docling with OCR fallback"
            )
```

### Pipeline Integration

```
Job Creation
    │
    ▼
[Stage 1: Ingest] ──► File validation, staging
    │
    ▼
[Stage 2: Detect] ──► Content detection (NEW)
    │                       │
    │                       ▼
    │               Store detection result
    │               Link to job
    │
    ▼
[Stage 3: Parse] ──► Parser Selection (MODIFIED)
    │                       │
    │                       ▼
    │               Read detection result
    │               Select primary/fallback
    │                       │
    │                       ▼
    │               Execute primary parser
    │               (Fallback if needed)
    │
    ▼
[Stage 4+: Enrich, Quality, Transform, Output]
```

### API Integration

```python
# GET /api/v1/jobs/{id}/parser-selection
{
  "data": {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "detection_result": {
      "content_type": "text_based",
      "confidence": 0.985
    },
    "selection": {
      "primary_parser": "docling",
      "fallback_parser": "azure_ocr",
      "rationale": "Text-based PDF detected (confidence: 0.99)",
      "overridden": false
    },
    "estimated_processing_time": "30s",
    "estimated_cost": 0.05  # USD
  }
}
```

### Scenarios

#### SC-1: Text-Based Auto-Routing
Job #100 processes "research_paper.pdf". Detection: TEXT_BASED (0.99 confidence). Parser selection chooses Docling primary, Azure OCR fallback. Processing completes in 15s using Docling only.

#### SC-2: Scanned Document Routing
Job #101 processes "scanned_contract.pdf". Detection: SCANNED (0.95 confidence). Parser selection chooses Azure OCR primary, Docling fallback. Azure OCR processes successfully in 45s.

#### SC-3: Explicit Override
Job #102 has detection: TEXT_BASED (0.85), but user specified "force_ocr: true". Parser selection uses Azure OCR primary despite detection result. Override is logged in audit trail.

#### SC-4: Low Confidence Conservative Strategy
Job #103 has detection: MIXED (0.55 confidence). Parser selection uses conservative strategy: Docling primary with Azure OCR fallback enabled. Both parsers attempted.

#### SC-5: Fallback Activation
Job #104: Text-based PDF (0.98 confidence), Docling primary selected. Docling fails on corrupted page 5. System automatically falls back to Azure OCR for complete extraction.

#### SC-6: Cost Estimation
User queries job before processing. API returns detection result + estimated cost: $0.05 for Docling (text-based) vs $0.50 if forced to Azure OCR. User accepts recommendation.
