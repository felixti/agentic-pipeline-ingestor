# Spec: PDF Analysis Engine

## Overview

Core engine for analyzing PDF structure and content to determine if it's text-based or scanned. Uses multiple heuristics for high-confidence classification.

### Requirements

#### REQ-1: Text Layer Detection
**Given** a PDF file with embedded text  
**When** the analysis engine processes it  
**Then** it detects the presence of extractable text layers

#### REQ-2: Image Analysis
**Given** a PDF with embedded images  
**When** the analysis engine processes it  
**Then** it calculates image-to-page ratio and image density

#### REQ-3: Text Statistics Extraction
**Given** any PDF file  
**When** analysis is performed  
**Then** it extracts: character count, word count, font information, encoding details

#### REQ-4: Page-by-Page Analysis
**Given** a multi-page PDF  
**When** analysis is performed  
**Then** results are provided per-page with overall document classification

#### REQ-5: Classification Decision Matrix
**Given** analysis results  
**When** applying classification rules  
**Then** the following matrix is used:
- Text ratio > 95% → TEXT_BASED
- Text ratio < 5% AND images > 90% → SCANNED
- Mixed content → MIXED (with confidence scores)

#### REQ-6: Confidence Scoring
**Given** classification results  
**When** confidence is calculated  
**Then** it returns a score 0.0-1.0 based on heuristic agreement

### Algorithm Details

```python
class PDFContentAnalyzer:
    def analyze(self, file_path: Path) -> ContentDetectionResult:
        # 1. Extract text layer statistics
        text_stats = self._extract_text_statistics(file_path)
        
        # 2. Analyze images
        image_stats = self._analyze_images(file_path)
        
        # 3. Calculate ratios
        text_ratio = text_stats.char_count / (text_stats.char_count + image_stats.estimated_chars)
        image_ratio = image_stats.total_image_area / image_stats.total_page_area
        
        # 4. Apply decision matrix
        if text_ratio > 0.95:
            content_type = ContentType.TEXT_BASED
            confidence = min(1.0, text_ratio)
        elif text_ratio < 0.05 and image_ratio > 0.90:
            content_type = ContentType.SCANNED
            confidence = min(1.0, image_ratio)
        else:
            content_type = ContentType.MIXED
            confidence = 0.5 + abs(text_ratio - 0.5)  # Higher confidence at extremes
        
        return ContentDetectionResult(
            content_type=content_type,
            confidence=confidence,
            text_ratio=text_ratio,
            image_ratio=image_ratio,
            page_results=[...]  # Per-page breakdown
        )
```

### Scenarios

#### SC-1: Text-Based PDF Detection
A 10-page research paper PDF with embedded text is analyzed. Engine detects:
- Text ratio: 98.5%
- Image ratio: 2%
- Classification: TEXT_BASED with 0.99 confidence

#### SC-2: Scanned Document Detection
A scanned invoice (10 pages, all images) is analyzed. Engine detects:
- Text ratio: 0.1%
- Image ratio: 95%
- Classification: SCANNED with 0.95 confidence

#### SC-3: Mixed Content Detection
A brochure PDF with text and photos is analyzed. Engine detects:
- Text ratio: 45%
- Image ratio: 55%
- Classification: MIXED with 0.55 confidence
- Recommends: Docling with OCR fallback

#### SC-4: Edge Case - Image-Heavy Text PDF
A textbook with diagrams on every page is analyzed:
- Text ratio: 85%
- Image ratio: 40% (overlapping elements)
- Classification: TEXT_BASED with 0.85 confidence
- Parser: Docling (sufficient for text extraction)
