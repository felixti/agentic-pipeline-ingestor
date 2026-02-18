# Proposal: Content Detection for PDFs

## Why

The pipeline currently lacks the ability to intelligently differentiate between **scanned PDFs** (image-based) and **text-based PDFs** (with embedded text layers). This leads to:

- **Suboptimal parser selection**: Using Docling on scanned PDFs wastes resources and produces poor results; using Azure OCR on text-based PDFs is unnecessarily expensive and slow
- **Processing failures**: Wrong parser choices cause extraction errors that require manual intervention
- **Cost inefficiency**: Azure OCR is 10x more expensive than text extraction; using it unnecessarily inflates costs
- **Poor user experience**: Users must manually specify content type or accept subpar results

According to the system spec (G3), we need automatic content detection with > 98% accuracy to intelligently route documents to the optimal parser.

## What Changes

Implement an **intelligent content detection system** for PDF documents that analyzes file characteristics and determines whether content is:

1. **Text-based PDF** - Has extractable text layer → Route to Docling
2. **Scanned PDF** - Image-based, needs OCR → Route to Azure OCR
3. **Mixed content** - Combination of both → Route to Docling with OCR fallback

The detection algorithm will analyze:
- Text-to-image ratio per page
- Presence of text layers in PDF structure
- Image characteristics (density, resolution)
- Font information and encoding

**Integration points:**
- New `POST /api/v1/detect` endpoint for on-demand detection
- Automatic detection during job creation (optional)
- Detection results stored in job metadata for downstream parser selection

## Capabilities

- [ ] PDF Analysis Engine
- [ ] Content Detection API
- [ ] Detection Result Storage
- [ ] Parser Selection Integration

## Impact

| Aspect | Impact |
|--------|--------|
| **Accuracy** | > 98% correct routing decisions (Target G3) |
| **Cost Reduction** | ~40% savings on OCR costs by avoiding unnecessary Azure OCR calls |
| **Performance** | 2-3x faster processing for text-based PDFs (Docling vs Azure OCR) |
| **Success Rate** | Reduced parser-related failures by routing to correct parser |
| **Complexity** | Medium - requires PDF structure analysis and heuristics |
| **Breaking Changes** | None - additive feature with backward compatibility |
| **Dependencies** | PyMuPDF (already in use), new: pdfplumber for text layer analysis |

### Success Metrics

- Detection accuracy > 98% on test corpus (1000 mixed PDFs)
- Average detection time < 500ms per PDF
- Zero false negatives (never classify scanned as text - quality over cost)
- < 2% false positives (text classified as scanned - acceptable cost)
