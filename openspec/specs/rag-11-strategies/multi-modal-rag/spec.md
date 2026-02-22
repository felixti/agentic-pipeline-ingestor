# Spec: Multi-Modal RAG

## Overview
Extend RAG capabilities to support images, audio, video, and other non-text content types alongside text documents.

## Requirements

### Functional Requirements
1. Extract text from images (OCR)
2. Generate image captions and descriptions
3. Create unified embeddings across modalities
4. Support audio transcription and video analysis
5. Cross-modal retrieval (text query → image results)

### Multi-Modal Processing Pipeline

```
Input Document
      │
      ▼
┌─────────────┐
│  Detect     │
│  Content    │
│   Type      │
└──────┬──────┘
       │
   ┌───┴───┐
   │       │
   ▼       ▼
┌──────┐ ┌──────┐
│ Text │ │Image │
└──┬───┘ └──┬───┘
   │        │
   ▼        ▼
┌──────┐ ┌─────────┐
│Chunk │ │Caption  │
│Embed │ │OCR Text │
└──┬───┘ └────┬────┘
   │          │
   └────┬─────┘
        ▼
   ┌─────────┐
   │ Unified │
   │ Embedding│
   └────┬────┘
        ▼
   ┌─────────┐
   │ Vector  │
   │  Store  │
   └─────────┘
```

## API Design

```python
class MultiModalProcessor:
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.audio_processor = AudioProcessor()
        self.video_processor = VideoProcessor()
    
    async def process(
        self,
        document: Document,
        extract_text: bool = True
    ) -> list[ContentChunk]:
        """
        Process multi-modal document into searchable chunks.
        
        Args:
            document: Input document with potentially mixed content
            extract_text: Whether to extract text from images
            
        Returns:
            List of content chunks with unified embeddings
        """
        chunks = []
        
        for element in document.elements:
            if element.type == "text":
                chunks.append(await self.process_text(element))
            elif element.type == "image":
                chunks.extend(await self.process_image(element, extract_text))
            elif element.type == "audio":
                chunks.append(await self.process_audio(element))
            elif element.type == "video":
                chunks.extend(await self.process_video(element))
        
        return chunks
    
    async def process_image(
        self,
        image: Image,
        extract_text: bool
    ) -> list[ContentChunk]:
        """Process image into searchable chunks."""
        chunks = []
        
        # Generate caption
        caption = await self.image_processor.caption(image)
        chunks.append(ContentChunk(
            type="image_caption",
            content=caption,
            embedding=await self.embed(caption),
            metadata={"source_image": image.id}
        ))
        
        # Extract OCR text if enabled
        if extract_text:
            ocr_text = await self.image_processor.ocr(image)
            if ocr_text.strip():
                chunks.append(ContentChunk(
                    type="image_text",
                    content=ocr_text,
                    embedding=await self.embed(ocr_text),
                    metadata={"source_image": image.id}
                ))
        
        return chunks

class UnifiedEmbedding:
    """Create embeddings that work across modalities."""
    
    async def embed_text(self, text: str) -> list[float]:
        """Generate text embedding."""
        pass
    
    async def embed_image(self, image: Image) -> list[float]:
        """Generate image embedding using CLIP or similar."""
        pass
    
    async def cross_modal_search(
        self,
        query: str,
        content_type: str = "all"
    ) -> list[SearchResult]:
        """Search across all modalities with text query."""
        # Use CLIP-style model for cross-modal search
        query_embedding = await self.embed_text(query)
        return await self.vector_search(query_embedding, content_type)
```

## Configuration
```yaml
multi_modal:
  enabled: true
  
  image:
    caption_model: "Salesforce/blip-image-captioning-base"
    ocr_model: "paddleocr"
    embedding_model: "openai/clip-vit-base-patch32"
    
  audio:
    transcription_model: "openai/whisper-base"
    chunk_duration: 30  # seconds
    
  video:
    frame_extraction_fps: 1  # Extract 1 frame per second
    max_frames: 10  # Max frames to process per video
    
  unified_embedding:
    model: "clip"  # CLIP for cross-modal compatibility
    dimensions: 512
  
  # Storage
  store_original_media: true
  media_storage: "s3"
  max_file_size: 100MB
```

## Database Schema
```sql
-- Add media type to chunks
ALTER TABLE chunks ADD COLUMN content_type VARCHAR(50) DEFAULT 'text';
ALTER TABLE chunks ADD COLUMN media_metadata JSONB;

-- Media storage table
CREATE TABLE media_files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id),
    file_type VARCHAR(50),
    file_path VARCHAR(500),
    file_size BIGINT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Multi-modal search index
CREATE INDEX idx_chunks_content_type ON chunks(content_type);
CREATE INDEX idx_chunks_media ON chunks USING GIN(media_metadata);
```

## Acceptance Criteria
- [ ] Images processed with caption and OCR
- [ ] Audio transcribed and chunked
- [ ] Videos analyzed with frame extraction
- [ ] Cross-modal search works (text → image)
- [ ] Unified embeddings compatible across types

## Performance Expectations
| Content Type | Processing Time | Storage |
|--------------|-----------------|---------|
| Image        | <2s             | 5MB     |
| Audio (1min) | <5s             | 1MB     |
| Video (5min) | <30s            | 50MB    |

## Dependencies
- CLIP (OpenAI) for cross-modal embeddings
- BLIP for image captioning
- PaddleOCR or Tesseract for OCR
- Whisper for audio transcription
- FFmpeg for video processing
