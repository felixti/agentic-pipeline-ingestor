# Spec: Contextual Retrieval

## Overview
Enhance chunks with surrounding context before embedding to improve semantic understanding of chunk boundaries and document structure.

## Requirements

### Functional Requirements
1. Add parent document context to each chunk
2. Include section headers and document metadata
3. Preserve hierarchical relationships (document → section → chunk)
4. Support different context strategies per document type
5. Enable parent document lookup for retrieved chunks

### Context Strategies

#### 1. Parent Document Enhancement
```
Chunk: "The system uses pgvector for vector storage."
Enhanced: "Document: Database Architecture Guide
Section: Vector Storage
Content: The system uses pgvector for vector storage."
```

#### 2. Window Context
```
Previous: "...semantic search requires vector storage."
Chunk: "The system uses pgvector for vector storage."
Next: "This enables efficient similarity queries."
Enhanced: "[Previous: semantic search requires vector storage] 
           The system uses pgvector for vector storage. 
           [Next: This enables efficient similarity queries]"
```

#### 3. Hierarchical Context
```
Document: API Documentation v2.0
Section: Authentication
Subsection: JWT Tokens
Chunk: "JWT tokens expire after 24 hours."
```

## API Design

```python
class ContextualRetrieval:
    async def enhance_chunk(
        self,
        chunk: Chunk,
        context_type: ContextType = ContextType.PARENT_DOCUMENT
    ) -> EnhancedChunk:
        """
        Enhance chunk with contextual information.
        
        Args:
            chunk: Original chunk
            context_type: Type of context to add
            
        Returns:
            Chunk with enhanced context for embedding
        """
        context = await self.get_context(chunk, context_type)
        
        return EnhancedChunk(
            original=chunk,
            context=context,
            enhanced_text=self.format_context(chunk, context),
            embedding=await self.embed(self.format_context(chunk, context))
        )
    
    async def get_parent_document(self, chunk_id: str) -> Document:
        """Retrieve parent document for chunk."""
        pass
    
    async def get_section_headers(self, chunk_id: str) -> list[str]:
        """Get section headers that contain this chunk."""
        pass
```

## Database Schema
```sql
-- Parent document reference
ALTER TABLE chunks ADD COLUMN parent_document_id UUID;
ALTER TABLE chunks ADD COLUMN section_headers TEXT[];
ALTER TABLE chunks ADD COLUMN document_metadata JSONB;

-- Hierarchical relationships
CREATE TABLE document_hierarchy (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id),
    chunk_id UUID REFERENCES chunks(id),
    level INTEGER,  -- 0=document, 1=section, 2=subsection, 3=chunk
    parent_id UUID REFERENCES document_hierarchy(id),
    path LTREE  -- For efficient tree queries
);

CREATE INDEX idx_hierarchy_path ON document_hierarchy USING GIST(path);
```

## Configuration
```yaml
contextual_retrieval:
  enabled: true
  
  # Default strategy
  default_strategy: "parent_document"
  
  # Strategy configs
  strategies:
    parent_document:
      include_metadata: true
      metadata_fields:
        - title
        - author
        - category
      max_context_length: 256
    
    window:
      window_size: 1  # chunks before and after
      separator: " | "
    
    hierarchical:
      max_depth: 3
      include_path: true
  
  # Per-document-type config
  document_types:
    technical_docs:
      strategy: hierarchical
      include_code_blocks: true
    
    articles:
      strategy: parent_document
      include_summary: true
```

## Acceptance Criteria
- [ ] Context is correctly added to chunks
- [ ] Parent document lookup works efficiently
- [ ] Enhanced embeddings show improved retrieval
- [ ] Hierarchical queries work with ltree
- [ ] Backward compatibility with existing chunks

## Performance Expectations
| Metric | Target |
|--------|--------|
| Context lookup | <10ms |
| Embedding overhead | <50ms |
| Storage increase | <30% |
| Retrieval improvement | +10% |

## Dependencies
- PostgreSQL ltree extension
- Existing chunking service
- Document storage
