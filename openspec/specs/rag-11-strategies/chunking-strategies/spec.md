# Spec: Advanced Chunking Strategies

## Overview
Implement intelligent document segmentation based on content structure with multiple strategies optimized for different document types.

## Requirements

### Functional Requirements
1. Semantic chunking based on meaning boundaries
2. Hierarchical chunking for structured documents
3. Fixed-size chunking with overlap
4. Agentic chunking strategy selection
5. Preserve code blocks, tables, and special formatting

### Chunking Strategies

#### 1. Semantic Chunking
```python
class SemanticChunker:
    """Chunk based on semantic boundaries."""
    
    def chunk(self, text: str) -> list[Chunk]:
        # Split into sentences
        sentences = self.split_sentences(text)
        
        # Group by semantic similarity
        chunks = []
        current_chunk = [sentences[0]]
        
        for sentence in sentences[1:]:
            # Check semantic similarity with current chunk
            if self.semantic_similarity(current_chunk, sentence) > threshold:
                current_chunk.append(sentence)
            else:
                chunks.append(self.create_chunk(current_chunk))
                current_chunk = [sentence]
        
        if current_chunk:
            chunks.append(self.create_chunk(current_chunk))
        
        return chunks
```

#### 2. Hierarchical Chunking
```python
class HierarchicalChunker:
    """Chunk based on document structure."""
    
    def chunk(self, document: Document) -> list[Chunk]:
        chunks = []
        
        for section in document.sections:
            section_chunk = Chunk(
                content=section.content,
                metadata={
                    "level": section.level,
                    "header": section.header,
                    "parent_id": section.parent_id
                }
            )
            chunks.append(section_chunk)
            
            # Recursively process subsections
            if section.subsections:
                chunks.extend(self.chunk_subsections(section))
        
        return chunks
```

#### 3. Fixed-Size with Overlap
```python
class FixedSizeChunker:
    """Standard fixed-size chunking with overlap."""
    
    def chunk(
        self,
        text: str,
        chunk_size: int = 512,
        overlap: int = 50
    ) -> list[Chunk]:
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            
            chunks.append(Chunk(
                content=chunk_text,
                metadata={
                    "start": start,
                    "end": end
                }
            ))
            
            start = end - overlap  # Move with overlap
        
        return chunks
```

## API Design

```python
class ChunkingService:
    def __init__(self):
        self.strategies = {
            "semantic": SemanticChunker(),
            "hierarchical": HierarchicalChunker(),
            "fixed": FixedSizeChunker(),
            "agentic": AgenticChunker()
        }
    
    async def chunk_document(
        self,
        document: Document,
        strategy: str = "agentic"
    ) -> list[Chunk]:
        """
        Chunk document using specified strategy.
        
        Args:
            document: Document to chunk
            strategy: Chunking strategy name
            
        Returns:
            List of chunks
        """
        chunker = self.strategies.get(strategy, self.strategies["fixed"])
        return await chunker.chunk(document)
    
    async def select_strategy(self, document: Document) -> str:
        """Agentically select best chunking strategy."""
        # Analyze document structure and content
        if document.has_clear_structure():
            return "hierarchical"
        elif document.is_technical():
            return "semantic"
        else:
            return "fixed"
```

## Configuration
```yaml
chunking:
  default_strategy: "agentic"
  
  strategies:
    semantic:
      similarity_threshold: 0.85
      min_chunk_size: 100
      max_chunk_size: 512
      embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
    
    hierarchical:
      max_depth: 4
      respect_headers: true
      preserve_code_blocks: true
    
    fixed:
      chunk_size: 512
      overlap: 50
      tokenizer: "cl100k_base"  # tiktoken
    
    agentic:
      selection_model: "gpt-4o-mini"
      decision_prompt: |
        Analyze this document and select the best chunking strategy:
        - "hierarchical" for structured docs with clear sections
        - "semantic" for technical content with logical breaks
        - "fixed" for narrative or mixed content
      
  # Special handling
  special_elements:
    code_blocks:
      preserve_integrity: true
      max_chunk_size: 1024
    
    tables:
      preserve_rows: true
      max_rows_per_chunk: 10
    
    images:
      extract_text: true
      create_alt_chunks: true
```

## Acceptance Criteria
- [ ] All 4 chunking strategies implemented
- [ ] Agentic strategy selection works correctly
- [ ] Code blocks preserved intact
- [ ] Hierarchical chunks maintain parent relationships
- [ ] Semantic chunks respect meaning boundaries

## Performance Expectations
| Metric | Target |
|--------|--------|
| Chunking speed | 1000 tokens/s |
| Agentic selection | <100ms |
| Chunk quality score | >0.85 |
| Boundary accuracy | >90% |

## Dependencies
- tiktoken for tokenization
- sentence-transformers for semantic similarity
- Document parsing service
