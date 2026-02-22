# Spec: Embedding Optimization

## Overview
Optimize embedding models and strategies to reduce memory costs and improve retrieval precision across different use cases.

## Requirements

### Functional Requirements
1. Support multiple embedding models with different characteristics
2. Model selection based on content type and query pattern
3. Dimensionality reduction for storage efficiency
4. Quantization for memory optimization
5. Embedding caching at multiple levels

### Supported Models

| Model | Dimensions | Speed | Quality | Use Case |
|-------|-----------|-------|---------|----------|
| text-embedding-3-small | 1536 | Fast | Good | General purpose |
| text-embedding-3-large | 3072 | Medium | Excellent | High precision |
| sentence-transformers/all-MiniLM-L6-v2 | 384 | Very Fast | Good | On-premise |
| BAAI/bge-large-en-v1.5 | 1024 | Medium | Excellent | Technical docs |
| voyage-2 | 1024 | Fast | Excellent | Enterprise |

## API Design

```python
class EmbeddingService:
    def __init__(self):
        self.models = {
            "fast": OpenAIEmbedder("text-embedding-3-small"),
            "precise": OpenAIEmbedder("text-embedding-3-large"),
            "local": SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
            "technical": BGEEmbedder("BAAI/bge-large-en-v1.5")
        }
        self.cache = EmbeddingCache()
    
    async def embed(
        self,
        texts: list[str],
        model: str = "auto",
        use_cache: bool = True
    ) -> list[Embedding]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: Texts to embed
            model: Model name or "auto" for selection
            use_cache: Whether to use caching
            
        Returns:
            List of embeddings
        """
        if model == "auto":
            model = self.select_model(texts)
        
        # Check cache
        if use_cache:
            cached = await self.cache.get_batch(texts, model)
            if all(cached):
                return cached
        
        # Generate embeddings
        embedder = self.models[model]
        embeddings = await embedder.embed(texts)
        
        # Cache results
        if use_cache:
            await self.cache.set_batch(texts, model, embeddings)
        
        return embeddings
    
    def select_model(self, texts: list[str]) -> str:
        """Select optimal model based on content."""
        # Analyze content characteristics
        avg_length = sum(len(t) for t in texts) / len(texts)
        has_technical_terms = any(self.is_technical(t) for t in texts)
        
        if has_technical_terms and avg_length > 500:
            return "technical"
        elif avg_length > 1000:
            return "precise"
        else:
            return "fast"
```

## Optimization Techniques

### 1. Dimensionality Reduction
```python
class EmbeddingOptimizer:
    def __init__(self, target_dims: int = 256):
        self.pca = PCA(n_components=target_dims)
        self.target_dims = target_dims
    
    def reduce_dimensions(self, embedding: list[float]) -> list[float]:
        """Reduce embedding dimensions using PCA."""
        return self.pca.transform([embedding])[0]
    
    def reconstruct(self, reduced: list[float]) -> list[float]:
        """Reconstruct original dimension embedding."""
        return self.pca.inverse_transform([reduced])[0]
```

### 2. Quantization
```python
class QuantizedEmbedding:
    """8-bit quantization for storage efficiency."""
    
    def quantize(self, embedding: list[float]) -> bytes:
        """Quantize float32 to int8."""
        arr = np.array(embedding)
        min_val, max_val = arr.min(), arr.max()
        
        # Scale to 0-255
        scaled = (arr - min_val) / (max_val - min_val) * 255
        quantized = scaled.astype(np.uint8)
        
        return quantized.tobytes(), (min_val, max_val)
    
    def dequantize(self, quantized: bytes, scale: tuple) -> list[float]:
        """Dequantize int8 back to float32."""
        arr = np.frombuffer(quantized, dtype=np.uint8)
        min_val, max_val = scale
        
        # Scale back to original range
        return (arr / 255 * (max_val - min_val) + min_val).tolist()
```

## Configuration
```yaml
embedding:
  default_model: "text-embedding-3-small"
  
  models:
    text-embedding-3-small:
      provider: "openai"
      dimensions: 1536
      batch_size: 100
      
    text-embedding-3-large:
      provider: "openai"
      dimensions: 3072
      batch_size: 50
      
    all-MiniLM-L6-v2:
      provider: "sentence-transformers"
      dimensions: 384
      device: "cuda"
  
  optimization:
    dimensionality_reduction:
      enabled: true
      target_dimensions: 256
      preserve_quality_threshold: 0.95
    
    quantization:
      enabled: true
      bits: 8
      compression_ratio: 4.0
    
    caching:
      enabled: true
      ttl: 86400  # 24 hours
      max_size: 100000  # entries
  
  auto_selection:
    enabled: true
    rules:
      - condition: "technical_content AND length > 500"
        model: "bge-large"
      - condition: "length > 2000"
        model: "text-embedding-3-large"
      - condition: "default"
        model: "text-embedding-3-small"
```

## Acceptance Criteria
- [ ] All 5 embedding models supported
- [ ] Model selection improves retrieval by >5%
- [ ] Dimensionality reduction preserves >95% quality
- [ ] Quantization achieves 4x compression
- [ ] Cache hit rate >70%

## Performance Expectations
| Metric | Target |
|--------|--------|
| Embedding latency (p95) | <100ms |
| Cache hit rate | >70% |
| Storage reduction | 75% |
| Quality retention | >95% |

## Dependencies
- OpenAI API or Azure OpenAI
- sentence-transformers
- scikit-learn (PCA)
- NumPy (quantization)
