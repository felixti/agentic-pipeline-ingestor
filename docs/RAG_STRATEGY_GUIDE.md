# RAG Strategy Guide

**Agentic Data Pipeline Ingestor**  
*Complete guide to Retrieval-Augmented Generation (RAG) strategies, when to use them, and how to configure them.*

> **Version**: 1.0.0  
> **Last Updated**: 2026-02-20

---

## Table of Contents

1. [Overview](#overview)
2. [Core RAG Strategies](#core-rag-strategies)
3. [Search Strategies](#search-strategies)
4. [Retrieval Enhancement Strategies](#retrieval-enhancement-strategies)
5. [Query Processing Strategies](#query-processing-strategies)
6. [Advanced RAG Strategies](#advanced-rag-strategies)
7. [Configuration Guide](#configuration-guide)
8. [Strategy Selection Decision Tree](#strategy-selection-decision-tree)
9. [Performance Optimization](#performance-optimization)
10. [API Usage Examples](#api-usage-examples)

---

## Overview

The Agentic Data Pipeline Ingestor implements a sophisticated multi-strategy RAG system that combines traditional vector search with advanced techniques including hybrid search, query rewriting, re-ranking, contextual retrieval, and knowledge graphs (GraphRAG).

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RAG System Architecture                          │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │   Query     │→ │   Query     │→ │  Retrieval  │→ │  Response   │   │
│  │   Input     │  │ Processing  │  │   Engine    │  │ Generation  │   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
│                          │                    │                         │
│                   ┌──────┴──────┐      ┌──────┴──────┐                 │
│                   │  Rewriting  │      │ Multi-Modal │                 │
│                   │  Classification│   │  Search     │                 │
│                   │  HyDE         │    │  (Vector/Text/Hybrid)        │
│                   └─────────────┘      └─────────────┘                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Core RAG Strategies

### 1. Vector Search (Semantic Search)

**What it is**: Pure semantic similarity search using embedding vectors and cosine similarity.

**When to use**:
- ✅ Searching for conceptually similar content
- ✅ Natural language queries where exact keywords don't matter
- ✅ Finding documents with similar meaning but different wording
- ✅ Large knowledge bases where semantic understanding is critical

**When NOT to use**:
- ❌ Queries with specific identifiers, codes, or names
- ❌ Exact phrase matching requirements
- ❌ When you need keyword highlighting

**Configuration**:
```python
# config.py
class VectorSearchConfig:
    default_top_k: int = 10
    default_min_similarity: float = 0.7  # 0-1 range
    embedding_dimensions: int = 1536
    max_top_k: int = 100
```

**API Usage**:
```bash
curl -X POST "http://localhost:8000/api/v1/search/semantic" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query_embedding": [0.1, 0.2, 0.3, ...],  # 1536 dimensions
    "top_k": 10,
    "min_similarity": 0.7,
    "filters": {"job_id": "uuid-here"}
  }'
```

---

### 2. Text Search (Lexical/BM25)

**What it is**: Traditional full-text search using PostgreSQL's tsvector/tsquery with BM25 ranking and optional fuzzy trigram matching.

**When to use**:
- ✅ Searching for specific keywords or phrases
- ✅ Exact term matching (product codes, names, identifiers)
- ✅ When highlighting matched terms is needed
- ✅ Handling typos with fuzzy matching
- ✅ Multi-language content

**When NOT to use**:
- ❌ Conceptual/semantic similarity needs
- ❌ Finding paraphrased content

**Configuration**:
```python
# config.py
class TextSearchConfig:
    default_language: str = "english"
    bm25_weights: tuple = (0.1, 0.2, 0.4, 1.0)
    default_similarity_threshold: float = 0.3  # For fuzzy matching
    highlight_start_tag: str = "<mark>"
    highlight_end_tag: str = "</mark>"
```

**API Usage**:
```bash
curl -X POST "http://localhost:8000/api/v1/search/text" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning architecture",
    "top_k": 10,
    "language": "english",
    "use_fuzzy": true,
    "highlight": true,
    "filters": {"job_id": "uuid-here"}
  }'
```

---

### 3. Hybrid Search

**What it is**: Combines vector and text search results using fusion methods (Weighted Sum or Reciprocal Rank Fusion).

**When to use**:
- ✅ Best of both semantic and lexical search
- ✅ Complex queries with both concepts and specific terms
- ✅ When you're unsure which search type works best
- ✅ Production systems requiring robust retrieval

**When NOT to use**:
- ❌ Extremely latency-sensitive applications (slower than single search)
- ❌ Simple queries where one search type is clearly better

**Fusion Methods**:

| Method | Formula | Best For |
|--------|---------|----------|
| **Weighted Sum** | `hybrid_score = (v_weight × v_score) + (t_weight × t_score)` | When you know the balance between semantic/lexical |
| **Reciprocal Rank Fusion (RRF)** | `score = Σ(weight / (k + rank))` | When rank matters more than raw scores |

**Configuration**:
```python
# config.py
class HybridSearchSettings:
    default_vector_weight: float = 0.7
    default_text_weight: float = 0.3
    rrf_k: int = 60
    default_fusion_method: str = "rrf"  # or "weighted_sum"
    query_expansion_enabled: bool = True
```

**Weight Presets**:
- `semantic_focus`: Vector 0.9, Text 0.1 - Emphasizes semantic similarity
- `balanced`: Vector 0.7, Text 0.3 - Balanced approach (default)
- `lexical_focus`: Vector 0.3, Text 0.7 - Emphasizes keyword matching

**API Usage**:
```bash
curl -X POST "http://localhost:8000/api/v1/search/hybrid" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "neural network architecture",
    "top_k": 10,
    "vector_weight": 0.7,
    "text_weight": 0.3,
    "fusion_method": "rrf",
    "filters": {"source_type": "pdf"}
  }'
```

---

## Search Strategies

### Strategy Comparison

| Strategy | Speed | Accuracy | Best Use Case |
|----------|-------|----------|---------------|
| Vector Search | Fast | High semantic | Conceptual queries |
| Text Search | Fast | High lexical | Keyword/phrase queries |
| Hybrid Search | Medium | Highest overall | Production systems |
| + Re-ranking | Slower | Highest precision | Quality-critical apps |
| + HyDE | Slower | Better for vague | Ambiguous queries |

---

## Retrieval Enhancement Strategies

### 4. Re-Ranking (Cross-Encoder)

**What it is**: Uses a cross-encoder model to re-score and re-order retrieved chunks for improved relevance.

**When to use**:
- ✅ Maximum retrieval quality is critical
- ✅ When initial retrieval returns many false positives
- ✅ Precision-focused applications

**When NOT to use**:
- ❌ Latency-sensitive applications (adds 50-200ms)
- ❌ Cost-constrained environments

**How it works**:
1. Retrieve top-K chunks using fast method (vector/hybrid)
2. Score each (query, chunk) pair with cross-encoder
3. Re-order by cross-encoder scores
4. Return top-N results

**Configuration**:
```python
# config.py
class ReRankingSettings:
    enabled: bool = True
    models: dict = {
        "default": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "high_precision": "cross-encoder/ms-marco-electra-base",
        "fast": "cross-encoder/ms-marco-TinyBERT-L-2"
    }
    initial_retrieval_k: int = 20  # Retrieve this many
    final_k: int = 5               # Return this many
```

**Available Models**:
- `cross-encoder/ms-marco-MiniLM-L-6-v2` - Balanced speed/quality
- `cross-encoder/ms-marco-electra-base` - High precision, slower
- `cross-encoder/ms-marco-TinyBERT-L-2` - Fast, lower quality
- `BAAI/bge-reranker-base` - Alternative high-quality model

---

### 5. Contextual Retrieval

**What it is**: Enhances chunks with surrounding context (parent document, neighboring chunks, hierarchy) before embedding.

**When to use**:
- ✅ Documents with clear structure (sections, chapters)
- ✅ When chunks lose meaning without context
- ✅ Technical documentation
- ✅ Legal or academic documents

**Context Types**:

| Type | Description | Best For |
|------|-------------|----------|
| **parent_document** | Adds document title/metadata as context | General documents |
| **window** | Includes neighboring chunks | Sequential content |
| **hierarchical** | Uses document hierarchy (sections/subsections) | Structured documents |

**Configuration**:
```python
# config.py
class ContextualRetrievalSettings:
    enabled: bool = True
    default_strategy: str = "parent_document"  # or "window", "hierarchical"
    strategies: dict = {
        "parent_document": {
            "include_metadata": True,
            "metadata_fields": ["title", "author", "category"],
            "max_context_length": 256
        },
        "window": {
            "window_size": 1,  # chunks before/after
            "separator": " | "
        },
        "hierarchical": {
            "max_depth": 3,
            "include_path": True
        }
    }
```

---

### 6. Query Expansion

**What it is**: Expands queries with synonyms and related terms to improve recall.

**When to use**:
- ✅ Acronyms and technical jargon
- ✅ When users might use different terminology
- ✅ Improving lexical search recall

**Built-in Expansions**:
```python
SYNONYMS = {
    "ml": ["machine learning"],
    "ai": ["artificial intelligence"],
    "nlp": ["natural language processing"],
    "rag": ["retrieval augmented generation"],
    "api": ["application programming interface"],
    # ... and more
}
```

**Configuration**:
```python
# config.py - enabled via HybridSearchSettings
query_expansion_enabled: bool = True
query_expansion_max_terms: int = 5
```

---

## Query Processing Strategies

### 7. Query Rewriting

**What it is**: Separates user intent into search-optimized and generation-optimized components.

**When to use**:
- ✅ Complex user queries with instructions
- ✅ When search keywords differ from generation context
- ✅ Conversational queries

**Example Transformation**:
```
User: "@knowledgebase explain what vibe coding is and its pros/cons"

Rewritten:
- search_rag: True
- embedding_source_text: "vibe coding programming approach"
- llm_query: "Based on the provided context, explain what vibe coding is, 
             including its pros and cons, and cite sources."
```

**Configuration**:
```python
# config.py
class QueryRewritingSettings:
    enabled: bool = True
    model: str = "agentic-decisions"
    temperature: float = 0.1
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 hour
```

---

### 8. Query Classification

**What it is**: Automatically classifies queries into types to select optimal RAG strategies.

**Query Types**:

| Type | Description | Recommended Strategy |
|------|-------------|---------------------|
| **factual** | Simple fact lookup | Hybrid search + re-ranking |
| **analytical** | Explanation/synthesis | Query rewrite + HyDE + re-ranking |
| **comparative** | Compare multiple items | Multi-hop retrieval + re-ranking |
| **vague** | Unclear/broad queries | HyDE + expanded retrieval |
| **multi_hop** | Requires multiple steps | Multi-hop + GraphRAG |

**Configuration**:
```python
# config.py
class ClassificationSettings:
    enabled: bool = True
    min_confidence_threshold: float = 0.7
    use_pattern_fallback: bool = True
    timeout_ms: int = 1000
```

---

### 9. HyDE (Hypothetical Document Embeddings)

**What it is**: Generates a hypothetical document that answers the query, then uses that for vector search instead of the raw query.

**When to use**:
- ✅ Vague or out-of-domain queries
- ✅ Complex questions requiring synthesis
- ✅ When standard retrieval returns poor results

**When NOT to use**:
- ❌ Simple, direct queries (adds unnecessary latency)
- ❌ Factual lookups with clear entities

**How it works**:
1. LLM generates hypothetical answer document
2. Embed the hypothetical document
3. Search using hypothetical embedding
4. Retrieve real documents similar to hypothetical

**Configuration**:
```python
# config.py
class HyDESettings:
    enabled: bool = True
    model: str = "agentic-decisions"
    temperature: float = 0.7  # Higher for creativity
    max_tokens: int = 300
    cache_enabled: bool = True
    cache_ttl: int = 7200  # 2 hours
    enable_for_query_types: list = [
        "complex_questions",
        "vague_queries", 
        "out_of_domain"
    ]
```

---

## Advanced RAG Strategies

### 10. GraphRAG (Knowledge Graph RAG)

**What it is**: Builds a knowledge graph from documents and uses graph traversal for multi-hop retrieval.

**When to use**:
- ✅ Multi-hop questions ("Who founded the company that acquired X?")
- ✅ Entity-rich documents (people, organizations, locations)
- ✅ Relationship-focused queries
- ✅ Complex interconnected knowledge

**Components**:
- **Entity Extraction**: Identifies people, orgs, locations, etc.
- **Relationship Detection**: Maps connections between entities
- **Community Detection**: Groups related entities
- **Graph Traversal**: Multi-hop reasoning

**Entity Types**:
```python
EntityType = {
    PERSON: "person",
    ORGANIZATION: "organization", 
    LOCATION: "location",
    EVENT: "event",
    PRODUCT: "product",
    TECHNOLOGY: "technology",
    CONCEPT: "concept",
    DOCUMENT: "document",
    CHUNK: "chunk"
}
```

**Configuration**:
```python
# Part of GraphRAG destination configuration
graphrag_config = {
    "entity_extraction_model": "spacy/en_core_web_lg",
    "enable_community_detection": True,
    "similarity_threshold": 0.75,
    "max_communities": 50
}
```

---

### 11. Multi-Hop Retrieval

**What it is**: Performs multiple retrieval steps to answer complex questions requiring information from multiple sources.

**When to use**:
- ✅ Questions requiring connecting multiple facts
- ✅ "What is the relationship between X and Y?"
- ✅ Comparative analysis across documents

**Example**:
```
Query: "What technologies do the companies founded by Elon Musk use?"

Hop 1: Find companies founded by Elon Musk
Hop 2: For each company, find their technologies
Hop 3: Aggregate and present results
```

**Configuration**:
```python
# config.py
class AgenticRAGSettings:
    multi_hop_enabled: bool = True
    max_iterations: int = 3  # Maximum hops
```

---

### 12. Agentic RAG Router

**What it is**: Orchestrates all RAG strategies based on query classification and quality thresholds.

**When to use**:
- ✅ Production systems with diverse query types
- ✅ When you want automatic strategy selection
- ✅ Systems requiring self-correction

**Features**:
- Automatic strategy selection based on query type
- Self-correction with iterative improvement
- Quality threshold enforcement
- Latency optimization

**Strategy Presets**:

| Preset | Speed | Quality | Strategies Enabled |
|--------|-------|---------|-------------------|
| **fast** | ⚡ Fast | Good | Basic hybrid search |
| **balanced** | 🚀 Medium | Better | Hybrid + re-ranking |
| **thorough** | 🐢 Slower | Best | All strategies |
| **auto** | Variable | Optimized | Dynamic selection |

**Configuration**:
```python
# config.py
class AgenticRAGSettings:
    enabled: bool = True
    quality_threshold: float = 0.7
    max_iterations: int = 3
    default_preset: str = "auto"  # fast, balanced, thorough, auto
    latency_target_ms: int = 500
```

---

## Configuration Guide

### Environment Variables

```bash
# Hybrid Search
HYBRID_SEARCH_DEFAULT_VECTOR_WEIGHT=0.7
HYBRID_SEARCH_DEFAULT_TEXT_WEIGHT=0.3
HYBRID_SEARCH_RRF_K=60
HYBRID_SEARCH_DEFAULT_FUSION_METHOD=rrf
HYBRID_SEARCH_QUERY_EXPANSION_ENABLED=true

# Re-Ranking
RERANKING_ENABLED=true
RERANKING_INITIAL_RETRIEVAL_K=20
RERANKING_FINAL_K=5

# Query Rewriting
QUERY_REWRITE_ENABLED=true
QUERY_REWRITE_MODEL=agentic-decisions
QUERY_REWRITE_CACHE_ENABLED=true

# HyDE
HYDE_ENABLED=true
HYDE_TEMPERATURE=0.7
HYDE_CACHE_ENABLED=true

# Classification
CLASSIFICATION_ENABLED=true
CLASSIFICATION_MIN_CONFIDENCE_THRESHOLD=0.7

# Contextual Retrieval
CONTEXTUAL_RETRIEVAL_ENABLED=true
CONTEXTUAL_RETRIEVAL_DEFAULT_STRATEGY=parent_document

# Agentic RAG
AGENTIC_RAG_ENABLED=true
AGENTIC_RAG_QUALITY_THRESHOLD=0.7
AGENTIC_RAG_DEFAULT_PRESET=auto
```

### RAGConfig Model

```python
from src.rag.models import RAGConfig

config = RAGConfig(
    query_rewrite=True,
    hyde=False,           # Enable for vague queries
    reranking=True,       # Enable for quality
    hybrid_search=True,
    strategy_preset="balanced"  # fast, balanced, thorough, auto
)
```

---

## Strategy Selection Decision Tree

```
┌────────────────────────────────────────────────────────────────┐
│                    Query Type Decision Tree                     │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  START: What type of query do you have?                         │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Simple factual lookup?                                  │   │
│  │ ("What is X?", "Who is Y?")                             │   │
│  └──────────────────┬──────────────────────────────────────┘   │
│                     │ YES                                      │
│                     ▼                                          │
│         ┌─────────────────────┐                                │
│         │ Hybrid Search +     │                                │
│         │ Basic Re-ranking    │                                │
│         └─────────────────────┘                                │
│                     │                                          │
│                     ▼ NO                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Complex analytical query?                               │   │
│  │ ("Explain X", "How does Y work?")                       │   │
│  └──────────────────┬──────────────────────────────────────┘   │
│                     │ YES                                      │
│                     ▼                                          │
│         ┌─────────────────────┐                                │
│         │ Query Rewrite +     │                                │
│         │ HyDE + Re-ranking   │                                │
│         └─────────────────────┘                                │
│                     │                                          │
│                     ▼ NO                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Vague or ambiguous query?                               │   │
│  │ ("Tell me about...", "Information on...")               │   │
│  └──────────────────┬──────────────────────────────────────┘   │
│                     │ YES                                      │
│                     ▼                                          │
│         ┌─────────────────────┐                                │
│         │ HyDE + Expanded     │                                │
│         │ Retrieval           │                                │
│         └─────────────────────┘                                │
│                     │                                          │
│                     ▼ NO                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Multi-hop or relational?                                │   │
│  │ ("Author of X on topic Y")                              │   │
│  └──────────────────┬──────────────────────────────────────┘   │
│                     │ YES                                      │
│                     ▼                                          │
│         ┌─────────────────────┐                                │
│         │ GraphRAG +          │                                │
│         │ Multi-hop Retrieval │                                │
│         └─────────────────────┘                                │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Performance Optimization

### Latency vs Quality Trade-offs

| Configuration | Latency | Quality | Use Case |
|--------------|---------|---------|----------|
| Fast preset | ~50ms | Good | Chatbots, real-time |
| Balanced preset | ~150ms | Better | General production |
| Thorough preset | ~500ms | Best | Research, analysis |
| Custom: Hybrid only | ~30ms | Good | High-throughput |
| Custom: +Re-ranking | ~100ms | Better | Quality-critical |
| Custom: +HyDE | ~200ms | Better | Vague queries |

### Caching Strategy

The system implements multi-layer caching:

```python
# L1: Redis (fastest, 1 hour TTL)
- Query embeddings
- Query results
- LLM responses

# L2: PostgreSQL (persistent, 24 hour TTL)
- Embeddings
- Query results
- Classification results

# L3: Semantic (longest, 7 days TTL)
- Similar query matching
```

**Cache Configuration**:
```python
# config.py
class CachingSettings:
    enabled: bool = True
    layers: dict = {
        "l1_redis": {"enabled": True, "ttl": 3600},
        "l2_postgres": {"enabled": True, "ttl": 86400},
        "l3_semantic": {"enabled": True, "ttl": 604800, "similarity_threshold": 0.95}
    }
```

---

## API Usage Examples

### Example 1: Basic Hybrid Search

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/search/hybrid",
    headers={"X-API-Key": "your-api-key"},
    json={
        "query": "machine learning applications in healthcare",
        "top_k": 10,
        "vector_weight": 0.7,
        "text_weight": 0.3,
        "fusion_method": "rrf",
        "filters": {"source_type": "pdf"}
    }
)

results = response.json()
for item in results["results"]:
    print(f"{item['rank']}. {item['content'][:100]}...")
    print(f"   Hybrid: {item['hybrid_score']:.2f}, Vector: {item['vector_score']:.2f}")
```

### Example 2: Semantic Search with Embedding

```python
# First, generate embedding using your embedding model
embedding = embedding_model.encode("neural network architecture")

response = requests.post(
    "http://localhost:8000/api/v1/search/semantic",
    headers={"X-API-Key": "your-api-key"},
    json={
        "query_embedding": embedding.tolist(),
        "top_k": 10,
        "min_similarity": 0.75,
        "filters": {"job_id": "uuid-here"}
    }
)
```

### Example 3: Find Similar Chunks

```python
# Find chunks similar to a reference chunk
response = requests.get(
    "http://localhost:8000/api/v1/search/similar/{chunk_id}",
    headers={"X-API-Key": "your-api-key"},
    params={
        "top_k": 5,
        "exclude_self": True
    }
)
```

### Example 4: Text Search with Highlighting

```python
response = requests.post(
    "http://localhost:8000/api/v1/search/text",
    headers={"X-API-Key": "your-api-key"},
    json={
        "query": "kubernetes deployment strategies",
        "top_k": 10,
        "highlight": True,
        "use_fuzzy": True,
        "language": "english"
    }
)

for item in response.json()["results"]:
    print(f"Rank {item['rank']}: {item['highlighted_content']}")
    print(f"Matched terms: {item.get('matched_terms', [])}")
```

---

## Summary

### Quick Reference: When to Use Each Strategy

| Strategy | Use When | Don't Use When |
|----------|----------|----------------|
| **Vector Search** | Conceptual similarity | Exact keyword matching |
| **Text Search** | Keywords, identifiers | Semantic understanding needed |
| **Hybrid Search** | Best overall retrieval | Ultra-low latency required |
| **Re-Ranking** | Maximum quality needed | Latency-sensitive |
| **Query Rewriting** | Complex instructions | Simple, direct queries |
| **HyDE** | Vague/out-of-domain | Clear, specific queries |
| **GraphRAG** | Multi-hop reasoning | Simple lookups |
| **Contextual Retrieval** | Structured documents | Unstructured content |

### Recommended Starting Configuration

For most production use cases:

```python
RAGConfig(
    query_rewrite=True,
    hyde=False,              # Enable selectively for vague queries
    reranking=True,          # Enable for quality
    hybrid_search=True,
    strategy_preset="balanced"
)
```

Adjust based on your specific latency/quality requirements and query patterns.

---

*For more information, see the [API Guide](./API_GUIDE.md) and [Architecture Documentation](../ARCHITECTURE.md).*
