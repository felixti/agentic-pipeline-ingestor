# Product Requirements Document (PRD)
# GraphRAG UI - Cognee & HippoRAG Interface

**Version:** 1.1.0  
**Date:** 2026-03-01  
**Status:** Draft  
**Author:** AI Engineering Team

---

## 1. Executive Summary

### Overview
This PRD defines the requirements for a web-based UI to interact with the new GraphRAG capabilities (Cognee and HippoRAG) in the Agentic Data Pipeline Ingestor. The UI will provide intuitive interfaces for knowledge graph exploration, multi-hop reasoning queries, and entity extraction.

### Goals
- Provide an intuitive interface for GraphRAG search capabilities
- Enable users to visualize and explore knowledge graphs
- Support both simple (Cognee) and complex multi-hop (HippoRAG) queries
- Offer side-by-side comparison of search approaches
- **Integrate with existing RAG strategies** (Hybrid, Re-ranking, HyDE)

### Target Users
- Data analysts exploring document relationships
- Researchers conducting literature reviews
- Legal/medical professionals analyzing case connections
- Data scientists validating GraphRAG implementations

### Integration with Existing RAG System

The GraphRAG UI is part of a comprehensive RAG ecosystem:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RAG System Architecture                          │
├─────────────────────────────────────────────────────────────────────────┤
│  Standard RAG          │  GraphRAG (NEW)                                │
│  ───────────────────── │  ─────────────────────────────────────────     │
│  • Vector Search       │  • Cognee (Hybrid Vector+Graph)                │
│  • Text Search         │  • HippoRAG (Multi-hop PPR)                    │
│  • Hybrid Search       │                                                │
│  • Re-ranking          │  Enhanced By:                                  │
│  • Query Rewriting     │  • Entity Extraction                           │
│  • HyDE                │  • Knowledge Graph Visualization               │
│  • Classification      │  • Multi-hop Reasoning                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Context & Background

### What is GraphRAG?

GraphRAG (Graph Retrieval-Augmented Generation) enhances traditional RAG by using knowledge graphs to:
- **Capture entity relationships** beyond simple text similarity
- **Enable multi-hop reasoning** across document connections
- **Improve answer accuracy** for complex questions requiring inference

### Two Approaches

#### Cognee GraphRAG
| Aspect | Details |
|--------|---------|
| **Storage** | Neo4j (graph) + PostgreSQL/pgvector (embeddings) |
| **Best For** | Production workloads, entity visualization, hybrid search |
| **Search Types** | Vector, Graph, Hybrid |
| **Multi-hop** | Multi-step graph traversal |
| **Speed** | ~100ms query latency |
| **Use Case** | "Find all documents about companies founded by Elon Musk" |

#### HippoRAG Multi-Hop
| Aspect | Details |
|--------|---------|
| **Storage** | File-based (persistent volume) |
| **Best For** | Complex reasoning, research synthesis, legal analysis |
| **Search Type** | Personalized PageRank (single-step multi-hop) |
| **Multi-hop** | Single-step (neurobiological memory model) |
| **Speed** | ~350ms query latency |
| **Advantage** | +20% better multi-hop QA accuracy |
| **Use Case** | "What county is Erik Hort's birthplace a part of?" (3 hops) |

### Complete Strategy Comparison

| Strategy | Speed | Accuracy | Best Use Case | Multi-hop |
|----------|-------|----------|---------------|-----------|
| **Vector Search** | Fast (~30ms) | High semantic | Conceptual queries | ❌ |
| **Text Search** | Fast (~30ms) | High lexical | Keyword/phrase queries | ❌ |
| **Hybrid Search** | Medium (~50ms) | Highest overall | Production systems | ❌ |
| **Cognee GraphRAG** | Fast (~100ms) | High + Relations | Entity relationships | ✅ Multi-step |
| **HippoRAG** | Medium (~350ms) | **+20% better** | Complex multi-hop | ✅ Single-step |

### Decision Tree

```
START: What type of query do you have?
│
├─► Simple similarity? 
│   └─► Use Standard Vector Search
│
├─► Need entity relationships & knowledge graph?
│   ├─► Production workload?
│   │   ├─► Multi-hop questions (2+ steps)?
│   │   │   ├─► YES → Use HippoRAG (single-step, +20% accuracy)
│   │   │   └─► NO  → Use Cognee (hybrid search, visualization)
│   │   └─► General use?
│   │       └─► Use Cognee (production-ready, multi-modal)
│   │
│   └─► Research/Complex reasoning?
│       └─► Use HippoRAG (neurobiological memory model)
│
└─► Research/analysis requiring synthesis?
    └─► Use HippoRAG (+20% multi-hop accuracy)
```

---

## 3. API Reference

### 3.1 Cognee GraphRAG Endpoints

#### POST `/api/v1/cognee/search`
Search the Cognee knowledge graph.

**Request:**
```json
{
  "query": "machine learning applications in healthcare",
  "search_type": "hybrid",  // Options: "vector", "graph", "hybrid"
  "top_k": 10,              // Range: 1-100, default: 10
  "dataset_id": "default"   // Optional: specify dataset
}
```

**Response:**
```json
{
  "results": [
    {
      "chunk_id": "uuid",
      "content": "Machine learning is transforming healthcare...",
      "score": 0.89,
      "source_document": "doc_123.pdf",
      "entities": ["Machine Learning", "Healthcare", "AI"]
    }
  ],
  "search_type": "hybrid",
  "dataset_id": "default",
  "query_time_ms": 95
}
```

**Search Type Details:**

| Type | Description | Best For | Latency |
|------|-------------|----------|---------|
| **vector** | Pure semantic similarity using pgvector embeddings | Conceptual queries | ~30ms |
| **graph** | Graph traversal using Neo4j relationships | Entity connections | ~50ms |
| **hybrid** | Combined vector + graph (recommended) | Best overall results | ~100ms |

#### POST `/api/v1/cognee/extract-entities`
Extract entities and relationships from text.

**Request:**
```json
{
  "text": "Google Cloud Platform provides machine learning services through Vertex AI.",
  "dataset_id": "default"
}
```

**Response:**
```json
{
  "entities": [
    {"name": "Google Cloud Platform", "type": "ORG", "description": "Cloud computing service"},
    {"name": "Vertex AI", "type": "PRODUCT", "description": "Machine learning platform"}
  ],
  "relationships": [
    {"source": "Google Cloud Platform", "target": "Vertex AI", "type": "PROVIDES"}
  ]
}
```

**Entity Types:**

| Type | Description | Color Code |
|------|-------------|------------|
| **PERSON** | People, individuals | 🟠 Amber |
| **ORG** | Organizations, companies | 🔵 Blue |
| **LOCATION** | Places, geographical | 🟢 Green |
| **PRODUCT** | Products, services | 🟣 Purple |
| **TECHNOLOGY** | Technologies, frameworks | 🔴 Red |
| **CONCEPT** | Abstract concepts | 🟡 Yellow |

#### GET `/api/v1/cognee/stats`
Get knowledge graph statistics.

**Response:**
```json
{
  "dataset_id": "default",
  "document_count": 150,
  "chunk_count": 1200,
  "entity_count": 450,
  "relationship_count": 890,
  "graph_density": 0.15,
  "last_updated": "2026-03-01T10:30:00Z"
}
```

### 3.2 HippoRAG Multi-Hop Endpoints

#### POST `/api/v1/hipporag/retrieve`
Multi-hop retrieval using Personalized PageRank.

**Request:**
```json
{
  "queries": ["What county is Erik Hort's birthplace part of?"],
  "num_to_retrieve": 10  // Range: 1-50, default: 10
}
```

**Response:**
```json
{
  "results": [
    {
      "query": "What county is Erik Hort's birthplace part of?",
      "passages": [
        "Erik Hort was born in San Francisco...",
        "San Francisco is located in San Francisco County..."
      ],
      "scores": [0.92, 0.88],
      "source_documents": ["doc_1.pdf", "doc_2.pdf"],
      "entities": ["Erik Hort", "San Francisco", "San Francisco County"]
    }
  ],
  "query_time_ms": 320
}
```

**How Multi-Hop Works:**

```
Traditional RAG (Iterative):
Query: "What county is Erik Hort's birthplace part of?"
Step 1: Search "Erik Hort birthplace" → Montebello
Step 2: Search "Montebello county" → Rockland County
Step 3: Combine results
Latency: ~200ms × 2 = 400ms

HippoRAG (Single-Step):
Query: "What county is Erik Hort's birthplace part of?"
Step 1: Extract query nodes → [Erik Hort]
Step 2: PPR traversal: Erik Hort → birthplace → Montebello → part_of → Rockland County
Step 3: Single retrieval returns answer
Latency: ~350ms (single LLM call)
```

#### POST `/api/v1/hipporag/qa`
Full RAG QA with multi-hop retrieval.

**Request:**
```json
{
  "queries": ["What technologies do companies founded by Elon Musk use?"],
  "num_to_retrieve": 10
}
```

**Response:**
```json
{
  "results": [
    {
      "query": "What technologies do companies founded by Elon Musk use?",
      "answer": "Elon Musk's companies use various advanced technologies: Tesla uses electric vehicle battery technology and autonomous driving AI, SpaceX uses reusable rocket technology and Starlink satellite systems, Neuralink uses brain-computer interface technology...",
      "sources": ["tesla_report.pdf", "spacex_overview.pdf", "neuralink_docs.pdf"],
      "confidence": 0.91,
      "retrieval_results": {
        "passages": [...],
        "entities": ["Elon Musk", "Tesla", "SpaceX", "Neuralink"]
      }
    }
  ],
  "total_tokens": 850,
  "query_time_ms": 1250
}
```

#### POST `/api/v1/hipporag/extract-triples`
Extract OpenIE triples (subject-predicate-object).

**Request:**
```json
{
  "text": "Google developed TensorFlow for machine learning."
}
```

**Response:**
```json
{
  "triples": [
    {"subject": "Google", "predicate": "developed", "object": "TensorFlow"},
    {"subject": "TensorFlow", "predicate": "used for", "object": "machine learning"}
  ]
}
```

---

## 4. UI Requirements

### 4.1 Page Structure

```
┌─────────────────────────────────────────────────────────────────┐
│  HEADER: GraphRAG Explorer                    [Cognee] [Hippo]  │
├─────────────────────────────────────────────────────────────────┤
│  SIDEBAR                          │  MAIN CONTENT AREA          │
│  ┌─────────────────────────────┐  │  ┌───────────────────────┐  │
│  │ Strategy Preset Selector    │  │  │ Search Interface      │  │
│  │ ⚡ Fast                     │  │  │ ┌───────────────────┐ │  │
│  │ 🚀 Balanced (default)       │  │  │ │ Query Input       │ │  │
│  │ 🐢 Thorough                 │  │  │ └───────────────────┘ │  │
│  │                             │  │  │ ┌───────────────────┐ │  │
│  │ Search Mode Selector        │  │  │ │ Results Panel     │ │  │
│  │ • Vector                    │  │  │ │                   │ │  │
│  │ • Graph                     │  │  │ │ [Cards/Graph]     │ │  │
│  │ • Hybrid (default)          │  │  │ └───────────────────┘ │  │
│  │                             │  │  └───────────────────────┘  │
│  │ Dataset Selector            │  │                             │
│  │ - default                   │  │  PERFORMANCE METRICS        │
│  │ - research-papers           │  │  ┌───────────────────────┐  │
│  │ - business-reports          │  │  │ Query Time: 95ms      │  │
│  │                             │  │  │ Strategy: Hybrid      │  │
│  │ Parameters                  │  │  │ Cache: L1 Hit         │  │
│  │ - Top K: [10 ▼]             │  │  └───────────────────────┘  │
│  │ - Temperature: [0.7]        │  │                             │
│  │                             │  │  STATS PANEL               │
│  │ Enhancement Toggles         │  │  ┌───────────────────────┐  │
│  │ ☑️ Re-ranking               │  │  │ Entities: 450         │  │
│  │ ☐ HyDE (for vague)          │  │  │ Relations: 890        │  │
│  │ ☑️ Query Rewrite            │  │  │ Documents: 150        │  │
│  │                             │  │  │ Density: 0.15         │  │
│  │ [Stats Button]              │  │  └───────────────────────┘  │
│  │ [Compare Button]            │  │                             │
│  └─────────────────────────────┘  │                             │
│                                   │                             │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Strategy Presets (Integration with RAG System)

The UI should support **strategy presets** that combine GraphRAG with other RAG strategies:

| Preset | Speed | Quality | Configuration | Best For |
|--------|-------|---------|---------------|----------|
| **⚡ Fast** | ~50ms | Good | Basic hybrid/graph search | Chatbots, real-time |
| **🚀 Balanced** | ~150ms | Better | Hybrid + Re-ranking | General production |
| **🐢 Thorough** | ~500ms | Best | All strategies + HyDE | Research, analysis |
| **🎯 GraphRAG** | ~350ms | Best multi-hop | Cognee/HippoRAG + Re-ranking | Complex reasoning |

**Preset Configuration:**

```typescript
const strategyPresets = {
  fast: {
    graphSearch: true,
    reRanking: false,
    hyde: false,
    queryRewrite: false,
    topK: 5,
    description: "Fast retrieval for real-time applications"
  },
  balanced: {
    graphSearch: true,
    reRanking: true,
    hyde: false,
    queryRewrite: true,
    topK: 10,
    description: "Balanced speed and quality (recommended)"
  },
  thorough: {
    graphSearch: true,
    reRanking: true,
    hyde: true,
    queryRewrite: true,
    topK: 20,
    description: "Maximum quality with all enhancements"
  },
  graphrag: {
    graphSearch: true,
    reRanking: true,
    multiHop: true,
    hyde: false,
    topK: 15,
    description: "Optimized for knowledge graph reasoning"
  }
};
```

### 4.3 Feature Requirements

#### FR-1: Search Interface with Strategy Integration

**Cognee Search Tab**
- Query input with placeholder examples
- Search type selector (Vector/Graph/Hybrid)
- **Strategy preset selector** (Fast/Balanced/Thorough/GraphRAG)
- Enhancement toggles:
  - ☑️ Re-ranking (on by default for Balanced/Thorough)
  - ☐ HyDE (for vague queries)
  - ☑️ Query Rewrite (on by default)
- Dataset dropdown
- Top-K slider (1-100)
- Temperature slider (for HyDE)
- Search button with loading state
- **Performance metrics display** (query time, cache hit, strategy used)

**HippoRAG Search Tab**
- Query input (supports multiple queries)
- Strategy preset selector
- Num-to-retrieve slider (1-50)
- QA toggle (on/off for answer generation)
- **Multi-hop visualization** showing reasoning steps
- Search button

**Results Display**
- Result cards showing:
  - Content snippet (expandable)
  - Relevance score
  - Source document
  - Associated entities (as color-coded tags)
  - **Retrieval path** (for HippoRAG multi-hop)
- Graph visualization toggle (for Cognee graph search)
- **Confidence indicator** based on strategy preset

#### FR-2: Knowledge Graph Visualization (Cognee)

- Interactive network graph showing:
  - Entity nodes (color-coded by type)
  - Relationship edges
  - Document clusters
- Features:
  - Zoom and pan
  - Node click → show entity details sidebar
  - Double-click → expand related entities
  - Filter by entity type (toggle ORG, PERSON, etc.)
  - Search within graph
  - **Community detection visualization** (grouped entities)

#### FR-3: Multi-Hop Path Visualization (HippoRAG)

For HippoRAG results, show the reasoning path:

```
Query: "What county is Erik Hort's birthplace part of?"

Reasoning Path:
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│Erik Hort │────▶│birthplace│────▶│Montebello│────▶│Rockland  │
│  PERSON  │     │  REL     │     │ LOCATION │     │  County  │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
     │                                                   │
     └───────────────────────────────────────────────────┘
                    Answer Found!
```

- Visual flow diagram of entity hops
- Each hop shows:
  - Entity/node
  - Relationship type
  - Confidence score
- Collapsible detail view for each hop

#### FR-4: Entity Extraction Tool

- Text input area (paste text)
- "Extract" button with loading state
- Results panel showing:
  - Entity list with types (color-coded)
  - Relationship table
  - Visual entity-relationship diagram
  - **Export options** (JSON, CSV, PNG diagram)
- **Batch processing** for multiple texts

#### FR-5: Statistics Dashboard

For Cognee:
- Document count
- Entity count by type (pie chart)
- Relationship count
- Graph density meter
- Last updated timestamp
- Dataset selector with health indicators

For HippoRAG:
- Indexed documents count
- Knowledge graph size
- Entity extraction stats
- Average processing time
- **Cache hit rate** (L1/L2/L3)

#### FR-6: Comparison Mode

- Side-by-side comparison of:
  - Cognee results (left)
  - HippoRAG results (right)
  - Standard Hybrid Search (optional third column)
- Same query executed on all
- Metrics comparison table:
  - Query time
  - Result count
  - Answer quality score
  - Token usage
  - Strategy complexity
- **Winner highlighting** based on preset criteria
- "Use This Result" action button

#### FR-7: Query Classification Display

Show automatic query classification in UI:

| Query Type | Icon | Recommended Strategy |
|------------|------|---------------------|
| **Factual** | 📋 | Hybrid Search |
| **Analytical** | 🔍 | Query Rewrite + HyDE |
| **Comparative** | ⚖️ | Multi-hop + Re-ranking |
| **Vague** | ❓ | HyDE + Expanded Retrieval |
| **Multi-hop** | 🔗 | **GraphRAG** |

- Display classification badge on results
- Suggest strategy preset based on query type
- Allow manual override

### 4.4 Caching & Performance Indicators

Display cache layer information:

```
Query Performance:
├─ Latency: 95ms
├─ Strategy: Hybrid + Re-ranking
├─ Cache: L1 Redis Hit ⚡
└─ Token Usage: 1,250
```

Cache Layers:
- **L1 (Redis)**: 1-hour TTL - Query embeddings, results, LLM responses
- **L2 (PostgreSQL)**: 24-hour TTL - Embeddings, results
- **L3 (Semantic)**: 7-day TTL - Similar query matching

### 4.5 UI Components

#### Component Library Requirements
- **SearchInput**: Auto-suggest, history dropdown, query classification badge
- **StrategyPresetSelector**: Visual cards for Fast/Balanced/Thorough/GraphRAG
- **ResultCard**: Expandable, entity tags, source link, confidence indicator
- **EntityGraph**: D3.js or Cytoscape.js network visualization
- **MultiHopPath**: Flow diagram for HippoRAG reasoning
- **StatsPanel**: Mini charts, metrics display, health indicators
- **ComparisonView**: Split-pane layout with metrics table
- **TripleViewer**: Subject-Predicate-Object table with visualization
- **PerformanceMetrics**: Latency, cache hit, token usage display

#### Responsive Design
- **Desktop** (>1200px): Full sidebar + main content + comparison mode
- **Tablet** (768-1200px): Collapsible sidebar, stacked comparison
- **Mobile** (<768px): Tab-based navigation, single column, simplified graph

### 4.6 User Flows

#### Flow 1: Simple Cognee Search with Presets
1. User selects "Cognee" tab
2. Selects "🚀 Balanced" preset (auto-configures re-ranking, query rewrite)
3. Enters query: "machine learning in healthcare"
4. Selects search type: "hybrid"
5. Clicks "Search"
6. Views result cards with performance metrics (95ms, L1 cache hit)
7. Clicks entity tag to filter by that entity
8. Toggles graph view to see relationships

#### Flow 2: Complex Multi-Hop Query with Path Visualization
1. User selects "HippoRAG" tab
2. Selects "🐢 Thorough" preset
3. Enters query: "What county is Erik Hort's birthplace part of?"
4. UI shows "🔗 Multi-hop" classification badge
5. Enables QA mode
6. Clicks "Search"
7. Views **reasoning path visualization** showing entity hops
8. Views generated answer with supporting passages
9. Expands retrieval results to see entity connections
10. Exports answer and sources

#### Flow 3: Strategy Comparison
1. User enters query: "Compare Tesla and SpaceX technologies"
2. Clicks "Compare Strategies" button
3. Views 3-column comparison:
   - Column 1: Standard Hybrid (50ms, basic results)
   - Column 2: Cognee GraphRAG (120ms, entity relationships)
   - Column 3: HippoRAG (380ms, synthesized answer)
4. Compares metrics table
5. UI highlights HippoRAG as "🏆 Best for Multi-hop"
6. User selects preferred result set

#### Flow 4: Vague Query with HyDE
1. User enters vague query: "Tell me about cool tech stuff"
2. UI classifies as "❓ Vague" query type
3. Suggests enabling HyDE
4. User clicks "Apply HyDE"
5. UI generates hypothetical document preview
6. User clicks "Search with HyDE"
7. Views improved results based on hypothetical embedding

---

## 5. Technical Requirements

### 5.1 API Integration

```typescript
// API Client Interface
interface GraphRAGAPI {
  // Cognee endpoints
  cogneeSearch(request: CogneeSearchRequest): Promise<CogneeSearchResponse>;
  cogneeExtractEntities(request: ExtractRequest): Promise<ExtractResponse>;
  cogneeStats(datasetId?: string): Promise<StatsResponse>;
  
  // HippoRAG endpoints
  hipporagRetrieve(request: HippoRetrieveRequest): Promise<HippoRetrieveResponse>;
  hipporagQA(request: HippoQARequest): Promise<HippoQAResponse>;
  hipporagExtractTriples(request: ExtractRequest): Promise<TripleResponse>;
  
  // Strategy endpoints
  classifyQuery(query: string): Promise<QueryClassification>;
  getCacheStatus(): Promise<CacheMetrics>;
}

interface QueryClassification {
  type: 'factual' | 'analytical' | 'comparative' | 'vague' | 'multi_hop';
  confidence: number;
  recommendedStrategy: string;
  recommendedPreset: 'fast' | 'balanced' | 'thorough' | 'graphrag';
}

interface CacheMetrics {
  l1_hit_rate: number;
  l2_hit_rate: number;
  l3_hit_rate: number;
  avg_latency_ms: number;
}
```

### 5.2 Authentication
- Use existing X-API-Key header
- Support API key input in UI settings
- Role-based access (view vs. admin for entity extraction)

### 5.3 Error Handling
- Network error: Retry with exponential backoff (3 attempts)
- API error: Display user-friendly message with error code
- Timeout: Show partial results with warning banner
- Rate limiting: Display countdown timer for retry

### 5.4 Performance Requirements
- Debounce search input: 300ms
- Virtual scrolling for result sets >50 items
- Lazy load graph visualization (render first 50 nodes, then expand)
- Cache stats data: 30-second TTL
- Progressive loading for multi-hop paths

---

## 6. Acceptance Criteria

### AC-1: Search Functionality
- [ ] User can perform Cognee search with all 3 types
- [ ] User can perform HippoRAG retrieve and QA
- [ ] Results display within 3 seconds (or show loading state)
- [ ] Error messages are user-friendly with retry option
- [ ] **Strategy presets apply correct configurations**

### AC-2: Visualization
- [ ] Graph renders with <100 nodes without lag
- [ ] Entity types are color-coded correctly
- [ ] Graph is interactive (zoom, pan, click)
- [ ] Graph legend is visible and toggleable
- [ ] **Multi-hop path visualization shows reasoning steps**

### AC-3: Entity Extraction
- [ ] Extracts entities from pasted text
- [ ] Shows confidence scores
- [ ] Allows exporting results (JSON, CSV, PNG)
- [ ] **Color-coded entity type badges**

### AC-4: Comparison Mode
- [ ] Side-by-side layout works on screens >1200px
- [ ] Metrics are clearly displayed with units
- [ ] User can select preferred results
- [ ] **Winner highlighting based on query type**

### AC-5: Performance Indicators
- [ ] Query latency displayed in ms
- [ ] Cache hit/miss shown with color coding
- [ ] Token usage displayed for QA mode
- [ ] **Strategy preset used is shown**

### AC-6: Mobile Responsiveness
- [ ] Layout adapts to mobile screens
- [ ] All features accessible on mobile
- [ ] Touch interactions work correctly
- [ ] **Simplified graph view for mobile**

### AC-7: Query Classification
- [ ] Auto-classifies queries correctly
- [ ] Shows classification badge
- [ ] Suggests appropriate preset
- [ ] **Allows manual override**

---

## 7. Integration with Existing RAG System

### 8.1 Configuration Precedence

```
User Selection → Preset Defaults → System Defaults
     │               │                │
     ▼               ▼                ▼
 Custom Settings  Preset Config   Base Config
 (highest)        (medium)        (lowest)
```

### 8.2 Environment Variables

```bash
# GraphRAG Core
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=cognee-graph-db
HIPPO_SAVE_DIR=/data/hipporag

# Strategy Integration
COGNEE_LLM_MODEL=azure/gpt-4.1
HIPPO_LLM_MODEL=azure/gpt-4.1
HYBRID_SEARCH_DEFAULT_VECTOR_WEIGHT=0.7
RERANKING_ENABLED=true
HYDE_ENABLED=true
QUERY_REWRITE_ENABLED=true
```

### 8.3 Feature Flags

```typescript
const featureFlags = {
  graphragEnabled: true,
  hipporagEnabled: true,
  cogneeEnabled: true,
  comparisonMode: true,
  multiHopVisualization: true,
  queryClassification: true,
  strategyPresets: true,
  cachingIndicators: true
};
```

---

## 8. Out of Scope

- Real-time collaborative editing
- Custom model training UI
- Advanced graph analytics (centrality, PageRank calculation)
- PDF viewer integration
- User authentication (use existing system)
- Data ingestion UI (separate feature)
- **Custom strategy preset builder** (future enhancement)

---

## 9. Future Enhancements

- **Dataset Management UI**: Create, delete, manage Cognee datasets
- **Query History**: Save and replay previous queries
- **Saved Searches**: Bookmark useful queries
- **Export Formats**: PDF reports, PowerPoint slides
- **Advanced Filters**: Date ranges, document types, confidence thresholds
- **Chat Interface**: Conversational GraphRAG interface
- **Custom Strategy Builder**: User-defined preset creation
- **A/B Testing**: Compare strategy effectiveness over time
- **Performance Analytics**: Query performance trends

---

## 10. Appendix

### A. HTTP Request Examples

See:
- `http/cognee.graphrag.http` - Local development examples
- `http/hipporag.graphrag.http` - Local development examples  
- `http/production/cognee.graphrag.http` - Production examples
- `http/production/hipporag.graphrag.http` - Production examples

### B. OpenAPI Spec

Full API specification: `api/openapi.graphrag.yaml`

### C. Related Documentation

- `docs/GRAPHRAG_OVERVIEW.md` - Architecture overview
- `docs/usage/cognee-local.md` - Cognee usage guide
- `docs/usage/hipporag.md` - HippoRAG usage guide
- `docs/API_GUIDE.md` - Complete API reference
- `docs/RAG_STRATEGY_GUIDE.md` - RAG strategy selection and configuration

### D. Performance Benchmarks

| Strategy | Query Time | Accuracy | Token Usage |
|----------|------------|----------|-------------|
| Standard Hybrid | 50ms | 75% | 500 |
| + Re-ranking | 100ms | 82% | 500 |
| + HyDE | 200ms | 85% | 800 |
| Cognee Hybrid | 95ms | 80% | 500 |
| Cognee + Re-ranking | 150ms | 87% | 500 |
| HippoRAG Retrieve | 320ms | 85% | 300 |
| HippoRAG QA | 1250ms | 91% | 1500 |

---

## 11. Review & Approval

| Role | Name | Date | Status |
|------|------|------|--------|
| Product Owner | | | ⬜ Pending |
| Tech Lead | | | ⬜ Pending |
| UX Designer | | | ⬜ Pending |
| RAG Specialist | | | ⬜ Pending |

---

**END OF DOCUMENT**
