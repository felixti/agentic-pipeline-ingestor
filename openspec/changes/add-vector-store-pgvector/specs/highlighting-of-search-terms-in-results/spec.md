# Spec: Highlighting of Search Terms in Results

## Purpose
Highlight matching search terms in result snippets using PostgreSQL's ts_headline() function with customizable tags and snippet generation.

## Interface

### SQL Implementation
```sql
-- Basic highlighting with default <b> tags
SELECT 
    id,
    content,
    ts_headline(
        'english',
        content,
        plainto_tsquery('english', 'machine learning'),
        'StartSel=<mark>, StopSel=</mark>, MaxWords=35, MinWords=15, MaxFragments=3, FragmentDelimiter=" ... "'
    ) AS highlighted_snippet
FROM document_chunks
WHERE to_tsvector('english', content) @@ plainto_tsquery('english', 'machine learning')
LIMIT 10;
```

### Advanced Highlighting Options
```sql
-- Multi-fragment highlighting with custom tags
SELECT 
    id,
    ts_headline(
        'english',
        content,
        query,
        ts_headline_options(
            startsel := '<mark class="search-match">',
            stopsel := '</mark>',
            maxwords := 50,
            minwords := 10,
            shortword := 3,
            highlightall := false,
            maxfragments := 3,
            fragmentdelimiter := ' ... '
        )
    ) AS highlighted_content
FROM document_chunks,
     plainto_tsquery('english', 'search terms') query
WHERE to_tsvector('english', content) @@ query;
```

### API Parameters
```json
{
  "query": "artificial intelligence",
  "highlight_options": {
    "start_tag": "<mark>",
    "end_tag": "</mark>",
    "max_words": 50,
    "min_words": 10,
    "max_fragments": 3,
    "fragment_delimiter": " ... ",
    "highlight_all": false
  },
  "include_full_content": false,
  "top_k": 10
}
```

### Highlight Options Reference
| Option | Default | Description |
|--------|---------|-------------|
| StartSel | `<b>` | Opening tag for highlighted terms |
| StopSel | `</b>` | Closing tag for highlighted terms |
| MaxWords | 35 | Maximum words per fragment |
| MinWords | 15 | Minimum words per fragment |
| ShortWord | 3 | Minimum word length to include |
| HighlightAll | false | Highlight entire content vs fragments |
| MaxFragments | 0 (auto) | Number of fragments to return |
| FragmentDelimiter | ` ... ` | Separator between fragments |

### SQLAlchemy Integration
```python
from sqlalchemy import func
from typing import Optional

class SearchService:
    DEFAULT_HIGHLIGHT_OPTIONS = {
        'StartSel': '<mark>',
        'StopSel': '</mark>',
        'MaxWords': 50,
        'MinWords': 15,
        'MaxFragments': 3,
        'FragmentDelimiter': ' ... '
    }
    
    def search_with_highlighting(
        self, 
        query: str, 
        highlight_options: Optional[dict] = None,
        top_k: int = 10
    ):
        options = highlight_options or self.DEFAULT_HIGHLIGHT_OPTIONS
        options_str = ', '.join(f"{k}={v}" for k, v in options.items())
        
        tsvector = func.to_tsvector('english', DocumentChunk.content)
        tsquery = func.plainto_tsquery('english', query)
        
        highlighted = func.ts_headline(
            'english',
            DocumentChunk.content,
            tsquery,
            options_str
        )
        
        return (
            self.db.query(
                DocumentChunk.id,
                DocumentChunk.job_id,
                highlighted.label('highlighted_snippet'),
                DocumentChunk.content.label('full_content')
            )
            .filter(tsvector.op('@@')(tsquery))
            .limit(top_k)
            .all()
        )
```

## Behavior

### Snippet Generation
1. **Fragment Selection**: Identify text regions containing query terms
2. **Context Expansion**: Add surrounding words (MinWords to MaxWords)
3. **Multiple Fragments**: Return up to MaxFragments separated by delimiter
4. **Overlap Prevention**: Avoid repeating content across fragments

### Highlight Example
```
Query: "neural networks"

Original Content:
"Deep neural networks have revolutionized machine learning. Neural network 
architectures like CNNs and transformers power modern AI systems. The neural 
network paradigm continues to evolve."

Highlighted Result:
"Deep <mark>neural</mark> <mark>networks</mark> have revolutionized machine 
learning. ... <mark>Neural</mark> <mark>network</mark> architectures like CNNs 
and transformers..."
```

### Fragment Extraction Logic
```python
def extract_fragments(content: str, query_terms: list, options: dict):
    """
    1. Find all positions of query terms in content
    2. For each position, expand by MinWords/MaxWords
    3. Merge overlapping fragments
    4. Return up to MaxFragments best matches
    """
    matches = find_term_positions(content, query_terms)
    fragments = []
    
    for match in matches:
        start = max(0, match.start - options['MinWords'])
        end = min(len(content), match.end + options['MinWords'])
        
        # Expand to MaxWords if possible
        while end - start < options['MaxWords'] and (start > 0 or end < len(content)):
            if start > 0:
                start -= 1
            if end < len(content):
                end += 1
        
        fragments.append(content[start:end])
    
    # Merge overlapping and return top fragments
    return merge_and_select_fragments(fragments, options['MaxFragments'])
```

### Search Result Format
```json
{
  "results": [
    {
      "id": "chunk-uuid",
      "job_id": "job-uuid",
      "highlighted_snippet": "Deep <mark>neural</mark> <mark>networks</mark> have revolutionized machine learning...",
      "full_content": "Deep neural networks have revolutionized...",
      "matched_terms": ["neural", "networks"],
      "fragment_positions": [
        {"start": 0, "end": 60},
        {"start": 145, "end": 220}
      ]
    }
  ],
  "highlight_options_used": {
    "start_tag": "<mark>",
    "end_tag": "</mark>",
    "max_fragments": 3
  }
}
```

### Highlighting Modes

#### Fragment Mode (Default)
```sql
-- Returns relevant excerpts with matches
MaxFragments=3, FragmentDelimiter=' ... '
```

#### Full Content Mode
```sql
-- Highlights all matches in complete content
HighlightAll=true
```

#### Single Best Fragment
```sql
-- Returns only the highest-scoring fragment
MaxFragments=1, MaxWords=100
```

## Error Handling

| Error Case | Error Type | Handling |
|------------|------------|----------|
| Invalid tag characters | ValidationError | Escape special HTML chars in tags |
| MaxWords < MinWords | ValidationError | Swap values or return 400 |
| Empty highlight result | Edge Case | Return original content with warning |
| Tag injection attempt | SecurityError | Sanitize user-provided tags |
| Unicode highlighting failure | EncodingError | Fallback to ASCII-safe content |
| Content too long for processing | PerformanceError | Truncate to 10K chars with warning |

## Performance Considerations

### ts_headline Performance
- **CPU Intensive**: Text parsing and tag insertion is expensive
- **Sequential Processing**: Each document processed individually
- **Memory Usage**: Creates copy of content with tags

### Optimization Strategies

#### 1. Limit Content Size
```sql
-- Only highlight first N characters
SELECT ts_headline(
    'english',
    LEFT(content, 10000),  -- Limit input size
    query
) FROM document_chunks;
```

#### 2. Conditional Highlighting
```python
# Only highlight top N results fully
top_results = search(query, top_k=100)
for i, result in enumerate(top_results):
    if i < 10:  # Full highlighting for top 10
        result.highlighted = generate_highlight(result, query)
    else:  # Simple snippet for rest
        result.highlighted = generate_simple_snippet(result, query)
```

#### 3. Pre-computed Snippets
```sql
-- Materialize common search result formats
CREATE TABLE search_snippets (
    chunk_id UUID PRIMARY KEY,
    snippet_50w TEXT,  -- Pre-computed 50-word snippets
    snippet_100w TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Benchmark Targets
| Operation | Target Latency |
|-----------|----------------|
| Highlight 1K chars | < 5ms |
| Highlight 10K chars | < 20ms |
| Highlight 100K chars | < 100ms |

### Resource Usage
- **Memory**: ~2x content size during processing
- **CPU**: Proportional to content length and match count
- **I/O**: Minimal (uses already-fetched content)

### Caching Recommendations
1. Cache highlighted results for popular queries
2. Use Redis/Memcached for snippet storage
3. Invalidate cache on content updates
4. Pre-generate snippets for trending content
