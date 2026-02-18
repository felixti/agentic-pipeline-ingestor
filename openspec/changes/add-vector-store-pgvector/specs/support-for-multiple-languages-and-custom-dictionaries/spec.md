# Spec: Support for Multiple Languages and Custom Dictionaries

## Purpose
Support non-English text search by using language-specific text search configurations and custom dictionaries for domain-specific terminology.

## Interface

### Language-Specific Setup
```sql
-- Create indexes for multiple languages
CREATE INDEX idx_document_chunks_content_en 
ON document_chunks USING gin (to_tsvector('english', content));

CREATE INDEX idx_document_chunks_content_es 
ON document_chunks USING gin (to_tsvector('spanish', content));

CREATE INDEX idx_document_chunks_content_fr 
ON document_chunks USING gin (to_tsvector('french', content));

CREATE INDEX idx_document_chunks_content_de 
ON document_chunks USING gin (to_tsvector('german', content));
```

### Language Detection and Search
```sql
-- Search with language parameter
SELECT 
    id,
    content,
    ts_rank_cd(
        to_tsvector('spanish', content),
        plainto_tsquery('spanish', 'inteligencia artificial')
    ) AS rank
FROM document_chunks
WHERE to_tsvector('spanish', content) @@ plainto_tsquery('spanish', 'inteligencia artificial')
ORDER BY rank DESC;
```

### API Parameters
```json
{
  "query": "inteligencia artificial",
  "language": "spanish",
  "fallback_languages": ["english", "simple"],
  "auto_detect": true,
  "top_k": 10
}
```

### Supported Languages
| Language | Config Name | pg_trgm Support |
|----------|-------------|-----------------|
| Arabic | arabic | ✓ |
| Danish | danish | ✓ |
| Dutch | dutch | ✓ |
| English | english | ✓ |
| Finnish | finnish | ✓ |
| French | french | ✓ |
| German | german | ✓ |
| Hungarian | hungarian | ✓ |
| Indonesian | indonesian | ✓ |
| Irish | irish | ✓ |
| Italian | italian | ✓ |
| Lithuanian | lithuanian | ✓ |
| Nepali | nepali | ✓ |
| Norwegian | norwegian | ✓ |
| Portuguese | portuguese | ✓ |
| Romanian | romanian | ✓ |
| Russian | russian | ✓ |
| Simple | simple | ✓ |
| Spanish | spanish | ✓ |
| Swedish | swedish | ✓ |
| Tamil | tamil | ✓ |
| Turkish | turkish | ✓ |

### Custom Dictionary Creation
```sql
-- Create custom dictionary for domain terms
CREATE TEXT SEARCH DICTIONARY tech_terms (
    TEMPLATE = pg_catalog.simple,
    STOPWORDS = english
);

-- Create custom text search configuration
CREATE TEXT SEARCH CONFIGURATION tech_english (COPY = english);

-- Add custom mapping
ALTER TEXT SEARCH CONFIGURATION tech_english
    ALTER MAPPING FOR asciiword, word
    WITH tech_terms, english_stem;

-- Create custom thesaurus for synonyms
CREATE TEXT SEARCH DICTIONARY tech_thesaurus (
    TEMPLATE = thesaurus,
    DictFile = tech_synonyms,
    Dictionary = english_stem
);
```

### Synonym Dictionary (thesaurus/tech_synonyms.ths)
```
AI : artificial intelligence
ML : machine learning
NN : neural network
DL : deep learning
CV : computer vision
NLP : natural language processing
```

### SQLAlchemy Integration
```python
from sqlalchemy import func, Index, text
from enum import Enum

class LanguageCode(str, Enum):
    ENGLISH = "english"
    SPANISH = "spanish"
    FRENCH = "french"
    GERMAN = "german"
    PORTUGUESE = "portuguese"
    SIMPLE = "simple"

class SearchService:
    SUPPORTED_LANGUAGES = [lang.value for lang in LanguageCode]
    DEFAULT_LANGUAGE = "english"
    
    def search(
        self, 
        query: str, 
        language: str = "english",
        fallback_languages: list = None
    ):
        # Validate language
        if language not in self.SUPPORTED_LANGUAGES:
            language = self.DEFAULT_LANGUAGE
        
        tsvector = func.to_tsvector(language, DocumentChunk.content)
        tsquery = func.plainto_tsquery(language, query)
        
        results = (
            self.db.query(DocumentChunk)
            .filter(tsvector.op('@@')(tsquery))
            .order_by(func.ts_rank_cd(tsvector, tsquery).desc())
            .all()
        )
        
        # Fallback if no results
        if not results and fallback_languages:
            for fallback in fallback_languages:
                if fallback != language:
                    results = self._search_with_language(query, fallback)
                    if results:
                        break
        
        return results
    
    def _search_with_language(self, query: str, language: str):
        tsvector = func.to_tsvector(language, DocumentChunk.content)
        tsquery = func.plainto_tsquery(language, query)
        return (
            self.db.query(DocumentChunk)
            .filter(tsvector.op('@@')(tsquery))
            .order_by(func.ts_rank_cd(tsvector, tsquery).desc())
            .all()
        )
```

## Behavior

### Language-Specific Processing

#### English
```sql
-- Stemming: running → run, algorithms → algorithm
-- Stop words: the, and, is, of
SELECT to_tsvector('english', 'The running algorithms');
-- Returns: 'algorithm':3 'run':2
```

#### Spanish
```sql
-- Stemming: corriendo → corr, algoritmos → algoritm
-- Stop words: el, la, de, y
SELECT to_tsvector('spanish', 'Los algoritmos corriendo');
-- Returns: 'algoritm':2 'corr':3
```

#### French
```sql
-- Stemming: algorithmes → algorithm, en cours → cour
-- Stop words: le, la, les, et
SELECT to_tsvector('french', 'Les algorithmes en cours');
-- Returns: 'algorithm':2 'cour':4
```

### Auto-Detection Strategy
```python
import langdetect

def detect_language(text: str) -> str:
    """
    Detect language with confidence threshold.
    Falls back to 'simple' for short or ambiguous text.
    """
    try:
        if len(text) < 20:
            return 'simple'  # No stemming for short text
        
        detected = langdetect.detect(text)
        lang_map = {
            'en': 'english',
            'es': 'spanish',
            'fr': 'french',
            'de': 'german',
            'pt': 'portuguese',
            # ... more mappings
        }
        return lang_map.get(detected, 'simple')
    except:
        return 'simple'
```

### Custom Dictionary Behavior

#### Thesaurus Expansion
```sql
-- With tech_thesaurus configured:
-- Query "AI" expands to "artificial intelligence"
SELECT plainto_tsquery('tech_english', 'AI');
-- Returns: 'artifici' <-> 'intellig'
```

#### Domain-Specific Stop Words
```sql
-- Custom stop words for programming docs
-- Remove: function, class, method, variable
-- Keep: specific API names, domain terms
```

### Search Result Format
```json
{
  "results": [
    {
      "id": "chunk-uuid",
      "content": "La inteligencia artificial...",
      "detected_language": "spanish",
      "rank": 0.785,
      "highlighted": "La <mark>inteligencia</mark> <mark>artificial</mark>..."
    }
  ],
  "query_info": {
    "original": "inteligencia artificial",
    "requested_language": "spanish",
    "detected_language": "spanish",
    "fallback_used": false
  },
  "languages_available": ["english", "spanish", "french", "german"]
}
```

### Multi-Language Search Strategy
```sql
-- Search across all language configs and combine results
WITH ranked_results AS (
    SELECT 
        id,
        content,
        'english' AS lang,
        ts_rank_cd(to_tsvector('english', content), 
                   plainto_tsquery('english', 'query')) AS rank
    FROM document_chunks
    WHERE to_tsvector('english', content) @@ plainto_tsquery('english', 'query')
    
    UNION ALL
    
    SELECT 
        id,
        content,
        'spanish' AS lang,
        ts_rank_cd(to_tsvector('spanish', content), 
                   plainto_tsquery('spanish', 'query')) AS rank
    FROM document_chunks
    WHERE to_tsvector('spanish', content) @@ plainto_tsquery('spanish', 'query')
)
SELECT * FROM ranked_results
ORDER BY rank DESC
LIMIT 10;
```

## Error Handling

| Error Case | Error Type | Handling |
|------------|------------|----------|
| Unsupported language | ValidationError | Return 400 with list of supported languages |
| Language config missing | RuntimeError | Fallback to 'simple' (no stemming), log warning |
| Custom dictionary not found | RuntimeError | Use base language config, log error |
| Auto-detection failure | RuntimeError | Fallback to 'simple' or user-specified default |
| Mixed language content | Edge Case | Search in requested language + 'simple' |
| Dictionary file permission error | RuntimeError | Return 500 with "Dictionary unavailable" |

## Performance Considerations

### Index Strategy Per Language
```sql
-- Option 1: Separate index per language (recommended for known distribution)
CREATE INDEX idx_en ON chunks USING gin(to_tsvector('english', content)) 
WHERE language = 'en';

-- Option 2: Functional index with language column
CREATE INDEX idx_multi ON chunks USING gin(to_tsvector(language, content));

-- Option 3: Simple config for mixed content (no stemming)
CREATE INDEX idx_simple ON chunks USING gin(to_tsvector('simple', content));
```

### Index Size Comparison
| Configuration | Relative Size | Best For |
|---------------|---------------|----------|
| english | 1.0x | English content |
| simple | 0.7x | Mixed/unknown languages |
| Custom dictionary | 1.1x | Domain-specific terms |

### Query Performance
| Scenario | Latency Impact |
|----------|----------------|
| Single language, indexed | Baseline |
| Language fallback (2 attempts) | 1.5-2x |
| Multi-language UNION | Nx (N languages) |
| Auto-detection | +5-10ms |

### Resource Usage
- **Memory**: ~10MB per loaded dictionary
- **Disk**: Indexes add 30-50% storage overhead per language
- **CPU**: Stemming adds ~20% query overhead vs simple config

### Optimization Guidelines
1. **Prefer 'simple' for mixed content**: No stemming = faster indexing
2. **Language column**: Store detected language for index partitioning
3. **Lazy dictionary loading**: Load custom dictionaries on first use
4. **Cache language detection**: Store detected language with content

### Benchmark Targets
- Language detection: < 10ms per query
- Single-language search: Same as English baseline
- Multi-language search: < 2x single-language time
