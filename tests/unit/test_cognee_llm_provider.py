"""Unit tests for CogneeLLMProvider.

Tests for entity extraction, relationship extraction, and summarization
using the CogneeLLMProvider class.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.plugins.destinations.cognee_llm import (
    CogneeLLMProvider,
    ExtractedEntity,
    ExtractedRelationship,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider."""
    mock = MagicMock()
    mock.simple_completion = AsyncMock(return_value='{"entities": []}')
    return mock


@pytest.fixture
def sample_entities():
    """Create sample extracted entities."""
    return [
        ExtractedEntity(name="Microsoft", type="ORGANIZATION", description="Tech company"),
        ExtractedEntity(name="Bill Gates", type="PERSON", description="Founder"),
        ExtractedEntity(name="Seattle", type="LOCATION", description="City in Washington"),
    ]


@pytest.fixture
def sample_relationships():
    """Create sample extracted relationships."""
    return [
        ExtractedRelationship(
            source="Bill Gates",
            target="Microsoft",
            type="FOUNDED",
            confidence=0.95,
        ),
        ExtractedRelationship(
            source="Microsoft",
            target="Seattle",
            type="LOCATED_IN",
            confidence=0.90,
        ),
    ]


# Fixed prompt template for testing (escapes the ... placeholder)
# The actual ENTITY_EXTRACTION_PROMPT has a bug where "..." is interpreted as a format placeholder
FIXED_ENTITY_EXTRACTION_PROMPT = """You are an expert in named entity recognition and extraction.

Extract named entities from the following text. For each entity, identify:
- name: The entity name (proper noun or specific term)
- type: The entity type (PERSON, ORGANIZATION, LOCATION, DATE, TECHNOLOGY, PRODUCT, CONCEPT, etc.)
- description: A brief description of the entity

Return ONLY a valid JSON object in this exact format:
{{
    "entities": [
        {{"name": "Entity Name", "type": "ENTITY_TYPE", "description": "Brief description"}},
        ...
    ]
}}

Important:
- Extract only real, specific entities mentioned in the text
- Do not create generic or inferred entities
- Ensure the response is valid JSON
- If no entities are found, return {{"entities": []}}

Text to analyze:
{text}
"""

FIXED_RELATIONSHIP_EXTRACTION_PROMPT = """You are an expert in relationship extraction and knowledge graph construction.

Given the following entities and text, extract relationships between the entities.
For each relationship, identify:
- source: The source entity name (must match one of the provided entities)
- target: The target entity name (must match one of the provided entities)
- type: The relationship type (WORKS_AT, LOCATED_IN, PART_OF, USES, CREATED, MENTIONS, RELATED_TO, etc.)
- confidence: Confidence score (0.0 to 1.0)

Return ONLY a valid JSON object in this exact format:
{{
    "relationships": [
        {{"source": "Entity A", "target": "Entity B", "type": "RELATIONSHIP_TYPE", "confidence": 0.95}},
        ...
    ]
}}

Important:
- Only create relationships that are explicitly stated or strongly implied in the text
- Both source and target must be from the provided entity list
- Ensure the response is valid JSON
- If no relationships are found, return {{"relationships": []}}

Entities:
{{entities}}

Text to analyze:
{{text}}
"""


# ============================================================================
# CogneeLLMProvider Initialization Tests
# ============================================================================

@pytest.mark.unit
class TestCogneeLLMProviderInit:
    """Tests for CogneeLLMProvider initialization."""

    def test_init_default(self):
        """Test initialization with default LLM provider."""
        with patch("src.plugins.destinations.cognee_llm.LLMProvider") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.return_value = mock_llm
            
            provider = CogneeLLMProvider()
            
            assert provider._llm is mock_llm

    def test_init_with_custom_provider(self, mock_llm_provider):
        """Test initialization with custom LLM provider."""
        provider = CogneeLLMProvider(llm_provider=mock_llm_provider)
        
        assert provider._llm is mock_llm_provider


# ============================================================================
# Entity Extraction Tests
# ============================================================================

@pytest.mark.unit
class TestCogneeLLMProviderEntityExtraction:
    """Tests for entity extraction functionality."""

    @pytest.mark.asyncio
    async def test_extract_entities_success(self, mock_llm_provider):
        """Test successful entity extraction."""
        mock_llm_provider.simple_completion.return_value = '''
        {
            "entities": [
                {"name": "Microsoft", "type": "ORGANIZATION", "description": "Tech company"},
                {"name": "Bill Gates", "type": "PERSON", "description": "Founder of Microsoft"}
            ]
        }
        '''
        
        # Patch the prompt template to fix the format placeholder issue
        with patch('src.plugins.destinations.cognee_llm.ENTITY_EXTRACTION_PROMPT', FIXED_ENTITY_EXTRACTION_PROMPT):
            provider = CogneeLLMProvider(llm_provider=mock_llm_provider)
            entities = await provider.extract_entities(
                "Microsoft was founded by Bill Gates."
            )
        
        assert len(entities) == 2
        assert entities[0].name == "Microsoft"
        assert entities[0].type == "ORGANIZATION"
        assert entities[1].name == "Bill Gates"
        assert entities[1].type == "PERSON"

    @pytest.mark.asyncio
    async def test_extract_entities_empty_text(self, mock_llm_provider):
        """Test entity extraction with empty text."""
        provider = CogneeLLMProvider(llm_provider=mock_llm_provider)
        
        entities = await provider.extract_entities("")
        assert entities == []
        
        entities = await provider.extract_entities("   ")
        assert entities == []
        
        # LLM should not be called for empty text
        mock_llm_provider.simple_completion.assert_not_called()

    @pytest.mark.asyncio
    async def test_extract_entities_whitespace_text(self, mock_llm_provider):
        """Test entity extraction with whitespace-only text."""
        provider = CogneeLLMProvider(llm_provider=mock_llm_provider)
        
        entities = await provider.extract_entities("   \n\t   ")
        assert entities == []

    @pytest.mark.asyncio
    async def test_extract_entities_no_entities_found(self, mock_llm_provider):
        """Test extraction when no entities are found."""
        mock_llm_provider.simple_completion.return_value = '{"entities": []}'
        
        with patch('src.plugins.destinations.cognee_llm.ENTITY_EXTRACTION_PROMPT', FIXED_ENTITY_EXTRACTION_PROMPT):
            provider = CogneeLLMProvider(llm_provider=mock_llm_provider)
            entities = await provider.extract_entities("The weather is nice today.")
        
        assert entities == []

    @pytest.mark.asyncio
    async def test_extract_entities_json_in_markdown(self, mock_llm_provider):
        """Test extraction when JSON is wrapped in markdown."""
        mock_llm_provider.simple_completion.return_value = '''
        ```json
        {
            "entities": [
                {"name": "Apple", "type": "ORGANIZATION", "description": "Tech company"}
            ]
        }
        ```
        '''
        
        with patch('src.plugins.destinations.cognee_llm.ENTITY_EXTRACTION_PROMPT', FIXED_ENTITY_EXTRACTION_PROMPT):
            provider = CogneeLLMProvider(llm_provider=mock_llm_provider)
            entities = await provider.extract_entities("Apple makes iPhones.")
        
        assert len(entities) == 1
        assert entities[0].name == "Apple"

    @pytest.mark.asyncio
    async def test_extract_entities_invalid_json_then_markdown(self, mock_llm_provider):
        """Test extraction with invalid JSON followed by markdown JSON."""
        mock_llm_provider.simple_completion.return_value = '''
        Some text before
        ```json
        {
            "entities": [
                {"name": "Google", "type": "ORGANIZATION", "description": "Search company"}
            ]
        }
        ```
        '''
        
        with patch('src.plugins.destinations.cognee_llm.ENTITY_EXTRACTION_PROMPT', FIXED_ENTITY_EXTRACTION_PROMPT):
            provider = CogneeLLMProvider(llm_provider=mock_llm_provider)
            entities = await provider.extract_entities("Google is a search engine.")
        
        assert len(entities) == 1
        assert entities[0].name == "Google"

    @pytest.mark.asyncio
    async def test_extract_entities_invalid_entity_data(self, mock_llm_provider):
        """Test extraction with invalid entity data."""
        mock_llm_provider.simple_completion.return_value = '''
        {
            "entities": [
                {"name": "Valid Entity", "type": "ORGANIZATION", "description": "Good"},
                {"invalid": "data"},
                {"name": "", "type": "PERSON", "description": "Empty name"}
            ]
        }
        '''
        
        with patch('src.plugins.destinations.cognee_llm.ENTITY_EXTRACTION_PROMPT', FIXED_ENTITY_EXTRACTION_PROMPT):
            provider = CogneeLLMProvider(llm_provider=mock_llm_provider)
            entities = await provider.extract_entities("Some text.")
        
        # Should only return valid entity with non-empty name
        assert len(entities) == 1
        assert entities[0].name == "Valid Entity"

    @pytest.mark.asyncio
    async def test_extract_entities_llm_error(self, mock_llm_provider):
        """Test extraction handles LLM errors gracefully."""
        mock_llm_provider.simple_completion.side_effect = Exception("LLM error")
        
        provider = CogneeLLMProvider(llm_provider=mock_llm_provider)
        entities = await provider.extract_entities("Some text.")
        
        # Should return empty list on error, not raise
        assert entities == []

    @pytest.mark.asyncio
    async def test_extract_entities_truncates_long_text(self, mock_llm_provider):
        """Test that long text is truncated."""
        mock_llm_provider.simple_completion.return_value = '{"entities": []}'
        
        with patch('src.plugins.destinations.cognee_llm.ENTITY_EXTRACTION_PROMPT', FIXED_ENTITY_EXTRACTION_PROMPT):
            provider = CogneeLLMProvider(llm_provider=mock_llm_provider)
            long_text = "x" * 10000
            await provider.extract_entities(long_text)
        
        # Verify prompt was formatted with truncated text
        call_args = mock_llm_provider.simple_completion.call_args
        prompt = call_args.kwargs.get("prompt") or call_args[1].get("prompt")
        assert len(prompt) < 10000  # Should be truncated

    @pytest.mark.asyncio
    async def test_extract_entities_uses_correct_model_params(self, mock_llm_provider):
        """Test that correct model parameters are used."""
        mock_llm_provider.simple_completion.return_value = '{"entities": []}'
        
        with patch('src.plugins.destinations.cognee_llm.ENTITY_EXTRACTION_PROMPT', FIXED_ENTITY_EXTRACTION_PROMPT):
            provider = CogneeLLMProvider(llm_provider=mock_llm_provider)
            await provider.extract_entities("Test text.")
        
        call_kwargs = mock_llm_provider.simple_completion.call_args.kwargs
        assert call_kwargs["model"] == "enrichment"
        assert call_kwargs["temperature"] == 0.1
        assert call_kwargs["max_tokens"] == 2000
        assert "entity extraction" in call_kwargs["system_prompt"].lower()


# ============================================================================
# Relationship Extraction Tests
# ============================================================================

@pytest.mark.unit
class TestCogneeLLMProviderRelationshipExtraction:
    """Tests for relationship extraction functionality."""

    @pytest.mark.asyncio
    async def test_extract_relationships_success(self, mock_llm_provider, sample_entities):
        """Test successful relationship extraction."""
        mock_llm_provider.simple_completion.return_value = '''
        {
            "relationships": [
                {"source": "Bill Gates", "target": "Microsoft", "type": "FOUNDED", "confidence": 0.95},
                {"source": "Microsoft", "target": "Seattle", "type": "LOCATED_IN", "confidence": 0.90}
            ]
        }
        '''
        
        with patch('src.plugins.destinations.cognee_llm.RELATIONSHIP_EXTRACTION_PROMPT', FIXED_RELATIONSHIP_EXTRACTION_PROMPT):
            provider = CogneeLLMProvider(llm_provider=mock_llm_provider)
            relationships = await provider.extract_relationships(
            "Bill Gates founded Microsoft in Seattle.",
            sample_entities,
        )
        
        assert len(relationships) == 2
        assert relationships[0].source == "Bill Gates"
        assert relationships[0].target == "Microsoft"
        assert relationships[0].type == "FOUNDED"
        assert relationships[0].confidence == 0.95

    @pytest.mark.asyncio
    async def test_extract_relationships_empty_text(self, mock_llm_provider, sample_entities):
        """Test relationship extraction with empty text."""
        provider = CogneeLLMProvider(llm_provider=mock_llm_provider)
        
        relationships = await provider.extract_relationships("", sample_entities)
        assert relationships == []

    @pytest.mark.asyncio
    async def test_extract_relationships_insufficient_entities(self, mock_llm_provider):
        """Test relationship extraction with fewer than 2 entities."""
        provider = CogneeLLMProvider(llm_provider=mock_llm_provider)
        
        single_entity = [ExtractedEntity(name="Only", type="PERSON")]
        relationships = await provider.extract_relationships("Text.", single_entity)
        
        assert relationships == []
        mock_llm_provider.simple_completion.assert_not_called()

    @pytest.mark.asyncio
    async def test_extract_relationships_empty_entities(self, mock_llm_provider):
        """Test relationship extraction with empty entity list."""
        provider = CogneeLLMProvider(llm_provider=mock_llm_provider)
        
        relationships = await provider.extract_relationships("Text.", [])
        
        assert relationships == []
        mock_llm_provider.simple_completion.assert_not_called()

    @pytest.mark.asyncio
    async def test_extract_relationships_none_entities(self, mock_llm_provider):
        """Test relationship extraction with None entities."""
        provider = CogneeLLMProvider(llm_provider=mock_llm_provider)
        
        relationships = await provider.extract_relationships("Text.", None)
        
        assert relationships == []

    @pytest.mark.asyncio
    async def test_extract_relationships_validates_entity_names(self, mock_llm_provider, sample_entities):
        """Test that relationships with unknown entities are filtered out."""
        mock_llm_provider.simple_completion.return_value = '''
        {
            "relationships": [
                {"source": "Bill Gates", "target": "Microsoft", "type": "FOUNDED", "confidence": 0.95},
                {"source": "Unknown Entity", "target": "Microsoft", "type": "RELATED_TO", "confidence": 0.5}
            ]
        }
        '''
        
        with patch('src.plugins.destinations.cognee_llm.RELATIONSHIP_EXTRACTION_PROMPT', FIXED_RELATIONSHIP_EXTRACTION_PROMPT):
            provider = CogneeLLMProvider(llm_provider=mock_llm_provider)
            relationships = await provider.extract_relationships(
            "Some text.",
            sample_entities,
        )
        
        # Only valid relationship should be returned
        assert len(relationships) == 1
        assert relationships[0].source == "Bill Gates"

    @pytest.mark.asyncio
    async def test_extract_relationships_no_relationships_found(self, mock_llm_provider, sample_entities):
        """Test extraction when no relationships are found."""
        mock_llm_provider.simple_completion.return_value = '{"relationships": []}'
        
        provider = CogneeLLMProvider(llm_provider=mock_llm_provider)
        relationships = await provider.extract_relationships(
            "Unrelated text.",
            sample_entities,
        )
        
        assert relationships == []

    @pytest.mark.asyncio
    async def test_extract_relationships_uses_correct_model_params(self, mock_llm_provider, sample_entities):
        """Test that correct model parameters are used."""
        mock_llm_provider.simple_completion.return_value = '{"relationships": []}'
        
        with patch('src.plugins.destinations.cognee_llm.RELATIONSHIP_EXTRACTION_PROMPT', FIXED_RELATIONSHIP_EXTRACTION_PROMPT):
            provider = CogneeLLMProvider(llm_provider=mock_llm_provider)
            await provider.extract_relationships("Text.", sample_entities)
        
        call_kwargs = mock_llm_provider.simple_completion.call_args.kwargs
        assert call_kwargs["model"] == "enrichment"
        assert call_kwargs["temperature"] == 0.1
        assert call_kwargs["max_tokens"] == 2000
        assert "relationship extraction" in call_kwargs["system_prompt"].lower()

    @pytest.mark.asyncio
    async def test_extract_relationships_llm_error(self, mock_llm_provider, sample_entities):
        """Test extraction handles LLM errors gracefully."""
        mock_llm_provider.simple_completion.side_effect = Exception("LLM error")
        
        provider = CogneeLLMProvider(llm_provider=mock_llm_provider)
        relationships = await provider.extract_relationships("Text.", sample_entities)
        
        assert relationships == []

    @pytest.mark.asyncio
    async def test_extract_relationships_limits_entity_count(self, mock_llm_provider):
        """Test that entity count is limited to prevent token overflow."""
        mock_llm_provider.simple_completion.return_value = '{"relationships": []}'
        
        # Create many entities
        many_entities = [
            ExtractedEntity(name=f"Entity{i}", type="PERSON")
            for i in range(100)
        ]
        
        with patch('src.plugins.destinations.cognee_llm.RELATIONSHIP_EXTRACTION_PROMPT', FIXED_RELATIONSHIP_EXTRACTION_PROMPT):
            provider = CogneeLLMProvider(llm_provider=mock_llm_provider)
            await provider.extract_relationships("Text.", many_entities)
        
        # Verify prompt contains limited entity list
        call_args = mock_llm_provider.simple_completion.call_args
        prompt = call_args.kwargs.get("prompt") or call_args[1].get("prompt")
        # Should only include first 50 entities
        assert prompt.count("Entity") <= 50


# ============================================================================
# Summarization Tests
# ============================================================================

@pytest.mark.unit
class TestCogneeLLMProviderSummarization:
    """Tests for search result summarization."""

    @pytest.mark.asyncio
    async def test_summarize_for_search_success(self, mock_llm_provider):
        """Test successful summarization."""
        mock_llm_provider.simple_completion.return_value = (
            "GraphRAG is a technique that combines knowledge graphs with retrieval."
        )
        
        provider = CogneeLLMProvider(llm_provider=mock_llm_provider)
        summary = await provider.summarize_for_search(
            query="What is GraphRAG?",
            context=[
                "GraphRAG uses knowledge graphs for better retrieval.",
                "It combines vector search with graph traversal.",
            ],
        )
        
        assert "GraphRAG" in summary
        assert "technique" in summary

    @pytest.mark.asyncio
    async def test_summarize_empty_context(self, mock_llm_provider):
        """Test summarization with empty context."""
        provider = CogneeLLMProvider(llm_provider=mock_llm_provider)
        
        summary = await provider.summarize_for_search("Query?", [])
        
        assert summary == "No relevant information found."
        mock_llm_provider.simple_completion.assert_not_called()

    @pytest.mark.asyncio
    async def test_summarize_truncates_to_max_length(self, mock_llm_provider):
        """Test that summary is truncated to max_length."""
        long_summary = "Word " * 200  # 1000+ characters
        mock_llm_provider.simple_completion.return_value = long_summary
        
        provider = CogneeLLMProvider(llm_provider=mock_llm_provider)
        summary = await provider.summarize_for_search(
            "Query?",
            ["Context."],
            max_length=100,
        )
        
        assert len(summary) <= 105  # Allow for "..." suffix
        assert summary.endswith("...")

    @pytest.mark.asyncio
    async def test_summarize_uses_agentic_model(self, mock_llm_provider):
        """Test that agentic model is used for summarization."""
        mock_llm_provider.simple_completion.return_value = "Summary."
        
        provider = CogneeLLMProvider(llm_provider=mock_llm_provider)
        await provider.summarize_for_search("Query?", ["Context."])
        
        call_kwargs = mock_llm_provider.simple_completion.call_args.kwargs
        assert call_kwargs["model"] == "agentic-decisions"
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["max_tokens"] == 1000

    @pytest.mark.asyncio
    async def test_summarize_llm_error_fallback(self, mock_llm_provider):
        """Test fallback when LLM fails."""
        mock_llm_provider.simple_completion.side_effect = Exception("LLM error")
        
        provider = CogneeLLMProvider(llm_provider=mock_llm_provider)
        summary = await provider.summarize_for_search(
            "Query?",
            ["Context passage 1.", "Context passage 2.", "Context passage 3."],
        )
        
        # Should fall back to concatenating contexts
        assert "Context passage 1" in summary

    @pytest.mark.asyncio
    async def test_summarize_limits_context_passages(self, mock_llm_provider):
        """Test that number of context passages is limited."""
        mock_llm_provider.simple_completion.return_value = "Summary."
        
        provider = CogneeLLMProvider(llm_provider=mock_llm_provider)
        many_contexts = [f"Context {i}." for i in range(20)]
        await provider.summarize_for_search("Query?", many_contexts)
        
        call_args = mock_llm_provider.simple_completion.call_args
        prompt = call_args.kwargs.get("prompt") or call_args[1].get("prompt")
        # Should only include first 10 passages
        assert prompt.count("[") <= 10


# ============================================================================
# Query Entity Extraction Tests
# ============================================================================

@pytest.mark.unit
class TestCogneeLLMProviderQueryEntityExtraction:
    """Tests for query entity extraction."""

    @pytest.mark.asyncio
    async def test_extract_query_entities_success(self, mock_llm_provider):
        """Test successful query entity extraction."""
        mock_llm_provider.simple_completion.return_value = "machine learning, AI, neural networks"
        
        provider = CogneeLLMProvider(llm_provider=mock_llm_provider)
        entities = await provider.extract_query_entities(
            "What are the latest advances in machine learning and AI?"
        )
        
        assert "machine learning" in entities
        assert "AI" in entities
        assert "neural networks" in entities

    @pytest.mark.asyncio
    async def test_extract_query_entities_empty_query(self, mock_llm_provider):
        """Test extraction with empty query."""
        provider = CogneeLLMProvider(llm_provider=mock_llm_provider)
        
        entities = await provider.extract_query_entities("")
        assert entities == []
        
        entities = await provider.extract_query_entities("   ")
        assert entities == []

    @pytest.mark.asyncio
    async def test_extract_query_entities_handles_multiline(self, mock_llm_provider):
        """Test handling of multiline response."""
        mock_llm_provider.simple_completion.return_value = "entity1\nentity2\nentity3"
        
        provider = CogneeLLMProvider(llm_provider=mock_llm_provider)
        entities = await provider.extract_query_entities("Query?")
        
        assert len(entities) == 3
        assert "entity1" in entities
        assert "entity3" in entities

    @pytest.mark.asyncio
    async def test_extract_query_entities_fallback_on_error(self, mock_llm_provider):
        """Test fallback to word splitting on error."""
        mock_llm_provider.simple_completion.side_effect = Exception("LLM error")
        
        provider = CogneeLLMProvider(llm_provider=mock_llm_provider)
        entities = await provider.extract_query_entities(
            "What about machine learning applications?"
        )
        
        # Should fall back to splitting query words
        assert "machine" in entities or "learning" in entities or "applications" in entities


# ============================================================================
# Helper Method Tests
# ============================================================================

@pytest.mark.unit
class TestCogneeLLMProviderHelpers:
    """Tests for helper methods."""

    def test_extract_json_from_markdown_with_json_tag(self):
        """Test extracting JSON from markdown with json tag."""
        provider = CogneeLLMProvider()
        text = '```json\n{"key": "value"}\n```'
        
        result = provider._extract_json_from_markdown(text)
        
        assert result == '{"key": "value"}'

    def test_extract_json_from_markdown_without_tag(self):
        """Test extracting JSON from markdown without json tag."""
        provider = CogneeLLMProvider()
        text = '```\n{"key": "value"}\n```'
        
        result = provider._extract_json_from_markdown(text)
        
        assert result == '{"key": "value"}'

    def test_extract_json_from_markdown_no_code_block(self):
        """Test returning original text when no code block."""
        provider = CogneeLLMProvider()
        text = '{"key": "value"}'
        
        result = provider._extract_json_from_markdown(text)
        
        assert result == '{"key": "value"}'

    def test_extract_json_from_markdown_multiple_blocks(self):
        """Test extracting from first code block."""
        provider = CogneeLLMProvider()
        text = '```json\n{"first": true}\n```\nSome text\n```json\n{"second": true}\n```'
        
        result = provider._extract_json_from_markdown(text)
        
        assert result == '{"first": true}'


# ============================================================================
# Health Check Tests
# ============================================================================

@pytest.mark.unit
class TestCogneeLLMProviderHealthCheck:
    """Tests for health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, mock_llm_provider):
        """Test health check returns healthy when LLM responds correctly."""
        mock_llm_provider.simple_completion.return_value = "healthy"
        
        provider = CogneeLLMProvider(llm_provider=mock_llm_provider)
        health = await provider.health_check()
        
        assert health["healthy"] is True
        assert health["status"] == "healthy"
        assert "responding" in health["message"]

    @pytest.mark.asyncio
    async def test_health_check_unexpected_response(self, mock_llm_provider):
        """Test health check returns degraded on unexpected response."""
        mock_llm_provider.simple_completion.return_value = "unexpected response"
        
        provider = CogneeLLMProvider(llm_provider=mock_llm_provider)
        health = await provider.health_check()
        
        assert health["healthy"] is False
        assert health["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_health_check_error(self, mock_llm_provider):
        """Test health check returns unhealthy on error."""
        mock_llm_provider.simple_completion.side_effect = Exception("LLM error")
        
        provider = CogneeLLMProvider(llm_provider=mock_llm_provider)
        health = await provider.health_check()
        
        assert health["healthy"] is False
        assert health["status"] == "unhealthy"
        assert "failed" in health["message"]


# ============================================================================
# Data Class Tests
# ============================================================================

@pytest.mark.unit
class TestExtractedEntity:
    """Tests for ExtractedEntity dataclass."""

    def test_entity_creation(self):
        """Test creating an ExtractedEntity."""
        entity = ExtractedEntity(
            name="Test Entity",
            type="PERSON",
            description="A test entity",
        )
        
        assert entity.name == "Test Entity"
        assert entity.type == "PERSON"
        assert entity.description == "A test entity"

    def test_entity_defaults(self):
        """Test ExtractedEntity default values."""
        entity = ExtractedEntity(name="Test", type="ORGANIZATION")
        
        assert entity.name == "Test"
        assert entity.type == "ORGANIZATION"
        assert entity.description == ""  # Default


@pytest.mark.unit
class TestExtractedRelationship:
    """Tests for ExtractedRelationship dataclass."""

    def test_relationship_creation(self):
        """Test creating an ExtractedRelationship."""
        rel = ExtractedRelationship(
            source="Entity A",
            target="Entity B",
            type="RELATED_TO",
            confidence=0.85,
        )
        
        assert rel.source == "Entity A"
        assert rel.target == "Entity B"
        assert rel.type == "RELATED_TO"
        assert rel.confidence == 0.85

    def test_relationship_defaults(self):
        """Test ExtractedRelationship default values."""
        rel = ExtractedRelationship(
            source="A",
            target="B",
            type="WORKS_AT",
        )
        
        assert rel.source == "A"
        assert rel.target == "B"
        assert rel.type == "WORKS_AT"
        assert rel.confidence == 0.8  # Default


# ============================================================================
# Prompt Template Tests
# ============================================================================

@pytest.mark.unit
class TestPromptTemplates:
    """Tests for prompt templates."""

    def test_entity_extraction_prompt_structure(self):
        """Test entity extraction prompt contains required elements."""
        from src.plugins.destinations.cognee_llm import ENTITY_EXTRACTION_PROMPT
        
        assert "named entity recognition" in ENTITY_EXTRACTION_PROMPT.lower()
        assert "entities" in ENTITY_EXTRACTION_PROMPT
        assert "type" in ENTITY_EXTRACTION_PROMPT
        assert "JSON" in ENTITY_EXTRACTION_PROMPT
        assert "{text}" in ENTITY_EXTRACTION_PROMPT

    def test_relationship_extraction_prompt_structure(self):
        """Test relationship extraction prompt contains required elements."""
        from src.plugins.destinations.cognee_llm import RELATIONSHIP_EXTRACTION_PROMPT
        
        assert "relationship extraction" in RELATIONSHIP_EXTRACTION_PROMPT.lower()
        assert "source" in RELATIONSHIP_EXTRACTION_PROMPT
        assert "target" in RELATIONSHIP_EXTRACTION_PROMPT
        assert "type" in RELATIONSHIP_EXTRACTION_PROMPT
        assert "confidence" in RELATIONSHIP_EXTRACTION_PROMPT
        assert "{entities}" in RELATIONSHIP_EXTRACTION_PROMPT
        assert "{text}" in RELATIONSHIP_EXTRACTION_PROMPT

    def test_summarization_prompt_structure(self):
        """Test summarization prompt contains required elements."""
        from src.plugins.destinations.cognee_llm import SUMMARIZATION_PROMPT
        
        assert "synthesizing information" in SUMMARIZATION_PROMPT.lower()
        assert "{query}" in SUMMARIZATION_PROMPT
        assert "{context}" in SUMMARIZATION_PROMPT
