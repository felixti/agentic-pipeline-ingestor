"""Enrichment module for document processing.

This module provides various enrichment capabilities for documents,
including entity extraction, summarization, sentiment analysis, and more.
"""

from src.core.enrichment.advanced import (
    AdvancedEnricher,
    EnrichmentResult,
    SentimentResult,
    SummarizationResult,
    TopicClassificationResult,
)

__all__ = [
    "AdvancedEnricher",
    "EnrichmentResult",
    "SummarizationResult",
    "SentimentResult",
    "TopicClassificationResult",
]
