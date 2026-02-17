"""Advanced enrichment features for document processing.

This module provides advanced enrichment capabilities including:
- Document summarization
- Sentiment analysis
- Topic classification
- Keyword extraction
"""

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.llm.provider import LLMProvider

logger = logging.getLogger(__name__)


class SentimentLabel(str, Enum):
    """Sentiment classification labels."""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


@dataclass
class SummarizationResult:
    """Result of document summarization.
    
    Attributes:
        summary: Generated summary text
        method: Summarization method used
        original_length: Original document length
        summary_length: Summary length
        compression_ratio: Compression ratio
        processing_time_ms: Processing time
    """
    summary: str
    method: str
    original_length: int
    summary_length: int
    compression_ratio: float = 0.0
    processing_time_ms: int = 0
    key_points: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.original_length > 0:
            self.compression_ratio = self.summary_length / self.original_length


@dataclass
class SentimentResult:
    """Result of sentiment analysis.
    
    Attributes:
        label: Sentiment label
        score: Sentiment score (-1.0 to 1.0)
        confidence: Confidence score (0.0 to 1.0)
        aspects: Aspect-based sentiment
        emotions: Detected emotions
    """
    label: SentimentLabel
    score: float
    confidence: float
    aspects: dict[str, dict[str, Any]] = field(default_factory=dict)
    emotions: dict[str, float] = field(default_factory=dict)
    processing_time_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label.value,
            "score": self.score,
            "confidence": self.confidence,
            "aspects": self.aspects,
            "emotions": self.emotions,
        }


@dataclass
class TopicClassificationResult:
    """Result of topic classification.
    
    Attributes:
        primary_topic: Primary topic label
        confidence: Primary topic confidence
        all_topics: All detected topics with scores
        categories: Broader categories
    """
    primary_topic: str
    confidence: float
    all_topics: list[dict[str, Any]] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    processing_time_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_topic": self.primary_topic,
            "confidence": self.confidence,
            "all_topics": self.all_topics,
            "categories": self.categories,
        }


@dataclass
class EnrichmentResult:
    """Complete enrichment result.
    
    Attributes:
        summarization: Summarization result
        sentiment: Sentiment analysis result
        topics: Topic classification result
        keywords: Extracted keywords
        language: Detected language
        reading_level: Estimated reading level
        processing_time_ms: Total processing time
    """
    summarization: SummarizationResult | None = None
    sentiment: SentimentResult | None = None
    topics: TopicClassificationResult | None = None
    keywords: list[str] = field(default_factory=list)
    language: str | None = None
    reading_level: str | None = None
    processing_time_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "summarization": self.summarization.__dict__ if self.summarization else None,
            "sentiment": self.sentiment.to_dict() if self.sentiment else None,
            "topics": self.topics.to_dict() if self.topics else None,
            "keywords": self.keywords,
            "language": self.language,
            "reading_level": self.reading_level,
        }


class AdvancedEnricher:
    """Advanced document enrichment using LLM.
    
    This class provides advanced enrichment capabilities for documents:
    - Summarization (extractive and abstractive)
    - Sentiment analysis
    - Topic classification
    - Keyword extraction
    - Language detection
    - Reading level estimation
    
    Example:
        >>> enricher = AdvancedEnricher(llm_provider)
        >>> result = await enricher.enrich(text, include_summary=True)
        >>> print(result.summarization.summary)
    """

    def __init__(self, llm_provider: LLMProvider | None = None):
        """Initialize the advanced enricher.
        
        Args:
            llm_provider: LLM provider for enrichment tasks
        """
        self._llm = llm_provider
        self._topic_categories = {
            "technology": ["software", "hardware", "AI", "cloud", "data"],
            "business": ["finance", "marketing", "sales", "strategy", "operations"],
            "science": ["research", "biology", "physics", "chemistry", "medicine"],
            "health": ["healthcare", "wellness", "fitness", "nutrition", "mental health"],
            "politics": ["government", "policy", "elections", "law", "diplomacy"],
            "entertainment": ["movies", "music", "games", "sports", "arts"],
            "education": ["learning", "teaching", "research", "academia", "training"],
        }

    async def enrich(
        self,
        text: str,
        include_summary: bool = True,
        include_sentiment: bool = True,
        include_topics: bool = True,
        include_keywords: bool = True,
        summary_length: str = "medium",  # short, medium, long
    ) -> EnrichmentResult:
        """Perform comprehensive document enrichment.
        
        Args:
            text: Document text to enrich
            include_summary: Include summarization
            include_sentiment: Include sentiment analysis
            include_topics: Include topic classification
            include_keywords: Include keyword extraction
            summary_length: Desired summary length
            
        Returns:
            Complete enrichment result
        """
        import time

        start_time = time.time()

        result = EnrichmentResult()

        # Summarization
        if include_summary and self._llm:
            result.summarization = await self.summarize(text, summary_length)

        # Sentiment analysis
        if include_sentiment:
            result.sentiment = await self.analyze_sentiment(text)

        # Topic classification
        if include_topics:
            result.topics = await self.classify_topics(text)

        # Keyword extraction
        if include_keywords:
            result.keywords = await self.extract_keywords(text)

        # Language detection (simple heuristic)
        result.language = self._detect_language(text)

        # Reading level estimation
        result.reading_level = self._estimate_reading_level(text)

        result.processing_time_ms = int((time.time() - start_time) * 1000)

        return result

    async def summarize(
        self,
        text: str,
        length: str = "medium",
        method: str = "abstractive",
    ) -> SummarizationResult:
        """Generate document summary.
        
        Args:
            text: Document text
            length: Summary length (short, medium, long)
            method: Summarization method (extractive, abstractive)
            
        Returns:
            Summarization result
        """
        import time

        start_time = time.time()

        length_map = {
            "short": (50, 100),
            "medium": (100, 250),
            "long": (250, 500),
        }
        min_len, max_len = length_map.get(length, (100, 250))

        original_length = len(text.split())

        if method == "extractive":
            summary, key_points = await self._extractive_summarize(text, max_len)
        else:
            summary, key_points = await self._abstractive_summarize(text, min_len, max_len)

        summary_length = len(summary.split())
        processing_time = int((time.time() - start_time) * 1000)

        return SummarizationResult(
            summary=summary,
            method=method,
            original_length=original_length,
            summary_length=summary_length,
            key_points=key_points,
            processing_time_ms=processing_time,
        )

    async def _extractive_summarize(
        self,
        text: str,
        max_words: int,
    ) -> tuple[str, list[str]]:
        """Generate extractive summary using sentence scoring."""
        sentences = self._split_sentences(text)

        if len(sentences) <= 3:
            return text, sentences[:3]

        # Score sentences based on word frequency
        word_freq = {}
        words = re.findall(r"\b\w+\b", text.lower())

        for word in words:
            if len(word) > 3:  # Skip short words
                word_freq[word] = word_freq.get(word, 0) + 1

        # Score each sentence
        sentence_scores = []
        for sentence in sentences:
            score = 0
            sentence_words = re.findall(r"\b\w+\b", sentence.lower())
            for word in sentence_words:
                score += word_freq.get(word, 0)
            sentence_scores.append((sentence, score / max(len(sentence_words), 1)))

        # Sort by score and select top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)

        # Select sentences while respecting word limit
        selected = []
        word_count = 0

        for sentence, _ in sentence_scores[:5]:  # Top 5 sentences
            sent_words = len(sentence.split())
            if word_count + sent_words <= max_words:
                selected.append(sentence)
                word_count += sent_words

        # Sort selected sentences by original order
        selected.sort(key=lambda s: text.find(s))

        summary = " ".join(selected)
        key_points = selected[:3]

        return summary, key_points

    async def _abstractive_summarize(
        self,
        text: str,
        min_words: int,
        max_words: int,
    ) -> tuple[str, list[str]]:
        """Generate abstractive summary using LLM."""
        if not self._llm:
            return await self._extractive_summarize(text, max_words)

        # Truncate text if too long
        truncated = text[:4000] if len(text) > 4000 else text

        prompt = f"""Summarize the following text in {min_words}-{max_words} words.

Text:
{truncated}

Provide:
1. A concise summary
2. 3-5 key points as bullet points

Format your response as:
Summary: <summary text>

Key Points:
- <point 1>
- <point 2>
- <point 3>"""

        try:
            response = await self._llm.chat_completion(
                messages=[
                    {"role": "system", "content": "You are a summarization assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )

            content = response.choices[0].message.content

            # Parse summary and key points
            summary = content
            key_points = []

            if "Key Points:" in content:
                parts = content.split("Key Points:", 1)
                summary = parts[0].replace("Summary:", "").strip()
                points_text = parts[1]
                key_points = [
                    line.strip("- ").strip()
                    for line in points_text.split("\n")
                    if line.strip().startswith("-")
                ]

            return summary, key_points[:5]

        except Exception as e:
            logger.warning(f"Abstractive summarization failed: {e}")
            return await self._extractive_summarize(text, max_words)

    async def analyze_sentiment(self, text: str) -> SentimentResult:
        """Analyze sentiment of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment analysis result
        """
        import time

        start_time = time.time()

        # Use LLM for sentiment if available
        if self._llm:
            return await self._llm_sentiment(text, start_time)

        # Fallback to lexicon-based approach
        return self._lexicon_sentiment(text, start_time)

    async def _llm_sentiment(self, text: str, start_time: float) -> SentimentResult:
        """Analyze sentiment using LLM."""
        truncated = text[:2000] if len(text) > 2000 else text

        prompt = f"""Analyze the sentiment of the following text.

Text:
{truncated}

Provide:
1. Overall sentiment: very_positive, positive, neutral, negative, or very_negative
2. Sentiment score: -1.0 (very negative) to 1.0 (very positive)
3. Confidence: 0.0 to 1.0
4. Detected emotions and their intensities (0.0 to 1.0)

Respond in JSON format:
{{
    "sentiment": "positive",
    "score": 0.7,
    "confidence": 0.85,
    "emotions": {{
        "joy": 0.6,
        "anger": 0.1,
        "sadness": 0.1
    }}
}}"""

        try:
            response = await self._llm.chat_completion(
                messages=[
                    {"role": "system", "content": "You are a sentiment analysis assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=300,
            )

            content = response.choices[0].message.content

            # Parse JSON response
            import json
            try:
                data = json.loads(content)
                return SentimentResult(
                    label=SentimentLabel(data.get("sentiment", "neutral")),
                    score=float(data.get("score", 0)),
                    confidence=float(data.get("confidence", 0.5)),
                    emotions=data.get("emotions", {}),
                    processing_time_ms=int((time.time() - start_time) * 1000),
                )
            except json.JSONDecodeError:
                pass

        except Exception as e:
            logger.warning(f"LLM sentiment analysis failed: {e}")

        return self._lexicon_sentiment(text, start_time)

    def _lexicon_sentiment(self, text: str, start_time: float) -> SentimentResult:
        """Analyze sentiment using lexicon approach."""
        # Simple lexicon-based sentiment
        positive_words = {
            "good", "great", "excellent", "amazing", "wonderful", "fantastic",
            "love", "happy", "best", "beautiful", "perfect", "awesome",
            "positive", "success", "win", "winning", "joy", "pleased",
            "satisfied", "recommend", "impressive", "outstanding"
        }

        negative_words = {
            "bad", "terrible", "awful", "horrible", "hate", "worst",
            "poor", "disappointing", "failed", "failure", "problem",
            "issue", "error", "wrong", "negative", "angry", "sad",
            "disappointed", "frustrated", "unhappy", "regret"
        }

        words = re.findall(r"\b\w+\b", text.lower())

        pos_count = sum(1 for w in words if w in positive_words)
        neg_count = sum(1 for w in words if w in negative_words)
        total = pos_count + neg_count

        if total == 0:
            score = 0.0
            label = SentimentLabel.NEUTRAL
            confidence = 0.5
        else:
            score = (pos_count - neg_count) / max(total, 10)
            score = max(-1, min(1, score))

            if score > 0.6:
                label = SentimentLabel.VERY_POSITIVE
            elif score > 0.2:
                label = SentimentLabel.POSITIVE
            elif score < -0.6:
                label = SentimentLabel.VERY_NEGATIVE
            elif score < -0.2:
                label = SentimentLabel.NEGATIVE
            else:
                label = SentimentLabel.NEUTRAL

            confidence = min(total / 20, 0.9)

        emotions = {}
        if pos_count > neg_count:
            emotions["joy"] = score
        elif neg_count > pos_count:
            emotions["sadness"] = -score

        return SentimentResult(
            label=label,
            score=score,
            confidence=confidence,
            emotions=emotions,
            processing_time_ms=int((time.time() - start_time) * 1000),
        )

    async def classify_topics(self, text: str) -> TopicClassificationResult:
        """Classify topics of text.
        
        Args:
            text: Text to classify
            
        Returns:
            Topic classification result
        """
        import time

        start_time = time.time()

        # Score each category based on keyword matches
        text_lower = text.lower()
        category_scores = {}

        for category, keywords in self._topic_categories.items():
            score = sum(1 for keyword in keywords if keyword.lower() in text_lower)
            if score > 0:
                category_scores[category] = score

        # Sort by score
        sorted_categories = sorted(
            category_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        if sorted_categories:
            primary_topic = sorted_categories[0][0]
            confidence = min(sorted_categories[0][1] / 3, 0.95)
            all_topics = [
                {"topic": cat, "score": score / max(category_scores.values())}
                for cat, score in sorted_categories
            ]
            categories = [cat for cat, _ in sorted_categories[:3]]
        else:
            primary_topic = "general"
            confidence = 0.3
            all_topics = [{"topic": "general", "score": 0.3}]
            categories = ["general"]

        # Use LLM to refine if available
        if self._llm and confidence < 0.7:
            llm_topics = await self._llm_topic_classification(text)
            if llm_topics:
                primary_topic = llm_topics.get("topic", primary_topic)
                confidence = llm_topics.get("confidence", confidence)

        processing_time = int((time.time() - start_time) * 1000)

        return TopicClassificationResult(
            primary_topic=primary_topic,
            confidence=confidence,
            all_topics=all_topics,
            categories=categories,
            processing_time_ms=processing_time,
        )

    async def _llm_topic_classification(self, text: str) -> dict[str, Any] | None:
        """Refine topic classification using LLM."""
        truncated = text[:2000] if len(text) > 2000 else text

        prompt = f"""Classify the main topic of this text from these categories:
technology, business, science, health, politics, entertainment, education, general

Text:
{truncated}

Respond with just the topic name and confidence (0-1)."""

        try:
            response = await self._llm.chat_completion(
                messages=[
                    {"role": "system", "content": "You are a topic classification assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=50,
            )

            content = response.choices[0].message.content.lower().strip()

            # Parse response
            for category in self._topic_categories.keys():
                if category in content:
                    # Try to extract confidence
                    import re
                    conf_match = re.search(r"(\d+\.?\d*)", content)
                    confidence = float(conf_match.group(1)) if conf_match else 0.7
                    if confidence > 1:
                        confidence = confidence / 100

                    return {"topic": category, "confidence": confidence}

        except Exception as e:
            logger.warning(f"LLM topic classification failed: {e}")

        return None

    async def extract_keywords(self, text: str, max_keywords: int = 10) -> list[str]:
        """Extract keywords from text.
        
        Args:
            text: Text to analyze
            max_keywords: Maximum number of keywords
            
        Returns:
            List of keywords
        """
        # Use TF-IDF-like approach
        words = re.findall(r"\b[A-Za-z][A-Za-z]+\b", text.lower())

        # Filter out common stop words
        stop_words = {
            "the", "and", "for", "are", "but", "not", "you", "all",
            "can", "had", "her", "was", "one", "our", "out", "day",
            "get", "has", "him", "his", "how", "its", "may", "new",
            "now", "old", "see", "two", "who", "boy", "did", "she",
            "use", "way", "many", "oil", "sit", "set", "run",
            "eat", "far", "sea", "eye", "ask", "own", "say", "too",
            "any", "try", "three", "also", "after", "back", "other",
            "than", "them", "these", "would", "there", "their",
            "what", "said", "each", "which", "will", "about", "could",
            "this", "with", "from", "they", "been", "have", "were",
            "more", "very", "when", "come", "made", "find", "part", "over", "such", "take", "only", "think", "know",
            "just", "first"
        }

        # Count word frequencies
        word_freq = {}
        for word in words:
            if word not in stop_words and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Score by frequency and position (words in first 20% get boost)
        word_scores = {}
        early_text_end = len(text) // 5

        for word, freq in word_freq.items():
            score = freq
            if text[:early_text_end].lower().find(word) != -1:
                score *= 1.5
            word_scores[word] = score

        # Return top keywords
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:max_keywords]]

    def _detect_language(self, text: str) -> str:
        """Detect language of text (simple heuristic)."""
        # Simple language detection based on common words
        english_words = {"the", "and", "for", "are", "but", "not", "with", "have", "this"}
        spanish_words = {"el", "la", "de", "que", "en", "un", "ser", "se", "no"}
        french_words = {"le", "de", "et", "à", "un", "il", "être", "par", "pour"}
        german_words = {"der", "die", "und", "in", "den", "von", "zu", "das", "mit"}

        words = set(re.findall(r"\b\w+\b", text.lower()))

        scores = {
            "en": len(words & english_words),
            "es": len(words & spanish_words),
            "fr": len(words & french_words),
            "de": len(words & german_words),
        }

        best_lang = max(scores, key=scores.get)
        return best_lang if scores[best_lang] > 0 else "unknown"

    def _estimate_reading_level(self, text: str) -> str:
        """Estimate reading level using Flesch-Kincaid-like metric."""
        sentences = self._split_sentences(text)
        words = re.findall(r"\b\w+\b", text)

        if not sentences or not words:
            return "unknown"

        avg_words_per_sentence = len(words) / len(sentences)
        avg_syllables_per_word = sum(self._count_syllables(w) for w in words) / len(words)

        # Simplified reading ease score
        score = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)

        if score >= 90:
            return "elementary"
        elif score >= 70:
            return "middle_school"
        elif score >= 50:
            return "high_school"
        elif score >= 30:
            return "college"
        else:
            return "graduate"

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simple heuristic)."""
        word = word.lower()
        vowels = "aeiouy"
        count = 0
        prev_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel

        # Adjust for silent e
        if word.endswith("e") and count > 1:
            count -= 1

        return max(1, count)
