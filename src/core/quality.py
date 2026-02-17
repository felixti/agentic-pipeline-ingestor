"""Quality assessment module for evaluating document parsing results.

This module provides quality scoring, threshold checking, and retry
recommendations for parsed document content.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any

from src.api.models import QualityConfig
from src.plugins.base import ParsingResult

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Quality score for parsed content.
    
    Attributes:
        overall_score: Overall quality score (0.0 - 1.0)
        text_quality: Text extraction quality score
        structure_quality: Document structure quality score
        ocr_confidence: OCR confidence if applicable
        completeness: Content completeness score
        issues: List of quality issues found
        recommendations: Recommendations for improvement
    """
    overall_score: float
    text_quality: float
    structure_quality: float
    ocr_confidence: float
    completeness: float
    issues: list[str]
    recommendations: list[str]

    @property
    def passed(self) -> bool:
        """Check if quality passes minimum threshold."""
        return self.overall_score >= 0.7

    @property
    def needs_retry(self) -> bool:
        """Check if retry is recommended."""
        return self.overall_score < 0.5


class TextQualityAnalyzer:
    """Analyzer for text extraction quality."""

    # Common OCR error patterns
    OCR_ERROR_PATTERNS = [
        r"[\x00-\x08\x0b-\x0c\x0e-\x1f]",  # Control characters
        r"[^\x00-\x7F]",  # Non-ASCII characters (may indicate encoding issues)
    ]

    # Readable word patterns
    WORD_PATTERN = re.compile(r"\b[a-zA-Z]{2,}\b")

    def analyze(self, text: str) -> dict[str, Any]:
        """Analyze text quality.
        
        Args:
            text: Extracted text to analyze
            
        Returns:
            Dictionary with quality metrics
        """
        if not text:
            return {
                "score": 0.0,
                "issues": ["No text extracted"],
                "char_count": 0,
                "word_count": 0,
                "avg_word_length": 0.0,
                "garbage_ratio": 1.0,
            }

        issues: list[str] = []
        char_count = len(text)
        word_count = len(text.split())

        # Count garbage characters
        garbage_count = 0
        for pattern in self.OCR_ERROR_PATTERNS:
            garbage_count += len(re.findall(pattern, text))

        garbage_ratio = garbage_count / char_count if char_count > 0 else 0

        # Average word length
        words = self.WORD_PATTERN.findall(text)
        avg_word_length = sum(len(w) for w in words) / len(words) if words else 0

        # Calculate score
        score = 1.0

        # Penalize for garbage characters
        score -= garbage_ratio * 2  # Heavy penalty for garbage

        # Penalize very short content
        if char_count < 100:
            score -= 0.3
            issues.append("Very short content")

        # Penalize very long words (possible OCR errors)
        if avg_word_length > 15:
            score -= 0.2
            issues.append("Unusually long words detected")

        # Check for repeated characters (common OCR error)
        repeated_pattern = re.search(r"(.)\1{10,}", text)
        if repeated_pattern:
            score -= 0.3
            issues.append("Repeated characters detected")

        # Check for reasonable word distribution
        if word_count > 0:
            lines = text.split("\n")
            avg_words_per_line = word_count / len(lines) if lines else 0
            if avg_words_per_line > 50:
                score -= 0.1
                issues.append("Unusually dense text")

        return {
            "score": max(0.0, min(1.0, score)),
            "issues": issues,
            "char_count": char_count,
            "word_count": word_count,
            "avg_word_length": avg_word_length,
            "garbage_ratio": garbage_ratio,
        }


class StructureQualityAnalyzer:
    """Analyzer for document structure quality."""

    def analyze(self, result: ParsingResult) -> dict[str, Any]:
        """Analyze document structure quality.
        
        Args:
            result: Parsing result to analyze
            
        Returns:
            Dictionary with quality metrics
        """
        issues: list[str] = []

        # Check for page consistency
        page_count = len(result.pages) if result.pages else 0

        # Check metadata presence
        has_metadata = bool(result.metadata)

        # Analyze page distribution
        if page_count > 0:
            page_lengths = [len(p) for p in result.pages]
            avg_length = sum(page_lengths) / len(page_lengths)

            # Check for empty pages
            empty_pages = sum(1 for l in page_lengths if l < 10)

            # Check for extreme length variance
            if page_lengths:
                max_length = max(page_lengths)
                min_length = min(page_lengths)
                if max_length > 0 and min_length / max_length < 0.1:
                    issues.append("Large variance in page content lengths")
        else:
            avg_length = 0
            empty_pages = 0

        # Calculate score
        score = 1.0

        if empty_pages > 0:
            score -= (empty_pages / page_count) * 0.3 if page_count > 0 else 0.3
            issues.append(f"{empty_pages} near-empty pages detected")

        if not has_metadata:
            score -= 0.1

        # Check tables and images extraction
        if result.tables:
            logger.debug(f"Extracted {len(result.tables)} tables")

        if result.images:
            logger.debug(f"Extracted {len(result.images)} images")

        return {
            "score": max(0.0, min(1.0, score)),
            "issues": issues,
            "page_count": page_count,
            "has_metadata": has_metadata,
            "avg_page_length": avg_length,
            "empty_pages": empty_pages,
            "tables_extracted": len(result.tables) if result.tables else 0,
            "images_extracted": len(result.images) if result.images else 0,
        }


class QualityAssessor:
    """Main quality assessment class.
    
    Evaluates parsing results and provides quality scores,
    recommendations, and retry decisions.
    
    Example:
        >>> assessor = QualityAssesser()
        >>> score = await assessor.assess(parsed_result, quality_config)
        >>> if not score.passed:
        ...     print("Quality check failed, retry needed")
    """

    def __init__(self) -> None:
        """Initialize the quality assessor."""
        self.text_analyzer = TextQualityAnalyzer()
        self.structure_analyzer = StructureQualityAnalyzer()

    async def assess(
        self,
        result: ParsingResult,
        config: QualityConfig | None = None,
    ) -> QualityScore:
        """Assess the quality of a parsing result.
        
        Args:
            result: Parsing result to assess
            config: Quality configuration
            
        Returns:
            QualityScore with detailed assessment
        """
        config = config or QualityConfig()

        if not result.success:
            return QualityScore(
                overall_score=0.0,
                text_quality=0.0,
                structure_quality=0.0,
                ocr_confidence=0.0,
                completeness=0.0,
                issues=["Parsing failed"],
                recommendations=["Retry with fallback parser"],
            )

        # Analyze text quality
        text_analysis = self.text_analyzer.analyze(result.text)

        # Analyze structure quality
        structure_analysis = self.structure_analyzer.analyze(result)

        # Calculate completeness
        completeness = self._calculate_completeness(result, text_analysis)

        # Calculate overall score with weights
        weights = {
            "text": 0.4,
            "structure": 0.3,
            "ocr": 0.2,
            "completeness": 0.1,
        }

        overall_score = (
            text_analysis["score"] * weights["text"] +
            structure_analysis["score"] * weights["structure"] +
            result.confidence * weights["ocr"] +
            completeness * weights["completeness"]
        )

        # Collect all issues
        all_issues = text_analysis["issues"] + structure_analysis["issues"]

        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_score,
            text_analysis,
            structure_analysis,
            result,
        )

        return QualityScore(
            overall_score=round(overall_score, 3),
            text_quality=round(text_analysis["score"], 3),
            structure_quality=round(structure_analysis["score"], 3),
            ocr_confidence=round(result.confidence, 3),
            completeness=round(completeness, 3),
            issues=all_issues,
            recommendations=recommendations,
        )

    def _calculate_completeness(
        self,
        result: ParsingResult,
        text_analysis: dict[str, Any],
    ) -> float:
        """Calculate content completeness score.
        
        Args:
            result: Parsing result
            text_analysis: Text analysis results
            
        Returns:
            Completeness score (0.0 - 1.0)
        """
        score = 1.0

        # Check if text was extracted
        if text_analysis["char_count"] == 0:
            return 0.0

        # Check for reasonable word count
        if text_analysis["word_count"] < 10:
            score -= 0.3

        # Check if pages were extracted
        if result.pages:
            expected_pages = len(result.pages)
            non_empty_pages = sum(1 for p in result.pages if len(p.strip()) > 10)
            if expected_pages > 0:
                page_completeness = non_empty_pages / expected_pages
                score = score * 0.5 + page_completeness * 0.5

        return max(0.0, min(1.0, score))

    def _generate_recommendations(
        self,
        overall_score: float,
        text_analysis: dict[str, Any],
        structure_analysis: dict[str, Any],
        result: ParsingResult,
    ) -> list[str]:
        """Generate recommendations based on analysis.
        
        Args:
            overall_score: Overall quality score
            text_analysis: Text analysis results
            structure_analysis: Structure analysis results
            result: Original parsing result
            
        Returns:
            List of recommendations
        """
        recommendations: list[str] = []

        if overall_score < 0.5:
            recommendations.append("Consider retry with fallback parser")

        if text_analysis["garbage_ratio"] > 0.1:
            recommendations.append("High garbage character ratio - OCR may have failed")

        if text_analysis["avg_word_length"] > 15:
            recommendations.append("Check for concatenated words or OCR errors")

        if structure_analysis["empty_pages"] > 0:
            recommendations.append("Some pages appear empty - verify extraction")

        if result.confidence < 0.7:
            recommendations.append("Low OCR confidence - consider preprocessing")

        if not recommendations and overall_score < 0.9:
            recommendations.append("Minor quality issues detected but acceptable")

        return recommendations

    def should_retry(
        self,
        score: QualityScore,
        config: QualityConfig,
        current_attempt: int = 1,
    ) -> bool:
        """Determine if retry should be attempted.
        
        Args:
            score: Quality score
            config: Quality configuration
            current_attempt: Current retry attempt number
            
        Returns:
            True if retry should be attempted
        """
        if not config.enabled:
            return False

        if current_attempt >= config.max_retries:
            return False

        if score.overall_score >= config.min_quality_score:
            return False

        if not config.auto_retry:
            return False

        return True

    def get_retry_strategy(
        self,
        score: QualityScore,
        current_parser: str,
    ) -> str:
        """Determine the best retry strategy.
        
        Args:
            score: Quality score
            current_parser: Parser that was used
            
        Returns:
            Retry strategy name
        """
        # If text quality is very poor, try OCR
        if score.text_quality < 0.3 and current_parser != "azure_ocr":
            return "fallback_parser"

        # If OCR confidence is low, try alternative parser
        if score.ocr_confidence < 0.5:
            return "fallback_parser"

        # Default: retry with same parser (transient failure)
        return "same_parser"


# Convenience function
async def assess_quality(
    result: ParsingResult,
    config: QualityConfig | None = None,
) -> QualityScore:
    """Assess quality of parsing result.
    
    Args:
        result: Parsing result to assess
        config: Quality configuration
        
    Returns:
        QualityScore
    """
    assessor = QualityAssessor()
    return await assessor.assess(result, config)
