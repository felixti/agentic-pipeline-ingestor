"""Unit tests for quality assessment module."""

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from src.api.models import QualityConfig
from src.core.quality import (
    QualityAssessor,
    QualityScore,
    StructureQualityAnalyzer,
    TextQualityAnalyzer,
    assess_quality,
)
from src.plugins.base import ParsingResult


@pytest.mark.unit
class TestQualityScore:
    """Tests for QualityScore dataclass."""

    def test_quality_score_creation(self):
        """Test creating a QualityScore instance."""
        score = QualityScore(
            overall_score=0.85,
            text_quality=0.9,
            structure_quality=0.8,
            ocr_confidence=0.75,
            completeness=0.95,
            issues=["Minor formatting issue"],
            recommendations=["Consider preprocessing"],
        )

        assert score.overall_score == 0.85
        assert score.text_quality == 0.9
        assert score.structure_quality == 0.8
        assert score.ocr_confidence == 0.75
        assert score.completeness == 0.95
        assert score.issues == ["Minor formatting issue"]
        assert score.recommendations == ["Consider preprocessing"]

    def test_quality_score_passed_true(self):
        """Test that passed property returns True when score >= 0.7."""
        score = QualityScore(
            overall_score=0.75,
            text_quality=0.8,
            structure_quality=0.7,
            ocr_confidence=0.9,
            completeness=0.85,
            issues=[],
            recommendations=[],
        )

        assert score.passed is True

    def test_quality_score_passed_false(self):
        """Test that passed property returns False when score < 0.7."""
        score = QualityScore(
            overall_score=0.65,
            text_quality=0.6,
            structure_quality=0.7,
            ocr_confidence=0.5,
            completeness=0.8,
            issues=["Low quality"],
            recommendations=["Retry"],
        )

        assert score.passed is False

    def test_quality_score_passed_at_boundary(self):
        """Test that passed property works at exact boundary (0.7)."""
        score = QualityScore(
            overall_score=0.7,
            text_quality=0.7,
            structure_quality=0.7,
            ocr_confidence=0.7,
            completeness=0.7,
            issues=[],
            recommendations=[],
        )

        assert score.passed is True

    def test_quality_score_needs_retry_true(self):
        """Test that needs_retry returns True when score < 0.5."""
        score = QualityScore(
            overall_score=0.45,
            text_quality=0.3,
            structure_quality=0.4,
            ocr_confidence=0.5,
            completeness=0.6,
            issues=["Very low quality"],
            recommendations=["Retry with different parser"],
        )

        assert score.needs_retry is True

    def test_quality_score_needs_retry_false(self):
        """Test that needs_retry returns False when score >= 0.5."""
        score = QualityScore(
            overall_score=0.5,
            text_quality=0.5,
            structure_quality=0.5,
            ocr_confidence=0.5,
            completeness=0.5,
            issues=[],
            recommendations=[],
        )

        assert score.needs_retry is False

    def test_quality_score_needs_retry_at_boundary(self):
        """Test that needs_retry returns True at exact boundary (0.49...)."""
        score = QualityScore(
            overall_score=0.49,
            text_quality=0.49,
            structure_quality=0.49,
            ocr_confidence=0.49,
            completeness=0.49,
            issues=[],
            recommendations=[],
        )

        assert score.needs_retry is True


@pytest.mark.unit
class TestTextQualityAnalyzer:
    """Tests for TextQualityAnalyzer class."""

    def test_analyze_empty_text(self):
        """Test analyzing empty text returns zero scores."""
        analyzer = TextQualityAnalyzer()
        result = analyzer.analyze("")

        assert result["score"] == 0.0
        assert result["issues"] == ["No text extracted"]
        assert result["char_count"] == 0
        assert result["word_count"] == 0
        assert result["avg_word_length"] == 0.0
        assert result["garbage_ratio"] == 1.0

    def test_analyze_high_quality_text(self):
        """Test analyzing high quality text returns high score."""
        analyzer = TextQualityAnalyzer()
        text = "This is a well formatted document with proper content and structure. It has sufficient length to avoid the very short content penalty that applies to text under 100 characters."
        result = analyzer.analyze(text)

        assert result["score"] > 0.8
        assert result["char_count"] == len(text)
        assert result["word_count"] >= 20
        assert result["garbage_ratio"] == 0.0
        assert len(result["issues"]) == 0

    def test_analyze_text_with_garbage_characters(self):
        """Test that garbage characters reduce the score."""
        analyzer = TextQualityAnalyzer()
        # Text with control characters
        text = "Hello\x00World\x01\x02Test"
        result = analyzer.analyze(text)

        assert result["garbage_ratio"] > 0
        assert result["score"] < 1.0

    def test_analyze_very_short_content(self):
        """Test that very short content triggers penalty."""
        analyzer = TextQualityAnalyzer()
        text = "Hi"
        result = analyzer.analyze(text)

        assert "Very short content" in result["issues"]
        assert result["score"] <= 0.7  # Score may equal boundary

    def test_analyze_long_words(self):
        """Test that unusually long words are flagged."""
        analyzer = TextQualityAnalyzer()
        # Word longer than 15 characters
        text = "Thisisaverylongwordthatexceedsfifteenchars"
        result = analyzer.analyze(text)

        assert "Unusually long words detected" in result["issues"]

    def test_analyze_repeated_characters(self):
        """Test that repeated characters are detected."""
        analyzer = TextQualityAnalyzer()
        text = "Normal text aaaaaaaaaaaaaaaaaaaaa"
        result = analyzer.analyze(text)

        assert "Repeated characters detected" in result["issues"]
        assert result["score"] < 1.0

    def test_analyze_dense_text(self):
        """Test detection of unusually dense text."""
        analyzer = TextQualityAnalyzer()
        # Create text with many words per line
        words = "word " * 60
        text = words + "\n" + words
        result = analyzer.analyze(text)

        assert "Unusually dense text" in result["issues"]

    def test_score_bounds(self):
        """Test that score is always between 0 and 1."""
        analyzer = TextQualityAnalyzer()

        # Very bad text
        bad_text = "\x00" * 100 + "a" * 1000
        result = analyzer.analyze(bad_text)
        assert 0.0 <= result["score"] <= 1.0

        # Very good text
        good_text = "This is a high quality document with proper formatting and content."
        result = analyzer.analyze(good_text)
        assert 0.0 <= result["score"] <= 1.0


@pytest.mark.unit
class TestStructureQualityAnalyzer:
    """Tests for StructureQualityAnalyzer class."""

    def test_analyze_empty_result(self):
        """Test analyzing an empty parsing result."""
        analyzer = StructureQualityAnalyzer()
        result = ParsingResult(
            success=True,
            text="",
            pages=[],
            metadata={},
        )
        analysis = analyzer.analyze(result)

        assert analysis["score"] < 1.0
        assert analysis["page_count"] == 0
        assert analysis["has_metadata"] is False
        assert analysis["empty_pages"] == 0

    def test_analyze_with_pages(self):
        """Test analyzing a result with pages."""
        analyzer = StructureQualityAnalyzer()
        result = ParsingResult(
            success=True,
            text="Page 1\n\nPage 2",
            pages=["Page 1", "Page 2"],
            metadata={"title": "Test"},
        )
        analysis = analyzer.analyze(result)

        assert analysis["page_count"] == 2
        assert analysis["has_metadata"] is True
        assert analysis["avg_page_length"] > 0

    def test_analyze_empty_pages_detection(self):
        """Test detection of empty pages."""
        analyzer = StructureQualityAnalyzer()
        result = ParsingResult(
            success=True,
            text="Content",
            pages=["Some content here", "x", ""],
            metadata={},
        )
        analysis = analyzer.analyze(result)

        assert analysis["empty_pages"] > 0
        assert "near-empty pages detected" in " ".join(analysis["issues"])

    def test_analyze_page_length_variance(self):
        """Test detection of large variance in page lengths."""
        analyzer = StructureQualityAnalyzer()
        result = ParsingResult(
            success=True,
            text="Short\n\n" + "Long " * 100,
            pages=["Short", "Long " * 100],
            metadata={},
        )
        analysis = analyzer.analyze(result)

        assert "Large variance in page content lengths" in analysis["issues"]

    def test_analyze_with_tables(self):
        """Test analyzing a result with extracted tables."""
        analyzer = StructureQualityAnalyzer()
        result = ParsingResult(
            success=True,
            text="Content",
            pages=["Page 1"],
            metadata={},
            tables=[{"data": []}, {"data": []}],
        )
        analysis = analyzer.analyze(result)

        assert analysis["tables_extracted"] == 2

    def test_analyze_with_images(self):
        """Test analyzing a result with extracted images."""
        analyzer = StructureQualityAnalyzer()
        result = ParsingResult(
            success=True,
            text="Content",
            pages=["Page 1"],
            metadata={},
            images=[{"path": "img1.png"}, {"path": "img2.png"}],
        )
        analysis = analyzer.analyze(result)

        assert analysis["images_extracted"] == 2


@pytest.mark.unit
class TestQualityAssessor:
    """Tests for QualityAssessor class."""

    @pytest.fixture
    def assessor(self):
        """Create a QualityAssessor instance."""
        return QualityAssessor()

    @pytest.fixture
    def successful_result(self):
        """Create a successful parsing result."""
        return ParsingResult(
            success=True,
            text="This is a well formatted document with proper content.",
            pages=["This is a well formatted document with proper content."],
            metadata={"title": "Test Document"},
            confidence=0.9,
            parser_used="docling",
        )

    @pytest.fixture
    def failed_result(self):
        """Create a failed parsing result."""
        return ParsingResult(
            success=False,
            text="",
            error="Parsing failed",
            confidence=0.0,
        )

    @pytest.mark.asyncio
    async def test_assess_successful_result(self, assessor, successful_result):
        """Test assessing a successful parsing result."""
        config = QualityConfig()
        score = await assessor.assess(successful_result, config)

        assert isinstance(score, QualityScore)
        assert score.overall_score > 0
        assert score.text_quality > 0
        assert score.structure_quality > 0
        assert score.ocr_confidence == 0.9

    @pytest.mark.asyncio
    async def test_assess_failed_result(self, assessor, failed_result):
        """Test assessing a failed parsing result."""
        config = QualityConfig()
        score = await assessor.assess(failed_result, config)

        assert isinstance(score, QualityScore)
        assert score.overall_score == 0.0
        assert score.text_quality == 0.0
        assert score.structure_quality == 0.0
        assert "Parsing failed" in score.issues
        assert "Retry with fallback parser" in score.recommendations

    @pytest.mark.asyncio
    async def test_assess_uses_default_config(self, assessor, successful_result):
        """Test that default config is used when none provided."""
        score = await assessor.assess(successful_result)

        assert isinstance(score, QualityScore)
        assert score.overall_score > 0

    @pytest.mark.asyncio
    async def test_assess_calculates_completeness(self, assessor):
        """Test that completeness is calculated correctly."""
        result = ParsingResult(
            success=True,
            text="Short",
            pages=["Short"],
            metadata={},
            confidence=0.8,
        )
        score = await assessor.assess(result)

        assert score.completeness < 1.0  # Should be penalized for short content

    @pytest.mark.asyncio
    async def test_assess_generates_recommendations(self, assessor):
        """Test that recommendations are generated based on analysis."""
        result = ParsingResult(
            success=True,
            text="Low confidence text",
            pages=["Low confidence text"],
            metadata={},
            confidence=0.5,
        )
        score = await assessor.assess(result)

        assert len(score.recommendations) > 0
        assert any("OCR" in r or "confidence" in r for r in score.recommendations)

    def test_should_retry_when_disabled(self, assessor):
        """Test that retry is not attempted when disabled."""
        score = QualityScore(
            overall_score=0.5,
            text_quality=0.5,
            structure_quality=0.5,
            ocr_confidence=0.5,
            completeness=0.5,
            issues=[],
            recommendations=[],
        )
        config = QualityConfig(enabled=False, auto_retry=True)

        assert assessor.should_retry(score, config) is False

    def test_should_retry_when_auto_retry_disabled(self, assessor):
        """Test that retry is not attempted when auto_retry is disabled."""
        score = QualityScore(
            overall_score=0.5,
            text_quality=0.5,
            structure_quality=0.5,
            ocr_confidence=0.5,
            completeness=0.5,
            issues=[],
            recommendations=[],
        )
        config = QualityConfig(enabled=True, auto_retry=False)

        assert assessor.should_retry(score, config) is False

    def test_should_retry_when_score_high_enough(self, assessor):
        """Test that retry is not attempted when score is above threshold."""
        score = QualityScore(
            overall_score=0.8,
            text_quality=0.8,
            structure_quality=0.8,
            ocr_confidence=0.8,
            completeness=0.8,
            issues=[],
            recommendations=[],
        )
        config = QualityConfig(enabled=True, auto_retry=True, min_quality_score=0.7)

        assert assessor.should_retry(score, config) is False

    def test_should_retry_when_max_retries_reached(self, assessor):
        """Test that retry is not attempted when max retries reached."""
        score = QualityScore(
            overall_score=0.5,
            text_quality=0.5,
            structure_quality=0.5,
            ocr_confidence=0.5,
            completeness=0.5,
            issues=[],
            recommendations=[],
        )
        config = QualityConfig(enabled=True, auto_retry=True, max_retries=3)

        assert assessor.should_retry(score, config, current_attempt=3) is False

    def test_should_retry_when_appropriate(self, assessor):
        """Test that retry is attempted when conditions are met."""
        score = QualityScore(
            overall_score=0.5,
            text_quality=0.5,
            structure_quality=0.5,
            ocr_confidence=0.5,
            completeness=0.5,
            issues=[],
            recommendations=[],
        )
        config = QualityConfig(enabled=True, auto_retry=True, max_retries=3)

        assert assessor.should_retry(score, config, current_attempt=1) is True

    def test_get_retry_strategy_fallback_for_poor_text(self, assessor):
        """Test fallback strategy is recommended for poor text quality."""
        score = QualityScore(
            overall_score=0.5,
            text_quality=0.2,
            structure_quality=0.5,
            ocr_confidence=0.5,
            completeness=0.5,
            issues=[],
            recommendations=[],
        )

        strategy = assessor.get_retry_strategy(score, "docling")
        assert strategy == "fallback_parser"

    def test_get_retry_strategy_fallback_for_low_ocr(self, assessor):
        """Test fallback strategy is recommended for low OCR confidence."""
        score = QualityScore(
            overall_score=0.5,
            text_quality=0.6,
            structure_quality=0.5,
            ocr_confidence=0.4,
            completeness=0.5,
            issues=[],
            recommendations=[],
        )

        strategy = assessor.get_retry_strategy(score, "docling")
        assert strategy == "fallback_parser"

    def test_get_retry_strategy_same_parser_for_transient(self, assessor):
        """Test same parser strategy for transient failures."""
        score = QualityScore(
            overall_score=0.6,
            text_quality=0.6,
            structure_quality=0.6,
            ocr_confidence=0.6,
            completeness=0.6,
            issues=[],
            recommendations=[],
        )

        strategy = assessor.get_retry_strategy(score, "docling")
        assert strategy == "same_parser"

    def test_get_retry_strategy_already_using_ocr(self, assessor):
        """Test same parser strategy when already using OCR."""
        score = QualityScore(
            overall_score=0.4,
            text_quality=0.2,
            structure_quality=0.5,
            ocr_confidence=0.5,
            completeness=0.5,
            issues=[],
            recommendations=[],
        )

        strategy = assessor.get_retry_strategy(score, "azure_ocr")
        assert strategy == "same_parser"

    def test_calculate_completeness_empty_text(self, assessor):
        """Test completeness calculation for empty text."""
        result = ParsingResult(success=True, text="", pages=[])
        text_analysis = {"char_count": 0, "word_count": 0}

        completeness = assessor._calculate_completeness(result, text_analysis)
        assert completeness == 0.0

    def test_calculate_completeness_few_words(self, assessor):
        """Test completeness penalty for very few words."""
        result = ParsingResult(success=True, text="Hi there", pages=["Hi there"])
        text_analysis = {"char_count": 8, "word_count": 2}

        completeness = assessor._calculate_completeness(result, text_analysis)
        assert completeness < 1.0

    def test_calculate_completeness_with_pages(self, assessor):
        """Test completeness calculation with multiple pages."""
        result = ParsingResult(
            success=True,
            text="Page 1 content here",
            pages=["Page 1 content here", "Page 2 content here"],
        )
        text_analysis = {"char_count": 40, "word_count": 10}

        completeness = assessor._calculate_completeness(result, text_analysis)
        assert 0.0 <= completeness <= 1.0

    def test_generate_recommendations_low_score(self, assessor):
        """Test recommendation generation for low score."""
        text_analysis = {"garbage_ratio": 0.05, "avg_word_length": 5.0}
        structure_analysis = {"empty_pages": 0}
        result = ParsingResult(success=True, text="", confidence=0.8)

        recs = assessor._generate_recommendations(0.4, text_analysis, structure_analysis, result)
        assert any("fallback parser" in r.lower() for r in recs)

    def test_generate_recommendations_high_garbage(self, assessor):
        """Test recommendation for high garbage ratio."""
        text_analysis = {"garbage_ratio": 0.15, "avg_word_length": 5.0}
        structure_analysis = {"empty_pages": 0}
        result = ParsingResult(success=True, text="", confidence=0.8)

        recs = assessor._generate_recommendations(0.7, text_analysis, structure_analysis, result)
        assert any("garbage" in r.lower() for r in recs)

    def test_generate_recommendations_long_words(self, assessor):
        """Test recommendation for long words."""
        text_analysis = {"garbage_ratio": 0.05, "avg_word_length": 20.0}
        structure_analysis = {"empty_pages": 0}
        result = ParsingResult(success=True, text="", confidence=0.8)

        recs = assessor._generate_recommendations(0.7, text_analysis, structure_analysis, result)
        assert any("concatenated" in r.lower() or "OCR error" in r for r in recs)

    def test_generate_recommendations_empty_pages(self, assessor):
        """Test recommendation for empty pages."""
        text_analysis = {"garbage_ratio": 0.05, "avg_word_length": 5.0}
        structure_analysis = {"empty_pages": 2}
        result = ParsingResult(success=True, text="", confidence=0.8)

        recs = assessor._generate_recommendations(0.7, text_analysis, structure_analysis, result)
        assert any("empty" in r.lower() for r in recs)

    def test_generate_recommendations_low_ocr(self, assessor):
        """Test recommendation for low OCR confidence."""
        text_analysis = {"garbage_ratio": 0.05, "avg_word_length": 5.0}
        structure_analysis = {"empty_pages": 0}
        result = ParsingResult(success=True, text="", confidence=0.5)

        recs = assessor._generate_recommendations(0.7, text_analysis, structure_analysis, result)
        assert any("preprocessing" in r.lower() for r in recs)


@pytest.mark.unit
class TestAssessQualityFunction:
    """Tests for the assess_quality convenience function."""

    @pytest.mark.asyncio
    async def test_assess_quality_convenience_function(self):
        """Test that assess_quality works as a convenience function."""
        result = ParsingResult(
            success=True,
            text="This is test content.",
            pages=["This is test content."],
            metadata={},
            confidence=0.9,
        )

        score = await assess_quality(result)

        assert isinstance(score, QualityScore)
        assert score.overall_score > 0
