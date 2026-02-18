"""Unit tests for the agentic decision engine module.

Tests cover:
- AgenticDecisionEngine class
- Decision types and enums
- Parser selection decisions
- Enrichment decisions
- Quality assessment decisions
- All decision strategies
"""

import json
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import UUID, uuid4

import pytest

from src.api.models import (
    ContentDetectionResult,
    ContentType,
    QualityConfig,
    TextStatistics,
)
from src.core.decisions import (
    AdvancedDecisionEngine,
    AgenticDecisionEngine,
    DecisionContext,
    DecisionResult,
    ParserSelection,
    RetryDecision,
)
from src.core.quality import QualityScore
from src.llm.provider import ChatMessage, LLMProvider


@pytest.mark.unit
class TestParserSelection:
    """Tests for ParserSelection dataclass."""

    def test_parser_selection_creation(self):
        """Test creating a ParserSelection instance."""
        selection = ParserSelection(
            parser="docling",
            reason="Text-based PDF detected",
            confidence=0.95,
            fallback_parser="azure_ocr",
            preprocessing_steps=["deskew"],
        )

        assert selection.parser == "docling"
        assert selection.reason == "Text-based PDF detected"
        assert selection.confidence == 0.95
        assert selection.fallback_parser == "azure_ocr"
        assert selection.preprocessing_steps == ["deskew"]

    def test_parser_selection_defaults(self):
        """Test ParserSelection default values."""
        selection = ParserSelection(
            parser="azure_ocr",
            reason="Scanned document",
            confidence=0.8,
        )

        assert selection.fallback_parser is None
        assert selection.preprocessing_steps == []

    def test_parser_selection_post_init(self):
        """Test that __post_init__ initializes preprocessing_steps."""
        selection = ParserSelection(
            parser="docling",
            reason="Test",
            confidence=0.9,
        )

        assert selection.preprocessing_steps == []
        assert isinstance(selection.preprocessing_steps, list)


@pytest.mark.unit
class TestRetryDecision:
    """Tests for RetryDecision dataclass."""

    def test_retry_decision_creation(self):
        """Test creating a RetryDecision instance."""
        decision = RetryDecision(
            should_retry=True,
            strategy="fallback_parser",
            reason="Low quality score",
            updated_config={"parser": {"primary": "azure_ocr"}},
        )

        assert decision.should_retry is True
        assert decision.strategy == "fallback_parser"
        assert decision.reason == "Low quality score"
        assert decision.updated_config == {"parser": {"primary": "azure_ocr"}}

    def test_retry_decision_no_retry(self):
        """Test RetryDecision when retry is not needed."""
        decision = RetryDecision(
            should_retry=False,
            strategy="none",
            reason="Quality meets threshold",
        )

        assert decision.should_retry is False
        assert decision.strategy == "none"
        assert decision.updated_config is None


@pytest.mark.unit
class TestDecisionContext:
    """Tests for DecisionContext dataclass."""

    def test_decision_context_creation(self):
        """Test creating a DecisionContext instance."""
        detection_result = ContentDetectionResult(
            detected_type=ContentType.TEXT_BASED_PDF,
            confidence=0.95,
            recommended_parser="docling",
        )

        context = DecisionContext(
            job_id=uuid4(),
            file_type="pdf",
            content_detection=detection_result,
            historical_success_rate=0.85,
            parser_performance={"docling": {"avg_time": 1.2}},
            system_load=0.4,
            time_of_day=14,
        )

        assert context.file_type == "pdf"
        assert context.historical_success_rate == 0.85
        assert context.system_load == 0.4
        assert context.time_of_day == 14

    def test_decision_context_defaults(self):
        """Test DecisionContext default values."""
        context = DecisionContext(
            job_id=uuid4(),
            file_type="docx",
        )

        assert context.content_detection is None
        assert context.historical_success_rate == 0.0
        assert context.parser_performance == {}
        assert context.current_config is None
        assert context.system_load == 0.5
        assert context.time_of_day == 12


@pytest.mark.unit
class TestDecisionResult:
    """Tests for DecisionResult dataclass."""

    def test_decision_result_creation(self):
        """Test creating a DecisionResult instance."""
        result = DecisionResult(
            decision_type="parser_selection",
            selected_option="docling",
            confidence=0.9,
            reasoning="Text-based PDF with good structure",
            alternatives=["azure_ocr"],
            expected_outcome={"success_rate": 0.95},
        )

        assert result.decision_type == "parser_selection"
        assert result.selected_option == "docling"
        assert result.confidence == 0.9
        assert result.reasoning == "Text-based PDF with good structure"
        assert result.alternatives == ["azure_ocr"]
        assert result.expected_outcome == {"success_rate": 0.95}

    def test_decision_result_defaults(self):
        """Test DecisionResult default values."""
        result = DecisionResult(
            decision_type="retry_strategy",
            selected_option="same_parser",
            confidence=0.75,
            reasoning="Transient failure",
        )

        assert result.alternatives == []
        assert result.expected_outcome == {}


@pytest.mark.unit
class TestAgenticDecisionEngine:
    """Tests for AgenticDecisionEngine class."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM provider."""
        llm = MagicMock(spec=LLMProvider)
        llm.chat_completion = AsyncMock()
        return llm

    @pytest.fixture
    def engine(self, mock_llm):
        """Create an AgenticDecisionEngine instance."""
        return AgenticDecisionEngine(llm_provider=mock_llm)

    @pytest.fixture
    def detection_result(self):
        """Create a sample content detection result."""
        return ContentDetectionResult(
            detected_type=ContentType.TEXT_BASED_PDF,
            confidence=0.95,
            detection_method="hybrid",
            page_count=10,
            has_text_layer=True,
            has_images=False,
            image_count=0,
            text_statistics=TextStatistics(
                total_characters=5000,
                total_words=800,
                text_ratio=0.98,
                avg_chars_per_page=500,
            ),
            recommended_parser="docling",
        )

    def test_engine_initialization(self, mock_llm):
        """Test that the engine initializes correctly."""
        engine = AgenticDecisionEngine(llm_provider=mock_llm)

        assert engine.llm == mock_llm
        assert engine.logger is not None

    @pytest.mark.asyncio
    async def test_select_parser_success(self, engine, mock_llm, detection_result):
        """Test successful parser selection via LLM."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "parser": "docling",
            "reason": "Text-based PDF with good structure",
            "confidence": 0.95,
            "fallback_parser": "azure_ocr",
            "preprocessing_steps": [],
        })
        mock_llm.chat_completion.return_value = mock_response

        result = await engine.select_parser(detection_result)

        assert isinstance(result, ParserSelection)
        assert result.parser == "docling"
        assert result.confidence == 0.95
        assert result.fallback_parser == "azure_ocr"
        mock_llm.chat_completion.assert_called_once()

    @pytest.mark.asyncio
    async def test_select_parser_json_decode_error(self, engine, mock_llm, detection_result):
        """Test parser selection fallback on JSON decode error."""
        mock_response = MagicMock()
        mock_response.content = "Invalid JSON response"
        mock_llm.chat_completion.return_value = mock_response

        result = await engine.select_parser(detection_result)

        assert isinstance(result, ParserSelection)
        assert result.parser == "docling"  # Rule-based fallback for text-based PDF
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_select_parser_llm_exception(self, engine, mock_llm, detection_result):
        """Test parser selection fallback on LLM exception."""
        mock_llm.chat_completion.side_effect = Exception("LLM timeout")

        result = await engine.select_parser(detection_result)

        assert isinstance(result, ParserSelection)
        assert result.parser == "docling"

    @pytest.mark.asyncio
    async def test_select_parser_scanned_pdf(self, engine, mock_llm):
        """Test parser selection for scanned PDF."""
        detection_result = ContentDetectionResult(
            detected_type=ContentType.SCANNED_PDF,
            confidence=0.9,
            recommended_parser="azure_ocr",
            text_statistics=TextStatistics(text_ratio=0.02),
        )
        mock_response = MagicMock()
        mock_response.content = "Invalid JSON"  # Force fallback
        mock_llm.chat_completion.return_value = mock_response

        result = await engine.select_parser(detection_result)

        assert result.parser == "azure_ocr"
        assert "Scanned PDF" in result.reason

    @pytest.mark.asyncio
    async def test_select_parser_image(self, engine, mock_llm):
        """Test parser selection for image file."""
        detection_result = ContentDetectionResult(
            detected_type=ContentType.IMAGE,
            confidence=0.95,
            recommended_parser="azure_ocr",
            text_statistics=TextStatistics(),
        )
        mock_response = MagicMock()
        mock_response.content = "Invalid JSON"  # Force fallback
        mock_llm.chat_completion.return_value = mock_response

        result = await engine.select_parser(detection_result)

        assert result.parser == "azure_ocr"
        assert "Image" in result.reason

    @pytest.mark.asyncio
    async def test_select_parser_mixed_pdf(self, engine, mock_llm):
        """Test parser selection for mixed PDF."""
        detection_result = ContentDetectionResult(
            detected_type=ContentType.MIXED_PDF,
            confidence=0.8,
            recommended_parser="docling",
            text_statistics=TextStatistics(text_ratio=0.5),
        )
        mock_response = MagicMock()
        mock_response.content = "Invalid JSON"  # Force fallback
        mock_llm.chat_completion.return_value = mock_response

        result = await engine.select_parser(detection_result)

        assert result.parser == "docling"
        assert "Mixed PDF" in result.reason
        assert result.fallback_parser == "azure_ocr"

    def test_rule_based_parser_selection_text_pdf(self, engine):
        """Test rule-based selection for text-based PDF."""
        detection_result = ContentDetectionResult(
            detected_type=ContentType.TEXT_BASED_PDF,
            confidence=0.95,
            recommended_parser="docling",
            text_statistics=TextStatistics(),
        )

        result = engine._rule_based_parser_selection(detection_result)

        assert result.parser == "docling"
        assert result.confidence == 0.9
        assert result.fallback_parser == "azure_ocr"

    def test_rule_based_parser_selection_office_doc(self, engine):
        """Test rule-based selection for Office document."""
        detection_result = ContentDetectionResult(
            detected_type=ContentType.OFFICE_DOC,
            confidence=0.9,
            recommended_parser="docling",
            text_statistics=TextStatistics(),
        )

        result = engine._rule_based_parser_selection(detection_result)

        assert result.parser == "docling"

    @pytest.mark.asyncio
    async def test_decide_retry_max_retries_reached(self, engine):
        """Test retry decision when max retries reached."""
        quality_score = QualityScore(
            overall_score=0.5,
            text_quality=0.5,
            structure_quality=0.5,
            ocr_confidence=0.5,
            completeness=0.5,
            issues=[],
            recommendations=[],
        )
        config = QualityConfig(max_retries=3)

        result = await engine.decide_retry(quality_score, config, attempt_number=3)

        assert result.should_retry is False
        assert result.strategy == "none"
        assert "Maximum retry" in result.reason

    @pytest.mark.asyncio
    async def test_decide_retry_quality_good(self, engine):
        """Test retry decision when quality is good."""
        quality_score = QualityScore(
            overall_score=0.8,
            text_quality=0.8,
            structure_quality=0.8,
            ocr_confidence=0.8,
            completeness=0.8,
            issues=[],
            recommendations=[],
        )
        config = QualityConfig(min_quality_score=0.7, max_retries=3)

        result = await engine.decide_retry(quality_score, config, attempt_number=1)

        assert result.should_retry is False
        assert "Quality meets" in result.reason

    @pytest.mark.asyncio
    async def test_decide_retry_llm_success(self, engine, mock_llm):
        """Test retry decision via LLM."""
        quality_score = QualityScore(
            overall_score=0.4,
            text_quality=0.3,
            structure_quality=0.5,
            ocr_confidence=0.4,
            completeness=0.5,
            issues=["Low OCR confidence"],
            recommendations=[],
        )
        config = QualityConfig(min_quality_score=0.7, max_retries=3)

        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "should_retry": True,
            "strategy": "fallback_parser",
            "reason": "Low quality, try alternative parser",
            "updated_config": {"parser": {"primary": "azure_ocr"}},
        })
        mock_llm.chat_completion.return_value = mock_response

        result = await engine.decide_retry(quality_score, config, attempt_number=1)

        assert result.should_retry is True
        assert result.strategy == "fallback_parser"
        assert result.updated_config is not None

    @pytest.mark.asyncio
    async def test_decide_retry_llm_exception(self, engine, mock_llm):
        """Test retry decision fallback on LLM exception."""
        quality_score = QualityScore(
            overall_score=0.4,
            text_quality=0.4,
            structure_quality=0.4,
            ocr_confidence=0.4,
            completeness=0.4,
            issues=[],
            recommendations=[],
        )
        config = QualityConfig(max_retries=3)

        mock_llm.chat_completion.side_effect = Exception("LLM error")

        result = await engine.decide_retry(quality_score, config, attempt_number=1)

        assert result.should_retry is True
        assert result.strategy == "fallback_parser"

    def test_rule_based_retry_decision_very_low_score(self, engine):
        """Test rule-based retry for very low quality score."""
        quality_score = QualityScore(
            overall_score=0.2,
            text_quality=0.2,
            structure_quality=0.2,
            ocr_confidence=0.2,
            completeness=0.2,
            issues=[],
            recommendations=[],
        )
        config = QualityConfig()

        result = engine._rule_based_retry_decision(quality_score, config, 1, [])

        assert result.should_retry is True
        assert result.strategy == "fallback_parser"

    def test_rule_based_retry_decision_medium_score(self, engine):
        """Test rule-based retry for medium quality score."""
        quality_score = QualityScore(
            overall_score=0.4,
            text_quality=0.4,
            structure_quality=0.4,
            ocr_confidence=0.4,
            completeness=0.4,
            issues=[],
            recommendations=[],
        )
        config = QualityConfig()

        result = engine._rule_based_retry_decision(quality_score, config, 1, [])

        assert result.should_retry is True
        assert result.strategy == "fallback_parser"

    def test_rule_based_retry_decision_already_tried_fallback(self, engine):
        """Test rule-based retry when fallback already tried."""
        quality_score = QualityScore(
            overall_score=0.4,
            text_quality=0.4,
            structure_quality=0.4,
            ocr_confidence=0.4,
            completeness=0.4,
            issues=[],
            recommendations=[],
        )
        config = QualityConfig()
        previous_strategies = ["fallback_parser"]

        result = engine._rule_based_retry_decision(
            quality_score, config, 2, previous_strategies
        )

        assert result.should_retry is True
        assert result.strategy == "same_parser"

    def test_rule_based_retry_decision_good_score(self, engine):
        """Test rule-based retry for good quality score."""
        quality_score = QualityScore(
            overall_score=0.7,
            text_quality=0.7,
            structure_quality=0.7,
            ocr_confidence=0.7,
            completeness=0.7,
            issues=[],
            recommendations=[],
        )
        config = QualityConfig()

        result = engine._rule_based_retry_decision(quality_score, config, 1, [])

        assert result.should_retry is False
        assert result.strategy == "none"

    @pytest.mark.asyncio
    async def test_analyze_error_success(self, engine, mock_llm):
        """Test successful error analysis."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "category": "transient",
            "recoverable": True,
            "suggested_action": "Retry with exponential backoff",
            "retry_immediately": True,
        })
        mock_llm.chat_completion.return_value = mock_response

        error = Exception("Connection timeout")
        context = {"stage": "parsing", "parser": "docling"}

        result = await engine.analyze_error(error, context)

        assert result["category"] == "transient"
        assert result["recoverable"] is True
        assert result["retry_immediately"] is True

    @pytest.mark.asyncio
    async def test_analyze_error_exception(self, engine, mock_llm):
        """Test error analysis fallback on exception."""
        mock_llm.chat_completion.side_effect = Exception("LLM error")

        error = Exception("Some error")
        context = {}

        result = await engine.analyze_error(error, context)

        assert result["category"] == "unknown"
        assert result["recoverable"] is True
        assert result["retry_immediately"] is True

    @pytest.mark.asyncio
    async def test_optimize_quality_thresholds_empty(self, engine):
        """Test threshold optimization with empty history."""
        result = await engine.optimize_quality_thresholds([])

        assert result["min_quality_score"] == 0.7

    @pytest.mark.asyncio
    async def test_optimize_quality_thresholds_low_success_rate(self, engine):
        """Test threshold optimization when success rate is low."""
        scores = [
            QualityScore(overall_score=0.9, text_quality=0.9, structure_quality=0.9,
                        ocr_confidence=0.9, completeness=0.9, issues=[], recommendations=[]),
            QualityScore(overall_score=0.5, text_quality=0.5, structure_quality=0.5,
                        ocr_confidence=0.5, completeness=0.5, issues=[], recommendations=[]),
            QualityScore(overall_score=0.4, text_quality=0.4, structure_quality=0.4,
                        ocr_confidence=0.4, completeness=0.4, issues=[], recommendations=[]),
        ]

        result = await engine.optimize_quality_thresholds(scores, target_success_rate=0.95)

        assert "min_quality_score" in result
        assert "current_success_rate" in result
        assert "average_score" in result
        assert abs(result["current_success_rate"] - 1 / 3) < 0.01

    @pytest.mark.asyncio
    async def test_optimize_quality_thresholds_high_success_rate(self, engine):
        """Test threshold optimization when success rate is high."""
        scores = [
            QualityScore(overall_score=0.9, text_quality=0.9, structure_quality=0.9,
                        ocr_confidence=0.9, completeness=0.9, issues=[], recommendations=[]),
            QualityScore(overall_score=0.85, text_quality=0.85, structure_quality=0.85,
                        ocr_confidence=0.85, completeness=0.85, issues=[], recommendations=[]),
        ]

        result = await engine.optimize_quality_thresholds(scores, target_success_rate=0.95)

        assert result["current_success_rate"] == 1.0
        assert result["min_quality_score"] >= 0.7


@pytest.mark.unit
class TestAdvancedDecisionEngine:
    """Tests for AdvancedDecisionEngine class."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM provider."""
        llm = MagicMock(spec=LLMProvider)
        llm.chat_completion = AsyncMock()
        return llm

    @pytest.fixture
    def advanced_engine(self, mock_llm):
        """Create an AdvancedDecisionEngine instance."""
        return AdvancedDecisionEngine(
            llm_provider=mock_llm,
            use_historical_data=True,
            ml_weight=0.6,
        )

    def test_advanced_engine_initialization(self, mock_llm):
        """Test that the advanced engine initializes correctly."""
        engine = AdvancedDecisionEngine(
            llm_provider=mock_llm,
            use_historical_data=False,
            ml_weight=0.7,
        )

        assert engine.llm == mock_llm
        assert engine.use_historical_data is False
        assert engine.ml_weight == 0.7
        assert engine._history is None

    def test_set_history_provider(self, advanced_engine):
        """Test setting the history provider."""
        mock_history = MagicMock()

        advanced_engine.set_history_provider(mock_history)

        assert advanced_engine._history == mock_history

    @pytest.mark.asyncio
    async def test_decide_with_ml_high_confidence_historical(self, advanced_engine):
        """Test ML decision with high confidence historical data."""
        mock_history = MagicMock()
        mock_history.get_parser_recommendation = AsyncMock(
            return_value=("docling", 0.85)
        )
        advanced_engine.set_history_provider(mock_history)

        detection_result = ContentDetectionResult(
            detected_type=ContentType.TEXT_BASED_PDF,
            confidence=0.9,
            recommended_parser="docling",
            text_statistics=TextStatistics(),
        )

        context = DecisionContext(
            job_id=uuid4(),
            file_type="pdf",
            content_detection=detection_result,
        )

        # Mock LLM to return different parser
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "parser": "azure_ocr",
            "confidence": 0.7,
            "reason": "LLM suggestion",
        })
        advanced_engine.llm.chat_completion.return_value = mock_response

        result = await advanced_engine.decide_with_ml(context)

        assert isinstance(result, DecisionResult)
        assert result.selected_option == "docling"  # Historical takes precedence
        assert result.confidence == 0.85

    @pytest.mark.asyncio
    async def test_decide_with_ml_blend_decisions(self, advanced_engine):
        """Test ML decision blending when ML and LLM agree."""
        mock_history = MagicMock()
        mock_history.get_parser_recommendation = AsyncMock(
            return_value=("docling", 0.6)
        )
        advanced_engine.set_history_provider(mock_history)

        detection_result = ContentDetectionResult(
            detected_type=ContentType.TEXT_BASED_PDF,
            confidence=0.9,
            recommended_parser="docling",
            text_statistics=TextStatistics(),
        )

        context = DecisionContext(
            job_id=uuid4(),
            file_type="pdf",
            content_detection=detection_result,
        )

        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "parser": "docling",
            "confidence": 0.8,
            "reason": "LLM agrees",
        })
        advanced_engine.llm.chat_completion.return_value = mock_response

        result = await advanced_engine.decide_with_ml(context)

        assert result.selected_option == "docling"
        assert result.confidence == 0.8  # Max of historical and LLM

    @pytest.mark.asyncio
    async def test_decide_with_ml_llm_fallback(self, advanced_engine):
        """Test ML decision fallback to LLM when no historical data."""
        context = DecisionContext(
            job_id=uuid4(),
            file_type="pdf",
            content_detection=None,
        )

        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "parser": "azure_ocr",
            "confidence": 0.8,
            "reason": "LLM decision",
        })
        advanced_engine.llm.chat_completion.return_value = mock_response

        result = await advanced_engine.decide_with_ml(context)

        assert result.selected_option == "azure_ocr"
        assert result.confidence == 0.8

    @pytest.mark.asyncio
    async def test_adaptive_parser_selection(self, advanced_engine):
        """Test adaptive parser selection."""
        detection_result = ContentDetectionResult(
            detected_type=ContentType.TEXT_BASED_PDF,
            confidence=0.9,
            recommended_parser="docling",
            text_statistics=TextStatistics(),
        )

        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "parser": "docling",
            "confidence": 0.9,
            "reason": "Good text ratio",
        })
        advanced_engine.llm.chat_completion.return_value = mock_response

        result = await advanced_engine.adaptive_parser_selection(
            detection_result, "document.pdf"
        )

        assert isinstance(result, ParserSelection)
        assert result.parser == "docling"
        assert result.fallback_parser == "azure_ocr"

    @pytest.mark.asyncio
    async def test_context_aware_routing_high_load(self, advanced_engine):
        """Test context-aware routing under high load."""
        mock_dest1 = MagicMock()
        mock_dest1.name = "dest_b"
        mock_dest2 = MagicMock()
        mock_dest2.name = "dest_a"

        destinations = [mock_dest1, mock_dest2]

        result = await advanced_engine.context_aware_routing(
            {}, destinations, system_load=0.9
        )

        # Should sort by name under high load
        assert result[0].name == "dest_a"
        assert result[1].name == "dest_b"

    @pytest.mark.asyncio
    async def test_context_aware_routing_low_load(self, advanced_engine):
        """Test context-aware routing under low load."""
        destinations = [MagicMock(), MagicMock()]

        result = await advanced_engine.context_aware_routing(
            {}, destinations, system_load=0.2
        )

        assert result == destinations

    @pytest.mark.asyncio
    async def test_context_aware_routing_normal_load(self, advanced_engine):
        """Test context-aware routing under normal load."""
        destinations = [MagicMock(), MagicMock()]

        result = await advanced_engine.context_aware_routing(
            {}, destinations, system_load=0.5
        )

        assert result == destinations

    @pytest.mark.asyncio
    async def test_adaptive_threshold_adjustment_with_history(self, advanced_engine):
        """Test adaptive threshold adjustment with history."""
        mock_history = MagicMock()
        mock_stats = MagicMock()
        mock_stats.total_count = 20
        mock_stats.avg_quality_score = 0.75
        mock_history.get_content_type_stats = AsyncMock(return_value={
            "text_based_pdf": mock_stats
        })
        advanced_engine.set_history_provider(mock_history)

        result = await advanced_engine.adaptive_threshold_adjustment(
            "pdf", "text_based_pdf", 0.8
        )

        assert 0.5 <= result <= 0.95

    @pytest.mark.asyncio
    async def test_adaptive_threshold_adjustment_challenging_type(self, advanced_engine):
        """Test adaptive threshold adjustment for challenging file types."""
        result = await advanced_engine.adaptive_threshold_adjustment(
            "tiff", "image", 0.8
        )

        # Should be lower than base due to challenging type
        assert result < 0.8
        assert result >= 0.5

    @pytest.mark.asyncio
    async def test_adaptive_threshold_adjustment_bounds(self, advanced_engine):
        """Test adaptive threshold adjustment respects bounds."""
        result = await advanced_engine.adaptive_threshold_adjustment(
            "pdf", "text_based_pdf", 1.5  # Above max
        )

        assert result <= 0.95

        result = await advanced_engine.adaptive_threshold_adjustment(
            "pdf", "text_based_pdf", 0.1  # Below min
        )

        assert result >= 0.5

    @pytest.mark.asyncio
    async def test_get_llm_decision_success(self, advanced_engine):
        """Test getting LLM decision."""
        detection_result = ContentDetectionResult(
            detected_type=ContentType.TEXT_BASED_PDF,
            confidence=0.9,
            recommended_parser="docling",
            has_text_layer=True,
            text_statistics=TextStatistics(text_ratio=0.95),
        )

        context = DecisionContext(
            job_id=uuid4(),
            file_type="pdf",
            content_detection=detection_result,
            system_load=0.4,
            historical_success_rate=0.85,
            time_of_day=14,
        )

        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "parser": "docling",
            "confidence": 0.9,
            "reason": "Text-based PDF",
        })
        advanced_engine.llm.chat_completion.return_value = mock_response

        result = await advanced_engine._get_llm_decision(context)

        assert result["parser"] == "docling"
        assert result["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_get_llm_decision_exception(self, advanced_engine):
        """Test LLM decision fallback on exception."""
        context = DecisionContext(
            job_id=uuid4(),
            file_type="pdf",
        )

        advanced_engine.llm.chat_completion.side_effect = Exception("LLM error")

        result = await advanced_engine._get_llm_decision(context)

        assert result["parser"] == "docling"
        assert result["confidence"] == 0.7
