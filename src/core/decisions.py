"""Agentic decision engine for intelligent pipeline decisions.

This module provides LLM-based decision making for:
- Parser selection based on content analysis
- Retry strategy selection
- Quality threshold adjustments
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from src.api.models import ContentDetectionResult, QualityConfig
from src.core.quality import QualityScore
from src.llm.provider import ChatMessage, LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class ParserSelection:
    """Result of parser selection decision.
    
    Attributes:
        parser: Selected parser ID
        reason: Explanation for selection
        confidence: Confidence in decision (0.0 - 1.0)
        fallback_parser: Recommended fallback parser
        preprocessing_steps: Recommended preprocessing steps
    """
    parser: str
    reason: str
    confidence: float
    fallback_parser: str | None = None
    preprocessing_steps: list[str] = None  # type: ignore

    def __post_init__(self) -> None:
        """Initialize default values."""
        if self.preprocessing_steps is None:
            self.preprocessing_steps = []


@dataclass
class RetryDecision:
    """Result of retry decision.
    
    Attributes:
        should_retry: Whether to retry
        strategy: Retry strategy to use
        reason: Explanation for decision
        updated_config: Updated configuration for retry
    """
    should_retry: bool
    strategy: str
    reason: str
    updated_config: dict[str, Any] | None = None


class AgenticDecisionEngine:
    """LLM-powered decision engine for pipeline decisions.
    
    Uses LLM to make intelligent decisions about parser selection,
    retry strategies, and quality assessments.
    
    Example:
        >>> engine = AgenticDecisionEngine(llm_provider)
        >>> selection = await engine.select_parser(detection_result)
        >>> print(f"Selected parser: {selection.parser}")
    """

    def __init__(self, llm_provider: LLMProvider) -> None:
        """Initialize the decision engine.
        
        Args:
            llm_provider: LLM provider for making decisions
        """
        self.llm = llm_provider
        self.logger = logger

    async def select_parser(
        self,
        detection_result: ContentDetectionResult,
    ) -> ParserSelection:
        """Select the optimal parser based on content detection.
        
        Uses LLM to make an intelligent parser selection based on
the content analysis results.
        
        Args:
            detection_result: Content detection results
            
        Returns:
            ParserSelection with chosen parser and rationale
        """
        # Build decision prompt
        system_prompt = """You are a document processing expert. Your task is to select the optimal parser for a document based on its content analysis.

Available parsers:
- docling: Primary parser for PDFs, Office docs. Best for text-based documents with good structure.
- azure_ocr: OCR-based parser. Best for scanned documents, images, and documents with poor text layers.

Decision rules:
1. Text-based PDFs (>95% text) → docling
2. Scanned PDFs (<5% text, >90% images) → azure_ocr
3. Mixed PDFs → docling with azure_ocr fallback
4. Office documents → docling
5. Images → azure_ocr

Respond with valid JSON only in this format:
{
    "parser": "docling|azure_ocr",
    "reason": "explanation of decision",
    "confidence": 0.95,
    "fallback_parser": "parser to use if primary fails",
    "preprocessing_steps": []
}"""

        user_prompt = f"""Analyze this content detection result and select the best parser:

Detected type: {detection_result.detected_type.value}
Confidence: {detection_result.confidence}
Detection method: {detection_result.detection_method}
Page count: {detection_result.page_count}
Has text layer: {detection_result.has_text_layer}
Has images: {detection_result.has_images}
Image count: {detection_result.image_count}
Text statistics:
  - Total characters: {detection_result.text_statistics.total_characters}
  - Total words: {detection_result.text_statistics.total_words}
  - Text ratio: {detection_result.text_statistics.text_ratio:.2f}
  - Avg chars per page: {detection_result.text_statistics.avg_chars_per_page or 'N/A'}

Select the optimal parser."""

        try:
            # Get LLM decision
            response = await self.llm.chat_completion(
                messages=[
                    ChatMessage.system(system_prompt),
                    ChatMessage.user(user_prompt),
                ],
                temperature=0.1,  # Low temperature for deterministic decisions
                max_tokens=500,
            )

            # Parse JSON response
            decision = json.loads(response.content)

            return ParserSelection(
                parser=decision.get("parser", "docling"),
                reason=decision.get("reason", "Default selection"),
                confidence=decision.get("confidence", 0.8),
                fallback_parser=decision.get("fallback_parser"),
                preprocessing_steps=decision.get("preprocessing_steps", []),
            )

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            # Fallback to rule-based decision
            return self._rule_based_parser_selection(detection_result)
        except Exception as e:
            self.logger.error(f"LLM decision failed: {e}")
            return self._rule_based_parser_selection(detection_result)

    def _rule_based_parser_selection(
        self,
        detection_result: ContentDetectionResult,
    ) -> ParserSelection:
        """Fallback rule-based parser selection.
        
        Args:
            detection_result: Content detection results
            
        Returns:
            ParserSelection based on rules
        """
        from src.api.models import ContentType

        detected_type = detection_result.detected_type

        if detected_type == ContentType.SCANNED_PDF:
            return ParserSelection(
                parser="azure_ocr",
                reason="Scanned PDF detected - OCR required",
                confidence=0.9,
                fallback_parser="docling",
            )
        elif detected_type == ContentType.IMAGE:
            return ParserSelection(
                parser="azure_ocr",
                reason="Image file - OCR required",
                confidence=0.95,
                fallback_parser="docling",
            )
        elif detected_type == ContentType.MIXED_PDF:
            return ParserSelection(
                parser="docling",
                reason="Mixed PDF - try docling first with OCR fallback",
                confidence=0.8,
                fallback_parser="azure_ocr",
            )
        else:
            return ParserSelection(
                parser="docling",
                reason="Text-based or Office document - docling is optimal",
                confidence=0.9,
                fallback_parser="azure_ocr",
            )

    async def decide_retry(
        self,
        quality_score: QualityScore,
        current_config: QualityConfig,
        attempt_number: int,
        previous_strategies: list[str] | None = None,
    ) -> RetryDecision:
        """Decide whether to retry and with what strategy.
        
        Args:
            quality_score: Quality assessment results
            current_config: Current quality configuration
            attempt_number: Current retry attempt number
            previous_strategies: List of previously tried strategies
            
        Returns:
            RetryDecision with recommendation
        """
        previous_strategies = previous_strategies or []

        # Quick check: if max retries reached, don't retry
        if attempt_number >= current_config.max_retries:
            return RetryDecision(
                should_retry=False,
                strategy="none",
                reason="Maximum retry attempts reached",
            )

        # If quality is good enough, don't retry
        if quality_score.overall_score >= current_config.min_quality_score:
            return RetryDecision(
                should_retry=False,
                strategy="none",
                reason="Quality meets minimum threshold",
            )

        # Build decision prompt
        system_prompt = """You are a document processing expert. Decide whether to retry a failed parsing job and select the best retry strategy.

Available strategies:
- same_parser: Retry with same parser (for transient failures)
- fallback_parser: Switch to alternative parser
- preprocess_then_retry: Apply image enhancement then retry
- split_processing: Process document in parts

Decision rules:
- If quality_score < 0.3 → fallback_parser
- If quality_score 0.3-0.5 and text_quality low → fallback_parser
- If quality_score 0.3-0.5 and structure_quality low → same_parser
- If garbage_ratio high → preprocess_then_retry

Respond with valid JSON only:
{
    "should_retry": true|false,
    "strategy": "strategy_name",
    "reason": "explanation",
    "updated_config": {"key": "value"}
}"""

        user_prompt = f"""Decide retry strategy based on:

Quality Score: {quality_score.overall_score}
Text Quality: {quality_score.text_quality}
Structure Quality: {quality_score.structure_quality}
OCR Confidence: {quality_score.ocr_confidence}
Completeness: {quality_score.completeness}
Issues: {', '.join(quality_score.issues) if quality_score.issues else 'None'}
Attempt: {attempt_number}/{current_config.max_retries}
Previous Strategies: {', '.join(previous_strategies) if previous_strategies else 'None'}

Decision:"""

        try:
            response = await self.llm.chat_completion(
                messages=[
                    ChatMessage.system(system_prompt),
                    ChatMessage.user(user_prompt),
                ],
                temperature=0.2,
                max_tokens=300,
            )

            decision = json.loads(response.content)

            return RetryDecision(
                should_retry=decision.get("should_retry", False),
                strategy=decision.get("strategy", "same_parser"),
                reason=decision.get("reason", "LLM decision"),
                updated_config=decision.get("updated_config"),
            )

        except Exception as e:
            self.logger.error(f"LLM retry decision failed: {e}")
            # Fallback to rule-based
            return self._rule_based_retry_decision(
                quality_score, current_config, attempt_number, previous_strategies
            )

    def _rule_based_retry_decision(
        self,
        quality_score: QualityScore,
        config: QualityConfig,
        attempt_number: int,
        previous_strategies: list[str],
    ) -> RetryDecision:
        """Fallback rule-based retry decision.
        
        Args:
            quality_score: Quality assessment results
            config: Quality configuration
            attempt_number: Current attempt number
            previous_strategies: Previously tried strategies
            
        Returns:
            RetryDecision based on rules
        """
        score = quality_score.overall_score

        if score < 0.3:
            return RetryDecision(
                should_retry=True,
                strategy="fallback_parser",
                reason="Very low quality score, try alternative parser",
            )
        elif score < 0.5:
            if "fallback_parser" not in previous_strategies:
                return RetryDecision(
                    should_retry=True,
                    strategy="fallback_parser",
                    reason="Low quality, trying alternative parser",
                )
            else:
                return RetryDecision(
                    should_retry=True,
                    strategy="same_parser",
                    reason="Retrying with same parser",
                )
        else:
            return RetryDecision(
                should_retry=False,
                strategy="none",
                reason="Quality acceptable, no retry needed",
            )

    async def analyze_error(
        self,
        error: Exception,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Analyze an error and suggest recovery actions.
        
        Args:
            error: The exception that occurred
            context: Context about where/when error occurred
            
        Returns:
            Dictionary with analysis and recommendations
        """
        system_prompt = """You are a document processing expert. Analyze errors and suggest recovery actions.

Categories:
- transient: Temporary failure, retry may succeed
- parsing: Document format issue, try different parser
- resource: Out of memory/resources, try splitting
- config: Configuration issue
- unknown: Unknown error

Respond with JSON:
{
    "category": "transient|parsing|resource|config|unknown",
    "recoverable": true|false,
    "suggested_action": "action description",
    "retry_immediately": true|false
}"""

        user_prompt = f"""Analyze this error:

Error Type: {type(error).__name__}
Error Message: {error!s}
Context: {json.dumps(context, default=str)}

Analysis:"""

        try:
            response = await self.llm.chat_completion(
                messages=[
                    ChatMessage.system(system_prompt),
                    ChatMessage.user(user_prompt),
                ],
                temperature=0.2,
                max_tokens=300,
            )

            return json.loads(response.content)

        except Exception as e:
            self.logger.error(f"Error analysis failed: {e}")
            return {
                "category": "unknown",
                "recoverable": True,
                "suggested_action": "Retry with same configuration",
                "retry_immediately": True,
            }

    async def optimize_quality_thresholds(
        self,
        historical_scores: list[QualityScore],
        target_success_rate: float = 0.95,
    ) -> dict[str, float]:
        """Suggest optimized quality thresholds based on historical data.
        
        Args:
            historical_scores: List of past quality scores
            target_success_rate: Desired success rate
            
        Returns:
            Dictionary with suggested thresholds
        """
        if not historical_scores:
            return {"min_quality_score": 0.7}

        # Calculate statistics
        scores = [s.overall_score for s in historical_scores]
        avg_score = sum(scores) / len(scores)
        passed_count = sum(1 for s in historical_scores if s.passed)
        current_success_rate = passed_count / len(scores)

        # Sort scores to find percentiles
        sorted_scores = sorted(scores)

        if current_success_rate < target_success_rate:
            # Need to lower threshold
            percentile_idx = int(len(sorted_scores) * target_success_rate)
            suggested_threshold = sorted_scores[min(percentile_idx, len(sorted_scores) - 1)]
        else:
            # Can afford higher threshold
            suggested_threshold = max(0.7, avg_score * 0.9)

        return {
            "min_quality_score": round(min(0.95, max(0.5, suggested_threshold)), 2),
            "current_success_rate": round(current_success_rate, 3),
            "average_score": round(avg_score, 3),
        }


@dataclass
class DecisionContext:
    """Context for ML-based decisions.
    
    Attributes:
        job_id: Job ID
        file_type: File extension/type
        content_detection: Content detection result
        historical_success_rate: Historical success rate for similar files
        parser_performance: Performance metrics for available parsers
        current_config: Current pipeline configuration
        system_load: Current system load (0-1)
        time_of_day: Current hour (0-23)
    """
    job_id: UUID
    file_type: str
    content_detection: ContentDetectionResult | None = None
    historical_success_rate: float = 0.0
    parser_performance: dict[str, Any] = field(default_factory=dict)
    current_config: QualityConfig | None = None
    system_load: float = 0.5
    time_of_day: int = 12


@dataclass
class DecisionResult:
    """Result of an ML-based decision.
    
    Attributes:
        decision_type: Type of decision made
        selected_option: Selected option/parser/strategy
        confidence: Confidence in decision (0-1)
        reasoning: Explanation of decision
        alternatives: Alternative options considered
        expected_outcome: Expected outcome metrics
    """
    decision_type: str
    selected_option: str
    confidence: float
    reasoning: str
    alternatives: list[str] = field(default_factory=list)
    expected_outcome: dict[str, Any] = field(default_factory=dict)


class AdvancedDecisionEngine(AgenticDecisionEngine):
    """Advanced decision engine with ML-based decision making.
    
    Extends the base AgenticDecisionEngine with:
    - Machine learning-based parser selection
    - Historical success tracking
    - Adaptive thresholds
    - Context-aware routing
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        use_historical_data: bool = True,
        ml_weight: float = 0.6,
    ) -> None:
        """Initialize the advanced decision engine.
        
        Args:
            llm_provider: LLM provider for decisions
            use_historical_data: Whether to use historical processing data
            ml_weight: Weight given to ML vs rule-based decisions (0-1)
        """
        super().__init__(llm_provider)
        self.use_historical_data = use_historical_data
        self.ml_weight = ml_weight
        self._history: Any | None = None

    def set_history_provider(self, history: Any) -> None:
        """Set the processing history provider.
        
        Args:
            history: ProcessingHistory instance
        """
        self._history = history

    async def decide_with_ml(
        self,
        context: DecisionContext,
    ) -> DecisionResult:
        """Make a decision using ML-enhanced reasoning.
        
        Combines LLM reasoning with historical data for optimal decisions.
        
        Args:
            context: Decision context with file info and system state
            
        Returns:
            DecisionResult with optimal choice and confidence
        """
        # Get historical recommendation
        historical_parser = None
        historical_confidence = 0.0

        if self.use_historical_data and self._history:
            try:
                content_type = (
                    context.content_detection.detected_type.value
                    if context.content_detection
                    else "unknown"
                )
                historical_parser, historical_confidence = (
                    await self._history.get_parser_recommendation(
                        content_type=content_type,
                        file_type=context.file_type,
                    )
                )
            except Exception as e:
                self.logger.warning(f"Failed to get historical recommendation: {e}")

        # Get LLM-based decision
        llm_decision = await self._get_llm_decision(context)

        # Combine decisions
        if historical_parser and historical_confidence > 0.7:
            # High confidence historical data - use it
            selected_parser = historical_parser
            confidence = historical_confidence
            reasoning = (
                f"Selected {selected_parser} based on strong historical performance "
                f"({historical_confidence:.0%} confidence). "
                f"LLM suggestion was {llm_decision.get('parser', 'unknown')}."
            )
        elif historical_parser and self.ml_weight > 0.5:
            # Blend ML and LLM decisions
            if historical_parser == llm_decision.get("parser"):
                selected_parser = historical_parser
                confidence = max(historical_confidence, 0.8)
                reasoning = f"Both ML and LLM agree on {selected_parser}"
            else:
                # Use ML weight to decide
                if self.ml_weight > 0.6:
                    selected_parser = historical_parser
                    confidence = historical_confidence * self.ml_weight
                    reasoning = f"ML-weighted selection: {selected_parser}"
                else:
                    selected_parser = llm_decision.get("parser", "docling")
                    confidence = 0.75
                    reasoning = f"LLM-weighted selection: {selected_parser}"
        else:
            # Fall back to LLM decision
            selected_parser = llm_decision.get("parser", "docling")
            confidence = llm_decision.get("confidence", 0.75)
            reasoning = llm_decision.get("reason", "LLM-based selection")

        # Get alternatives
        alternatives = ["azure_ocr", "docling"]
        alternatives = [a for a in alternatives if a != selected_parser]

        return DecisionResult(
            decision_type="parser_selection",
            selected_option=selected_parser,
            confidence=confidence,
            reasoning=reasoning,
            alternatives=alternatives,
            expected_outcome={
                "estimated_success_rate": confidence,
                "estimated_quality_score": 0.8 if selected_parser == "docling" else 0.75,
            },
        )

    async def _get_llm_decision(self, context: DecisionContext) -> dict[str, Any]:
        """Get LLM-based decision for parser selection.
        
        Args:
            context: Decision context
            
        Returns:
            Dictionary with LLM decision
        """
        system_prompt = """You are an ML-enhanced document processing expert. 
Select the optimal parser based on the file type, content detection, and system context.

Available parsers:
- docling: Best for structured text-based documents, Office files
- azure_ocr: Best for scanned documents, images, poor quality scans

Consider:
1. Content type and structure
2. Historical performance patterns
3. Current system load
4. Time of day (resource availability)

Respond with valid JSON:
{
    "parser": "docling|azure_ocr",
    "confidence": 0.95,
    "reason": "explanation"
}"""

        detection_info = ""
        if context.content_detection:
            detection_info = f"""
Content Detection:
- Type: {context.content_detection.detected_type.value}
- Has text layer: {context.content_detection.has_text_layer}
- Text ratio: {context.content_detection.text_statistics.text_ratio:.2f}
- Confidence: {context.content_detection.confidence}
"""

        user_prompt = f"""Select the best parser for:

File Type: {context.file_type}
{detection_info}
System Load: {context.system_load:.0%}
Historical Success Rate: {context.historical_success_rate:.0%}
Time: {context.time_of_day}:00

Decision:"""

        try:
            response = await self.llm.chat_completion(
                messages=[
                    ChatMessage.system(system_prompt),
                    ChatMessage.user(user_prompt),
                ],
                temperature=0.1,
                max_tokens=300,
            )

            import json
            return json.loads(response.content)

        except Exception as e:
            self.logger.error(f"LLM decision failed: {e}")
            return {"parser": "docling", "confidence": 0.7, "reason": "Fallback to default"}

    async def adaptive_parser_selection(
        self,
        detection_result: ContentDetectionResult,
        file_name: str,
    ) -> ParserSelection:
        """Parser selection with adaptive learning.
        
        Uses both detection results and historical performance
        to make optimal parser selections.
        
        Args:
            detection_result: Content detection results
            file_name: Name of the file
            
        Returns:
            ParserSelection with adaptive choice
        """
        # Build context
        from pathlib import Path
        from uuid import uuid4

        file_type = Path(file_name).suffix.lower().lstrip(".") or "unknown"

        context = DecisionContext(
            job_id=uuid4(),  # Placeholder
            file_type=file_type,
            content_detection=detection_result,
            system_load=0.5,  # Default, should come from monitoring
            time_of_day=datetime.utcnow().hour,
        )

        # Get ML-based decision
        result = await self.decide_with_ml(context)

        # Determine fallback
        fallback = "azure_ocr" if result.selected_option == "docling" else "docling"

        return ParserSelection(
            parser=result.selected_option,
            reason=result.reasoning,
            confidence=result.confidence,
            fallback_parser=fallback,
            preprocessing_steps=[],
        )

    async def context_aware_routing(
        self,
        data: Any,
        destinations: list[Any],
        system_load: float,
    ) -> list[Any]:
        """Make context-aware routing decisions.
        
        Selects optimal destinations based on:
        - Current system load
        - Destination health
        - Data characteristics
        
        Args:
            data: Data to route
            destinations: Available destinations
            system_load: Current system load
            
        Returns:
            Prioritized list of destinations
        """
        if system_load > 0.8:
            # High load - prioritize reliable destinations
            # Sort by reliability (would use historical data in practice)
            return sorted(destinations, key=lambda d: d.name if hasattr(d, "name") else "")

        if system_load < 0.3:
            # Low load - can use all destinations
            return destinations

        # Normal load - use default order
        return destinations

    async def adaptive_threshold_adjustment(
        self,
        file_type: str,
        content_type: str,
        base_threshold: float,
    ) -> float:
        """Adaptively adjust quality threshold based on context.
        
        Args:
            file_type: File type
            content_type: Content type
            base_threshold: Base quality threshold
            
        Returns:
            Adjusted threshold
        """
        adjusted = base_threshold

        # Adjust based on historical performance
        if self._history:
            try:
                stats = await self._history.get_content_type_stats()
                pattern = stats.get(content_type)
                if pattern and pattern.total_count > 10:
                    # Adjust toward historical average
                    historical_avg = pattern.avg_quality_score
                    adjusted = (adjusted * 0.7) + (historical_avg * 0.3)
            except Exception as e:
                self.logger.warning(f"Failed to get content type stats: {e}")

        # Adjust for challenging file types
        challenging_types = ["tiff", "bmp", "gif", "zip"]
        if file_type.lower() in challenging_types:
            adjusted *= 0.9  # Lower threshold for difficult formats

        # Ensure bounds
        return max(0.5, min(0.95, adjusted))
