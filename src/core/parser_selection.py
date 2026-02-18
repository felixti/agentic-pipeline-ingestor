"""Parser selection service for routing documents to optimal parsers."""

from typing import List, Optional

from src.core.content_detection.models import (
    ContentAnalysisResult,
    ContentDetectionRecord,
    ContentType,
)


class ParserSelection:
    """Result of parser selection."""
    
    def __init__(
        self,
        primary_parser: str,
        fallback_parser: Optional[str],
        rationale: str,
        overridden: bool = False
    ):
        """Initialize parser selection.
        
        Args:
            primary_parser: Primary parser to use
            fallback_parser: Fallback parser (None if no fallback)
            rationale: Explanation for selection
            overridden: Whether explicit config overrode detection
        """
        self.primary_parser = primary_parser
        self.fallback_parser = fallback_parser
        self.rationale = rationale
        self.overridden = overridden
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "primary_parser": self.primary_parser,
            "fallback_parser": self.fallback_parser,
            "rationale": self.rationale,
            "overridden": self.overridden
        }


class ParserConfig:
    """Explicit parser configuration."""
    
    def __init__(
        self,
        primary_parser: str,
        fallback_parser: Optional[str] = None,
        force_ocr: bool = False
    ):
        """Initialize parser config.
        
        Args:
            primary_parser: Explicit primary parser
            fallback_parser: Explicit fallback parser
            force_ocr: Force OCR even for text-based docs
        """
        self.primary_parser = primary_parser
        self.fallback_parser = fallback_parser
        self.force_ocr = force_ocr


class ParserSelector:
    """Selects optimal parser based on content detection results."""
    
    # Parser configuration
    PARSER_DOC = "docling"
    PARSER_OCR = "azure_ocr"
    
    # Confidence threshold for conservative strategy
    LOW_CONFIDENCE_THRESHOLD = 0.70
    
    @classmethod
    def select_parser(
        cls,
        detection_result: ContentAnalysisResult,
        explicit_config: Optional[ParserConfig] = None
    ) -> ParserSelection:
        """Select parser based on detection result.
        
        Selection logic:
        1. If explicit config provided, use it (highest priority)
        2. If confidence < 0.70, use conservative strategy (both parsers)
        3. Based on content type:
           - TEXT_BASED: Docling primary, Azure OCR fallback
           - SCANNED: Azure OCR primary, Docling fallback
           - MIXED: Docling primary with OCR fallback
        
        Args:
            detection_result: Content detection result
            explicit_config: Optional explicit parser configuration
            
        Returns:
            Parser selection result
        """
        # Priority 1: Explicit configuration override
        if explicit_config is not None:
            return cls._apply_explicit_config(explicit_config, detection_result)
        
        # Priority 2: Low confidence - use conservative approach
        if detection_result.confidence < cls.LOW_CONFIDENCE_THRESHOLD:
            return ParserSelection(
                primary=cls.PARSER_DOC,
                fallback=cls.PARSER_OCR,
                rationale=(
                    f"Low detection confidence ({detection_result.confidence:.2f}), "
                    f"using conservative strategy with both parsers"
                )
            )
        
        # Priority 3: Content-based selection
        if detection_result.content_type == ContentType.TEXT_BASED:
            return cls._select_for_text_based(detection_result)
        elif detection_result.content_type == ContentType.SCANNED:
            return cls._select_for_scanned(detection_result)
        else:  # MIXED
            return cls._select_for_mixed(detection_result)
    
    @classmethod
    def _apply_explicit_config(
        cls,
        config: ParserConfig,
        detection_result: ContentAnalysisResult
    ) -> ParserSelection:
        """Apply explicit parser configuration.
        
        Args:
            config: Explicit parser configuration
            detection_result: Detection result for context
            
        Returns:
            Parser selection
        """
        if config.force_ocr:
            return ParserSelection(
                primary=cls.PARSER_OCR,
                fallback=config.fallback_parser or cls.PARSER_DOC,
                rationale=(
                    f"OCR forced by user configuration. "
                    f"Detected type was {detection_result.content_type} "
                    f"(confidence: {detection_result.confidence:.2f})"
                ),
                overridden=True
            )
        
        return ParserSelection(
            primary=config.primary_parser,
            fallback=config.fallback_parser,
            rationale=(
                f"User-specified configuration. "
                f"Detected type was {detection_result.content_type} "
                f"(confidence: {detection_result.confidence:.2f})"
            ),
            overridden=True
        )
    
    @classmethod
    def _select_for_text_based(
        cls,
        detection_result: ContentAnalysisResult
    ) -> ParserSelection:
        """Select parser for text-based document.
        
        Args:
            detection_result: Detection result
            
        Returns:
            Parser selection
        """
        return ParserSelection(
            primary=cls.PARSER_DOC,
            fallback=cls.PARSER_OCR,
            rationale=(
                f"Text-based PDF detected (confidence: {detection_result.confidence:.2f}, "
                f"text ratio: {detection_result.text_statistics.average_chars_per_page:.0f} chars/page). "
                f"Using Docling for fast text extraction."
            )
        )
    
    @classmethod
    def _select_for_scanned(
        cls,
        detection_result: ContentAnalysisResult
    ) -> ParserSelection:
        """Select parser for scanned document.
        
        Args:
            detection_result: Detection result
            
        Returns:
            Parser selection
        """
        return ParserSelection(
            primary=cls.PARSER_OCR,
            fallback=cls.PARSER_DOC,
            rationale=(
                f"Scanned PDF detected (confidence: {detection_result.confidence:.2f}, "
                f"image ratio: {detection_result.image_statistics.image_area_ratio:.1%}). "
                f"Using Azure OCR for image-based text extraction."
            )
        )
    
    @classmethod
    def _select_for_mixed(
        cls,
        detection_result: ContentAnalysisResult
    ) -> ParserSelection:
        """Select parser for mixed content document.
        
        Args:
            detection_result: Detection result
            
        Returns:
            Parser selection
        """
        return ParserSelection(
            primary=cls.PARSER_DOC,
            fallback=cls.PARSER_OCR,
            rationale=(
                f"Mixed content detected (confidence: {detection_result.confidence:.2f}, "
                f"text ratio: {detection_result.text_statistics.total_characters} chars, "
                f"image ratio: {detection_result.image_statistics.image_area_ratio:.1%}). "
                f"Using Docling with OCR fallback for comprehensive extraction."
            )
        )
    
    @classmethod
    def estimate_processing_time(
        cls,
        selection: ParserSelection,
        page_count: int
    ) -> str:
        """Estimate processing time.
        
        Args:
            selection: Parser selection
            page_count: Number of pages
            
        Returns:
            Estimated time as human-readable string
        """
        # Rough estimates:
        # Docling: ~1s per page
        # Azure OCR: ~5s per page
        
        if selection.primary_parser == cls.PARSER_DOC:
            seconds = page_count * 1
        else:
            seconds = page_count * 5
        
        # Add fallback time if needed
        if selection.fallback_parser:
            seconds += page_count * 0.5  # Partial fallback
        
        if seconds < 60:
            return f"{seconds}s"
        else:
            minutes = seconds // 60
            remaining_seconds = seconds % 60
            return f"{minutes}m {remaining_seconds}s"
    
    @classmethod
    def estimate_cost(
        cls,
        selection: ParserSelection,
        page_count: int
    ) -> float:
        """Estimate processing cost in USD.
        
        Args:
            selection: Parser selection
            page_count: Number of pages
            
        Returns:
            Estimated cost in USD
        """
        # Rough cost estimates:
        # Docling: $0.001 per page (local processing, minimal cost)
        # Azure OCR: $0.01 per page
        
        if selection.primary_parser == cls.PARSER_DOC:
            cost = page_count * 0.001
        else:
            cost = page_count * 0.01
        
        # Fallback adds partial cost
        if selection.fallback_parser == cls.PARSER_OCR:
            cost += page_count * 0.005
        elif selection.fallback_parser == cls.PARSER_DOC:
            cost += page_count * 0.0005
        
        return round(cost, 3)


def select_parser_for_job(
    detection_result: ContentAnalysisResult,
    explicit_config: Optional[dict] = None
) -> ParserSelection:
    """Convenience function to select parser for a job.
    
    Args:
        detection_result: Content detection result
        explicit_config: Optional explicit configuration dict with keys:
            - primary_parser: str
            - fallback_parser: str (optional)
            - force_ocr: bool (optional)
            
    Returns:
        Parser selection
    """
    config = None
    if explicit_config:
        config = ParserConfig(
            primary_parser=explicit_config.get("primary_parser", "docling"),
            fallback_parser=explicit_config.get("fallback_parser"),
            force_ocr=explicit_config.get("force_ocr", False)
        )
    
    return ParserSelector.select_parser(detection_result, config)
