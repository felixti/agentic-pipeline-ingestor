"""GenAI-specific span attributes for OpenTelemetry.

This module defines GenAI-specific span attributes following the OpenTelemetry
semantic conventions for Generative AI operations.

Reference: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class GenAISpanAttributes:
    """GenAI-specific span attributes per OpenTelemetry semantic conventions.
    
    These attributes provide standardized instrumentation for LLM operations,
    enabling observability across different AI systems and providers.
    
    Attributes:
        GEN_AI_SYSTEM: The Generative AI product as identified by the client or server instrumentation.
        GEN_AI_REQUEST_MODEL: The name of the GenAI model a request is being made to.
        GEN_AI_REQUEST_MAX_TOKENS: The maximum number of tokens the model generates for a request.
        GEN_AI_REQUEST_TEMPERATURE: The temperature setting for the model request.
        GEN_AI_REQUEST_TOP_P: The top_p sampling setting for the model request.
        GEN_AI_USAGE_INPUT_TOKENS: The number of tokens used in the prompt.
        GEN_AI_USAGE_OUTPUT_TOKENS: The number of tokens used in the completion.
        GEN_AI_RESPONSE_FINISH_REASON: The reason the model stopped generating tokens.
        GEN_AI_RESPONSE_MODEL: The model used for the response.
    """
    
    # System attributes
    GEN_AI_SYSTEM = "gen_ai.system"
    GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
    
    # Request attributes
    GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
    GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
    GEN_AI_REQUEST_PRESENCE_PENALTY = "gen_ai.request.presence_penalty"
    GEN_AI_REQUEST_FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
    GEN_AI_REQUEST_STOP_SEQUENCES = "gen_ai.request.stop_sequences"
    
    # Usage attributes
    GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    GEN_AI_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"
    
    # Response attributes
    GEN_AI_RESPONSE_FINISH_REASON = "gen_ai.response.finish_reason"
    GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
    GEN_AI_RESPONSE_ID = "gen_ai.response.id"
    
    # Custom attributes for our pipeline
    GEN_AI_PIPELINE_STAGE = "gen_ai.pipeline.stage"
    GEN_AI_DECISION_TYPE = "gen_ai.decision.type"
    GEN_AI_FALLBACK_USED = "gen_ai.fallback.used"
    GEN_AI_RETRY_COUNT = "gen_ai.retry.count"
    GEN_AI_LATENCY_MS = "gen_ai.latency_ms"


@dataclass
class GenAISpanContext:
    """Context for creating GenAI spans.
    
    This dataclass holds all the information needed to create a properly
    instrumented GenAI span following OpenTelemetry conventions.
    
    Attributes:
        system: The AI system (e.g., "azure_openai", "openrouter", "anthropic")
        operation: The operation name (e.g., "chat_completion", "embedding")
        request_model: The model requested
        max_tokens: Maximum tokens setting
        temperature: Temperature setting
        top_p: Top-p sampling setting
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        finish_reason: Why the generation stopped
        response_model: The actual model used (may differ from request)
        response_id: Unique response identifier
        latency_ms: Request latency in milliseconds
        custom_attributes: Additional custom attributes
    """
    
    system: str
    operation: str = "chat_completion"
    request_model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    response_model: Optional[str] = None
    response_id: Optional[str] = None
    latency_ms: Optional[float] = None
    custom_attributes: Optional[Dict[str, Any]] = None
    
    def to_attributes(self) -> Dict[str, Any]:
        """Convert context to OpenTelemetry attribute dictionary.
        
        Returns:
            Dictionary of span attributes following GenAI conventions
        """
        attrs: Dict[str, Any] = {
            GenAISpanAttributes.GEN_AI_SYSTEM: self.system,
            GenAISpanAttributes.GEN_AI_OPERATION_NAME: self.operation,
        }
        
        if self.request_model:
            attrs[GenAISpanAttributes.GEN_AI_REQUEST_MODEL] = self.request_model
        if self.max_tokens is not None:
            attrs[GenAISpanAttributes.GEN_AI_REQUEST_MAX_TOKENS] = self.max_tokens
        if self.temperature is not None:
            attrs[GenAISpanAttributes.GEN_AI_REQUEST_TEMPERATURE] = self.temperature
        if self.top_p is not None:
            attrs[GenAISpanAttributes.GEN_AI_REQUEST_TOP_P] = self.top_p
        if self.input_tokens is not None:
            attrs[GenAISpanAttributes.GEN_AI_USAGE_INPUT_TOKENS] = self.input_tokens
        if self.output_tokens is not None:
            attrs[GenAISpanAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] = self.output_tokens
        if self.finish_reason:
            attrs[GenAISpanAttributes.GEN_AI_RESPONSE_FINISH_REASON] = self.finish_reason
        if self.response_model:
            attrs[GenAISpanAttributes.GEN_AI_RESPONSE_MODEL] = self.response_model
        if self.response_id:
            attrs[GenAISpanAttributes.GEN_AI_RESPONSE_ID] = self.response_id
        if self.latency_ms is not None:
            attrs[GenAISpanAttributes.GEN_AI_LATENCY_MS] = self.latency_ms
        
        # Add custom attributes
        if self.custom_attributes:
            attrs.update(self.custom_attributes)
        
        return attrs
    
    @classmethod
    def from_litellm_response(
        cls,
        system: str,
        response: Any,
        request_params: Dict[str, Any],
        latency_ms: float,
    ) -> "GenAISpanContext":
        """Create context from a litellm response.
        
        Args:
            system: The AI system used
            response: The litellm response object
            request_params: The original request parameters
            latency_ms: Request latency
            
        Returns:
            GenAISpanContext populated from the response
        """
        usage = getattr(response, 'usage', None) or {}
        
        return cls(
            system=system,
            operation="chat_completion",
            request_model=request_params.get("model"),
            max_tokens=request_params.get("max_tokens"),
            temperature=request_params.get("temperature"),
            top_p=request_params.get("top_p"),
            input_tokens=getattr(usage, 'prompt_tokens', None) or usage.get('prompt_tokens'),
            output_tokens=getattr(usage, 'completion_tokens', None) or usage.get('completion_tokens'),
            finish_reason=response.choices[0].finish_reason if response.choices else None,
            response_model=getattr(response, 'model', None),
            response_id=getattr(response, 'id', None),
            latency_ms=latency_ms,
        )


# System identifiers for common providers
class GenAISystems:
    """Standard system identifiers for GenAI providers."""
    
    AZURE_OPENAI = "azure_openai"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    COHERE = "cohere"
    OLLAMA = "ollama"
    VLLM = "vllm"
    HUGGING_FACE = "huggingface"
    GOOGLE = "google"
    COGNEE = "cognee"
