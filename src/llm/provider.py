"""LLM Provider using litellm for multi-provider abstraction.

This module provides a unified interface for LLM operations using litellm,
supporting Azure OpenAI as primary and OpenRouter as fallback.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

# Try to import litellm, provide fallback if not available
try:
    import litellm
    from litellm import acompletion, Router
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    litellm = None  # type: ignore

from src.llm.config import LLMConfig, load_llm_config
from src.observability.genai_spans import GenAISpanContext, GenAISystems
from src.observability.tracing import get_tracer
from src.observability.metrics import get_metrics_manager

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """A message in a chat conversation.
    
    Attributes:
        role: Message role (system, user, assistant, tool)
        content: Message content
        name: Optional name for the message sender
        tool_calls: Optional tool calls for assistant messages
    """
    role: str
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for litellm.
        
        Returns:
            Dictionary representation of the message
        """
        msg: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            msg["name"] = self.name
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessage":
        """Create from dictionary.
        
        Args:
            data: Dictionary containing message data
            
        Returns:
            ChatMessage instance
        """
        return cls(
            role=data["role"],
            content=data["content"],
            name=data.get("name"),
            tool_calls=data.get("tool_calls"),
        )
    
    @classmethod
    def system(cls, content: str) -> "ChatMessage":
        """Create a system message.
        
        Args:
            content: Message content
            
        Returns:
            System ChatMessage
        """
        return cls(role="system", content=content)
    
    @classmethod
    def user(cls, content: str) -> "ChatMessage":
        """Create a user message.
        
        Args:
            content: Message content
            
        Returns:
            User ChatMessage
        """
        return cls(role="user", content=content)
    
    @classmethod
    def assistant(cls, content: str) -> "ChatMessage":
        """Create an assistant message.
        
        Args:
            content: Message content
            
        Returns:
            Assistant ChatMessage
        """
        return cls(role="assistant", content=content)


@dataclass
class ChatCompletionResponse:
    """Response from a chat completion request.
    
    Attributes:
        id: Unique response identifier
        model: Model used for the completion
        content: Generated content
        role: Role of the response (usually "assistant")
        finish_reason: Reason for completion finish
        usage: Token usage information
        raw_response: Original response object
    """
    id: str
    model: str
    content: str
    role: str = "assistant"
    finish_reason: str = "stop"
    usage: Dict[str, int] = None  # type: ignore
    raw_response: Any = None
    
    def __post_init__(self) -> None:
        """Initialize default usage if not provided."""
        if self.usage is None:
            self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    @classmethod
    def from_litellm(cls, response: Any) -> "ChatCompletionResponse":
        """Create from litellm response.
        
        Args:
            response: litellm response object
            
        Returns:
            ChatCompletionResponse instance
        """
        choice = response.choices[0]
        return cls(
            id=response.id,
            model=response.model,
            content=choice.message.content or "",
            role=choice.message.role,
            finish_reason=choice.finish_reason or "stop",
            usage=dict(response.usage) if response.usage else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            raw_response=response,
        )
    
    def to_json(self) -> str:
        """Convert content to JSON if it's a JSON string.
        
        Returns:
            JSON string or original content
        """
        try:
            parsed = json.loads(self.content)
            return json.dumps(parsed, indent=2)
        except json.JSONDecodeError:
            return self.content


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""
    pass


class LLMProviderUnavailableException(LLMProviderError):
    """Exception raised when all LLM providers fail."""
    pass


class LLMProvider:
    """LLM Provider using litellm for multi-provider support.
    
    This class provides a unified interface for LLM operations with
    automatic fallback chains between providers (Azure OpenAI → OpenRouter).
    
    Example:
        >>> config = load_llm_config()
        >>> provider = LLMProvider(config)
        >>> response = await provider.chat_completion(
        ...     messages=[ChatMessage.system("You are helpful"), ChatMessage.user("Hello")],
        ...     temperature=0.7
        ... )
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize the LLM provider.
        
        Args:
            config: LLM configuration. If None, loads from default config file.
            
        Raises:
            ImportError: If litellm is not installed
        """
        if not LITELLM_AVAILABLE:
            raise ImportError(
                "litellm is not installed. "
                "Install with: pip install litellm"
            )
        
        self.config = config or load_llm_config()
        self._router: Optional[Any] = None
        self._initialize_router()
    
    def _initialize_router(self) -> None:
        """Initialize the litellm router with fallback chains."""
        model_list = self.config.to_litellm_model_list()
        
        if not model_list:
            logger.warning("No models configured in LLM config")
            return
        
        try:
            self._router = Router(
                model_list=model_list,
                default_fallbacks=True,
                cooldown_time=60,  # Cooldown period for failed models
                num_retries=self.config.retry_attempts,
                retry_after=5,
                timeout=self.config.retry_timeout,
            )
            logger.info(f"Initialized LLM router with {len(model_list)} model entries")
        except Exception as e:
            logger.error(f"Failed to initialize LLM router: {e}")
            raise LLMProviderError(f"Router initialization failed: {e}") from e
    
    async def chat_completion(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[Dict[str, str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> ChatCompletionResponse:
        """Generate a chat completion.
        
        This method routes requests through the configured fallback chain:
        Azure GPT-4 → OpenRouter Claude-3 → Azure GPT-3.5
        
        Args:
            messages: List of chat messages
            model: Model group to use (e.g., "agentic-decisions", "enrichment")
            temperature: Sampling temperature (0.0 - 2.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty (-2.0 - 2.0)
            presence_penalty: Presence penalty (-2.0 - 2.0)
            response_format: Response format (e.g., {"type": "json_object"})
            tools: List of tools for function calling
            tool_choice: Tool choice configuration
            **kwargs: Additional provider-specific parameters
            
        Returns:
            ChatCompletionResponse with generated content
            
        Raises:
            LLMProviderUnavailableException: If all providers fail
        """
        import time
        
        if not self._router:
            raise LLMProviderError("LLM router not initialized")
        
        model = model or "agentic-decisions"
        temperature = temperature or self.config.default_temperature
        max_tokens = max_tokens or self.config.default_max_tokens
        
        # Convert messages to litellm format
        litellm_messages = [m.to_dict() for m in messages]
        
        # Build request parameters
        request_params: Dict[str, Any] = {
            "model": model,
            "messages": litellm_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if top_p is not None:
            request_params["top_p"] = top_p
        if frequency_penalty is not None:
            request_params["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            request_params["presence_penalty"] = presence_penalty
        if response_format is not None:
            request_params["response_format"] = response_format
        if tools is not None:
            request_params["tools"] = tools
        if tool_choice is not None:
            request_params["tool_choice"] = tool_choice
        
        request_params.update(kwargs)
        
        # Determine system type for span attributes
        system = self._get_system_from_model(model)
        
        # Start timing for metrics
        start_time = time.time()
        
        # Create tracer and start GenAI span
        tracer = get_tracer("llm")
        
        with tracer.start_as_current_span(
            name="llm.chat_completion",
            attributes={
                "gen_ai.operation.name": "chat_completion",
                "gen_ai.system": system,
                "gen_ai.request.model": model,
                "gen_ai.request.max_tokens": max_tokens or 0,
                "gen_ai.request.temperature": temperature or 0.7,
            },
        ) as span:
            try:
                # Use router for automatic fallback
                response = await self._router.acompletion(**request_params)
                
                chat_response = ChatCompletionResponse.from_litellm(response)
                
                # Calculate latency
                latency = time.time() - start_time
                
                # Update span with response attributes
                span.set_attribute("gen_ai.response.model", chat_response.model)
                span.set_attribute("gen_ai.response.id", chat_response.id)
                span.set_attribute("gen_ai.response.finish_reason", chat_response.finish_reason)
                
                # Add token usage to span
                usage = chat_response.usage
                if usage:
                    span.set_attribute("gen_ai.usage.input_tokens", usage.get("prompt_tokens", 0))
                    span.set_attribute("gen_ai.usage.output_tokens", usage.get("completion_tokens", 0))
                    span.set_attribute("gen_ai.usage.total_tokens", usage.get("total_tokens", 0))
                
                # Record metrics
                metrics = get_metrics_manager()
                metrics.record_llm_request(
                    model=chat_response.model,
                    operation="chat_completion",
                    status="success",
                    latency=latency,
                    input_tokens=usage.get("prompt_tokens", 0) if usage else 0,
                    output_tokens=usage.get("completion_tokens", 0) if usage else 0,
                )
                
                logger.debug(
                    f"LLM completion successful: model={chat_response.model}, "
                    f"tokens={chat_response.usage.get('total_tokens', 0)}"
                )
                
                return chat_response
                
            except Exception as e:
                latency = time.time() - start_time
                
                # Record error in span
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                span.record_exception(e)
                
                # Determine error type
                error_type = "error"
                if "rate limit" in str(e).lower():
                    error_type = "rate_limited"
                
                # Record error metrics
                metrics = get_metrics_manager()
                metrics.record_llm_request(
                    model=model,
                    operation="chat_completion",
                    status=error_type,
                    latency=latency,
                )
                
                logger.error(f"All LLM providers failed: {e}")
                raise LLMProviderUnavailableException(
                    f"Failed to get completion from any provider: {e}"
                ) from e
    
    async def simple_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Simple completion with a single prompt.
        
        Convenience method for simple use cases.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters for chat_completion
            
        Returns:
            Generated text content
        """
        messages: List[ChatMessage] = []
        
        if system_prompt:
            messages.append(ChatMessage.system(system_prompt))
        
        messages.append(ChatMessage.user(prompt))
        
        response = await self.chat_completion(messages=messages, **kwargs)
        return response.content
    
    async def json_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Get completion as parsed JSON.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters
            
        Returns:
            Parsed JSON response as dictionary
            
        Raises:
            json.JSONDecodeError: If response is not valid JSON
        """
        # Add JSON instruction to system prompt
        json_instruction = "You must respond with valid JSON only."
        if system_prompt:
            system_prompt = f"{system_prompt}\n\n{json_instruction}"
        else:
            system_prompt = json_instruction
        
        content = await self.simple_completion(
            prompt=prompt,
            system_prompt=system_prompt,
            response_format={"type": "json_object"},
            **kwargs,
        )
        
        # Parse and return JSON
        return json.loads(content)
    
    def _get_system_from_model(self, model: str) -> str:
        """Determine the AI system from model name.
        
        Args:
            model: Model name or group
            
        Returns:
            System identifier (azure_openai, openrouter, etc.)
        """
        model_lower = model.lower()
        
        if "azure" in model_lower:
            return GenAISystems.AZURE_OPENAI
        elif "openrouter" in model_lower:
            return GenAISystems.OPENROUTER
        elif "claude" in model_lower:
            return GenAISystems.ANTHROPIC
        elif "openai" in model_lower:
            return GenAISystems.OPENAI
        elif "ollama" in model_lower:
            return GenAISystems.OLLAMA
        else:
            return GenAISystems.AZURE_OPENAI  # Default
    
    def get_available_models(self) -> List[str]:
        """Get list of available model groups.
        
        Returns:
            List of configured model group names
        """
        return [router.model_name for router in self.config.routers]
    
    async def health_check(self, model: Optional[str] = None) -> Dict[str, Any]:
        """Check health of LLM providers.
        
        Args:
            model: Specific model group to check, or None for all
            
        Returns:
            Dictionary with health status information
        """
        results: Dict[str, Any] = {
            "healthy": False,
            "models": {},
        }
        
        if not self._router:
            results["error"] = "Router not initialized"
            return results
        
        test_messages = [ChatMessage.user("Say 'healthy' and nothing else.")]
        
        models_to_check = [model] if model else self.get_available_models()
        
        for model_name in models_to_check:
            try:
                response = await self.chat_completion(
                    messages=test_messages,
                    model=model_name,
                    max_tokens=10,
                )
                results["models"][model_name] = {
                    "healthy": True,
                    "model_used": response.model,
                }
            except Exception as e:
                results["models"][model_name] = {
                    "healthy": False,
                    "error": str(e),
                }
        
        # Overall health
        results["healthy"] = all(
            m.get("healthy", False) for m in results["models"].values()
        )
        
        return results


# Convenience function for simple use cases
async def get_completion(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Get a simple text completion.
    
    Convenience function for one-off completions without
    managing provider instances.
    
    Args:
        prompt: User prompt
        system_prompt: Optional system prompt
        model: Model group to use
        **kwargs: Additional parameters
        
    Returns:
        Generated text
    """
    config = load_llm_config()
    provider = LLMProvider(config)
    return await provider.simple_completion(prompt, system_prompt, model=model, **kwargs)
