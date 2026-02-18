"""Unit tests for GenAI-specific span attributes."""

from unittest.mock import MagicMock

import pytest

from src.observability.genai_spans import (
    GenAISpanAttributes,
    GenAISpanContext,
    GenAISystems,
)


@pytest.mark.unit
class TestGenAISpanAttributes:
    """Tests for GenAISpanAttributes class."""

    def test_system_attributes(self):
        """Test system attribute constants."""
        assert GenAISpanAttributes.GEN_AI_SYSTEM == "gen_ai.system"
        assert GenAISpanAttributes.GEN_AI_OPERATION_NAME == "gen_ai.operation.name"

    def test_request_attributes(self):
        """Test request attribute constants."""
        assert GenAISpanAttributes.GEN_AI_REQUEST_MODEL == "gen_ai.request.model"
        assert GenAISpanAttributes.GEN_AI_REQUEST_MAX_TOKENS == "gen_ai.request.max_tokens"
        assert GenAISpanAttributes.GEN_AI_REQUEST_TEMPERATURE == "gen_ai.request.temperature"
        assert GenAISpanAttributes.GEN_AI_REQUEST_TOP_P == "gen_ai.request.top_p"
        assert GenAISpanAttributes.GEN_AI_REQUEST_PRESENCE_PENALTY == "gen_ai.request.presence_penalty"
        assert GenAISpanAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY == "gen_ai.request.frequency_penalty"
        assert GenAISpanAttributes.GEN_AI_REQUEST_STOP_SEQUENCES == "gen_ai.request.stop_sequences"

    def test_usage_attributes(self):
        """Test usage attribute constants."""
        assert GenAISpanAttributes.GEN_AI_USAGE_INPUT_TOKENS == "gen_ai.usage.input_tokens"
        assert GenAISpanAttributes.GEN_AI_USAGE_OUTPUT_TOKENS == "gen_ai.usage.output_tokens"
        assert GenAISpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS == "gen_ai.usage.total_tokens"

    def test_response_attributes(self):
        """Test response attribute constants."""
        assert GenAISpanAttributes.GEN_AI_RESPONSE_FINISH_REASON == "gen_ai.response.finish_reason"
        assert GenAISpanAttributes.GEN_AI_RESPONSE_MODEL == "gen_ai.response.model"
        assert GenAISpanAttributes.GEN_AI_RESPONSE_ID == "gen_ai.response.id"

    def test_custom_pipeline_attributes(self):
        """Test custom pipeline attribute constants."""
        assert GenAISpanAttributes.GEN_AI_PIPELINE_STAGE == "gen_ai.pipeline.stage"
        assert GenAISpanAttributes.GEN_AI_DECISION_TYPE == "gen_ai.decision.type"
        assert GenAISpanAttributes.GEN_AI_FALLBACK_USED == "gen_ai.fallback.used"
        assert GenAISpanAttributes.GEN_AI_RETRY_COUNT == "gen_ai.retry.count"
        assert GenAISpanAttributes.GEN_AI_LATENCY_MS == "gen_ai.latency_ms"


@pytest.mark.unit
class TestGenAISystems:
    """Tests for GenAISystems class."""

    def test_azure_openai_system(self):
        """Test Azure OpenAI system identifier."""
        assert GenAISystems.AZURE_OPENAI == "azure_openai"

    def test_openai_system(self):
        """Test OpenAI system identifier."""
        assert GenAISystems.OPENAI == "openai"

    def test_anthropic_system(self):
        """Test Anthropic system identifier."""
        assert GenAISystems.ANTHROPIC == "anthropic"

    def test_openrouter_system(self):
        """Test OpenRouter system identifier."""
        assert GenAISystems.OPENROUTER == "openrouter"

    def test_cohere_system(self):
        """Test Cohere system identifier."""
        assert GenAISystems.COHERE == "cohere"

    def test_ollama_system(self):
        """Test Ollama system identifier."""
        assert GenAISystems.OLLAMA == "ollama"

    def test_vllm_system(self):
        """Test vLLM system identifier."""
        assert GenAISystems.VLLM == "vllm"

    def test_hugging_face_system(self):
        """Test Hugging Face system identifier."""
        assert GenAISystems.HUGGING_FACE == "huggingface"

    def test_google_system(self):
        """Test Google system identifier."""
        assert GenAISystems.GOOGLE == "google"

    def test_cognee_system(self):
        """Test Cognee system identifier."""
        assert GenAISystems.COGNEE == "cognee"


@pytest.mark.unit
class TestGenAISpanContext:
    """Tests for GenAISpanContext dataclass."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        ctx = GenAISpanContext(system="azure_openai")

        assert ctx.system == "azure_openai"
        assert ctx.operation == "chat_completion"
        assert ctx.request_model is None
        assert ctx.max_tokens is None
        assert ctx.temperature is None
        assert ctx.top_p is None
        assert ctx.input_tokens is None
        assert ctx.output_tokens is None
        assert ctx.finish_reason is None
        assert ctx.response_model is None
        assert ctx.response_id is None
        assert ctx.latency_ms is None
        assert ctx.custom_attributes is None

    def test_init_with_all_values(self):
        """Test initialization with all values."""
        ctx = GenAISpanContext(
            system="openai",
            operation="embedding",
            request_model="gpt-4",
            max_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            input_tokens=100,
            output_tokens=50,
            finish_reason="stop",
            response_model="gpt-4-1106-preview",
            response_id="resp_123",
            latency_ms=1500.5,
            custom_attributes={"custom_key": "custom_value"},
        )

        assert ctx.system == "openai"
        assert ctx.operation == "embedding"
        assert ctx.request_model == "gpt-4"
        assert ctx.max_tokens == 2048
        assert ctx.temperature == 0.7
        assert ctx.top_p == 0.9
        assert ctx.input_tokens == 100
        assert ctx.output_tokens == 50
        assert ctx.finish_reason == "stop"
        assert ctx.response_model == "gpt-4-1106-preview"
        assert ctx.response_id == "resp_123"
        assert ctx.latency_ms == 1500.5
        assert ctx.custom_attributes == {"custom_key": "custom_value"}

    def test_to_attributes_basic(self):
        """Test converting to attributes with basic values."""
        ctx = GenAISpanContext(system="azure_openai")
        attrs = ctx.to_attributes()

        assert attrs[GenAISpanAttributes.GEN_AI_SYSTEM] == "azure_openai"
        assert attrs[GenAISpanAttributes.GEN_AI_OPERATION_NAME] == "chat_completion"

    def test_to_attributes_with_all_values(self):
        """Test converting to attributes with all values."""
        ctx = GenAISpanContext(
            system="openai",
            operation="chat_completion",
            request_model="gpt-4",
            max_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            input_tokens=100,
            output_tokens=50,
            finish_reason="stop",
            response_model="gpt-4-1106-preview",
            response_id="resp_123",
            latency_ms=1500.5,
            custom_attributes={"pipeline.stage": "decision"},
        )

        attrs = ctx.to_attributes()

        assert attrs[GenAISpanAttributes.GEN_AI_SYSTEM] == "openai"
        assert attrs[GenAISpanAttributes.GEN_AI_OPERATION_NAME] == "chat_completion"
        assert attrs[GenAISpanAttributes.GEN_AI_REQUEST_MODEL] == "gpt-4"
        assert attrs[GenAISpanAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 2048
        assert attrs[GenAISpanAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.7
        assert attrs[GenAISpanAttributes.GEN_AI_REQUEST_TOP_P] == 0.9
        assert attrs[GenAISpanAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 100
        assert attrs[GenAISpanAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 50
        assert attrs[GenAISpanAttributes.GEN_AI_RESPONSE_FINISH_REASON] == "stop"
        assert attrs[GenAISpanAttributes.GEN_AI_RESPONSE_MODEL] == "gpt-4-1106-preview"
        assert attrs[GenAISpanAttributes.GEN_AI_RESPONSE_ID] == "resp_123"
        assert attrs[GenAISpanAttributes.GEN_AI_LATENCY_MS] == 1500.5
        assert attrs["pipeline.stage"] == "decision"

    def test_to_attributes_ignores_none_values(self):
        """Test that None values are not included in attributes."""
        ctx = GenAISpanContext(
            system="openai",
            request_model="gpt-4",
            max_tokens=None,
            temperature=0.7,
        )

        attrs = ctx.to_attributes()

        assert GenAISpanAttributes.GEN_AI_REQUEST_MODEL in attrs
        assert GenAISpanAttributes.GEN_AI_REQUEST_TEMPERATURE in attrs
        assert GenAISpanAttributes.GEN_AI_REQUEST_MAX_TOKENS not in attrs

    def test_to_attributes_zero_values_included(self):
        """Test that zero values are included in attributes."""
        ctx = GenAISpanContext(
            system="openai",
            max_tokens=0,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0.0,
        )

        attrs = ctx.to_attributes()

        assert attrs[GenAISpanAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 0
        assert attrs[GenAISpanAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 0
        assert attrs[GenAISpanAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 0
        assert attrs[GenAISpanAttributes.GEN_AI_LATENCY_MS] == 0.0

    def test_to_attributes_custom_attributes_override(self):
        """Test that custom attributes can override defaults."""
        ctx = GenAISpanContext(
            system="openai",
            custom_attributes={
                GenAISpanAttributes.GEN_AI_SYSTEM: "overridden",
                "custom.key": "value",
            },
        )

        attrs = ctx.to_attributes()

        # Custom attributes are added after defaults, so they override
        assert attrs[GenAISpanAttributes.GEN_AI_SYSTEM] == "overridden"
        assert attrs["custom.key"] == "value"

    def test_from_litellm_response_basic(self):
        """Test creating context from litellm response."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(finish_reason="stop")]
        mock_response.model = "gpt-4"
        mock_response.id = "resp_123"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        mock_response.usage = mock_usage

        request_params = {"model": "gpt-3.5-turbo"}

        ctx = GenAISpanContext.from_litellm_response(
            system="openai",
            response=mock_response,
            request_params=request_params,
            latency_ms=1500.0,
        )

        assert ctx.system == "openai"
        assert ctx.operation == "chat_completion"
        assert ctx.request_model == "gpt-3.5-turbo"
        assert ctx.input_tokens == 100
        assert ctx.output_tokens == 50
        assert ctx.finish_reason == "stop"
        assert ctx.response_model == "gpt-4"
        assert ctx.response_id == "resp_123"
        assert ctx.latency_ms == 1500.0

    def test_from_litellm_response_with_all_params(self):
        """Test creating context from litellm response with all request params."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(finish_reason="length")]
        mock_response.model = "claude-3-opus"
        mock_response.id = "resp_456"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 500
        mock_usage.completion_tokens = 1000
        mock_response.usage = mock_usage

        request_params = {
            "model": "claude-3-opus-20240229",
            "max_tokens": 4096,
            "temperature": 0.8,
            "top_p": 0.95,
        }

        ctx = GenAISpanContext.from_litellm_response(
            system="anthropic",
            response=mock_response,
            request_params=request_params,
            latency_ms=2500.0,
        )

        assert ctx.system == "anthropic"
        assert ctx.request_model == "claude-3-opus-20240229"
        assert ctx.max_tokens == 4096
        assert ctx.temperature == 0.8
        assert ctx.top_p == 0.95
        assert ctx.input_tokens == 500
        assert ctx.output_tokens == 1000
        assert ctx.finish_reason == "length"
        assert ctx.response_model == "claude-3-opus"

    def test_from_litellm_response_with_dict_usage(self):
        """Test creating context when usage is a dict instead of object."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(finish_reason="stop")]
        mock_response.model = "gpt-4"
        mock_response.id = "resp_789"

        # Usage as dict instead of object
        mock_response.usage = {
            "prompt_tokens": 200,
            "completion_tokens": 100,
        }

        request_params = {"model": "gpt-4"}

        ctx = GenAISpanContext.from_litellm_response(
            system="openai",
            response=mock_response,
            request_params=request_params,
            latency_ms=1200.0,
        )

        assert ctx.input_tokens == 200
        assert ctx.output_tokens == 100

    def test_from_litellm_response_no_choices(self):
        """Test creating context when response has no choices."""
        mock_response = MagicMock()
        # Create empty list for choices - but we need to handle the truthiness check
        mock_response.choices = []
        mock_response.model = "gpt-4"
        mock_response.id = "resp_000"

        # Use a dict for usage to avoid MagicMock behavior with 0 values
        mock_response.usage = {
            "prompt_tokens": 50,
            "completion_tokens": 0,
        }

        request_params = {"model": "gpt-4"}

        # When choices is empty, finish_reason should be None
        ctx = GenAISpanContext.from_litellm_response(
            system="openai",
            response=mock_response,
            request_params=request_params,
            latency_ms=500.0,
        )

        # Empty list is falsy, so finish_reason should be None
        assert ctx.finish_reason is None
        assert ctx.input_tokens == 50
        assert ctx.output_tokens == 0

    def test_from_litellm_response_no_usage(self):
        """Test creating context when response has no usage."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(finish_reason="stop")]
        mock_response.model = "gpt-4"
        mock_response.id = "resp_111"
        mock_response.usage = None

        request_params = {"model": "gpt-4"}

        ctx = GenAISpanContext.from_litellm_response(
            system="openai",
            response=mock_response,
            request_params=request_params,
            latency_ms=800.0,
        )

        assert ctx.input_tokens is None
        assert ctx.output_tokens is None

    def test_from_litellm_response_azure_openai(self):
        """Test creating context from Azure OpenAI response."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(finish_reason="content_filter")]
        mock_response.model = "gpt-4"
        mock_response.id = "resp_azure_123"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 1000
        mock_usage.completion_tokens = 500
        mock_response.usage = mock_usage

        request_params = {
            "model": "gpt-4",
            "max_tokens": 2048,
            "temperature": 0.5,
        }

        ctx = GenAISpanContext.from_litellm_response(
            system="azure_openai",
            response=mock_response,
            request_params=request_params,
            latency_ms=3000.0,
        )

        assert ctx.system == "azure_openai"
        assert ctx.finish_reason == "content_filter"
        assert ctx.input_tokens == 1000
        assert ctx.output_tokens == 500


@pytest.mark.unit
class TestGenAISpanContextTokenCounting:
    """Tests for token counting in GenAISpanContext."""

    def test_input_token_counting(self):
        """Test input token counting."""
        ctx = GenAISpanContext(
            system="openai",
            input_tokens=1000,
        )
        attrs = ctx.to_attributes()

        assert attrs[GenAISpanAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 1000

    def test_output_token_counting(self):
        """Test output token counting."""
        ctx = GenAISpanContext(
            system="openai",
            output_tokens=500,
        )
        attrs = ctx.to_attributes()

        assert attrs[GenAISpanAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 500

    def test_total_tokens_calculation(self):
        """Test that total tokens can be calculated from input + output."""
        ctx = GenAISpanContext(
            system="openai",
            input_tokens=100,
            output_tokens=50,
        )
        attrs = ctx.to_attributes()

        # Individual tokens are recorded
        assert attrs[GenAISpanAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 100
        assert attrs[GenAISpanAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 50


@pytest.mark.unit
class TestGenAISpanContextModelAttribution:
    """Tests for model attribution in GenAISpanContext."""

    def test_requested_model_vs_response_model(self):
        """Test tracking both requested and actual response models."""
        ctx = GenAISpanContext(
            system="openrouter",
            request_model="anthropic/claude-3-opus",
            response_model="claude-3-opus-20240229",
        )
        attrs = ctx.to_attributes()

        assert attrs[GenAISpanAttributes.GEN_AI_REQUEST_MODEL] == "anthropic/claude-3-opus"
        assert attrs[GenAISpanAttributes.GEN_AI_RESPONSE_MODEL] == "claude-3-opus-20240229"

    def test_model_attribution_with_fallback(self):
        """Test model attribution when fallback was used."""
        ctx = GenAISpanContext(
            system="openrouter",
            request_model="anthropic/claude-3-opus",
            response_model="openai/gpt-4",
            custom_attributes={"gen_ai.fallback.used": True},
        )
        attrs = ctx.to_attributes()

        assert attrs[GenAISpanAttributes.GEN_AI_REQUEST_MODEL] == "anthropic/claude-3-opus"
        assert attrs[GenAISpanAttributes.GEN_AI_RESPONSE_MODEL] == "openai/gpt-4"
        assert attrs["gen_ai.fallback.used"] is True

    def test_latency_tracking(self):
        """Test latency tracking in milliseconds."""
        ctx = GenAISpanContext(
            system="openai",
            latency_ms=2500.5,
        )
        attrs = ctx.to_attributes()

        assert attrs[GenAISpanAttributes.GEN_AI_LATENCY_MS] == 2500.5

    def test_finish_reason_tracking(self):
        """Test tracking of finish reasons."""
        finish_reasons = ["stop", "length", "content_filter", "tool_calls", "function_call"]

        for reason in finish_reasons:
            ctx = GenAISpanContext(
                system="openai",
                finish_reason=reason,
            )
            attrs = ctx.to_attributes()
            assert attrs[GenAISpanAttributes.GEN_AI_RESPONSE_FINISH_REASON] == reason


@pytest.mark.unit
class TestGenAISpanContextIntegration:
    """Integration-style tests for GenAISpanContext."""

    def test_complete_workflow(self):
        """Test a complete workflow with all attributes."""
        # Create initial context from request
        ctx = GenAISpanContext(
            system=GenAISystems.AZURE_OPENAI,
            operation="chat_completion",
            request_model="gpt-4",
            max_tokens=4096,
            temperature=0.7,
            top_p=0.95,
        )

        # Simulate response received
        ctx.input_tokens = 150
        ctx.output_tokens = 250
        ctx.finish_reason = "stop"
        ctx.response_model = "gpt-4-1106-preview"
        ctx.response_id = "chatcmpl-123"
        ctx.latency_ms = 2345.6

        attrs = ctx.to_attributes()

        # Verify all expected attributes
        assert attrs[GenAISpanAttributes.GEN_AI_SYSTEM] == "azure_openai"
        assert attrs[GenAISpanAttributes.GEN_AI_REQUEST_MODEL] == "gpt-4"
        assert attrs[GenAISpanAttributes.GEN_AI_RESPONSE_MODEL] == "gpt-4-1106-preview"
        assert attrs[GenAISpanAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 150
        assert attrs[GenAISpanAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 250
        assert attrs[GenAISpanAttributes.GEN_AI_LATENCY_MS] == 2345.6

    def test_pipeline_stage_tracking(self):
        """Test tracking GenAI operations within pipeline stages."""
        ctx = GenAISpanContext(
            system=GenAISystems.AZURE_OPENAI,
            operation="chat_completion",
            request_model="gpt-4",
            input_tokens=100,
            output_tokens=50,
            latency_ms=1000.0,
            custom_attributes={
                GenAISpanAttributes.GEN_AI_PIPELINE_STAGE: "decision",
                GenAISpanAttributes.GEN_AI_DECISION_TYPE: "parser_selection",
            },
        )

        attrs = ctx.to_attributes()

        assert attrs[GenAISpanAttributes.GEN_AI_PIPELINE_STAGE] == "decision"
        assert attrs[GenAISpanAttributes.GEN_AI_DECISION_TYPE] == "parser_selection"

    def test_retry_tracking(self):
        """Test tracking retry attempts."""
        ctx = GenAISpanContext(
            system=GenAISystems.OPENROUTER,
            request_model="anthropic/claude-3-opus",
            response_model="anthropic/claude-3-opus",
            latency_ms=3000.0,
            custom_attributes={
                GenAISpanAttributes.GEN_AI_RETRY_COUNT: 2,
                GenAISpanAttributes.GEN_AI_FALLBACK_USED: False,
            },
        )

        attrs = ctx.to_attributes()

        assert attrs[GenAISpanAttributes.GEN_AI_RETRY_COUNT] == 2
        assert attrs[GenAISpanAttributes.GEN_AI_FALLBACK_USED] is False
