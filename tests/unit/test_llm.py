"""Unit tests for LLM provider and configuration."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.llm.config import LLMConfig, ModelConfig, RouterConfig, load_llm_config
from src.llm.provider import ChatMessage, ChatCompletionResponse, LLMProvider


# ============================================================================
# ChatMessage Tests
# ============================================================================

class TestChatMessage:
    """Tests for ChatMessage dataclass."""
    
    def test_create_user_message(self):
        """Test creating a user message."""
        msg = ChatMessage.user("Hello")
        
        assert msg.role == "user"
        assert msg.content == "Hello"
    
    def test_create_system_message(self):
        """Test creating a system message."""
        msg = ChatMessage.system("You are helpful")
        
        assert msg.role == "system"
        assert msg.content == "You are helpful"
    
    def test_create_assistant_message(self):
        """Test creating an assistant message."""
        msg = ChatMessage.assistant("I can help")
        
        assert msg.role == "assistant"
        assert msg.content == "I can help"
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        msg = ChatMessage(role="user", content="Hello", name="test")
        
        d = msg.to_dict()
        
        assert d == {"role": "user", "content": "Hello", "name": "test"}
    
    def test_to_dict_without_optional(self):
        """Test converting to dict without optional fields."""
        msg = ChatMessage(role="user", content="Hello")
        
        d = msg.to_dict()
        
        assert d == {"role": "user", "content": "Hello"}
    
    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {"role": "user", "content": "Hello"}
        
        msg = ChatMessage.from_dict(data)
        
        assert msg.role == "user"
        assert msg.content == "Hello"


# ============================================================================
# LLM Config Tests
# ============================================================================

class TestModelConfig:
    """Tests for ModelConfig."""
    
    def test_basic_config(self):
        """Test basic model configuration."""
        config = ModelConfig(
            model="azure/gpt-4",
            api_key="test-key",
            api_base="https://test.openai.azure.com",
        )
        
        assert config.model == "azure/gpt-4"
        assert config.api_key == "test-key"
    
    def test_to_litellm_params(self):
        """Test conversion to litellm params."""
        config = ModelConfig(
            model="azure/gpt-4",
            api_key="test-key",
            api_base="https://test.openai.azure.com",
            api_version="2024-02-01",
        )
        
        params = config.to_litellm_params()
        
        assert params["model"] == "azure/gpt-4"
        assert params["api_key"] == "test-key"
        assert params["api_base"] == "https://test.openai.azure.com"
        assert params["api_version"] == "2024-02-01"
    
    def test_env_var_resolution(self):
        """Test environment variable resolution."""
        import os
        os.environ["TEST_API_KEY"] = "resolved-key"
        
        config = ModelConfig(
            model="azure/gpt-4",
            api_key="${TEST_API_KEY}",
        )
        
        params = config.to_litellm_params()
        
        assert params["api_key"] == "resolved-key"
        
        del os.environ["TEST_API_KEY"]


class TestLLMConfig:
    """Tests for LLMConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = LLMConfig._default_config()
        
        assert len(config.routers) > 0
        assert config.routers[0].model_name == "agentic-decisions"
    
    def test_to_litellm_model_list(self):
        """Test conversion to litellm model list."""
        config = LLMConfig._default_config()
        
        model_list = config.to_litellm_model_list()
        
        assert len(model_list) > 0
        assert all("model_name" in entry for entry in model_list)
        assert all("litellm_params" in entry for entry in model_list)
    
    def test_get_router_for_model(self):
        """Test getting router by model name."""
        config = LLMConfig._default_config()
        
        router = config.get_router_for_model("agentic-decisions")
        
        assert router is not None
        assert router.model_name == "agentic-decisions"
    
    def test_get_router_for_nonexistent_model(self):
        """Test getting router for nonexistent model."""
        config = LLMConfig._default_config()
        
        router = config.get_router_for_model("nonexistent")
        
        assert router is None


# ============================================================================
# ChatCompletionResponse Tests
# ============================================================================

class TestChatCompletionResponse:
    """Tests for ChatCompletionResponse."""
    
    def test_basic_response(self):
        """Test basic response creation."""
        response = ChatCompletionResponse(
            id="test-id",
            model="gpt-4",
            content="Hello world",
        )
        
        assert response.id == "test-id"
        assert response.model == "gpt-4"
        assert response.content == "Hello world"
    
    def test_default_usage(self):
        """Test default usage values."""
        response = ChatCompletionResponse(
            id="test",
            model="gpt-4",
            content="Hello",
        )
        
        assert response.usage["prompt_tokens"] == 0
        assert response.usage["completion_tokens"] == 0
        assert response.usage["total_tokens"] == 0
    
    def test_to_json_valid(self):
        """Test converting valid JSON content."""
        response = ChatCompletionResponse(
            id="test",
            model="gpt-4",
            content='{"key": "value"}',
        )
        
        json_str = response.to_json()
        
        assert '"key": "value"' in json_str
    
    def test_to_json_invalid(self):
        """Test converting invalid JSON content."""
        response = ChatCompletionResponse(
            id="test",
            model="gpt-4",
            content="Plain text",
        )
        
        result = response.to_json()
        
        assert result == "Plain text"


# ============================================================================
# LLM Provider Tests
# ============================================================================

@pytest.mark.asyncio
class TestLLMProvider:
    """Tests for LLMProvider."""
    
    @patch("src.llm.provider.LITELLM_AVAILABLE", True)
    @patch("src.llm.provider.Router")
    def test_init_without_litellm(self, mock_router):
        """Test provider initialization."""
        config = LLMConfig._default_config()
        
        provider = LLMProvider(config)
        
        assert provider.config is config
        mock_router.assert_called_once()
    
    @patch("src.llm.provider.LITELLM_AVAILABLE", False)
    def test_init_raises_without_litellm(self):
        """Test that provider raises if litellm not installed."""
        config = LLMConfig._default_config()
        
        with pytest.raises(ImportError, match="litellm is not installed"):
            LLMProvider(config)
    
    @patch("src.llm.provider.LITELLM_AVAILABLE", True)
    @patch("src.llm.provider.Router")
    async def test_chat_completion(self, mock_router_class):
        """Test chat completion."""
        # Setup mock
        mock_router = MagicMock()
        mock_router_class.return_value = mock_router
        
        mock_response = MagicMock()
        mock_response.id = "test-id"
        mock_response.model = "gpt-4"
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = None
        
        mock_router.acompletion = AsyncMock(return_value=mock_response)
        
        config = LLMConfig._default_config()
        provider = LLMProvider(config)
        provider._router = mock_router
        
        messages = [ChatMessage.user("Hello")]
        response = await provider.chat_completion(messages)
        
        assert response.content == "Test response"
        assert response.model == "gpt-4"
        mock_router.acompletion.assert_called_once()
    
    @patch("src.llm.provider.LITELLM_AVAILABLE", True)
    @patch("src.llm.provider.Router")
    async def test_simple_completion(self, mock_router_class):
        """Test simple completion method."""
        # Setup mock
        mock_router = MagicMock()
        mock_router_class.return_value = mock_router
        
        mock_response = MagicMock()
        mock_response.id = "test-id"
        mock_response.model = "gpt-4"
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Simple response"
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = None
        
        mock_router.acompletion = AsyncMock(return_value=mock_response)
        
        config = LLMConfig._default_config()
        provider = LLMProvider(config)
        provider._router = mock_router
        
        result = await provider.simple_completion("Test prompt")
        
        assert result == "Simple response"
    
    def test_get_available_models(self):
        """Test getting available models."""
        with patch("src.llm.provider.LITELLM_AVAILABLE", True):
            with patch("src.llm.provider.Router"):
                config = LLMConfig._default_config()
                provider = LLMProvider(config)
                
                models = provider.get_available_models()
                
                assert "agentic-decisions" in models
                assert "enrichment" in models


# ============================================================================
# Load Config Tests
# ============================================================================

class TestLoadLLMConfig:
    """Tests for load_llm_config function."""
    
    def test_load_default(self):
        """Test loading default config."""
        config = load_llm_config("nonexistent.yaml")
        
        # Should return default config when file doesn't exist
        assert len(config.routers) > 0
    
    def test_load_with_none_path(self):
        """Test loading with None path."""
        config = load_llm_config(None)
        
        # Should return default config
        assert len(config.routers) > 0
