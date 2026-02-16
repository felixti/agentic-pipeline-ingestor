"""LLM configuration management for litellm integration.

This module provides configuration structures for LLM providers
including Azure OpenAI and OpenRouter fallback chains.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a single LLM model.
    
    Attributes:
        model: Model identifier (e.g., "azure/gpt-4", "openrouter/anthropic/claude-3-opus")
        api_key: API key for the provider
        api_base: Base URL for the API
        api_version: API version (for Azure)
        tpm: Tokens per minute limit
        rpm: Requests per minute limit
        timeout: Request timeout in seconds
        additional_params: Additional provider-specific parameters
    """
    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    tpm: Optional[int] = None
    rpm: Optional[int] = None
    timeout: int = 30
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_litellm_params(self) -> Dict[str, Any]:
        """Convert to litellm-compatible parameters.
        
        Returns:
            Dictionary of parameters for litellm
        """
        params: Dict[str, Any] = {"model": self.model}
        
        if self.api_key:
            params["api_key"] = self._resolve_env_vars(self.api_key)
        if self.api_base:
            params["api_base"] = self._resolve_env_vars(self.api_base)
        if self.api_version:
            params["api_version"] = self.api_version
        
        params.update(self.additional_params)
        return params
    
    @staticmethod
    def _resolve_env_vars(value: str) -> str:
        """Resolve environment variables in a string value.
        
        Args:
            value: String that may contain ${ENV_VAR} placeholders
            
        Returns:
            String with environment variables resolved
        """
        if not isinstance(value, str):
            return value
        
        import re
        pattern = r'\$\{([^}]+)\}'
        
        def replace_env_var(match: Any) -> str:
            env_var = match.group(1)
            env_value = os.getenv(env_var, "")
            if not env_value:
                logger.warning(f"Environment variable {env_var} not set")
            return env_value
        
        return re.sub(pattern, replace_env_var, value)


@dataclass
class RouterConfig:
    """Configuration for a litellm router entry.
    
    Attributes:
        model_name: Logical name for this model group
        litellm_params: Primary model parameters
        fallback_models: List of fallback model configurations
    """
    model_name: str
    litellm_params: ModelConfig
    fallback_models: List[ModelConfig] = field(default_factory=list)
    
    def to_litellm_router_format(self) -> List[Dict[str, Any]]:
        """Convert to litellm router format.
        
        Returns:
            List of router entries for litellm
        """
        entries: List[Dict[str, Any]] = []
        
        # Primary model
        primary_entry = {
            "model_name": self.model_name,
            "litellm_params": self.litellm_params.to_litellm_params(),
        }
        entries.append(primary_entry)
        
        # Fallback models
        for i, fallback in enumerate(self.fallback_models):
            fallback_entry = {
                "model_name": self.model_name,
                "litellm_params": fallback.to_litellm_params(),
            }
            entries.append(fallback_entry)
        
        return entries


@dataclass
class LLMConfig:
    """Complete LLM configuration.
    
    Attributes:
        routers: List of router configurations
        proxy_host: Host for litellm proxy
        proxy_port: Port for litellm proxy
        retry_attempts: Number of retry attempts
        retry_timeout: Timeout for retries
        backoff_factor: Exponential backoff factor
        default_temperature: Default temperature for completions
        default_max_tokens: Default max tokens for completions
        provider_settings: Provider-specific settings
    """
    routers: List[RouterConfig] = field(default_factory=list)
    proxy_host: str = "0.0.0.0"
    proxy_port: int = 4000
    retry_attempts: int = 3
    retry_timeout: int = 30
    backoff_factor: float = 2.0
    default_temperature: float = 0.3
    default_max_tokens: int = 2000
    provider_settings: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "LLMConfig":
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Loaded LLMConfig instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return cls._default_config()
        
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        if not data or "llm" not in data:
            logger.warning("Invalid config format, using defaults")
            return cls._default_config()
        
        llm_data = data["llm"]
        
        # Parse routers
        routers = []
        for router_data in llm_data.get("router", []):
            routers.append(cls._parse_router_config(router_data))
        
        # Parse proxy settings
        proxy = llm_data.get("proxy", {})
        
        # Parse retry settings
        retry = llm_data.get("retry", {})
        
        # Parse defaults
        defaults = llm_data.get("defaults", {})
        
        # Parse provider settings
        providers = llm_data.get("providers", {})
        
        return cls(
            routers=routers,
            proxy_host=proxy.get("host", "0.0.0.0"),
            proxy_port=proxy.get("port", 4000),
            retry_attempts=retry.get("num_retries", 3),
            retry_timeout=retry.get("timeout", 30),
            backoff_factor=retry.get("backoff_factor", 2.0),
            default_temperature=defaults.get("temperature", 0.3),
            default_max_tokens=defaults.get("max_tokens", 2000),
            provider_settings=providers,
        )
    
    @classmethod
    def _parse_router_config(cls, data: Dict[str, Any]) -> RouterConfig:
        """Parse router configuration from YAML data.
        
        Args:
            data: Router configuration dictionary
            
        Returns:
            Parsed RouterConfig
        """
        # Parse primary model
        primary_params = data.get("litellm_params", {})
        primary_model = ModelConfig(
            model=primary_params["model"],
            api_key=primary_params.get("api_key"),
            api_base=primary_params.get("api_base"),
            api_version=primary_params.get("api_version"),
            tpm=primary_params.get("tpm"),
            rpm=primary_params.get("rpm"),
            timeout=primary_params.get("timeout", 30),
            additional_params={
                k: v for k, v in primary_params.items()
                if k not in ["model", "api_key", "api_base", "api_version", "tpm", "rpm", "timeout"]
            },
        )
        
        # Parse fallback models
        fallback_models = []
        for fallback_data in data.get("fallback_models", []):
            fallback_models.append(ModelConfig(
                model=fallback_data["model"],
                api_key=fallback_data.get("api_key"),
                api_base=fallback_data.get("api_base"),
                api_version=fallback_data.get("api_version"),
                tpm=fallback_data.get("tpm"),
                rpm=fallback_data.get("rpm"),
                timeout=fallback_data.get("timeout", 30),
            ))
        
        return RouterConfig(
            model_name=data["model_name"],
            litellm_params=primary_model,
            fallback_models=fallback_models,
        )
    
    @classmethod
    def _default_config(cls) -> "LLMConfig":
        """Create default configuration.
        
        Returns:
            Default LLMConfig with Azure + OpenRouter fallback
        """
        return cls(
            routers=[
                RouterConfig(
                    model_name="agentic-decisions",
                    litellm_params=ModelConfig(
                        model="azure/gpt-4",
                        api_key="${AZURE_OPENAI_API_KEY}",
                        api_base="${AZURE_OPENAI_API_BASE}",
                        api_version="2024-02-01",
                        tpm=10000,
                        rpm=60,
                    ),
                    fallback_models=[
                        ModelConfig(
                            model="openrouter/anthropic/claude-3-opus",
                            api_key="${OPENROUTER_API_KEY}",
                            api_base="https://openrouter.ai/api/v1",
                            tpm=5000,
                        ),
                    ],
                ),
                RouterConfig(
                    model_name="enrichment",
                    litellm_params=ModelConfig(
                        model="azure/gpt-35-turbo",
                        api_key="${AZURE_OPENAI_API_KEY}",
                        api_base="${AZURE_OPENAI_API_BASE}",
                        api_version="2024-02-01",
                        tpm=20000,
                    ),
                    fallback_models=[
                        ModelConfig(
                            model="openrouter/anthropic/claude-3-haiku",
                            api_key="${OPENROUTER_API_KEY}",
                            api_base="https://openrouter.ai/api/v1",
                        ),
                    ],
                ),
            ],
        )
    
    def to_litellm_model_list(self) -> List[Dict[str, Any]]:
        """Convert to litellm model list format.
        
        Returns:
            List of model configurations for litellm Router
        """
        model_list: List[Dict[str, Any]] = []
        for router in self.routers:
            model_list.extend(router.to_litellm_router_format())
        return model_list
    
    def get_router_for_model(self, model_name: str) -> Optional[RouterConfig]:
        """Get router configuration for a model name.
        
        Args:
            model_name: Logical model name
            
        Returns:
            RouterConfig if found, None otherwise
        """
        for router in self.routers:
            if router.model_name == model_name:
                return router
        return None


def load_llm_config(config_path: Optional[str] = None) -> LLMConfig:
    """Load LLM configuration from file or return defaults.
    
    Args:
        config_path: Path to configuration file. If None, uses
                     default path "config/llm.yaml"
    
    Returns:
        Loaded LLMConfig
    """
    if config_path is None:
        config_path = "config/llm.yaml"
    
    return LLMConfig.from_yaml(config_path)
