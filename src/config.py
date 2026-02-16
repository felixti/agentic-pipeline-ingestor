"""Configuration management for the Agentic Data Pipeline Ingestor.

This module provides centralized configuration management using Pydantic Settings,
supporting environment variables, .env files, and YAML configuration files.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database connection settings."""

    url: PostgresDsn = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/pipeline",
        description="PostgreSQL connection URL",
    )
    pool_size: int = Field(default=10, ge=1, le=100)
    max_overflow: int = Field(default=20, ge=0, le=100)
    pool_timeout: int = Field(default=30, ge=1)
    echo: bool = Field(default=False, description="Enable SQL echo for debugging")

    model_config = SettingsConfigDict(env_prefix="DB_")


class RedisSettings(BaseSettings):
    """Redis connection settings."""

    url: RedisDsn = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
    )
    password: Optional[str] = Field(default=None)
    ssl: bool = Field(default=False)
    socket_timeout: int = Field(default=5)
    socket_connect_timeout: int = Field(default=5)
    retry_on_timeout: bool = Field(default=True)

    model_config = SettingsConfigDict(env_prefix="REDIS_")


class OpenSearchSettings(BaseSettings):
    """OpenSearch connection settings for audit logs."""

    hosts: List[str] = Field(default=["http://localhost:9200"])
    username: Optional[str] = Field(default=None)
    password: Optional[str] = Field(default=None)
    use_ssl: bool = Field(default=False)
    verify_certs: bool = Field(default=True)
    index_prefix: str = Field(default="pipeline-audit")

    model_config = SettingsConfigDict(env_prefix="OPENSEARCH_")


class AzureSettings(BaseSettings):
    """Azure service settings."""

    tenant_id: Optional[str] = Field(default=None)
    client_id: Optional[str] = Field(default=None)
    client_secret: Optional[str] = Field(default=None)
    subscription_id: Optional[str] = Field(default=None)
    storage_account: Optional[str] = Field(default=None)
    storage_key: Optional[str] = Field(default=None)
    queue_connection_string: Optional[str] = Field(default=None)

    model_config = SettingsConfigDict(env_prefix="AZURE_")


class SecuritySettings(BaseSettings):
    """Security and authentication settings."""

    secret_key: str = Field(
        default="change-me-in-production",
        description="Secret key for JWT signing",
    )
    algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=30, ge=1)
    api_key_header: str = Field(default="X-API-Key")
    rate_limit_default: int = Field(default=100, ge=1)
    rate_limit_window: int = Field(default=60, ge=1)
    allowed_hosts: List[str] = Field(default=["*"])
    cors_origins: List[str] = Field(default=["http://localhost:3000"])
    cors_allow_credentials: bool = Field(default=True)
    cors_allow_methods: List[str] = Field(default=["*"])
    cors_allow_headers: List[str] = Field(default=["*"])

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        """Validate that secret key is not the default in production."""
        if v == "change-me-in-production":
            import warnings
            warnings.warn(
                "Using default secret key. Please set a secure SECRET_KEY in production!",
                stacklevel=2,
            )
        return v

    model_config = SettingsConfigDict(env_prefix="SECURITY_")


class ProcessingSettings(BaseSettings):
    """Document processing settings."""

    max_file_size_mb: int = Field(default=100, ge=1, le=1000)
    max_pages_per_document: int = Field(default=1000, ge=1)
    allowed_mime_types: List[str] = Field(
        default=[
            "application/pdf",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-powerpoint",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "image/jpeg",
            "image/png",
            "image/tiff",
            "text/csv",
            "text/plain",
            "application/zip",
        ]
    )
    temp_dir: Path = Field(default=Path("/tmp/pipeline"))
    cleanup_temp_files: bool = Field(default=True)
    default_timeout_seconds: int = Field(default=300, ge=30)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay_seconds: int = Field(default=5, ge=1)

    model_config = SettingsConfigDict(env_prefix="PROCESSING_")


class ObservabilitySettings(BaseSettings):
    """Observability and monitoring settings."""

    service_name: str = Field(default="pipeline-ingestor")
    service_version: str = Field(default="1.0.0")
    environment: str = Field(default="development")
    otlp_endpoint: Optional[str] = Field(default=None)
    otlp_insecure: bool = Field(default=True)
    jaeger_enabled: bool = Field(default=False)
    prometheus_enabled: bool = Field(default=True)
    prometheus_port: int = Field(default=9090)
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")
    log_request_body: bool = Field(default=False)
    log_response_body: bool = Field(default=False)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()

    model_config = SettingsConfigDict(env_prefix="OTEL_")


class LLMYamlConfig(BaseSettings):
    """LLM configuration loaded from YAML file."""

    config_path: Path = Field(default=Path("config/llm.yaml"))
    _config_cache: Optional[Dict[str, Any]] = None

    def load_yaml(self) -> Dict[str, Any]:
        """Load LLM configuration from YAML file.
        
        Returns:
            Dictionary containing LLM configuration
        """
        if self._config_cache is not None:
            return self._config_cache

        if not self.config_path.exists():
            return self._default_config()

        with open(self.config_path, "r") as f:
            self._config_cache = yaml.safe_load(f)
        return self._config_cache or self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Return default LLM configuration."""
        return {
            "llm": {
                "router": [
                    {
                        "model_name": "agentic-decisions",
                        "litellm_params": {
                            "model": "azure/gpt-4",
                            "api_base": "${AZURE_OPENAI_API_BASE}",
                            "api_key": "${AZURE_OPENAI_API_KEY}",
                            "api_version": "2024-02-01",
                            "tpm": 10000,
                            "rpm": 60,
                        },
                        "fallback_models": [
                            {
                                "model": "openrouter/anthropic/claude-3-opus",
                                "api_key": "${OPENROUTER_API_KEY}",
                                "api_base": "https://openrouter.ai/api/v1",
                                "tpm": 5000,
                            }
                        ],
                    }
                ],
                "proxy": {"host": "0.0.0.0", "port": 4000},
                "retry": {"num_retries": 3, "timeout": 30, "backoff_factor": 2},
                "defaults": {"temperature": 0.3, "max_tokens": 2000},
            }
        }

    model_config = SettingsConfigDict(env_prefix="LLM_")


class Settings(BaseSettings):
    """Main application settings."""

    # Application
    app_name: str = Field(default="Agentic Data Pipeline Ingestor")
    app_version: str = Field(default="1.0.0")
    debug: bool = Field(default=False)
    env: str = Field(default="development")

    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=1, ge=1)

    # Sub-settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    opensearch: OpenSearchSettings = Field(default_factory=OpenSearchSettings)
    azure: AzureSettings = Field(default_factory=AzureSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    llm_yaml: LLMYamlConfig = Field(default_factory=LLMYamlConfig)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings.
    
    Returns:
        Settings instance with loaded configuration
    """
    return Settings()


def reload_settings() -> Settings:
    """Reload settings from environment.
    
    Returns:
        Fresh Settings instance
    """
    get_settings.cache_clear()
    return get_settings()


# Global settings instance for convenient access
settings = get_settings()
