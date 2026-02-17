"""Unit tests for configuration management."""

import pytest

from src.config import Settings, get_settings, reload_settings


def test_default_settings():
    """Test default settings are loaded correctly."""
    settings = Settings()

    assert settings.app_name == "Agentic Data Pipeline Ingestor"
    assert settings.app_version == "1.0.0"
    assert settings.port == 8000
    assert settings.host == "0.0.0.0"


def test_database_settings():
    """Test database settings defaults."""
    settings = Settings()

    assert settings.database.pool_size == 10
    assert settings.database.max_overflow == 20
    assert settings.database.echo is False


def test_security_settings_default_warning():
    """Test that default secret key triggers a warning."""

    with pytest.warns(UserWarning, match="default secret key"):
        settings = Settings()
        settings.security.secret_key = "change-me-in-production"
        # Trigger validation
        from pydantic import TypeAdapter
        TypeAdapter(str).validate_python(settings.security.secret_key)


def test_llm_yaml_default_config():
    """Test LLM YAML configuration defaults."""
    settings = Settings()
    config = settings.llm_yaml._default_config()

    assert "llm" in config
    assert "router" in config["llm"]
    assert len(config["llm"]["router"]) > 0

    # Check primary model
    primary = config["llm"]["router"][0]
    assert primary["model_name"] == "agentic-decisions"
    assert "litellm_params" in primary


def test_get_settings_cached():
    """Test that get_settings returns cached instance."""
    settings1 = get_settings()
    settings2 = get_settings()

    assert settings1 is settings2


def test_reload_settings():
    """Test that reload_settings clears cache."""
    settings1 = get_settings()
    settings2 = reload_settings()

    # They should be different instances with same values
    assert settings1 is not settings2
    assert settings1.app_name == settings2.app_name
