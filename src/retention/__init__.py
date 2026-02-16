"""Data retention management system for the Agentic Data Pipeline Ingestor.

This module provides configurable data retention policies for managing
storage lifecycle of processed data.
"""

from src.retention.manager import (
    DataRetentionManager,
    RetentionAction,
    RetentionPolicy,
    RetentionRule,
    get_retention_manager,
)

__all__ = [
    "DataRetentionManager",
    "RetentionAction",
    "RetentionPolicy",
    "RetentionRule",
    "get_retention_manager",
]
