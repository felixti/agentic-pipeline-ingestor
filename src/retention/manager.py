"""Data retention manager implementation.

This module provides configurable retention policies for managing
the lifecycle of data in the system.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol
from uuid import UUID

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class RetentionAction(str, Enum):
    """Actions to take when retention period expires."""
    DELETE = "delete"  # Permanently delete data
    ARCHIVE = "archive"  # Move to archive storage
    COMPRESS = "compress"  # Compress data
    ANONYMIZE = "anonymize"  # Remove PII but keep data
    KEEP = "keep"  # Keep indefinitely (compliance)


class RetentionRule(BaseModel):
    """A single retention rule.
    
    Attributes:
        name: Rule name/identifier
        description: Human-readable description
        data_type: Type of data this rule applies to
        retention_days: Number of days to retain data
        action: Action to take after retention period
        archive_location: Optional archive storage location
        conditions: Optional conditions for applying rule
        priority: Rule priority (higher = more specific)
    """
    
    name: str
    description: str
    data_type: str
    retention_days: int
    action: RetentionAction
    archive_location: Optional[str] = None
    conditions: Dict[str, Any] = Field(default_factory=dict)
    priority: int = 0
    
    def is_applicable(
        self,
        data_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Check if this rule applies to given data.
        
        Args:
            data_type: Type of data
            metadata: Additional data metadata
            
        Returns:
            True if rule applies
        """
        if self.data_type != data_type:
            return False
        
        # Check conditions
        if metadata and self.conditions:
            for key, value in self.conditions.items():
                if metadata.get(key) != value:
                    return False
        
        return True
    
    def get_expiration_date(self, created_at: datetime) -> datetime:
        """Calculate expiration date for data.
        
        Args:
            created_at: Data creation timestamp
            
        Returns:
            Expiration timestamp
        """
        return created_at + timedelta(days=self.retention_days)
    
    def is_expired(self, created_at: datetime) -> bool:
        """Check if data has expired according to this rule.
        
        Args:
            created_at: Data creation timestamp
            
        Returns:
            True if data has expired
        """
        return datetime.utcnow() > self.get_expiration_date(created_at)


class RetentionPolicy(BaseModel):
    """Collection of retention rules.
    
    Attributes:
        name: Policy name
        description: Policy description
        rules: List of retention rules
        default_action: Default action for unclassified data
    """
    
    name: str
    description: str
    rules: List[RetentionRule]
    default_action: RetentionAction = RetentionAction.KEEP
    
    def get_applicable_rule(
        self,
        data_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[RetentionRule]:
        """Get the applicable rule for data.
        
        Rules are sorted by priority (highest first), and the first
        matching rule is returned.
        
        Args:
            data_type: Type of data
            metadata: Additional metadata
            
        Returns:
            Applicable rule or None
        """
        # Sort by priority (highest first)
        sorted_rules = sorted(self.rules, key=lambda r: r.priority, reverse=True)
        
        for rule in sorted_rules:
            if rule.is_applicable(data_type, metadata):
                return rule
        
        return None


class RetentionHandler(Protocol):
    """Protocol for retention action handlers."""
    
    async def delete(self, data_id: str, data_type: str) -> bool:
        """Delete data."""
        ...
    
    async def archive(
        self,
        data_id: str,
        data_type: str,
        archive_location: str,
    ) -> bool:
        """Archive data to specified location."""
        ...
    
    async def compress(self, data_id: str, data_type: str) -> bool:
        """Compress data."""
        ...
    
    async def anonymize(self, data_id: str, data_type: str) -> bool:
        """Anonymize data (remove PII)."""
        ...


# Default retention rules as specified in Section 10.2
DEFAULT_RETENTION_RULES = [
    RetentionRule(
        name="raw_files",
        description="Raw uploaded files",
        data_type="raw_file",
        retention_days=30,
        action=RetentionAction.DELETE,
        priority=100,
    ),
    RetentionRule(
        name="processed_data",
        description="Processed document data",
        data_type="processed_data",
        retention_days=90,
        action=RetentionAction.ARCHIVE,
        archive_location="cold_storage",
        priority=100,
    ),
    RetentionRule(
        name="job_metadata",
        description="Job processing metadata",
        data_type="job_metadata",
        retention_days=2555,  # 7 years
        action=RetentionAction.KEEP,
        priority=100,
    ),
    RetentionRule(
        name="audit_logs",
        description="Audit log records",
        data_type="audit_log",
        retention_days=2555,  # 7 years
        action=RetentionAction.KEEP,
        priority=100,
    ),
    RetentionRule(
        name="lineage_records",
        description="Data lineage records",
        data_type="lineage_record",
        retention_days=2555,  # 7 years
        action=RetentionAction.KEEP,
        priority=100,
    ),
    RetentionRule(
        name="temporary_files",
        description="Temporary processing files",
        data_type="temp_file",
        retention_days=1,
        action=RetentionAction.DELETE,
        priority=100,
    ),
    RetentionRule(
        name="failed_job_data",
        description="Data from failed jobs (kept for debugging)",
        data_type="raw_file",
        retention_days=14,
        action=RetentionAction.DELETE,
        conditions={"job_status": "failed"},
        priority=200,  # Higher priority than raw_files
    ),
]


class DataRetentionManager:
    """Manager for data retention policies.
    
    This class manages retention policies and applies them to data
    in the system. It supports configurable policies per bucket/space.
    
    Example:
        manager = DataRetentionManager()
        
        # Apply default policy
        await manager.apply_retention_policy("raw_uploads")
        
        # Check data expiration
        is_expired = manager.is_data_expired(data_type, created_at)
    """
    
    def __init__(
        self,
        policy: Optional[RetentionPolicy] = None,
        handler: Optional[RetentionHandler] = None,
    ):
        """Initialize retention manager.
        
        Args:
            policy: Retention policy (defaults to built-in rules)
            handler: Handler for retention actions
        """
        self.policy = policy or RetentionPolicy(
            name="default",
            description="Default retention policy",
            rules=DEFAULT_RETENTION_RULES,
        )
        self.handler = handler
        self._action_handlers: Dict[RetentionAction, Callable] = {}
        self._lock = asyncio.Lock()
    
    def register_action_handler(
        self,
        action: RetentionAction,
        handler: Callable,
    ) -> None:
        """Register a handler for a retention action.
        
        Args:
            action: Action type
            handler: Handler function
        """
        self._action_handlers[action] = handler
    
    def get_retention_rule(
        self,
        data_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RetentionRule:
        """Get the retention rule for data.
        
        Args:
            data_type: Type of data
            metadata: Additional metadata
            
        Returns:
            Applicable retention rule
        """
        rule = self.policy.get_applicable_rule(data_type, metadata)
        
        if rule:
            return rule
        
        # Return default rule
        return RetentionRule(
            name="default",
            description="Default rule",
            data_type=data_type,
            retention_days=2555,  # 7 years default
            action=self.policy.default_action,
        )
    
    def is_data_expired(
        self,
        data_type: str,
        created_at: datetime,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Check if data has expired according to retention policy.
        
        Args:
            data_type: Type of data
            created_at: Data creation timestamp
            metadata: Additional metadata
            
        Returns:
            True if data has expired
        """
        rule = self.get_retention_rule(data_type, metadata)
        return rule.is_expired(created_at)
    
    def get_expiration_date(
        self,
        data_type: str,
        created_at: datetime,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> datetime:
        """Get expiration date for data.
        
        Args:
            data_type: Type of data
            created_at: Data creation timestamp
            metadata: Additional metadata
            
        Returns:
            Expiration timestamp
        """
        rule = self.get_retention_rule(data_type, metadata)
        return rule.get_expiration_date(created_at)
    
    async def apply_retention_policy(
        self,
        bucket: str,
        data_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Apply retention policy to a bucket.
        
        This is the main entry point for retention enforcement.
        It scans data in the bucket and applies retention rules.
        
        Args:
            bucket: Storage bucket/space name
            data_type: Optional data type filter
            
        Returns:
            Summary of actions taken
        """
        logger.info(f"Applying retention policy to bucket: {bucket}")
        
        summary = {
            "bucket": bucket,
            "processed": 0,
            "deleted": 0,
            "archived": 0,
            "compressed": 0,
            "anonymized": 0,
            "kept": 0,
            "errors": [],
        }
        
        # This is a placeholder - actual implementation would scan
        # the bucket and process each item
        # In production, this would integrate with the storage backend
        
        return summary
    
    async def process_data_item(
        self,
        data_id: str,
        data_type: str,
        created_at: datetime,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Process a single data item according to retention policy.
        
        Args:
            data_id: Data identifier
            data_type: Type of data
            created_at: Data creation timestamp
            metadata: Additional metadata
            
        Returns:
            True if action was successfully applied
        """
        # Check if data has expired
        if not self.is_data_expired(data_type, created_at, metadata):
            return True  # Not expired, nothing to do
        
        # Get applicable rule
        rule = self.get_retention_rule(data_type, metadata)
        
        logger.info(
            f"Applying retention action {rule.action.value} to {data_id} "
            f"(type: {data_type}, rule: {rule.name})"
        )
        
        # Apply action
        try:
            if rule.action == RetentionAction.DELETE:
                return await self._delete_data(data_id, data_type)
            
            elif rule.action == RetentionAction.ARCHIVE:
                return await self._archive_data(
                    data_id,
                    data_type,
                    rule.archive_location,
                )
            
            elif rule.action == RetentionAction.COMPRESS:
                return await self._compress_data(data_id, data_type)
            
            elif rule.action == RetentionAction.ANONYMIZE:
                return await self._anonymize_data(data_id, data_type)
            
            elif rule.action == RetentionAction.KEEP:
                return True  # Keep data, nothing to do
            
            else:
                logger.warning(f"Unknown retention action: {rule.action}")
                return False
        
        except Exception as e:
            logger.error(f"Failed to apply retention action: {e}")
            return False
    
    async def _delete_data(self, data_id: str, data_type: str) -> bool:
        """Delete data.
        
        Args:
            data_id: Data identifier
            data_type: Type of data
            
        Returns:
            True if deleted successfully
        """
        if RetentionAction.DELETE in self._action_handlers:
            return await self._action_handlers[RetentionAction.DELETE](
                data_id, data_type
            )
        
        if self.handler:
            return await self.handler.delete(data_id, data_type)
        
        logger.warning(f"No handler for delete action: {data_id}")
        return False
    
    async def _archive_data(
        self,
        data_id: str,
        data_type: str,
        archive_location: Optional[str],
    ) -> bool:
        """Archive data.
        
        Args:
            data_id: Data identifier
            data_type: Type of data
            archive_location: Archive location
            
        Returns:
            True if archived successfully
        """
        if RetentionAction.ARCHIVE in self._action_handlers:
            return await self._action_handlers[RetentionAction.ARCHIVE](
                data_id, data_type, archive_location
            )
        
        if self.handler:
            return await self.handler.archive(data_id, data_type, archive_location or "archive")
        
        logger.warning(f"No handler for archive action: {data_id}")
        return False
    
    async def _compress_data(self, data_id: str, data_type: str) -> bool:
        """Compress data.
        
        Args:
            data_id: Data identifier
            data_type: Type of data
            
        Returns:
            True if compressed successfully
        """
        if RetentionAction.COMPRESS in self._action_handlers:
            return await self._action_handlers[RetentionAction.COMPRESS](
                data_id, data_type
            )
        
        if self.handler:
            return await self.handler.compress(data_id, data_type)
        
        logger.warning(f"No handler for compress action: {data_id}")
        return False
    
    async def _anonymize_data(self, data_id: str, data_type: str) -> bool:
        """Anonymize data.
        
        Args:
            data_id: Data identifier
            data_type: Type of data
            
        Returns:
            True if anonymized successfully
        """
        if RetentionAction.ANONYMIZE in self._action_handlers:
            return await self._action_handlers[RetentionAction.ANONYMIZE](
                data_id, data_type
            )
        
        if self.handler:
            return await self.handler.anonymize(data_id, data_type)
        
        logger.warning(f"No handler for anonymize action: {data_id}")
        return False
    
    def get_policy_summary(self) -> Dict[str, Any]:
        """Get a summary of the current retention policy.
        
        Returns:
            Policy summary
        """
        return {
            "policy_name": self.policy.name,
            "description": self.policy.description,
            "default_action": self.policy.default_action.value,
            "rules": [
                {
                    "name": rule.name,
                    "data_type": rule.data_type,
                    "retention_days": rule.retention_days,
                    "action": rule.action.value,
                    "conditions": rule.conditions,
                }
                for rule in self.policy.rules
            ],
        }


# Global retention manager instance
_retention_manager: Optional[DataRetentionManager] = None


def get_retention_manager() -> DataRetentionManager:
    """Get global retention manager instance.
    
    Returns:
        Retention manager singleton
    """
    global _retention_manager
    if _retention_manager is None:
        _retention_manager = DataRetentionManager()
    return _retention_manager


def set_retention_manager(manager: DataRetentionManager) -> None:
    """Set the global retention manager.
    
    Args:
        manager: Retention manager instance
    """
    global _retention_manager
    _retention_manager = manager
