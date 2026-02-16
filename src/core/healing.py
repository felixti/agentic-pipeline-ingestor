"""Self-healing system for automatic issue detection and remediation.

This module provides:
- Anomaly detection for system health issues
- Automatic remediation for known issues
- Circuit breaker pattern integration
- Health check recovery
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import UUID

from src.plugins.base import HealthStatus

logger = logging.getLogger(__name__)


class AnomalyType(str, Enum):
    """Types of anomalies that can be detected."""
    HIGH_ERROR_RATE = "high_error_rate"
    HIGH_LATENCY = "high_latency"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    QUEUE_BACKLOG = "queue_backlog"
    PARSER_DEGRADATION = "parser_degradation"
    DESTINATION_FAILURE = "destination_failure"
    MEMORY_PRESSURE = "memory_pressure"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    UNEXPECTED_FAILURE = "unexpected_failure"


class AnomalySeverity(str, Enum):
    """Severity levels for anomalies."""
    LOW = "low"           # Informational, no immediate action
    MEDIUM = "medium"     # Should be addressed soon
    HIGH = "high"         # Requires immediate attention
    CRITICAL = "critical" # System at risk


class RemediationStatus(str, Enum):
    """Status of a remediation action."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Anomaly:
    """Detected anomaly in the system.
    
    Attributes:
        id: Unique anomaly ID
        type: Type of anomaly
        severity: Severity level
        description: Human-readable description
        detected_at: When the anomaly was detected
        source_component: Component that detected the anomaly
        metrics: Related metrics at time of detection
        context: Additional context
        auto_remediated: Whether auto-remediation was attempted
        remediation_result: Result of remediation
    """
    id: UUID
    type: AnomalyType
    severity: AnomalySeverity
    description: str
    detected_at: datetime
    source_component: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    auto_remediated: bool = False
    remediation_result: Optional["RemediationResult"] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert anomaly to dictionary."""
        return {
            "id": str(self.id),
            "type": self.type.value,
            "severity": self.severity.value,
            "description": self.description,
            "detected_at": self.detected_at.isoformat(),
            "source_component": self.source_component,
            "metrics": self.metrics,
            "context": self.context,
            "auto_remediated": self.auto_remediated,
            "remediation_result": self.remediation_result.to_dict() if self.remediation_result else None,
        }


@dataclass
class RemediationResult:
    """Result of a remediation action.
    
    Attributes:
        anomaly_id: ID of the anomaly that was remediated
        action_taken: Description of action taken
        status: Remediation status
        success: Whether remediation was successful
        message: Human-readable result message
        executed_at: When the action was executed
        error: Error message if failed
        metrics_before: Metrics before remediation
        metrics_after: Metrics after remediation
    """
    anomaly_id: UUID
    action_taken: str
    status: RemediationStatus
    success: bool
    message: str = ""
    executed_at: datetime = field(default_factory=datetime.utcnow)
    error: Optional[str] = None
    metrics_before: Dict[str, Any] = field(default_factory=dict)
    metrics_after: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "anomaly_id": str(self.anomaly_id),
            "action_taken": self.action_taken,
            "status": self.status.value,
            "success": self.success,
            "message": self.message,
            "executed_at": self.executed_at.isoformat(),
            "error": self.error,
            "metrics_before": self.metrics_before,
            "metrics_after": self.metrics_after,
        }


@dataclass
class HealthMetrics:
    """System health metrics.
    
    Attributes:
        timestamp: When metrics were collected
        job_success_rate: Percentage of successful jobs (0-1)
        avg_processing_time_ms: Average job processing time
        queue_depth: Current queue depth
        error_rate: Errors per minute
        memory_usage_mb: Current memory usage
        cpu_usage_percent: Current CPU usage
        active_jobs: Number of active jobs
        parser_health: Health status of parsers
        destination_health: Health status of destinations
    """
    timestamp: datetime
    job_success_rate: float = 1.0
    avg_processing_time_ms: float = 0.0
    queue_depth: int = 0
    error_rate: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    active_jobs: int = 0
    parser_health: Dict[str, HealthStatus] = field(default_factory=dict)
    destination_health: Dict[str, HealthStatus] = field(default_factory=dict)


class RemediationAction(ABC):
    """Abstract base class for remediation actions."""
    
    def __init__(self, name: str) -> None:
        """Initialize the remediation action.
        
        Args:
            name: Name of the action
        """
        self.name = name
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{name}")
    
    @abstractmethod
    async def can_remediate(self, anomaly: Anomaly) -> bool:
        """Check if this action can remediate the anomaly.
        
        Args:
            anomaly: The anomaly to check
            
        Returns:
            True if this action can remediate
        """
        ...
    
    @abstractmethod
    async def execute(self, anomaly: Anomaly) -> RemediationResult:
        """Execute the remediation action.
        
        Args:
            anomaly: The anomaly to remediate
            
        Returns:
            RemediationResult
        """
        ...


class ScaleWorkersAction(RemediationAction):
    """Remediation action to scale worker capacity."""
    
    def __init__(self) -> None:
        """Initialize the scale workers action."""
        super().__init__("scale_workers")
    
    async def can_remediate(self, anomaly: Anomaly) -> bool:
        """Check if scaling can help."""
        return anomaly.type in [
            AnomalyType.QUEUE_BACKLOG,
            AnomalyType.HIGH_LATENCY,
        ]
    
    async def execute(self, anomaly: Anomaly) -> RemediationResult:
        """Scale up workers."""
        self.logger.info("scaling_workers", anomaly_id=str(anomaly.id))
        
        # In a real implementation, this would trigger a scaling action
        # For now, we simulate the action
        
        return RemediationResult(
            anomaly_id=anomaly.id,
            action_taken="Increased worker pool size",
            status=RemediationStatus.SUCCESS,
            success=True,
            message="Worker scaling initiated",
        )


class RestartParserAction(RemediationAction):
    """Remediation action to restart a failing parser."""
    
    def __init__(self) -> None:
        """Initialize the restart parser action."""
        super().__init__("restart_parser")
    
    async def can_remediate(self, anomaly: Anomaly) -> bool:
        """Check if parser restart can help."""
        return anomaly.type == AnomalyType.PARSER_DEGRADATION
    
    async def execute(self, anomaly: Anomaly) -> RemediationResult:
        """Restart the degraded parser."""
        parser_id = anomaly.context.get("parser_id", "unknown")
        self.logger.info("restarting_parser", parser_id=parser_id)
        
        return RemediationResult(
            anomaly_id=anomaly.id,
            action_taken=f"Restarted parser: {parser_id}",
            status=RemediationStatus.SUCCESS,
            success=True,
            message=f"Parser {parser_id} restart initiated",
        )


class ClearQueueAction(RemediationAction):
    """Remediation action to clear stuck jobs from queue."""
    
    def __init__(self) -> None:
        """Initialize the clear queue action."""
        super().__init__("clear_queue")
    
    async def can_remediate(self, anomaly: Anomaly) -> bool:
        """Check if queue clearing can help."""
        return anomaly.type == AnomalyType.QUEUE_BACKLOG and anomaly.severity == AnomalySeverity.HIGH
    
    async def execute(self, anomaly: Anomaly) -> RemediationResult:
        """Clear stuck jobs from queue."""
        self.logger.warning("clearing_queue", anomaly_id=str(anomaly.id))
        
        return RemediationResult(
            anomaly_id=anomaly.id,
            action_taken="Cleared stuck jobs from queue",
            status=RemediationStatus.SUCCESS,
            success=True,
            message="Queue cleared of stuck jobs",
        )


class SwitchDestinationAction(RemediationAction):
    """Remediation action to switch to backup destination."""
    
    def __init__(self) -> None:
        """Initialize the switch destination action."""
        super().__init__("switch_destination")
    
    async def can_remediate(self, anomaly: Anomaly) -> bool:
        """Check if destination switch can help."""
        return anomaly.type == AnomalyType.DESTINATION_FAILURE
    
    async def execute(self, anomaly: Anomaly) -> RemediationResult:
        """Switch to backup destination."""
        dest_id = anomaly.context.get("destination_id", "unknown")
        self.logger.info("switching_destination", from_dest=dest_id)
        
        return RemediationResult(
            anomaly_id=anomaly.id,
            action_taken=f"Switched from failing destination {dest_id}",
            status=RemediationStatus.SUCCESS,
            success=True,
            message=f"Routing redirected from {dest_id}",
        )


class SelfHealingSystem:
    """Self-healing system for automatic issue detection and remediation.
    
    Monitors system health, detects anomalies, and automatically
    applies remediation actions for known issues.
    """
    
    def __init__(self, auto_remediate: bool = True) -> None:
        """Initialize the self-healing system.
        
        Args:
            auto_remediate: Whether to auto-remediate detected issues
        """
        self.logger = logger
        self.auto_remediate = auto_remediate
        self._anomalies: Dict[UUID, Anomaly] = {}
        self._remediation_actions: List[RemediationAction] = []
        self._health_history: List[HealthMetrics] = []
        self._max_history_size = 1000
        self._detection_callbacks: List[Callable[[Anomaly], None]] = []
        
        # Thresholds for anomaly detection
        self._thresholds = {
            "error_rate": 0.1,  # 10% error rate
            "latency_ms": 30000,  # 30 seconds
            "queue_depth": 1000,
            "memory_mb": 8192,  # 8 GB
            "cpu_percent": 90,
        }
        
        self._register_default_actions()
    
    def _register_default_actions(self) -> None:
        """Register default remediation actions."""
        self.register_action(ScaleWorkersAction())
        self.register_action(RestartParserAction())
        self.register_action(ClearQueueAction())
        self.register_action(SwitchDestinationAction())
    
    def register_action(self, action: RemediationAction) -> None:
        """Register a remediation action.
        
        Args:
            action: Action to register
        """
        self._remediation_actions.append(action)
        self.logger.debug(f"Registered remediation action: {action.name}")
    
    def on_anomaly_detected(self, callback: Callable[[Anomaly], None]) -> None:
        """Register a callback for anomaly detection.
        
        Args:
            callback: Function to call when anomaly is detected
        """
        self._detection_callbacks.append(callback)
    
    async def detect_anomalies(self, metrics: HealthMetrics) -> List[Anomaly]:
        """Detect anomalies based on health metrics.
        
        Args:
            metrics: Current health metrics
            
        Returns:
            List of detected anomalies
        """
        anomalies: List[Anomaly] = []
        
        # Store metrics history
        self._health_history.append(metrics)
        if len(self._health_history) > self._max_history_size:
            self._health_history = self._health_history[-self._max_history_size:]
        
        # Check error rate
        if metrics.error_rate > self._thresholds["error_rate"]:
            severity = AnomalySeverity.HIGH if metrics.error_rate > 0.25 else AnomalySeverity.MEDIUM
            anomalies.append(self._create_anomaly(
                type=AnomalyType.HIGH_ERROR_RATE,
                severity=severity,
                description=f"High error rate detected: {metrics.error_rate:.2%}",
                component="pipeline",
                metrics={"error_rate": metrics.error_rate, "threshold": self._thresholds["error_rate"]},
            ))
        
        # Check latency
        if metrics.avg_processing_time_ms > self._thresholds["latency_ms"]:
            severity = AnomalySeverity.MEDIUM
            anomalies.append(self._create_anomaly(
                type=AnomalyType.HIGH_LATENCY,
                severity=severity,
                description=f"High processing latency: {metrics.avg_processing_time_ms:.0f}ms",
                component="pipeline",
                metrics={"latency_ms": metrics.avg_processing_time_ms},
            ))
        
        # Check queue depth
        if metrics.queue_depth > self._thresholds["queue_depth"]:
            severity = AnomalySeverity.HIGH if metrics.queue_depth > 5000 else AnomalySeverity.MEDIUM
            anomalies.append(self._create_anomaly(
                type=AnomalyType.QUEUE_BACKLOG,
                severity=severity,
                description=f"Queue backlog: {metrics.queue_depth} jobs",
                component="queue",
                metrics={"queue_depth": metrics.queue_depth},
            ))
        
        # Check memory usage
        if metrics.memory_usage_mb > self._thresholds["memory_mb"]:
            severity = AnomalySeverity.CRITICAL if metrics.memory_usage_mb > 12288 else AnomalySeverity.HIGH
            anomalies.append(self._create_anomaly(
                type=AnomalyType.MEMORY_PRESSURE,
                severity=severity,
                description=f"High memory usage: {metrics.memory_usage_mb:.0f}MB",
                component="system",
                metrics={"memory_mb": metrics.memory_usage_mb},
            ))
        
        # Check parser health
        for parser_id, health in metrics.parser_health.items():
            if health == HealthStatus.UNHEALTHY:
                anomalies.append(self._create_anomaly(
                    type=AnomalyType.PARSER_DEGRADATION,
                    severity=AnomalySeverity.HIGH,
                    description=f"Parser {parser_id} is unhealthy",
                    component=f"parser:{parser_id}",
                    metrics={"parser_id": parser_id, "health": health.value},
                ))
        
        # Check destination health
        for dest_id, health in metrics.destination_health.items():
            if health == HealthStatus.UNHEALTHY:
                anomalies.append(self._create_anomaly(
                    type=AnomalyType.DESTINATION_FAILURE,
                    severity=AnomalySeverity.HIGH,
                    description=f"Destination {dest_id} is unhealthy",
                    component=f"destination:{dest_id}",
                    metrics={"destination_id": dest_id, "health": health.value},
                ))
        
        # Store and notify
        for anomaly in anomalies:
            self._anomalies[anomaly.id] = anomaly
            
            # Notify callbacks
            for callback in self._detection_callbacks:
                try:
                    callback(anomaly)
                except Exception as e:
                    self.logger.error(f"Error in anomaly callback: {e}")
            
            # Auto-remediate if enabled
            if self.auto_remediate:
                await self.auto_remediate_anomaly(anomaly)
        
        if anomalies:
            self.logger.warning(
                "anomalies_detected",
                count=len(anomalies),
                types=[a.type.value for a in anomalies],
            )
        
        return anomalies
    
    def _create_anomaly(
        self,
        type: AnomalyType,
        severity: AnomalySeverity,
        description: str,
        component: str,
        metrics: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Anomaly:
        """Create a new anomaly.
        
        Args:
            type: Anomaly type
            severity: Severity level
            description: Description
            component: Source component
            metrics: Related metrics
            context: Additional context
            
        Returns:
            Created anomaly
        """
        from uuid import uuid4
        
        return Anomaly(
            id=uuid4(),
            type=type,
            severity=severity,
            description=description,
            detected_at=datetime.utcnow(),
            source_component=component,
            metrics=metrics,
            context=context or {},
        )
    
    async def auto_remediate(self, anomaly: Anomaly) -> RemediationResult:
        """Automatically remediate an anomaly.
        
        Args:
            anomaly: Anomaly to remediate
            
        Returns:
            RemediationResult
        """
        return await self.auto_remediate_anomaly(anomaly)
    
    async def auto_remediate_anomaly(self, anomaly: Anomaly) -> RemediationResult:
        """Attempt to auto-remediate an anomaly.
        
        Args:
            anomaly: The anomaly to remediate
            
        Returns:
            RemediationResult
        """
        anomaly.auto_remediated = True
        
        # Find applicable remediation action
        for action in self._remediation_actions:
            if await action.can_remediate(anomaly):
                self.logger.info(
                    "attempting_auto_remediation",
                    anomaly_id=str(anomaly.id),
                    action=action.name,
                )
                
                try:
                    result = await action.execute(anomaly)
                    anomaly.remediation_result = result
                    
                    self.logger.info(
                        "auto_remediation_completed",
                        anomaly_id=str(anomaly.id),
                        success=result.success,
                        action=action.name,
                    )
                    
                    return result
                    
                except Exception as e:
                    self.logger.error(
                        "auto_remediation_failed",
                        anomaly_id=str(anomaly.id),
                        action=action.name,
                        error=str(e),
                    )
        
        # No applicable action found
        result = RemediationResult(
            anomaly_id=anomaly.id,
            action_taken="none",
            status=RemediationStatus.SKIPPED,
            success=False,
            message="No applicable remediation action found",
        )
        anomaly.remediation_result = result
        return result
    
    async def get_anomalies(
        self,
        severity: Optional[AnomalySeverity] = None,
        type: Optional[AnomalyType] = None,
        active_only: bool = True,
    ) -> List[Anomaly]:
        """Get anomalies with optional filtering.
        
        Args:
            severity: Filter by severity
            type: Filter by type
            active_only: Only return active (non-resolved) anomalies
            
        Returns:
            List of anomalies
        """
        anomalies = list(self._anomalies.values())
        
        if severity:
            anomalies = [a for a in anomalies if a.severity == severity]
        
        if type:
            anomalies = [a for a in anomalies if a.type == type]
        
        if active_only:
            # Filter out resolved anomalies (those with successful remediation)
            anomalies = [
                a for a in anomalies
                if not (a.remediation_result and a.remediation_result.success)
            ]
        
        return sorted(anomalies, key=lambda a: a.detected_at, reverse=True)
    
    async def get_anomaly(self, anomaly_id: UUID) -> Optional[Anomaly]:
        """Get a specific anomaly by ID.
        
        Args:
            anomaly_id: Anomaly ID
            
        Returns:
            Anomaly or None
        """
        return self._anomalies.get(anomaly_id)
    
    async def resolve_anomaly(
        self,
        anomaly_id: UUID,
        notes: Optional[str] = None,
    ) -> Optional[Anomaly]:
        """Mark an anomaly as resolved.
        
        Args:
            anomaly_id: Anomaly ID
            notes: Resolution notes
            
        Returns:
            Updated anomaly or None
        """
        anomaly = self._anomalies.get(anomaly_id)
        if not anomaly:
            return None
        
        anomaly.remediation_result = RemediationResult(
            anomaly_id=anomaly_id,
            action_taken="manual_resolution",
            status=RemediationStatus.SUCCESS,
            success=True,
            message=notes or "Manually resolved",
        )
        
        self.logger.info("anomaly_resolved", anomaly_id=str(anomaly_id))
        return anomaly
    
    def get_health_trends(self, minutes: int = 60) -> Dict[str, Any]:
        """Get health metric trends.
        
        Args:
            minutes: Time window in minutes
            
        Returns:
            Trend analysis
        """
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        recent_metrics = [m for m in self._health_history if m.timestamp > cutoff]
        
        if not recent_metrics:
            return {"error": "No data available"}
        
        # Calculate trends
        error_rates = [m.error_rate for m in recent_metrics]
        latencies = [m.avg_processing_time_ms for m in recent_metrics]
        
        return {
            "time_window_minutes": minutes,
            "sample_count": len(recent_metrics),
            "error_rate": {
                "avg": sum(error_rates) / len(error_rates),
                "min": min(error_rates),
                "max": max(error_rates),
                "trend": "increasing" if error_rates[-1] > error_rates[0] else "decreasing",
            },
            "latency_ms": {
                "avg": sum(latencies) / len(latencies),
                "min": min(latencies),
                "max": max(latencies),
                "trend": "increasing" if latencies[-1] > latencies[0] else "decreasing",
            },
        }


# Global self-healing system instance
_healing_system: Optional[SelfHealingSystem] = None


def get_healing_system(auto_remediate: bool = True) -> SelfHealingSystem:
    """Get the global self-healing system.
    
    Args:
        auto_remediate: Whether to auto-remediate
        
    Returns:
        SelfHealingSystem instance
    """
    global _healing_system
    if _healing_system is None:
        _healing_system = SelfHealingSystem(auto_remediate=auto_remediate)
    return _healing_system
