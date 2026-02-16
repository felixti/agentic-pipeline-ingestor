"""Multi-destination routing for processed data.

This module handles routing transformed data to multiple destinations
with parallel execution, filtering, and partial failure handling.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import UUID

from src.api.models import DestinationConfig, DestinationFilter, FilterOperator
from src.plugins.base import Connection, DestinationPlugin, TransformedData, WriteResult
from src.plugins.registry import PluginRegistry

logger = logging.getLogger(__name__)


class RoutingStatus(str, Enum):
    """Status of a routing operation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DestinationRouteResult:
    """Result of routing to a single destination.
    
    Attributes:
        destination_id: ID of the destination
        destination_type: Type of destination
        status: Routing status
        success: Whether routing was successful
        write_result: Result from destination write operation
        error: Error message if failed
        latency_ms: Time taken for routing
        filtered: Whether the data was filtered out
        filter_reason: Reason for filtering
    """
    destination_id: str
    destination_type: str
    status: RoutingStatus
    success: bool
    write_result: Optional[WriteResult] = None
    error: Optional[str] = None
    latency_ms: int = 0
    filtered: bool = False
    filter_reason: Optional[str] = None


@dataclass
class MultiDestinationResult:
    """Result of routing to multiple destinations.
    
    Attributes:
        job_id: ID of the job
        overall_status: Overall routing status
        destination_results: Results for each destination
        success_count: Number of successful destinations
        failure_count: Number of failed destinations
        skipped_count: Number of skipped destinations
        total_latency_ms: Total time taken
        partial_failure: Whether some destinations failed
        errors: List of error messages
    """
    job_id: UUID
    overall_status: RoutingStatus
    destination_results: List[DestinationRouteResult]
    success_count: int = 0
    failure_count: int = 0
    skipped_count: int = 0
    total_latency_ms: int = 0
    partial_failure: bool = False
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "job_id": str(self.job_id),
            "overall_status": self.overall_status.value,
            "destination_results": [
                {
                    "destination_id": r.destination_id,
                    "destination_type": r.destination_type,
                    "status": r.status.value,
                    "success": r.success,
                    "error": r.error,
                    "latency_ms": r.latency_ms,
                    "filtered": r.filtered,
                    "filter_reason": r.filter_reason,
                }
                for r in self.destination_results
            ],
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "skipped_count": self.skipped_count,
            "total_latency_ms": self.total_latency_ms,
            "partial_failure": self.partial_failure,
            "errors": self.errors,
        }


class DestinationRouter:
    """Router for sending data to multiple destinations.
    
    Features:
    - Parallel routing to multiple destinations
    - Filter-based routing decisions
    - Partial failure handling
    - Circuit breaker pattern for failing destinations
    - Retry logic per destination
    """
    
    def __init__(self, plugin_registry: Optional[PluginRegistry] = None) -> None:
        """Initialize the destination router.
        
        Args:
            plugin_registry: Plugin registry for accessing destinations
        """
        self.logger = logger
        self.registry = plugin_registry or PluginRegistry()
        self._circuit_breakers: Dict[str, "CircuitBreaker"] = {}
    
    async def route_to_multiple(
        self,
        data: TransformedData,
        destinations: List[DestinationConfig],
        parallel: bool = True,
        max_concurrent: int = 5,
    ) -> MultiDestinationResult:
        """Route data to multiple destinations.
        
        Args:
            data: Transformed data to route
            destinations: List of destination configurations
            parallel: Whether to route in parallel
            max_concurrent: Maximum concurrent routing operations
            
        Returns:
            MultiDestinationResult with results for all destinations
        """
        start_time = datetime.utcnow()
        
        if not destinations:
            return MultiDestinationResult(
                job_id=data.job_id,
                overall_status=RoutingStatus.SKIPPED,
                destination_results=[],
                skipped_count=1,
            )
        
        self.logger.info(
            "routing_to_destinations",
            job_id=str(data.job_id),
            destination_count=len(destinations),
            parallel=parallel,
        )
        
        results: List[DestinationRouteResult] = []
        
        if parallel:
            # Use semaphore to limit concurrency
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def route_with_semaphore(dest: DestinationConfig) -> DestinationRouteResult:
                async with semaphore:
                    return await self._route_to_single(data, dest)
            
            # Route to all destinations in parallel
            tasks = [route_with_semaphore(dest) for dest in destinations]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            processed_results: List[DestinationRouteResult] = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(
                        "routing_task_failed",
                        job_id=str(data.job_id),
                        destination=destinations[i].name or str(destinations[i].type),
                        error=str(result),
                    )
                    processed_results.append(DestinationRouteResult(
                        destination_id=str(destinations[i].id) if destinations[i].id else "unknown",
                        destination_type=destinations[i].type.value,
                        status=RoutingStatus.FAILED,
                        success=False,
                        error=str(result),
                    ))
                else:
                    processed_results.append(result)
            results = processed_results
        else:
            # Route sequentially
            for dest in destinations:
                result = await self._route_to_single(data, dest)
                results.append(result)
        
        # Calculate statistics
        success_count = sum(1 for r in results if r.success and not r.filtered)
        failure_count = sum(1 for r in results if not r.success and not r.filtered)
        skipped_count = sum(1 for r in results if r.filtered)
        
        # Determine overall status
        if failure_count == 0:
            overall_status = RoutingStatus.SUCCESS
        elif success_count > 0:
            overall_status = RoutingStatus.PARTIAL_SUCCESS
        else:
            overall_status = RoutingStatus.FAILED
        
        total_latency = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        errors = [r.error for r in results if r.error]
        
        self.logger.info(
            "routing_completed",
            job_id=str(data.job_id),
            success_count=success_count,
            failure_count=failure_count,
            total_latency_ms=total_latency,
        )
        
        return MultiDestinationResult(
            job_id=data.job_id,
            overall_status=overall_status,
            destination_results=results,
            success_count=success_count,
            failure_count=failure_count,
            skipped_count=skipped_count,
            total_latency_ms=total_latency,
            partial_failure=failure_count > 0 and success_count > 0,
            errors=errors,
        )
    
    async def _route_to_single(
        self,
        data: TransformedData,
        destination: DestinationConfig,
    ) -> DestinationRouteResult:
        """Route data to a single destination.
        
        Args:
            data: Data to route
            destination: Destination configuration
            
        Returns:
            DestinationRouteResult
        """
        start_time = datetime.utcnow()
        dest_id = str(destination.id) if destination.id else destination.type.value
        dest_type = destination.type.value
        
        # Check if destination is enabled
        if not destination.enabled:
            return DestinationRouteResult(
                destination_id=dest_id,
                destination_type=dest_type,
                status=RoutingStatus.SKIPPED,
                success=True,
                filtered=True,
                filter_reason="Destination disabled",
            )
        
        # Check circuit breaker
        if self._is_circuit_open(dest_id):
            return DestinationRouteResult(
                destination_id=dest_id,
                destination_type=dest_type,
                status=RoutingStatus.SKIPPED,
                success=False,
                filtered=True,
                filter_reason="Circuit breaker open - destination temporarily unavailable",
            )
        
        # Apply filters
        filter_result = self._apply_filters(data, destination.filters)
        if not filter_result["passed"]:
            return DestinationRouteResult(
                destination_id=dest_id,
                destination_type=dest_type,
                status=RoutingStatus.SKIPPED,
                success=True,
                filtered=True,
                filter_reason=filter_result["reason"],
            )
        
        try:
            # Get destination plugin
            plugin = self.registry.get_destination(dest_type)
            if not plugin:
                return DestinationRouteResult(
                    destination_id=dest_id,
                    destination_type=dest_type,
                    status=RoutingStatus.FAILED,
                    success=False,
                    error=f"Destination plugin not found: {dest_type}",
                )
            
            # Initialize and connect
            await plugin.initialize(destination.config)
            conn = await plugin.connect(destination.config)
            
            # Write data
            write_result = await plugin.write(conn, data)
            
            latency = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            # Record success for circuit breaker
            self._record_success(dest_id)
            
            return DestinationRouteResult(
                destination_id=dest_id,
                destination_type=dest_type,
                status=RoutingStatus.SUCCESS if write_result.success else RoutingStatus.FAILED,
                success=write_result.success,
                write_result=write_result,
                error=write_result.error,
                latency_ms=latency,
            )
            
        except Exception as e:
            latency = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            # Record failure for circuit breaker
            self._record_failure(dest_id)
            
            self.logger.error(
                "routing_to_destination_failed",
                job_id=str(data.job_id),
                destination=dest_id,
                error=str(e),
            )
            
            return DestinationRouteResult(
                destination_id=dest_id,
                destination_type=dest_type,
                status=RoutingStatus.FAILED,
                success=False,
                error=str(e),
                latency_ms=latency,
            )
    
    def _apply_filters(
        self,
        data: TransformedData,
        filters: List[DestinationFilter],
    ) -> Dict[str, Any]:
        """Apply filters to determine if data should be routed.
        
        Args:
            data: Data to filter
            filters: List of filters to apply
            
        Returns:
            Filter result with passed status and reason
        """
        if not filters:
            return {"passed": True, "reason": None}
        
        # Get metadata for filtering
        metadata = data.metadata
        
        for filter_config in filters:
            field_value = metadata.get(filter_config.field)
            filter_value = filter_config.value
            operator = filter_config.operator
            
            passed = self._evaluate_filter(field_value, filter_value, operator)
            
            if not passed:
                return {
                    "passed": False,
                    "reason": f"Filter failed: {filter_config.field} {operator.value} {filter_value}",
                }
        
        return {"passed": True, "reason": None}
    
    def _evaluate_filter(
        self,
        field_value: Any,
        filter_value: str,
        operator: FilterOperator,
    ) -> bool:
        """Evaluate a single filter condition.
        
        Args:
            field_value: Value from data field
            filter_value: Value to compare against
            operator: Comparison operator
            
        Returns:
            True if filter passes
        """
        # Convert field value to string for comparison
        field_str = str(field_value) if field_value is not None else ""
        filter_str = str(filter_value) if filter_value is not None else ""
        
        if operator == FilterOperator.EQUALS:
            return field_str == filter_str
        elif operator == FilterOperator.NOT_EQUALS:
            return field_str != filter_str
        elif operator == FilterOperator.CONTAINS:
            return filter_str in field_str
        elif operator == FilterOperator.STARTS_WITH:
            return field_str.startswith(filter_str)
        elif operator == FilterOperator.ENDS_WITH:
            return field_str.endswith(filter_str)
        elif operator == FilterOperator.REGEX:
            import re
            try:
                return bool(re.search(filter_str, field_str))
            except re.error:
                return False
        
        return True
    
    def _is_circuit_open(self, destination_id: str) -> bool:
        """Check if circuit breaker is open for a destination.
        
        Args:
            destination_id: Destination ID
            
        Returns:
            True if circuit is open (destination unavailable)
        """
        breaker = self._circuit_breakers.get(destination_id)
        if breaker:
            return breaker.is_open()
        return False
    
    def _record_success(self, destination_id: str) -> None:
        """Record a successful operation for circuit breaker.
        
        Args:
            destination_id: Destination ID
        """
        if destination_id not in self._circuit_breakers:
            self._circuit_breakers[destination_id] = CircuitBreaker()
        self._circuit_breakers[destination_id].record_success()
    
    def _record_failure(self, destination_id: str) -> None:
        """Record a failed operation for circuit breaker.
        
        Args:
            destination_id: Destination ID
        """
        if destination_id not in self._circuit_breakers:
            self._circuit_breakers[destination_id] = CircuitBreaker()
        self._circuit_breakers[destination_id].record_failure()
    
    async def get_destination_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for all destinations with circuit breakers.
        
        Returns:
            Dictionary of destination health information
        """
        health = {}
        for dest_id, breaker in self._circuit_breakers.items():
            health[dest_id] = {
                "circuit_state": breaker.state.value,
                "failure_count": breaker.failure_count,
                "success_count": breaker.success_count,
                "last_failure": breaker.last_failure_time.isoformat() if breaker.last_failure_time else None,
                "last_success": breaker.last_success_time.isoformat() if breaker.last_success_time else None,
            }
        return health


class CircuitBreakerState(str, Enum):
    """State of a circuit breaker."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if recovered


class CircuitBreaker:
    """Circuit breaker pattern for destination routing.
    
    Prevents cascading failures by temporarily disabling
    destinations that are experiencing issues.
    """
    
    # Thresholds
    FAILURE_THRESHOLD = 5
    SUCCESS_THRESHOLD = 3
    TIMEOUT_SECONDS = 60
    
    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 3,
        timeout_seconds: int = 60,
    ) -> None:
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Failures before opening circuit
            success_threshold: Successes needed to close circuit
            timeout_seconds: Time before attempting reset
        """
        self.state = CircuitBreakerState.CLOSED
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout_seconds = timeout_seconds
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_success_time: Optional[datetime] = None
        self.last_state_change: Optional[datetime] = None
    
    def is_open(self) -> bool:
        """Check if circuit is open (failing).
        
        Returns:
            True if circuit is open
        """
        if self.state == CircuitBreakerState.OPEN:
            # Check if timeout has elapsed
            if self.last_state_change:
                elapsed = (datetime.utcnow() - self.last_state_change).total_seconds()
                if elapsed > self.timeout_seconds:
                    # Transition to half-open
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.success_count = 0
                    self.last_state_change = datetime.utcnow()
                    logger.info("circuit_breaker_half_open")
                    return False
            return True
        return False
    
    def record_success(self) -> None:
        """Record a successful operation."""
        self.success_count += 1
        self.last_success_time = datetime.utcnow()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            if self.success_count >= self.success_threshold:
                # Close the circuit
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.last_state_change = datetime.utcnow()
                logger.info("circuit_breaker_closed")
    
    def record_failure(self) -> None:
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                # Open the circuit
                self.state = CircuitBreakerState.OPEN
                self.last_state_change = datetime.utcnow()
                logger.warning("circuit_breaker_opened", failure_count=self.failure_count)
        
        elif self.state == CircuitBreakerState.HALF_OPEN:
            # Go back to open
            self.state = CircuitBreakerState.OPEN
            self.last_state_change = datetime.utcnow()
            logger.warning("circuit_breaker_reopened")


class RoutingFilterBuilder:
    """Builder for creating destination filters.
    
    Provides a fluent interface for building complex filter conditions.
    """
    
    def __init__(self) -> None:
        """Initialize the filter builder."""
        self.filters: List[DestinationFilter] = []
    
    def equals(self, field: str, value: str) -> "RoutingFilterBuilder":
        """Add equals filter.
        
        Args:
            field: Field to filter on
            value: Value to match
            
        Returns:
            Self for chaining
        """
        self.filters.append(DestinationFilter(
            field=field,
            operator=FilterOperator.EQUALS,
            value=value,
        ))
        return self
    
    def not_equals(self, field: str, value: str) -> "RoutingFilterBuilder":
        """Add not equals filter.
        
        Args:
            field: Field to filter on
            value: Value to not match
            
        Returns:
            Self for chaining
        """
        self.filters.append(DestinationFilter(
            field=field,
            operator=FilterOperator.NOT_EQUALS,
            value=value,
        ))
        return self
    
    def contains(self, field: str, value: str) -> "RoutingFilterBuilder":
        """Add contains filter.
        
        Args:
            field: Field to filter on
            value: Value that should be contained
            
        Returns:
            Self for chaining
        """
        self.filters.append(DestinationFilter(
            field=field,
            operator=FilterOperator.CONTAINS,
            value=value,
        ))
        return self
    
    def starts_with(self, field: str, value: str) -> "RoutingFilterBuilder":
        """Add starts with filter.
        
        Args:
            field: Field to filter on
            value: Prefix to match
            
        Returns:
            Self for chaining
        """
        self.filters.append(DestinationFilter(
            field=field,
            operator=FilterOperator.STARTS_WITH,
            value=value,
        ))
        return self
    
    def regex(self, field: str, pattern: str) -> "RoutingFilterBuilder":
        """Add regex filter.
        
        Args:
            field: Field to filter on
            pattern: Regex pattern to match
            
        Returns:
            Self for chaining
        """
        self.filters.append(DestinationFilter(
            field=field,
            operator=FilterOperator.REGEX,
            value=pattern,
        ))
        return self
    
    def build(self) -> List[DestinationFilter]:
        """Build and return the filters.
        
        Returns:
            List of destination filters
        """
        return self.filters.copy()


# Global router instance
_router: Optional[DestinationRouter] = None


def get_router(plugin_registry: Optional[PluginRegistry] = None) -> DestinationRouter:
    """Get the global destination router.
    
    Args:
        plugin_registry: Optional plugin registry
        
    Returns:
        DestinationRouter instance
    """
    global _router
    if _router is None:
        _router = DestinationRouter(plugin_registry)
    return _router
