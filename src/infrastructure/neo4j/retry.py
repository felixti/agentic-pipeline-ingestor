"""Retry utilities for Neo4j operations.

This module provides retry decorators using tenacity for handling transient
Neo4j errors such as deadlocks, connection issues, and timeouts.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from neo4j.exceptions import Neo4jError, ServiceUnavailable
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.observability.logging import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Neo4j error codes that indicate transient failures worthy of retry
NEO4J_TRANSIENT_ERROR_CODES = {
    # Deadlock detected - concurrent transactions conflicting
    "Neo.TransientError.Transaction.DeadlockDetected",
    # Lock acquisition timeout
    "Neo.TransientError.Transaction.LockAcquisitionTimeout",
    # Transaction termination
    "Neo.TransientError.Transaction.Terminated",
    # Transaction timeout
    "Neo.TransientError.Transaction.TransactionTimedOut",
    # Cluster errors
    "Neo.ClientError.Cluster.NotALeader",
    "Neo.TransientError.Cluster.NoLeaderAvailable",
    "Neo.TransientError.Cluster.LeaderSwitch",
    # Database unavailable
    "Neo.TransientError.General.DatabaseUnavailable",
}

NEO4J_CONNECTION_ERRORS = (
    ServiceUnavailable,
    ConnectionError,
    TimeoutError,
)


def _is_transient_neo4j_error(exception: Exception) -> bool:
    """Check if an exception is a transient Neo4j error worth retrying.
    
    Args:
        exception: The exception to check
        
    Returns:
        True if the error is transient and should trigger a retry
    """
    # Check for Neo4jError with specific transient codes
    if isinstance(exception, Neo4jError):
        error_code = getattr(exception, "code", None) or str(exception)
        # Check if the error code matches known transient errors
        for transient_code in NEO4J_TRANSIENT_ERROR_CODES:
            if transient_code in error_code or transient_code in str(exception):
                return True
        # Check for deadlock in message
        if "DeadlockDetected" in str(exception) or "deadlock" in str(exception).lower():
            return True
        # Check for lock acquisition issues
        if "lock" in str(exception).lower() and ("timeout" in str(exception).lower() or "acquire" in str(exception).lower()):
            return True
    
    # Check for connection-related errors
    if isinstance(exception, NEO4J_CONNECTION_ERRORS):
        return True
    
    # Check for ForsetiClient lock errors (common Neo4j deadlock message)
    if "ForsetiClient" in str(exception) and "lock" in str(exception).lower():
        return True
    
    # Check for ExclusiveLock errors
    if "ExclusiveLock" in str(exception) or "can't acquire" in str(exception).lower():
        return True
    
    return False


def _before_retry_log(retry_state: RetryCallState) -> None:
    """Log retry attempts.
    
    Args:
        retry_state: Current state of the retry call
    """
    if retry_state.outcome is None:
        return
    
    exception = retry_state.outcome.exception()
    if exception is None:
        return
    
    logger.warning(
        "neo4j_retry_attempt",
        attempt=retry_state.attempt_number,
        max_attempts=4,
        error=str(exception),
        error_type=type(exception).__name__,
        wait_time=retry_state.next_action.sleep if retry_state.next_action else None,
    )


def neo4j_retry(
    max_attempts: int = 4,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
    exponential_base: float = 2.0,
) -> Callable[[F], F]:
    """Decorator for retrying Neo4j operations with exponential backoff.
    
    This decorator retries operations that fail due to transient Neo4j errors
    such as deadlocks, connection issues, and timeouts.
    
    Args:
        max_attempts: Maximum number of retry attempts (default: 4)
        min_wait: Minimum wait time in seconds before first retry (default: 1.0)
        max_wait: Maximum wait time in seconds between retries (default: 10.0)
        exponential_base: Base for exponential backoff calculation (default: 2.0)
        
    Returns:
        Decorated function with retry logic
        
    Example:
        >>> @neo4j_retry(max_attempts=4)
        ... async def create_node(session, data):
        ...     await session.run("CREATE (n:Node {data: $data})", data=data)
        
        >>> @neo4j_retry(max_attempts=3, min_wait=0.5)
        ... def get_node_data(tx, node_id):
        ...     return tx.run("MATCH (n {id: $id}) RETURN n", id=node_id).data()
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=1,
            min=min_wait,
            max=max_wait,
            exp_base=exponential_base,
        ),
        retry=retry_if_exception_type(
            (Neo4jError, ServiceUnavailable, ConnectionError, TimeoutError)
        ),
        before_sleep=_before_retry_log,
        reraise=True,
    )


def neo4j_retry_conditional(
    max_attempts: int = 4,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
    exponential_base: float = 2.0,
) -> Callable[[F], F]:
    """Decorator for retrying Neo4j operations with conditional error checking.
    
    Similar to neo4j_retry but only retries if the error is detected to be
    a transient error (deadlock, connection issue, etc.).
    
    Args:
        max_attempts: Maximum number of retry attempts (default: 4)
        min_wait: Minimum wait time in seconds before first retry (default: 1.0)
        max_wait: Maximum wait time in seconds between retries (default: 10.0)
        exponential_base: Base for exponential backoff calculation (default: 2.0)
        
    Returns:
        Decorated function with conditional retry logic
        
    Example:
        >>> @neo4j_retry_conditional(max_attempts=4)
        ... async def update_graph(session, data):
        ...     await session.run("MERGE (n:Node {id: $id}) SET n.data = $data", id=data.id, data=data)
    """
    from tenacity import retry_if_exception

    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=1,
            min=min_wait,
            max=max_wait,
            exp_base=exponential_base,
        ),
        retry=retry_if_exception(_is_transient_neo4j_error),
        before_sleep=_before_retry_log,
        reraise=True,
    )


class Neo4jRetryContext:
    """Context manager for Neo4j retry operations.
    
    Provides a context-based approach for retrying Neo4j operations
    with detailed logging and metrics.
    
    Example:
        >>> async with Neo4jRetryContext(operation="graph_write") as ctx:
        ...     await ctx.run(session.run, "CREATE (n:Node {id: $id})", id=node_id)
    """

    def __init__(
        self,
        operation: str,
        max_attempts: int = 4,
        min_wait: float = 1.0,
        max_wait: float = 10.0,
    ) -> None:
        """Initialize retry context.
        
        Args:
            operation: Name of the operation for logging
            max_attempts: Maximum retry attempts
            min_wait: Minimum wait time between retries
            max_wait: Maximum wait time between retries
        """
        self.operation = operation
        self.max_attempts = max_attempts
        self.min_wait = min_wait
        self.max_wait = max_wait
        self.attempt = 0
        self.last_error: Exception | None = None

    async def run(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute a function with retry logic.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Result of the function call
            
        Raises:
            Exception: The last exception if all retries fail
        """
        from asyncio import sleep
        import random

        for attempt in range(1, self.max_attempts + 1):
            self.attempt = attempt
            try:
                result = await func(*args, **kwargs)
                
                if attempt > 1:
                    logger.info(
                        "neo4j_retry_success",
                        operation=self.operation,
                        attempts=attempt,
                    )
                
                return result
                
            except Exception as e:
                self.last_error = e
                
                if not _is_transient_neo4j_error(e):
                    logger.error(
                        "neo4j_non_transient_error",
                        operation=self.operation,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    raise
                
                if attempt >= self.max_attempts:
                    logger.error(
                        "neo4j_retry_exhausted",
                        operation=self.operation,
                        attempts=attempt,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    raise
                
                # Calculate wait time with exponential backoff and jitter
                wait_time = min(
                    self.min_wait * (2 ** (attempt - 1)),
                    self.max_wait,
                )
                jitter = random.uniform(0, 1)
                total_wait = wait_time + jitter
                
                logger.warning(
                    "neo4j_retry_waiting",
                    operation=self.operation,
                    attempt=attempt,
                    max_attempts=self.max_attempts,
                    wait_seconds=round(total_wait, 2),
                    error=str(e),
                )
                
                await sleep(total_wait)
        
        # This should never be reached, but just in case
        if self.last_error:
            raise self.last_error
        return None


def is_transient_error(exception: Exception) -> bool:
    """Check if an exception is a transient Neo4j error.
    
    Utility function to check if an error is worth retrying.
    
    Args:
        exception: The exception to check
        
    Returns:
        True if the error is transient
        
    Example:
        >>> try:
        ...     await session.run(query)
        ... except Exception as e:
        ...     if is_transient_error(e):
        ...         # Handle retry logic
        ...     else:
        ...         raise
    """
    return _is_transient_neo4j_error(exception)
