"""Database transaction utilities for safe session management.

This module provides utilities for managing database transactions safely,
including proper commit/rollback handling and session lifecycle management.
"""

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable, TypeVar

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.observability.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


@asynccontextmanager
async def safe_transaction(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncSession, None]:
    """Context manager for safe database transactions.
    
    This context manager ensures:
    1. Session is properly committed on success
    2. Session is rolled back on exception
    3. Session is always closed
    
    Args:
        session_factory: SQLAlchemy async session factory
        
    Yields:
        AsyncSession that is managed by the context
        
    Example:
        >>> async with safe_transaction(session_factory) as session:
        ...     session.add(my_object)
        ...     # Automatically committed if no exception
    """
    session = session_factory()
    try:
        yield session
        await session.commit()
        logger.debug("transaction_committed")
    except SQLAlchemyError as e:
        await session.rollback()
        logger.error("transaction_rolled_back_due_to_sql_error", error=str(e))
        raise
    except Exception as e:
        await session.rollback()
        logger.error("transaction_rolled_back_due_to_error", error=str(e), error_type=type(e).__name__)
        raise
    finally:
        await session.close()
        logger.debug("transaction_session_closed")


async def execute_with_retry(
    session_factory: async_sessionmaker[AsyncSession],
    operation: Callable[[AsyncSession], Any],
    max_retries: int = 3,
    retry_delay: float = 0.1,
) -> Any:
    """Execute a database operation with retry logic.
    
    Useful for handling transient database errors like deadlocks.
    
    Args:
        session_factory: SQLAlchemy async session factory
        operation: Function that takes a session and performs database operations
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        Result of the operation
        
    Raises:
        SQLAlchemyError: If all retries are exhausted
    """
    import asyncio
    
    last_error = None
    
    for attempt in range(max_retries):
        async with safe_transaction(session_factory) as session:
            try:
                result = await operation(session)
                return result
            except SQLAlchemyError as e:
                last_error = e
                logger.warning(
                    "transaction_attempt_failed",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    error=str(e),
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))
    
    if last_error:
        raise last_error


@asynccontextmanager
async def verify_job_exists(
    session: AsyncSession,
    job_id: Any,
) -> AsyncGenerator[bool, None]:
    """Verify that a job exists in the database.
    
    Args:
        session: Database session
        job_id: Job ID to verify
        
    Yields:
        True if job exists, False otherwise
        
    Example:
        >>> async with verify_job_exists(session, job_id) as exists:
        ...     if not exists:
        ...         raise ValueError(f"Job {job_id} not found")
    """
    from uuid import UUID
    
    from sqlalchemy import select
    
    from src.db.models import JobModel
    
    try:
        # Convert string to UUID if needed
        if isinstance(job_id, str):
            job_id = UUID(job_id)
        
        # Query for job without caching to ensure fresh data
        result = await session.execute(
            select(JobModel).where(JobModel.id == job_id)
        )
        job = result.scalar_one_or_none()
        
        exists = job is not None
        if not exists:
            logger.error("job_not_found_for_verification", job_id=str(job_id))
        
        yield exists
        
    except Exception as e:
        logger.error("error_verifying_job_exists", job_id=str(job_id), error=str(e))
        yield False


async def verify_job_exists_simple(
    session: AsyncSession,
    job_id: Any,
) -> bool:
    """Simple check if a job exists.
    
    Args:
        session: Database session
        job_id: Job ID to verify
        
    Returns:
        True if job exists, False otherwise
    """
    from uuid import UUID
    
    from sqlalchemy import select
    
    from src.db.models import JobModel
    
    try:
        if isinstance(job_id, str):
            job_id = UUID(job_id)
        
        result = await session.execute(
            select(JobModel).where(JobModel.id == job_id)
        )
        job = result.scalar_one_or_none()
        return job is not None
    except Exception as e:
        logger.error("error_checking_job_exists", job_id=str(job_id), error=str(e))
        return False
