"""Webhook delivery service.

This module provides functionality for delivering webhook events
to subscribed endpoints with retry logic and HMAC signatures.
"""

import hashlib
import hmac
import json
import logging
from datetime import datetime
from typing import Any

import httpx

from src.config import settings
from src.db.models import get_async_engine
from src.db.repositories.webhook import WebhookRepository

logger = logging.getLogger(__name__)


class WebhookDeliveryService:
    """Service for delivering webhook events.
    
    Handles:
    - HTTP POST delivery
    - HMAC-SHA256 signature generation
    - Exponential backoff retry
    - Delivery status tracking
    
    Example:
        >>> service = WebhookDeliveryService()
        >>> await service.deliver_event("job.completed", {"job_id": "..."})
    """
    
    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 5,
    ) -> None:
        """Initialize webhook delivery service.
        
        Args:
            timeout: HTTP request timeout in seconds
            max_retries: Maximum delivery attempts
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: httpx.AsyncClient | None = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    async def _generate_signature(
        self,
        payload: str,
        secret: str,
    ) -> str:
        """Generate HMAC-SHA256 signature for webhook payload.
        
        Args:
            payload: JSON payload string
            secret: Webhook secret
            
        Returns:
            Hex-encoded signature
        """
        signature = hmac.new(
            secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return signature
    
    async def _deliver_to_subscription(
        self,
        subscription: Any,
        event_type: str,
        payload: dict[str, Any],
    ) -> tuple[bool, int | None, str | None]:
        """Deliver event to a single subscription.
        
        Args:
            subscription: WebhookSubscriptionModel
            event_type: Event type
            payload: Event payload
            
        Returns:
            Tuple of (success, http_status, error_message)
        """
        # Build payload
        event_payload = {
            "event": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": payload,
        }
        payload_str = json.dumps(event_payload, default=str)
        
        # Generate headers
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "PipelineIngestor/1.0",
            "X-Webhook-Event": event_type,
            "X-Webhook-Timestamp": str(int(datetime.utcnow().timestamp())),
        }
        
        # Add signature if secret is configured
        if subscription.secret:
            signature = await self._generate_signature(payload_str, subscription.secret)
            headers["X-Webhook-Signature"] = f"sha256={signature}"
        
        try:
            client = await self._get_client()
            response = await client.post(
                subscription.url,
                content=payload_str,
                headers=headers,
            )
            
            # Consider 2xx responses as success
            success = 200 <= response.status_code < 300
            
            if success:
                logger.info(  # type: ignore[call-arg]
                    "webhook_delivered",
                    subscription_id=str(subscription.id),
                    event=event_type,
                    status_code=response.status_code,
                )
            else:
                logger.warning(  # type: ignore[call-arg]
                    "webhook_delivery_failed",
                    subscription_id=str(subscription.id),
                    event=event_type,
                    status_code=response.status_code,
                    response=response.text[:200],
                )
            
            return success, response.status_code, None
            
        except httpx.TimeoutException:
            logger.warning(  # type: ignore[call-arg]
                "webhook_delivery_timeout",
                subscription_id=str(subscription.id),
                event=event_type,
                url=subscription.url,
            )
            return False, None, "Request timeout"
            
        except Exception as e:
            logger.error(  # type: ignore[call-arg]
                "webhook_delivery_error",
                subscription_id=str(subscription.id),
                event=event_type,
                error=str(e),
            )
            return False, None, str(e)
    
    async def deliver_event(
        self,
        event_type: str,
        payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Deliver event to all matching subscriptions.
        
        Args:
            event_type: Event type (e.g., "job.completed")
            payload: Event payload
            
        Returns:
            List of delivery results
        """
        from sqlalchemy.ext.asyncio import AsyncSession
        
        engine = get_async_engine()
        results = []
        
        async with AsyncSession(engine) as session:
            repo = WebhookRepository(session)
            
            # Get subscriptions for this event
            subscriptions = await repo.list_subscriptions_for_event(event_type)
            
            if not subscriptions:
                return []
            
            for sub in subscriptions:
                # Create delivery record
                delivery = await repo.create_delivery(
                    subscription_id=str(sub.id),
                    event_type=event_type,
                    payload=payload,
                )
                
                # Attempt delivery
                success, http_status, error = await self._deliver_to_subscription(
                    sub, event_type, payload
                )
                
                # Update delivery status
                if success:
                    await repo.update_delivery_status(
                        str(delivery.id),
                        status="delivered",
                        http_status=http_status,
                    )
                else:
                    await repo.update_delivery_status(
                        str(delivery.id),
                        status="pending",  # Will be retried
                        http_status=http_status,
                        error=error,
                    )
                
                results.append({
                    "subscription_id": str(sub.id),
                    "delivery_id": str(delivery.id),
                    "success": success,
                    "status_code": http_status,
                    "error": error,
                })
        
        return results
    
    async def process_pending_deliveries(self) -> int:
        """Process pending webhook deliveries with retry.
        
        Returns:
            Number of deliveries processed
        """
        from sqlalchemy.ext.asyncio import AsyncSession
        
        engine = get_async_engine()
        processed = 0
        
        async with AsyncSession(engine) as session:
            repo = WebhookRepository(session)
            
            # Get pending deliveries
            pending = await repo.get_pending_deliveries(limit=100)
            
            for delivery in pending:
                # Get subscription
                sub = await repo.get_subscription(str(delivery.subscription_id))
                if not sub or not sub.is_active:
                    # Mark as failed if subscription is gone
                    await repo.update_delivery_status(
                        str(delivery.id),
                        status="failed",
                        error="Subscription no longer active",
                    )
                    continue
                
                # Attempt delivery
                success, http_status, error = await self._deliver_to_subscription(
                    sub,
                    str(delivery.event_type),
                    dict(delivery.payload),
                )
                
                # Update status
                if success:
                    await repo.update_delivery_status(
                        str(delivery.id),
                        status="delivered",
                        http_status=http_status,
                    )
                    processed += 1
                else:
                    # Check if max retries reached
                    if delivery.attempts + 1 >= delivery.max_attempts:
                        await repo.update_delivery_status(
                            str(delivery.id),
                            status="failed",
                            http_status=http_status,
                            error=error,
                        )
                    else:
                        await repo.update_delivery_status(
                            str(delivery.id),
                            status="retrying",
                            http_status=http_status,
                            error=error,
                        )
                    processed += 1
        
        return processed
    
    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Global service instance
_webhook_service: WebhookDeliveryService | None = None


def get_webhook_service() -> WebhookDeliveryService:
    """Get or create global webhook service instance."""
    global _webhook_service
    if _webhook_service is None:
        _webhook_service = WebhookDeliveryService()
    return _webhook_service
