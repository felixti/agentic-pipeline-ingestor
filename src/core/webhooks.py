"""Webhook system for event notifications.

This module provides webhook functionality for sending event notifications
to external systems with HMAC signature verification and retry logic.
"""

import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

import httpx

logger = logging.getLogger(__name__)


class WebhookEventType(str, Enum):
    """Types of webhook events."""
    JOB_CREATED = "job.created"
    JOB_STARTED = "job.started"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"
    JOB_CANCELLED = "job.cancelled"
    JOB_RETRYING = "job.retrying"
    STAGE_COMPLETED = "stage.completed"
    STAGE_FAILED = "stage.failed"
    PARSER_FALLBACK = "parser.fallback"
    QUALITY_CHECK_FAILED = "quality.check_failed"
    EXPORT_COMPLETED = "export.completed"
    BULK_OPERATION_COMPLETED = "bulk_operation.completed"


class WebhookStatus(str, Enum):
    """Status of webhook delivery attempts."""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"
    EXHAUSTED = "exhausted"


@dataclass
class WebhookEvent:
    """Represents a webhook event."""
    id: UUID
    event_type: WebhookEventType
    timestamp: datetime
    payload: dict[str, Any]
    signature: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": str(self.id),
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
        }

    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict(), separators=(",", ":"))


@dataclass
class WebhookDelivery:
    """Represents a webhook delivery attempt."""
    id: UUID
    event_id: UUID
    webhook_id: UUID
    url: str
    status: WebhookStatus
    attempt_count: int = 0
    last_attempt_at: datetime | None = None
    next_attempt_at: datetime | None = None
    http_status: int | None = None
    response_body: str | None = None
    error_message: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None


@dataclass
class WebhookSubscription:
    """Represents a webhook subscription."""
    id: UUID
    url: str
    secret: str
    event_types: list[WebhookEventType]
    headers: dict[str, str] = field(default_factory=dict)
    active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime | None = None
    description: str | None = None

    # Retry configuration
    max_retries: int = 3
    retry_delays: list[int] = field(default_factory=lambda: [60, 300, 900])  # 1min, 5min, 15min
    timeout_seconds: int = 30

    # Delivery tracking
    total_deliveries: int = 0
    successful_deliveries: int = 0
    failed_deliveries: int = 0


class WebhookSigner:
    """HMAC signature generator and verifier for webhooks."""

    @staticmethod
    def generate_signature(payload: str, secret: str) -> str:
        """Generate HMAC-SHA256 signature for webhook payload.
        
        Args:
            payload: JSON payload string
            secret: Webhook secret key
            
        Returns:
            Hex-encoded HMAC signature
        """
        signature = hmac.new(
            secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        return signature

    @staticmethod
    def verify_signature(payload: str, signature: str, secret: str) -> bool:
        """Verify HMAC-SHA256 signature for webhook payload.
        
        Args:
            payload: JSON payload string
            signature: Expected signature
            secret: Webhook secret key
            
        Returns:
            True if signature is valid
        """
        expected = WebhookSigner.generate_signature(payload, secret)
        # Use constant-time comparison to prevent timing attacks
        return hmac.compare_digest(expected, signature)

    @staticmethod
    def generate_signature_header(payload: str, secret: str) -> str:
        """Generate signature header value.
        
        Args:
            payload: JSON payload string
            secret: Webhook secret key
            
        Returns:
            Signature header value (sha256=<signature>)
        """
        signature = WebhookSigner.generate_signature(payload, secret)
        return f"sha256={signature}"


class WebhookManager:
    """Manages webhook subscriptions and event delivery.
    
    This class handles:
    - Webhook subscription management
    - Event queuing and delivery
    - Retry logic with exponential backoff
    - HMAC signature generation
    
    Example:
        >>> manager = WebhookManager()
        >>> subscription = await manager.subscribe(
        ...     url="https://example.com/webhooks",
        ...     secret="my-secret",
        ...     event_types=[WebhookEventType.JOB_COMPLETED]
        ... )
        >>> await manager.emit(WebhookEventType.JOB_COMPLETED, {"job_id": "123"})
    """

    def __init__(self, http_client: httpx.AsyncClient | None = None):
        """Initialize the webhook manager.
        
        Args:
            http_client: Optional HTTP client for webhook delivery
        """
        self._subscriptions: dict[UUID, WebhookSubscription] = {}
        self._pending_deliveries: list[WebhookDelivery] = []
        self._delivered_events: dict[UUID, WebhookDelivery] = {}
        self._http_client = http_client or httpx.AsyncClient(timeout=30.0)
        self._signer = WebhookSigner()

    async def subscribe(
        self,
        url: str,
        secret: str,
        event_types: list[WebhookEventType],
        description: str | None = None,
        headers: dict[str, str] | None = None,
        max_retries: int = 3,
    ) -> WebhookSubscription:
        """Create a new webhook subscription.
        
        Args:
            url: Webhook endpoint URL
            secret: Secret key for HMAC signature
            event_types: List of event types to subscribe to
            description: Optional description
            headers: Optional custom headers
            max_retries: Maximum retry attempts
            
        Returns:
            Created subscription
        """
        subscription = WebhookSubscription(
            id=uuid4(),
            url=url,
            secret=secret,
            event_types=event_types,
            headers=headers or {},
            description=description,
            max_retries=max_retries,
        )

        self._subscriptions[subscription.id] = subscription
        logger.info(f"Created webhook subscription {subscription.id} for {url}")

        return subscription

    async def unsubscribe(self, subscription_id: UUID) -> bool:
        """Remove a webhook subscription.
        
        Args:
            subscription_id: ID of subscription to remove
            
        Returns:
            True if subscription was removed
        """
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            logger.info(f"Removed webhook subscription {subscription_id}")
            return True
        return False

    async def get_subscription(self, subscription_id: UUID) -> WebhookSubscription | None:
        """Get a webhook subscription by ID.
        
        Args:
            subscription_id: Subscription ID
            
        Returns:
            Subscription or None
        """
        return self._subscriptions.get(subscription_id)

    async def list_subscriptions(
        self,
        event_type: WebhookEventType | None = None,
        active_only: bool = True,
    ) -> list[WebhookSubscription]:
        """List webhook subscriptions.
        
        Args:
            event_type: Filter by event type
            active_only: Only return active subscriptions
            
        Returns:
            List of matching subscriptions
        """
        subscriptions = list(self._subscriptions.values())

        if active_only:
            subscriptions = [s for s in subscriptions if s.active]

        if event_type:
            subscriptions = [
                s for s in subscriptions
                if event_type in s.event_types
            ]

        return subscriptions

    async def emit(
        self,
        event_type: WebhookEventType,
        payload: dict[str, Any],
        event_id: UUID | None = None,
    ) -> list[WebhookDelivery]:
        """Emit a webhook event to all matching subscriptions.
        
        Args:
            event_type: Type of event
            payload: Event payload data
            event_id: Optional event ID (generated if not provided)
            
        Returns:
            List of delivery records
        """
        event = WebhookEvent(
            id=event_id or uuid4(),
            event_type=event_type,
            timestamp=datetime.utcnow(),
            payload=payload,
        )

        # Find matching subscriptions
        subscriptions = await self.list_subscriptions(event_type=event_type)

        if not subscriptions:
            logger.debug(f"No subscriptions for event type {event_type}")
            return []

        deliveries: list[WebhookDelivery] = []

        for subscription in subscriptions:
            delivery = WebhookDelivery(
                id=uuid4(),
                event_id=event.id,
                webhook_id=subscription.id,
                url=subscription.url,
                status=WebhookStatus.PENDING,
                next_attempt_at=datetime.utcnow(),
            )

            self._pending_deliveries.append(delivery)
            deliveries.append(delivery)

            # Attempt immediate delivery
            await self._attempt_delivery(delivery, subscription, event)

        return deliveries

    async def _attempt_delivery(
        self,
        delivery: WebhookDelivery,
        subscription: WebhookSubscription,
        event: WebhookEvent,
    ) -> bool:
        """Attempt to deliver a webhook event.
        
        Args:
            delivery: Delivery record
            subscription: Webhook subscription
            event: Event to deliver
            
        Returns:
            True if delivery was successful
        """
        delivery.attempt_count += 1
        delivery.last_attempt_at = datetime.utcnow()

        # Prepare payload
        payload = event.to_json()

        # Generate signature
        signature = self._signer.generate_signature_header(payload, subscription.secret)

        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "X-Webhook-ID": str(subscription.id),
            "X-Event-ID": str(event.id),
            "X-Event-Type": event.event_type.value,
            "X-Webhook-Timestamp": str(int(time.time())),
            "X-Webhook-Signature": signature,
            **subscription.headers,
        }

        try:
            response = await self._http_client.post(
                subscription.url,
                content=payload,
                headers=headers,
                timeout=subscription.timeout_seconds,
            )

            delivery.http_status = response.status_code
            delivery.response_body = response.text[:1000]  # Limit response size

            # Check for success (2xx status)
            if 200 <= response.status_code < 300:
                delivery.status = WebhookStatus.DELIVERED
                delivery.completed_at = datetime.utcnow()
                subscription.successful_deliveries += 1

                logger.info(
                    f"Webhook {delivery.id} delivered successfully "
                    f"to {subscription.url}"
                )
                return True
            else:
                raise httpx.HTTPStatusError(
                    f"HTTP {response.status_code}",
                    request=response.request,
                    response=response,
                )

        except Exception as e:
            delivery.error_message = str(e)[:500]
            subscription.failed_deliveries += 1

            logger.warning(
                f"Webhook delivery {delivery.id} failed (attempt {delivery.attempt_count}): {e}"
            )

            # Schedule retry if applicable
            if delivery.attempt_count <= subscription.max_retries:
                retry_delay = subscription.retry_delays[
                    min(delivery.attempt_count - 1, len(subscription.retry_delays) - 1)
                ]
                delivery.next_attempt_at = datetime.utcnow() + timedelta(seconds=retry_delay)
                delivery.status = WebhookStatus.RETRYING
            else:
                delivery.status = WebhookStatus.EXHAUSTED
                delivery.completed_at = datetime.utcnow()
                logger.error(
                    f"Webhook {delivery.id} exhausted all retry attempts"
                )

            return False

    async def process_retries(self) -> int:
        """Process pending webhook retries.
        
        Returns:
            Number of deliveries processed
        """
        now = datetime.utcnow()
        retry_deliveries = [
            d for d in self._pending_deliveries
            if d.status == WebhookStatus.RETRYING
            and d.next_attempt_at and d.next_attempt_at <= now
        ]

        count = 0
        for delivery in retry_deliveries:
            subscription = self._subscriptions.get(delivery.webhook_id)
            if not subscription or not subscription.active:
                delivery.status = WebhookStatus.FAILED
                delivery.error_message = "Subscription no longer active"
                continue

            # Recreate event from stored data
            # In a real implementation, events would be persisted
            # For now, we'll fail the delivery
            delivery.status = WebhookStatus.FAILED
            delivery.error_message = "Event data no longer available for retry"
            count += 1

        return count

    async def get_delivery_status(self, delivery_id: UUID) -> WebhookDelivery | None:
        """Get the status of a webhook delivery.
        
        Args:
            delivery_id: Delivery ID
            
        Returns:
            Delivery record or None
        """
        for delivery in self._pending_deliveries:
            if delivery.id == delivery_id:
                return delivery
        return self._delivered_events.get(delivery_id)

    async def verify_webhook(
        self,
        payload: str,
        signature: str,
        secret: str,
    ) -> bool:
        """Verify an incoming webhook signature.
        
        Args:
            payload: Raw request body
            signature: Signature from X-Webhook-Signature header
            secret: Webhook secret
            
        Returns:
            True if signature is valid
        """
        return self._signer.verify_signature(payload, signature, secret)

    async def update_subscription(
        self,
        subscription_id: UUID,
        **kwargs
    ) -> WebhookSubscription | None:
        """Update a webhook subscription.
        
        Args:
            subscription_id: Subscription ID
            **kwargs: Fields to update
            
        Returns:
            Updated subscription or None
        """
        subscription = self._subscriptions.get(subscription_id)
        if not subscription:
            return None

        # Update allowed fields
        allowed_fields = [
            "url", "event_types", "headers", "active", "description",
            "max_retries", "timeout_seconds"
        ]

        for field, value in kwargs.items():
            if field in allowed_fields and hasattr(subscription, field):
                setattr(subscription, field, value)

        subscription.updated_at = datetime.utcnow()
        return subscription

    async def get_delivery_stats(
        self,
        subscription_id: UUID | None = None,
    ) -> dict[str, Any]:
        """Get delivery statistics.
        
        Args:
            subscription_id: Optional subscription ID to filter by
            
        Returns:
            Statistics dictionary
        """
        if subscription_id:
            subscription = self._subscriptions.get(subscription_id)
            if subscription:
                return {
                    "subscription_id": str(subscription_id),
                    "total_deliveries": subscription.total_deliveries,
                    "successful_deliveries": subscription.successful_deliveries,
                    "failed_deliveries": subscription.failed_deliveries,
                    "success_rate": (
                        subscription.successful_deliveries / max(subscription.total_deliveries, 1)
                    ),
                }
            return {}

        # Aggregate stats for all subscriptions
        total = sum(s.total_deliveries for s in self._subscriptions.values())
        successful = sum(s.successful_deliveries for s in self._subscriptions.values())
        failed = sum(s.failed_deliveries for s in self._subscriptions.values())

        return {
            "total_subscriptions": len(self._subscriptions),
            "total_deliveries": total,
            "successful_deliveries": successful,
            "failed_deliveries": failed,
            "success_rate": successful / max(total, 1),
        }

    async def cleanup_old_deliveries(self, max_age_hours: int = 24) -> int:
        """Clean up old delivered events.
        
        Args:
            max_age_hours: Maximum age of deliveries to keep
            
        Returns:
            Number of deliveries removed
        """
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)

        # Move completed deliveries to archive
        completed = [
            d for d in self._pending_deliveries
            if d.status in [WebhookStatus.DELIVERED, WebhookStatus.EXHAUSTED]
            and d.completed_at and d.completed_at < cutoff
        ]

        for delivery in completed:
            self._pending_deliveries.remove(delivery)
            self._delivered_events[delivery.id] = delivery

        # Limit archive size
        if len(self._delivered_events) > 10000:
            oldest = sorted(
                self._delivered_events.items(),
                key=lambda x: x[1].completed_at or datetime.min
            )
            to_remove = len(self._delivered_events) - 10000
            for key, _ in oldest[:to_remove]:
                del self._delivered_events[key]

        return len(completed)


class WebhookMiddleware:
    """Middleware for handling incoming webhooks.
    
    Provides utilities for verifying and processing incoming webhooks
    from external systems.
    """

    def __init__(self, secret: str):
        """Initialize middleware.
        
        Args:
            secret: Secret key for signature verification
        """
        self._secret = secret
        self._signer = WebhookSigner()

    async def verify_request(
        self,
        body: bytes,
        signature_header: str,
    ) -> bool:
        """Verify an incoming webhook request.
        
        Args:
            body: Raw request body
            signature_header: X-Webhook-Signature header value
            
        Returns:
            True if signature is valid
        """
        # Extract signature from header (format: "sha256=<signature>")
        if "=" in signature_header:
            _, signature = signature_header.split("=", 1)
        else:
            signature = signature_header

        return self._signer.verify_signature(
            body.decode("utf-8"),
            signature,
            self._secret
        )

    async def parse_event(self, body: bytes) -> WebhookEvent:
        """Parse webhook event from request body.
        
        Args:
            body: Raw request body
            
        Returns:
            Parsed webhook event
        """
        data = json.loads(body)

        return WebhookEvent(
            id=UUID(data.get("event_id", str(uuid4()))),
            event_type=WebhookEventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            payload=data.get("payload", {}),
        )
