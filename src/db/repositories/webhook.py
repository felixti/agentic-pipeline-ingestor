"""Repository for webhook subscription and delivery data access."""

from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from sqlalchemy import asc, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import WebhookDeliveryModel, WebhookSubscriptionModel


class WebhookRepository:
    """Repository for webhook operations."""
    
    def __init__(self, session: AsyncSession):
        """Initialize repository.
        
        Args:
            session: SQLAlchemy async session
        """
        self.session = session
    
    # Subscription methods
    
    async def create_subscription(
        self,
        user_id: str,
        url: str,
        events: list[str],
        secret: str | None = None,
    ) -> WebhookSubscriptionModel:
        """Create a new webhook subscription.
        
        Args:
            user_id: User ID
            url: Webhook URL
            events: List of event types to subscribe to
            secret: Secret for HMAC signature
            
        Returns:
            Created WebhookSubscriptionModel
        """
        sub = WebhookSubscriptionModel(
            user_id=user_id,
            url=url,
            events=events,
            secret=secret,
            is_active=1,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        
        self.session.add(sub)
        await self.session.commit()
        await self.session.refresh(sub)
        
        return sub
    
    async def get_subscription(
        self,
        subscription_id: str | UUID,
        user_id: str | None = None,
    ) -> Optional[WebhookSubscriptionModel]:
        """Get subscription by ID.
        
        Args:
            subscription_id: Subscription ID
            user_id: Optional user ID filter
            
        Returns:
            WebhookSubscriptionModel if found, None otherwise
        """
        if isinstance(subscription_id, str):
            try:
                subscription_id = UUID(subscription_id)
            except ValueError:
                return None
        
        query = select(WebhookSubscriptionModel).where(
            WebhookSubscriptionModel.id == subscription_id
        )
        
        if user_id:
            query = query.where(WebhookSubscriptionModel.user_id == user_id)
        
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def list_subscriptions(
        self,
        user_id: str | None = None,
        event: str | None = None,
        page: int = 1,
        limit: int = 20,
    ) -> tuple[list[WebhookSubscriptionModel], int]:
        """List webhook subscriptions.
        
        Args:
            user_id: Filter by user
            event: Filter by event type
            page: Page number
            limit: Items per page
            
        Returns:
            Tuple of (subscriptions list, total count)
        """
        query = select(WebhookSubscriptionModel).where(
            WebhookSubscriptionModel.is_active == 1
        )
        count_query = select(func.count(WebhookSubscriptionModel.id)).where(
            WebhookSubscriptionModel.is_active == 1
        )
        
        if user_id:
            query = query.where(WebhookSubscriptionModel.user_id == user_id)
            count_query = count_query.where(WebhookSubscriptionModel.user_id == user_id)
        
        if event:
            query = query.where(WebhookSubscriptionModel.events.contains([event]))
            count_query = count_query.where(WebhookSubscriptionModel.events.contains([event]))
        
        # Get total count
        count_result = await self.session.execute(count_query)
        total = count_result.scalar()
        
        # Apply sorting and pagination
        query = query.order_by(desc(WebhookSubscriptionModel.created_at))
        offset = (page - 1) * limit
        query = query.offset(offset).limit(limit)
        
        result = await self.session.execute(query)
        subs = result.scalars().all()
        
        return list(subs), total
    
    async def list_subscriptions_for_event(
        self,
        event: str,
    ) -> list[WebhookSubscriptionModel]:
        """List all active subscriptions for an event type.
        
        Args:
            event: Event type
            
        Returns:
            List of matching subscriptions
        """
        query = select(WebhookSubscriptionModel).where(
            WebhookSubscriptionModel.is_active == 1
        ).where(
            WebhookSubscriptionModel.events.contains([event])
        )
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def update_subscription(
        self,
        subscription_id: str | UUID,
        url: str | None = None,
        events: list[str] | None = None,
        secret: str | None = None,
    ) -> Optional[WebhookSubscriptionModel]:
        """Update a subscription.
        
        Args:
            subscription_id: Subscription ID
            url: New URL
            events: New events list
            secret: New secret
            
        Returns:
            Updated model if found, None otherwise
        """
        sub = await self.get_subscription(subscription_id)
        if not sub:
            return None
        
        if url is not None:
            sub.url = url
        if events is not None:
            sub.events = events
        if secret is not None:
            sub.secret = secret
        
        sub.updated_at = datetime.utcnow()
        await self.session.commit()
        await self.session.refresh(sub)
        
        return sub
    
    async def delete_subscription(
        self,
        subscription_id: str | UUID,
        user_id: str | None = None,
    ) -> bool:
        """Delete a subscription.
        
        Args:
            subscription_id: Subscription ID
            user_id: Optional user ID filter
            
        Returns:
            True if deleted, False if not found
        """
        sub = await self.get_subscription(subscription_id, user_id)
        if not sub:
            return False
        
        await self.session.delete(sub)
        await self.session.commit()
        
        return True
    
    # Delivery methods
    
    async def create_delivery(
        self,
        subscription_id: str | UUID,
        event_type: str,
        payload: dict,
    ) -> WebhookDeliveryModel:
        """Create a delivery record.
        
        Args:
            subscription_id: Subscription ID
            event_type: Event type
            payload: Payload to deliver
            
        Returns:
            Created WebhookDeliveryModel
        """
        if isinstance(subscription_id, str):
            subscription_id = UUID(subscription_id)
        
        delivery = WebhookDeliveryModel(
            subscription_id=subscription_id,
            event_type=event_type,
            payload=payload,
            status="pending",
            attempts=0,
            max_attempts=5,
            created_at=datetime.utcnow(),
            next_retry_at=datetime.utcnow(),
        )
        
        self.session.add(delivery)
        await self.session.commit()
        await self.session.refresh(delivery)
        
        return delivery
    
    async def get_delivery(
        self,
        delivery_id: str | UUID,
    ) -> Optional[WebhookDeliveryModel]:
        """Get delivery by ID.
        
        Args:
            delivery_id: Delivery ID
            
        Returns:
            WebhookDeliveryModel if found, None otherwise
        """
        if isinstance(delivery_id, str):
            try:
                delivery_id = UUID(delivery_id)
            except ValueError:
                return None
        
        result = await self.session.execute(
            select(WebhookDeliveryModel).where(WebhookDeliveryModel.id == delivery_id)
        )
        return result.scalar_one_or_none()
    
    async def list_deliveries(
        self,
        subscription_id: str | UUID | None = None,
        status: str | None = None,
        page: int = 1,
        limit: int = 20,
    ) -> tuple[list[WebhookDeliveryModel], int]:
        """List webhook deliveries.
        
        Args:
            subscription_id: Filter by subscription
            status: Filter by status
            page: Page number
            limit: Items per page
            
        Returns:
            Tuple of (deliveries list, total count)
        """
        query = select(WebhookDeliveryModel)
        count_query = select(func.count(WebhookDeliveryModel.id))
        
        if subscription_id:
            if isinstance(subscription_id, str):
                subscription_id = UUID(subscription_id)
            query = query.where(WebhookDeliveryModel.subscription_id == subscription_id)
            count_query = count_query.where(WebhookDeliveryModel.subscription_id == subscription_id)
        
        if status:
            query = query.where(WebhookDeliveryModel.status == status)
            count_query = count_query.where(WebhookDeliveryModel.status == status)
        
        # Get total count
        count_result = await self.session.execute(count_query)
        total = count_result.scalar()
        
        # Apply sorting and pagination
        query = query.order_by(desc(WebhookDeliveryModel.created_at))
        offset = (page - 1) * limit
        query = query.offset(offset).limit(limit)
        
        result = await self.session.execute(query)
        deliveries = result.scalars().all()
        
        return list(deliveries), total
    
    async def update_delivery_status(
        self,
        delivery_id: str | UUID,
        status: str,
        http_status: int | None = None,
        error: str | None = None,
    ) -> Optional[WebhookDeliveryModel]:
        """Update delivery status.
        
        Args:
            delivery_id: Delivery ID
            status: New status
            http_status: HTTP response status
            error: Error message
            
        Returns:
            Updated model if found, None otherwise
        """
        delivery = await self.get_delivery(delivery_id)
        if not delivery:
            return None
        
        delivery.status = status
        delivery.attempts += 1
        
        if http_status is not None:
            delivery.http_status = http_status
        
        if error is not None:
            delivery.last_error = error
        
        if status == "delivered":
            delivery.delivered_at = datetime.utcnow()
            delivery.next_retry_at = None
        elif status == "failed":
            delivery.next_retry_at = None
        else:
            # Schedule next retry with exponential backoff
            retry_delays = [0, 60, 300, 900, 3600]  # 0, 1min, 5min, 15min, 1hour
            delay = retry_delays[min(delivery.attempts, len(retry_delays) - 1)]
            delivery.next_retry_at = datetime.utcnow() + timedelta(seconds=delay)
        
        await self.session.commit()
        await self.session.refresh(delivery)
        
        return delivery
    
    async def get_pending_deliveries(
        self,
        limit: int = 100,
    ) -> list[WebhookDeliveryModel]:
        """Get pending deliveries that are due for retry.
        
        Args:
            limit: Maximum number to return
            
        Returns:
            List of pending deliveries
        """
        now = datetime.utcnow()
        
        query = (
            select(WebhookDeliveryModel)
            .where(WebhookDeliveryModel.status.in_(["pending", "retrying"]))
            .where(WebhookDeliveryModel.next_retry_at <= now)
            .where(WebhookDeliveryModel.attempts < WebhookDeliveryModel.max_attempts)
            .order_by(WebhookDeliveryModel.next_retry_at)
            .limit(limit)
        )
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
