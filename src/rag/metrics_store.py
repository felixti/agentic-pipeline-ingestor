from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import redis.asyncio as redis

from src.config import get_settings
from src.rag.models import QueryType, RAGMetrics

if TYPE_CHECKING:
    from src.api.models.rag import RAGMetricsSummary

logger = logging.getLogger(__name__)


class RAGMetricsStore:
    KEY_PREFIX = "rag:metrics"
    DEFAULT_TTL_SECONDS = 7 * 24 * 60 * 60
    ROLLING_DAYS = 7

    def __init__(
        self,
        redis_client: redis.Redis | None = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        self.redis = redis_client
        self.ttl_seconds = ttl_seconds

    async def _get_redis(self) -> redis.Redis:
        if self.redis is None:
            settings = get_settings()
            self.redis = await redis.from_url(
                str(settings.redis.url),
                encoding="utf-8",
                decode_responses=True,
            )
        return self.redis

    def _date_key(self, day: datetime) -> str:
        return day.date().isoformat()

    def _make_key(self, day: datetime, metric_type: str) -> str:
        return f"{self.KEY_PREFIX}:{self._date_key(day)}:{metric_type}"

    async def record(
        self,
        metrics: RAGMetrics,
        strategy_used: str | None = None,
        query_type: QueryType | str | None = None,
    ) -> None:
        redis_client = await self._get_redis()
        now = datetime.now(UTC)

        summary_key = self._make_key(now, "summary")
        strategy_key = self._make_key(now, "strategy")
        query_type_key = self._make_key(now, "query_type")

        strategy_name = strategy_used or "unknown"
        query_type_name = (
            query_type.value if isinstance(query_type, QueryType) else (query_type or "unknown")
        )

        async with redis_client.pipeline(transaction=True) as pipe:
            pipe.hincrby(summary_key, "total_queries", 1)
            pipe.hincrbyfloat(summary_key, "latency_sum_ms", float(metrics.latency_ms))
            pipe.hincrbyfloat(summary_key, "retrieval_score_sum", float(metrics.retrieval_score))
            pipe.hincrbyfloat(
                summary_key,
                "classification_confidence_sum",
                float(metrics.classification_confidence),
            )
            pipe.hincrby(summary_key, "tokens_sum", int(metrics.tokens_used))
            pipe.hincrby(strategy_key, strategy_name, 1)
            pipe.hincrby(query_type_key, query_type_name, 1)
            pipe.expire(summary_key, self.ttl_seconds)
            pipe.expire(strategy_key, self.ttl_seconds)
            pipe.expire(query_type_key, self.ttl_seconds)
            await pipe.execute()

    async def get_summary(self) -> "RAGMetricsSummary":
        from src.api.models.rag import RAGMetricsSummary

        try:
            redis_client = await self._get_redis()
        except Exception as exc:
            logger.warning("rag_metrics_redis_init_failed", exc_info=exc)
            return RAGMetricsSummary()

        total_queries = 0
        latency_sum_ms = 0.0
        retrieval_score_sum = 0.0
        classification_confidence_sum = 0.0
        strategy_usage: dict[str, int] = {}
        query_type_distribution: dict[str, int] = {}

        today = datetime.now(UTC)

        try:
            for day_offset in range(self.ROLLING_DAYS):
                day = today - timedelta(days=day_offset)
                summary_key = self._make_key(day, "summary")
                strategy_key = self._make_key(day, "strategy")
                query_type_key = self._make_key(day, "query_type")

                summary_data = await redis_client.hgetall(summary_key)
                if summary_data:
                    total_queries += int(summary_data.get("total_queries", 0))
                    latency_sum_ms += float(summary_data.get("latency_sum_ms", 0.0))
                    retrieval_score_sum += float(summary_data.get("retrieval_score_sum", 0.0))
                    classification_confidence_sum += float(
                        summary_data.get("classification_confidence_sum", 0.0)
                    )

                strategy_data = await redis_client.hgetall(strategy_key)
                for strategy, count in strategy_data.items():
                    strategy_usage[strategy] = strategy_usage.get(strategy, 0) + int(count)

                query_type_data = await redis_client.hgetall(query_type_key)
                for qtype, count in query_type_data.items():
                    query_type_distribution[qtype] = query_type_distribution.get(qtype, 0) + int(
                        count
                    )

            avg_latency_ms = latency_sum_ms / total_queries if total_queries > 0 else 0.0
            avg_retrieval_score = retrieval_score_sum / total_queries if total_queries > 0 else 0.0
            avg_classification_confidence = (
                classification_confidence_sum / total_queries if total_queries > 0 else 0.0
            )

            return RAGMetricsSummary(
                total_queries=total_queries,
                avg_latency_ms=avg_latency_ms,
                avg_retrieval_score=avg_retrieval_score,
                avg_classification_confidence=avg_classification_confidence,
                strategy_usage=strategy_usage,
                query_type_distribution=query_type_distribution,
            )
        except Exception as exc:
            logger.warning("rag_metrics_summary_failed", exc_info=exc)
            return RAGMetricsSummary()
