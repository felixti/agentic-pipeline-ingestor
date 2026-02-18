# Spec: Retry Metrics and Observability

## Overview

Comprehensive observability for retry operations including metrics, events, and tracing to monitor retry effectiveness and diagnose issues.

### Requirements

#### REQ-1: Retry Attempt Events
**Given** a webhook retry attempt  
**When** the attempt occurs  
**Then** a structured event is emitted with attempt number, delay, and error details

#### REQ-2: Success Rate Metrics
**Given** the retry system is operational  
**When** metrics are collected  
**Then** counters track: total_deliveries, successful_first_attempt, successful_after_retry, failed_permanently

#### REQ-3: Retry Duration Histogram
**Given** completed webhook jobs  
**When** metrics are collected  
**Then** a histogram tracks time from first attempt to final resolution

#### REQ-4: Per-Endpoint Retry Stats
**Given** multiple webhook endpoints  
**When** viewing metrics  
**Then** retry statistics are available per endpoint for identifying problematic destinations

#### REQ-5: Distributed Tracing
**Given** a retry operation  
**When** traces are collected  
**Then** each retry attempt appears as a span with attributes: attempt_number, delay_ms, error_type

### Metrics Schema

```yaml
# Counters
webhook_delivery_total{endpoint_id, status}
webhook_retry_total{endpoint_id, attempt_number}

# Histograms
webhook_retry_delay_seconds{endpoint_id}
webhook_delivery_duration_seconds{endpoint_id}

# Gauges
webhook_pending_retries{endpoint_id}
webhook_dlq_size{endpoint_id}
```

### Event Structure

```json
{
  "event_type": "webhook.retry_attempt",
  "job_id": "uuid",
  "endpoint_id": "endpoint-123",
  "attempt_number": 2,
  "delay_ms": 2045,
  "error_code": 503,
  "error_message": "Service Unavailable",
  "timestamp": "2026-02-17T14:30:00Z"
}
```

### Scenarios

#### SC-1: Dashboard View
Operations team views Grafana dashboard showing 95% success rate, average 1.2 retries per failed delivery, 3 endpoints with elevated retry rates.

#### SC-2: Trace Investigation
Developer traces a specific job ID and sees: initial attempt (503) → 2s delay → retry 1 (503) → 4s delay → retry 2 (200 success). Total time: 7.2s.

#### SC-3: Alert on High Retry Rate
Alert fires when endpoint "payment-webhook" has >50% retry rate in 5-minute window. Team investigates and finds destination service degradation.
