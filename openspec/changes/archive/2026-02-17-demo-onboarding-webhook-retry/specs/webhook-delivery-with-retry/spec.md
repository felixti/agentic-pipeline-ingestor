# Spec: Webhook Delivery with Retry

## Overview

Enhanced webhook delivery service that integrates retry logic with the existing webhook system, supporting async execution and DLQ routing.

### Requirements

#### REQ-1: Async Delivery with Retry
**Given** a webhook job configured with retry  
**When** the initial delivery fails with a retryable error  
**Then** the job enters "retrying" state and schedules next attempt

#### REQ-2: Retry State Tracking
**Given** a job in retry state  
**When** checking job status  
**Then** it shows current attempt number, next retry time, and failure history

#### REQ-3: Success Completion
**Given** a job on its 3rd retry attempt  
**When** the delivery succeeds  
**Then** the job is marked complete with total_attempts = 3

#### REQ-4: Max Retries Exhausted
**Given** a job with max_retries = 3  
**When** all 3 retries fail  
**Then** the job is marked failed and routed to DLQ

#### REQ-5: Manual Retry Support
**Given** a failed job in DLQ  
**When** an operator triggers manual retry  
**Then** the job restarts from attempt 1 with fresh retry configuration

### State Machine

```
PENDING → DELIVERING → [SUCCESS] → COMPLETED
                    → [RETRYABLE] → RETRYING → DELIVERING
                                          → [MAX RETRIES] → FAILED → DLQ
                    → [NON-RETRYABLE] → FAILED → DLQ
```

### Scenarios

#### SC-1: Successful Retry
Initial delivery fails with 503. System waits 2s, retries, succeeds. Job shows: attempts=2, status=completed.

#### SC-2: All Retries Exhausted
Delivery fails 4 times (initial + 3 retries). Final error logged. Job routed to DLQ for manual inspection.

#### SC-3: Manual Recovery
Operator reviews DLQ job, fixes destination URL, triggers manual retry. Job restarts and succeeds.
