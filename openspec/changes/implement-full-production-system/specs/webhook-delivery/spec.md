# Spec: Webhook Delivery

## Overview
Deliver webhook notifications for job events to external systems.

### Requirements

#### R1: Webhook Subscription
**Given** a user wants notifications
**When** they subscribe to webhook
**Then** store subscription with:
  - URL
  - events (job.completed, job.failed)
  - secret for signature
  - retry configuration

#### R2: Deliver on Event
**Given** a job event occurs
**When** job status changes
**Then** POST to all subscribed webhooks

#### R3: Retry Failed Deliveries
**Given** webhook delivery fails
**When** HTTP error or timeout
**Then** retry with exponential backoff (max 5 attempts)

#### R4: Delivery Status Tracking
**Given** webhooks are configured
**When** checking status
**Then** show delivery attempts, status codes, next retry

#### R5: Webhook Signature
**Given** webhook is delivered
**When** payload is sent
**Then** include HMAC-SHA256 signature for verification

### Scenarios

#### SC1: Job Completion Notification
Job completes:
- POST to https://customer.com/webhook
- Payload: {event: "job.completed", job_id: "...", result: {...}}
- Signature: X-Webhook-Signature header

#### SC2: Retry Failed Webhook
Webhook returns 500:
- Attempt 1: immediate
- Attempt 2: after 1 minute
- Attempt 3: after 5 minutes
- Attempt 4: after 15 minutes
- Attempt 5: after 1 hour
- Then mark as failed
