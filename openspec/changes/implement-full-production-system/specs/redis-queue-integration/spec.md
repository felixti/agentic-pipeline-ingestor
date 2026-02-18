# Spec: Redis Queue Integration

## Overview
Use Redis as a message queue for job distribution across multiple workers.

### Requirements

#### R1: Enqueue Job on Creation
**Given** a job is created via API
**When** job is persisted to database
**Then** also push job_id to Redis queue

#### R2: Worker Consumes from Queue
**Given** workers are running
**When** job_id appears in queue
**Then** worker claims and processes the job

#### R3: Job Prioritization
**Given** jobs have different priorities
**When** enqueuing
**Then** use priority queues (high, normal, low)

#### R4: Handle Worker Crashes
**Given** a worker crashes mid-processing
**When** job timeout expires
**Then** requeue job for another worker

#### R5: Queue Monitoring
**Given** Redis is connected
**When** monitoring endpoint is called
**Then** return queue depth, processing rate, worker count

### Scenarios

#### SC1: High Priority Job
User submits urgent job:
- Job created in DB with priority='high'
- Pushed to 'jobs:high' queue
- Worker prioritizes high queue

#### SC2: Multiple Workers
3 workers running:
- All listen to same queue
- Redis distributes jobs round-robin
- Each worker processes different job

#### SC3: Worker Crash Recovery
Worker crashes while processing:
- Job has timeout of 5 minutes
- After timeout, job requeued
- Another worker picks it up
