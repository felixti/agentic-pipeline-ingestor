# Spec: Exponential Backoff Engine

## Overview

Core retry logic implementing exponential backoff with jitter to prevent thundering herd problems when services recover.

### Requirements

#### REQ-1: Exponential Delay Calculation
**Given** base_delay = 1s and attempt = 3  
**When** calculating the next retry delay  
**Then** the delay is base_delay × 2^(attempt-1) = 4s (before jitter)

#### REQ-2: Jitter Application
**Given** a calculated delay of 4s and jitter enabled  
**When** the actual delay is applied  
**Then** a random value between 0.5× and 1.5× is used (2s - 6s range)

#### REQ-3: Max Delay Cap
**Given** max_delay = 300s and calculated delay = 600s  
**When** the delay is applied  
**Then** it is capped at max_delay (300s)

#### REQ-4: Retryable Error Classification
**Given** an HTTP response with status 503  
**When** checking if retry should occur  
**Then** it is classified as retryable

#### REQ-5: Non-Retryable Error Classification
**Given** an HTTP response with status 404  
**When** checking if retry should occur  
**Then** it is classified as non-retryable (fail immediately)

### Algorithm

```python
def calculate_delay(attempt, base_delay, max_delay, enable_jitter):
    # Exponential calculation
    delay = base_delay * (2 ** (attempt - 1))
    delay = min(delay, max_delay)
    
    # Apply jitter (±50%)
    if enable_jitter:
        jitter = random.uniform(0.5, 1.5)
        delay = delay * jitter
    
    return delay
```

### Scenarios

#### SC-1: Normal Exponential Progression
Attempts with base_delay=1s: delays are ~1s, ~2s, ~4s, ~8s (with jitter variance)

#### SC-2: Max Delay Cap Applied
After 9 attempts with base_delay=1s, calculated delay would be 512s. With max_delay=300s, actual delay is capped at 300s.

#### SC-3: 4xx Error Fails Immediately
A 400 Bad Request response triggers immediate failure without any retries.
