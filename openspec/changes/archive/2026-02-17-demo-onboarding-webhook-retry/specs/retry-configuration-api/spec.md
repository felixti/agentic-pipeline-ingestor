# Spec: Retry Configuration API

## Overview

Configuration API for webhook retry behavior. Allows setting max retries, delays, and backoff strategy per webhook endpoint or globally.

### Requirements

#### REQ-1: Global Default Configuration
**Given** the system has no endpoint-specific configuration  
**When** a webhook delivery fails  
**Then** it uses the global default retry configuration

#### REQ-2: Endpoint-Specific Override
**Given** an endpoint has custom retry configuration  
**When** a webhook delivery to that endpoint fails  
**Then** it uses the endpoint-specific configuration instead of global defaults

#### REQ-3: Configuration Validation
**Given** a configuration with max_retries > 10  
**When** the configuration is saved  
**Then** the system rejects it with a validation error

### Configuration Schema

```yaml
retry_config:
  max_retries: 3              # 0-10
  base_delay_seconds: 1       # >= 1
  max_delay_seconds: 300      # >= base_delay
  backoff_strategy: exponential  # exponential | linear | fixed
  retryable_status_codes: [408, 429, 500, 502, 503, 504]
  enable_jitter: true
```

### Scenarios

#### SC-1: Default Configuration Applied
A new webhook endpoint is created without retry config. When a 503 error occurs, the system retries 3 times with delays of 1s, 2s, 4s.

#### SC-2: Custom Configuration Override
An endpoint is configured with `max_retries: 5` and `base_delay_seconds: 2`. When failures occur, it retries 5 times with delays of 2s, 4s, 8s, 16s, 32s.

#### SC-3: Invalid Configuration Rejected
User attempts to set `max_retries: 15`. API returns 400 Bad Request with error: "max_retries must be between 0 and 10".
