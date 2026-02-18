# Spec: Audit Logging

## Overview
Log all significant operations for compliance and debugging.

### Requirements

#### R1: Log All API Operations
**Given** any API call
**When** operation completes
**Then** create audit log entry with:
  - timestamp
  - user/api_key
  - action (create, read, update, delete)
  - resource_type and resource_id
  - request details
  - success/failure status

#### R2: Query Audit Logs
**Given** audit logs exist
**When** GET /api/v1/audit/logs is called
**Then** return paginated logs with filtering

#### R3: Sensitive Data Redaction
**Given** request contains sensitive data
**When** logging
**Then** redact passwords, tokens, API keys

#### R4: Audit Log Retention
**Given** audit logs accumulate
**When** retention period passes
**Then** archive or delete old logs (configurable)

### Scenarios

#### SC1: Job Creation Audited
User creates job:
```json
{
  "timestamp": "2026-02-18T10:00:00Z",
  "user": "john.doe@example.com",
  "action": "create",
  "resource_type": "job",
  "resource_id": "job-uuid",
  "details": {"source_type": "upload", "file_name": "doc.pdf"},
  "success": true
}
```

#### SC2: Failed Login Attempt
Invalid credentials:
- Log attempt with IP address
- Increment failed login counter
- Alert on multiple failures
