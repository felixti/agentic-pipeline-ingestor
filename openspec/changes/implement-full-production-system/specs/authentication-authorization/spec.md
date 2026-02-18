# Spec: Authentication & Authorization

## Overview
Secure all API endpoints with JWT and API key authentication, enforce RBAC.

### Requirements

#### R1: JWT Authentication
**Given** a protected endpoint
**When** request includes Authorization: Bearer <token> header
**Then** validate token and extract user identity

#### R2: API Key Authentication
**Given** service-to-service call
**When** request includes X-API-Key header
**Then** validate API key and extract permissions

#### R3: Role-Based Access Control
**Given** authenticated user
**When** accessing endpoint
**Then** check role has required permission:
  - admin: all operations
  - operator: create, read, cancel jobs
  - viewer: read-only

#### R4: Token Refresh
**Given** valid refresh token
**When** POST /auth/refresh is called
**Then** return new access token

#### R5: Permission Denied
**Given** user lacks permission
**When** accessing restricted endpoint
**Then** return 403 Forbidden

### Scenarios

#### SC1: Admin Access
Admin user accesses all endpoints:
- Authenticates with JWT
- Has 'admin' role
- Can create, delete, configure everything

#### SC2: Viewer Restricted
Viewer tries to create job:
- Authenticated successfully
- Has 'viewer' role
- POST /jobs returns 403

#### SC3: API Key Service Auth
External service calls API:
- Includes X-API-Key header
- Key validated against database
- Service has operator permissions
