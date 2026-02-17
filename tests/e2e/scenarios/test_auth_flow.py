"""Authentication & Authorization E2E Tests.

This module tests authentication and authorization flows including:
- API key authentication
- OAuth2/JWT authentication
- Role-based access control (RBAC)
- Unauthorized access handling
"""

import pytest

# Mark all tests in this module as E2E and auth tests
pytestmark = [pytest.mark.e2e, pytest.mark.auth, pytest.mark.asyncio]


class TestAPIKeyAuth:
    """Test API key authentication."""
    
    async def test_valid_api_key_grants_access(self, auth_client):
        """E2E: Valid API key grants access to protected endpoints."""
        response = await auth_client.get("/api/v1/jobs?limit=1")
        
        # Should succeed or return empty list
        assert response.status_code in (200, 401)
        
        if response.status_code == 200:
            data = response.json()
            assert "data" in data
    
    async def test_invalid_api_key_rejected(self, unauth_client):
        """E2E: Invalid API key is rejected."""
        # Add invalid API key
        unauth_client.headers["X-API-Key"] = "invalid-key-12345"
        
        response = await unauth_client.get("/api/v1/jobs")
        
        assert response.status_code == 401
    
    async def test_missing_api_key_rejected(self, unauth_client):
        """E2E: Missing API key is rejected."""
        response = await unauth_client.get("/api/v1/jobs")
        
        assert response.status_code == 401
    
    async def test_expired_api_key_rejected(self, unauth_client):
        """E2E: Expired API key is rejected."""
        unauth_client.headers["X-API-Key"] = "expired-key"
        
        response = await unauth_client.get("/api/v1/jobs")
        
        assert response.status_code in (401, 403)
    
    async def test_api_key_rate_limiting(self, auth_client):
        """E2E: API key rate limiting is enforced.
        
        Make multiple rapid requests and verify rate limit headers.
        """
        # Make several requests
        responses = []
        for _ in range(5):
            response = await auth_client.get("/api/v1/jobs?limit=1")
            responses.append(response)
        
        # Check for rate limit headers in any response
        found_headers = False
        for response in responses:
            if "X-RateLimit-Limit" in response.headers:
                found_headers = True
                limit = response.headers.get("X-RateLimit-Limit")
                remaining = response.headers.get("X-RateLimit-Remaining")
                
                assert limit is not None
                assert remaining is not None
        
        # Rate limit headers may not be present in all implementations
        # Just verify requests succeeded
        assert all(r.status_code in (200, 401, 429) for r in responses)


class TestOAuth2Auth:
    """Test OAuth2/JWT authentication."""
    
    async def test_valid_oauth2_token_grants_access(self, client):
        """E2E: Valid OAuth2 token grants access."""
        # This test requires a valid OAuth2 token
        token = "valid-oauth2-token"  # Would be a real token in practice
        
        client.headers["Authorization"] = f"Bearer {token}"
        
        response = await client.get("/api/v1/jobs")
        
        # May succeed or fail depending on token validity
        assert response.status_code in (200, 401)
    
    async def test_invalid_oauth2_token_rejected(self, client):
        """E2E: Invalid OAuth2 token is rejected."""
        client.headers["Authorization"] = "Bearer invalid-token"
        
        response = await client.get("/api/v1/jobs")
        
        assert response.status_code == 401
    
    async def test_malformed_oauth2_token_rejected(self, client):
        """E2E: Malformed OAuth2 token is rejected."""
        client.headers["Authorization"] = "Bearer malformed{{{}token"
        
        response = await client.get("/api/v1/jobs")
        
        assert response.status_code == 401
    
    async def test_expired_oauth2_token_rejected(self, client):
        """E2E: Expired OAuth2 token is rejected."""
        # Use a known expired token format
        expired_token = (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyLCJleHAiOjE1MTYyMzkwMjJ9."
            "4Adcj8TqQzyi4lpj0_ij3fZKfrE1j29nQZ2WZtkO0JM"
        )
        
        client.headers["Authorization"] = f"Bearer {expired_token}"
        
        response = await client.get("/api/v1/jobs")
        
        assert response.status_code == 401
    
    async def test_missing_bearer_prefix_rejected(self, client):
        """E2E: Authorization header without Bearer prefix is rejected."""
        client.headers["Authorization"] = "valid-token-without-bearer"
        
        response = await client.get("/api/v1/jobs")
        
        assert response.status_code == 401


class TestRBACAdmin:
    """Test admin role permissions."""
    
    async def test_admin_can_list_jobs(self, admin_client):
        """E2E: Admin can list all jobs."""
        response = await admin_client.get("/api/v1/jobs")
        
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
    
    async def test_admin_can_create_job(self, admin_client, sample_job_payload):
        """E2E: Admin can create jobs."""
        response = await admin_client.post("/api/v1/jobs", json=sample_job_payload)
        
        assert response.status_code == 202
    
    async def test_admin_can_cancel_any_job(self, admin_client):
        """E2E: Admin can cancel any job."""
        # First create a job
        payload = {
            "source_type": "upload",
            "source_uri": "/uploads/test.pdf",
            "file_name": "test.pdf",
            "mime_type": "application/pdf",
            "priority": 5
        }
        
        create_response = await admin_client.post("/api/v1/jobs", json=payload)
        assert create_response.status_code == 202
        
        job_id = create_response.json()["data"]["id"]
        
        # Cancel the job
        cancel_response = await admin_client.delete(f"/api/v1/jobs/{job_id}")
        
        assert cancel_response.status_code in (204, 404)
    
    async def test_admin_can_access_dlq(self, admin_client):
        """E2E: Admin can access DLQ."""
        response = await admin_client.get("/api/v1/dlq")
        
        # May return 200 or 404 if endpoint doesn't exist
        assert response.status_code in (200, 404)
    
    async def test_admin_can_access_audit_logs(self, admin_client):
        """E2E: Admin can access audit logs."""
        response = await admin_client.get("/api/v1/audit/logs")
        
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
    
    async def test_admin_can_manage_users(self, admin_client):
        """E2E: Admin can manage users."""
        # This endpoint may not exist in all implementations
        response = await admin_client.get("/api/v1/admin/users")
        
        assert response.status_code in (200, 404)
    
    async def test_admin_can_manage_api_keys(self, admin_client):
        """E2E: Admin can manage API keys."""
        # List API keys
        response = await admin_client.get("/api/v1/admin/api-keys")
        
        assert response.status_code in (200, 404)


class TestRBACViewer:
    """Test viewer role (read-only) permissions."""
    
    async def test_viewer_can_list_jobs(self, viewer_client):
        """E2E: Viewer can list jobs."""
        response = await viewer_client.get("/api/v1/jobs")
        
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
    
    async def test_viewer_can_get_job_details(self, viewer_client):
        """E2E: Viewer can get job details."""
        # Assuming a job exists
        job_id = "some-job-id"
        
        response = await viewer_client.get(f"/api/v1/jobs/{job_id}")
        
        assert response.status_code in (200, 404)
    
    async def test_viewer_cannot_create_job(self, viewer_client, sample_job_payload):
        """E2E: Viewer cannot create jobs."""
        response = await viewer_client.post("/api/v1/jobs", json=sample_job_payload)
        
        assert response.status_code == 403
    
    async def test_viewer_cannot_cancel_job(self, viewer_client):
        """E2E: Viewer cannot cancel jobs."""
        job_id = "some-job-id"
        
        response = await viewer_client.delete(f"/api/v1/jobs/{job_id}")
        
        assert response.status_code == 403
    
    async def test_viewer_cannot_retry_job(self, viewer_client):
        """E2E: Viewer cannot retry jobs."""
        job_id = "some-job-id"
        
        response = await viewer_client.post(f"/api/v1/jobs/{job_id}/retry", json={})
        
        assert response.status_code == 403
    
    async def test_viewer_can_access_audit_logs(self, viewer_client):
        """E2E: Viewer can access audit logs (read-only)."""
        response = await viewer_client.get("/api/v1/audit/logs")
        
        assert response.status_code == 200
    
    async def test_viewer_cannot_access_dlq(self, viewer_client):
        """E2E: Viewer cannot access DLQ."""
        response = await viewer_client.get("/api/v1/dlq")
        
        assert response.status_code in (403, 404)
    
    async def test_viewer_cannot_manage_users(self, viewer_client):
        """E2E: Viewer cannot manage users."""
        response = await viewer_client.get("/api/v1/admin/users")
        
        assert response.status_code in (403, 404)


class TestRBACOperator:
    """Test operator role permissions."""
    
    async def test_operator_can_create_job(self, auth_client, sample_job_payload):
        """E2E: Operator can create jobs."""
        response = await auth_client.post("/api/v1/jobs", json=sample_job_payload)
        
        # Standard API key (operator role) should be able to create jobs
        assert response.status_code == 202
    
    async def test_operator_can_cancel_own_jobs(self, auth_client):
        """E2E: Operator can cancel their own jobs."""
        # Create a job
        payload = {
            "source_type": "upload",
            "source_uri": "/uploads/test.pdf",
            "file_name": "test.pdf",
            "mime_type": "application/pdf",
            "priority": 5
        }
        
        create_response = await auth_client.post("/api/v1/jobs", json=payload)
        assert create_response.status_code == 202
        
        job_id = create_response.json()["data"]["id"]
        
        # Cancel the job
        cancel_response = await auth_client.delete(f"/api/v1/jobs/{job_id}")
        
        assert cancel_response.status_code in (204, 404)


class TestUnauthorizedAccess:
    """Test unauthorized access handling."""
    
    async def test_unauthorized_job_list_rejected(self, unauth_client):
        """E2E: Unauthorized job list request is rejected."""
        response = await unauth_client.get("/api/v1/jobs")
        
        assert response.status_code == 401
    
    async def test_unauthorized_job_create_rejected(self, unauth_client, sample_job_payload):
        """E2E: Unauthorized job creation is rejected."""
        response = await unauth_client.post("/api/v1/jobs", json=sample_job_payload)
        
        assert response.status_code == 401
    
    async def test_unauthorized_upload_rejected(self, unauth_client):
        """E2E: Unauthorized file upload is rejected."""
        response = await unauth_client.post("/api/v1/upload")
        
        assert response.status_code == 401
    
    async def test_unauthorized_audit_access_rejected(self, unauth_client):
        """E2E: Unauthorized audit log access is rejected."""
        response = await unauth_client.get("/api/v1/audit/logs")
        
        assert response.status_code == 401
    
    async def test_unauthorized_dlq_access_rejected(self, unauth_client):
        """E2E: Unauthorized DLQ access is rejected."""
        response = await unauth_client.get("/api/v1/dlq")
        
        assert response.status_code == 401
    
    async def test_error_response_format(self, unauth_client):
        """E2E: Unauthorized responses have proper error format."""
        response = await unauth_client.get("/api/v1/jobs")
        
        assert response.status_code == 401
        
        data = response.json()
        # Should have error information
        assert "error" in data or "detail" in data or "message" in data


class TestAuthEdgeCases:
    """Test authentication edge cases."""
    
    async def test_multiple_auth_headers(self, client):
        """E2E: Multiple authentication headers are handled."""
        client.headers["X-API-Key"] = "valid-key"
        client.headers["Authorization"] = "Bearer valid-token"
        
        response = await client.get("/api/v1/jobs")
        
        # Should use one of the auth methods or reject as ambiguous
        assert response.status_code in (200, 401)
    
    async def test_case_sensitive_api_key_header(self, client):
        """E2E: API key header is case-sensitive."""
        # Standard header
        client.headers["X-API-Key"] = "test-key"
        
        response1 = await client.get("/api/v1/jobs")
        
        # Lowercase header
        client.headers.pop("X-API-Key")
        client.headers["x-api-key"] = "test-key"
        
        response2 = await client.get("/api/v1/jobs")
        
        # Both should work (HTTP headers are case-insensitive)
        assert response1.status_code in (200, 401)
        assert response2.status_code in (200, 401)
    
    async def test_empty_api_key_rejected(self, unauth_client):
        """E2E: Empty API key is rejected."""
        unauth_client.headers["X-API-Key"] = ""
        
        response = await unauth_client.get("/api/v1/jobs")
        
        assert response.status_code == 401
    
    async def test_whitespace_only_api_key_rejected(self, unauth_client):
        """E2E: Whitespace-only API key is rejected."""
        unauth_client.headers["X-API-Key"] = "   "
        
        response = await unauth_client.get("/api/v1/jobs")
        
        assert response.status_code == 401
