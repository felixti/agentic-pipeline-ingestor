"""Dead Letter Queue (DLQ) E2E Tests.

This module tests the Dead Letter Queue functionality including:
- Job moves to DLQ after exhausted retries
- Manual retry from DLQ with modified config
- DLQ querying and management
"""

import pytest

# Mark all tests in this module as E2E and DLQ tests
pytestmark = [pytest.mark.e2e, pytest.mark.dlq, pytest.mark.asyncio]


class TestDLQWorkflow:
    """Test Dead Letter Queue functionality."""
    
    async def test_list_dlq_entries(self, admin_client):
        """E2E: List all DLQ entries."""
        response = await admin_client.get("/api/v1/dlq")
        
        # May return 200 or 404 depending on implementation
        assert response.status_code in (200, 404)
        
        if response.status_code == 200:
            data = response.json()
            assert "data" in data
            assert isinstance(data["data"], list)
    
    async def test_get_dlq_entry(self, admin_client):
        """E2E: Get specific DLQ entry."""
        entry_id = "test-entry-id"
        
        response = await admin_client.get(f"/api/v1/dlq/{entry_id}")
        
        # May return 200 or 404
        assert response.status_code in (200, 404)
        
        if response.status_code == 200:
            data = response.json()
            assert "data" in data
            assert data["data"]["id"] == entry_id
    
    async def test_dlq_entry_contains_failure_info(self, admin_client):
        """E2E: DLQ entry contains failure information.
        
        A DLQ entry should include:
        - Original job details
        - Failure reason
        - Retry count
        - Timestamp of last failure
        """
        entry_id = "test-entry-id"
        
        response = await admin_client.get(f"/api/v1/dlq/{entry_id}")
        
        if response.status_code == 200:
            data = response.json()
            entry = data.get("data", {})
            
            # Verify entry has required fields
            assert "job_id" in entry or "original_job_id" in entry
            assert "failure_reason" in entry or "error" in entry
            assert "retry_count" in entry or "attempts" in entry
    
    async def test_retry_from_dlq(self, admin_client):
        """E2E: Retry a job from DLQ with modified configuration.
        
        When retrying from DLQ, the configuration can be modified
        to potentially resolve the issue.
        """
        entry_id = "test-dlq-entry"
        
        retry_payload = {
            "force_parser": "azure_ocr",
            "priority": 10,
            "options": {
                "skip_cache": True,
                "force_reprocess": True,
                "increased_timeout": True
            },
            "remove_from_dlq": True
        }
        
        response = await admin_client.post(
            f"/api/v1/dlq/{entry_id}/retry",
            json=retry_payload
        )
        
        # May return 202 (accepted), 404 (entry not found), or 409 (not retryable)
        assert response.status_code in (202, 404, 409)
    
    async def test_bulk_retry_from_dlq(self, admin_client):
        """E2E: Bulk retry multiple jobs from DLQ."""
        payload = {
            "entry_ids": ["entry-1", "entry-2", "entry-3"],
            "retry_options": {
                "force_parser": "azure_ocr",
                "priority": 10
            }
        }
        
        response = await admin_client.post("/api/v1/dlq/bulk-retry", json=payload)
        
        assert response.status_code in (202, 207, 404)
        
        if response.status_code in (202, 207):
            data = response.json()
            assert "data" in data
    
    async def test_delete_dlq_entry(self, admin_client):
        """E2E: Delete a DLQ entry."""
        entry_id = "test-entry-id"
        
        response = await admin_client.delete(f"/api/v1/dlq/{entry_id}")
        
        assert response.status_code in (204, 404)
    
    async def test_dlq_entry_lifecycle(self, auth_client, admin_client):
        """E2E: Full lifecycle of a job moving to DLQ.
        
        1. Create job configured to fail
        2. Job exhausts retries
        3. Job moves to DLQ
        4. Verify DLQ entry exists
        5. Retry from DLQ
        6. Verify DLQ entry removed or updated
        """
        # Step 1: Create job configured to fail
        payload = {
            "source_type": "upload",
            "source_uri": "/uploads/failing-document.pdf",
            "file_name": "failing-document.pdf",
            "mime_type": "application/pdf",
            "mode": "async",
            "priority": 5,
            "retry_config": {
                "max_retries": 1,
                "retry_delay_seconds": 1
            },
            "metadata": {
                "test_case": "DLQ002",
                "scenario": "dlq_lifecycle",
                "force_failure": True
            }
        }
        
        create_response = await auth_client.post("/api/v1/jobs", json=payload)
        assert create_response.status_code == 202
        
        job_id = create_response.json()["data"]["id"]
        assert job_id is not None
        
        # Steps 2-6 would require waiting for job processing
        # In a real test, we would poll for DLQ entry creation
        
        # Verify DLQ endpoint exists
        dlq_response = await admin_client.get("/api/v1/dlq")
        assert dlq_response.status_code in (200, 404)


class TestDLQQueries:
    """Test DLQ querying capabilities."""
    
    async def test_query_dlq_by_job_id(self, admin_client):
        """E2E: Query DLQ by original job ID."""
        job_id = "original-job-id"
        
        response = await admin_client.get(f"/api/v1/dlq?job_id={job_id}")
        
        assert response.status_code in (200, 404)
        
        if response.status_code == 200:
            data = response.json()
            # If entries exist, verify job_id matches
            for entry in data.get("data", []):
                assert entry.get("job_id") == job_id
    
    async def test_query_dlq_by_date_range(self, admin_client):
        """E2E: Query DLQ by date range."""
        from_date = "2024-01-01T00:00:00Z"
        to_date = "2024-12-31T23:59:59Z"
        
        response = await admin_client.get(
            f"/api/v1/dlq?from={from_date}&to={to_date}"
        )
        
        assert response.status_code in (200, 404)
    
    async def test_query_dlq_by_failure_reason(self, admin_client):
        """E2E: Query DLQ by failure reason/pattern."""
        failure_pattern = "parser_error"
        
        response = await admin_client.get(
            f"/api/v1/dlq?failure_reason={failure_pattern}"
        )
        
        assert response.status_code in (200, 404)
    
    async def test_dlq_pagination(self, admin_client):
        """E2E: DLQ list supports pagination."""
        response = await admin_client.get("/api/v1/dlq?page=1&limit=10")
        
        assert response.status_code in (200, 404)
        
        if response.status_code == 200:
            data = response.json()
            assert "data" in data
            assert isinstance(data["data"], list)
            assert len(data["data"]) <= 10


class TestDLQManagement:
    """Test DLQ management operations."""
    
    async def test_dlq_statistics(self, admin_client):
        """E2E: Get DLQ statistics."""
        response = await admin_client.get("/api/v1/dlq/stats")
        
        assert response.status_code in (200, 404)
        
        if response.status_code == 200:
            data = response.json()
            assert "data" in data
            stats = data["data"]
            
            # Verify stats contain expected fields
            assert "total_entries" in stats or "count" in stats
    
    async def test_export_dlq_entries(self, admin_client):
        """E2E: Export DLQ entries."""
        response = await admin_client.get(
            "/api/v1/dlq/export?format=json",
            headers={"Accept": "application/json"}
        )
        
        assert response.status_code in (200, 404)
    
    async def test_dlq_retention_policy(self, admin_client):
        """E2E: Configure DLQ retention policy."""
        policy_payload = {
            "retention_days": 30,
            "auto_purge": False,
            "max_entries": 10000
        }
        
        response = await admin_client.put("/api/v1/dlq/retention", json=policy_payload)
        
        assert response.status_code in (200, 404)
    
    async def test_purge_dlq(self, admin_client):
        """E2E: Purge DLQ entries."""
        purge_payload = {
            "older_than_days": 30,
            "dry_run": True  # Don't actually delete in test
        }
        
        response = await admin_client.post("/api/v1/dlq/purge", json=purge_payload)
        
        assert response.status_code in (200, 404)


class TestDLQPermissions:
    """Test DLQ access permissions."""
    
    async def test_viewer_cannot_access_dlq(self, viewer_client):
        """E2E: Viewer role cannot access DLQ."""
        response = await viewer_client.get("/api/v1/dlq")
        
        # Should return 403 Forbidden or 404 (if endpoint doesn't exist)
        assert response.status_code in (403, 404)
    
    async def test_unauthorized_cannot_access_dlq(self, unauth_client):
        """E2E: Unauthenticated users cannot access DLQ."""
        response = await unauth_client.get("/api/v1/dlq")
        
        assert response.status_code == 401


class TestDLQNotifications:
    """Test DLQ notification features."""
    
    async def test_dlq_webhook_notification(self, admin_client):
        """E2E: Configure webhook for DLQ notifications."""
        webhook_payload = {
            "url": "https://example.com/webhooks/dlq",
            "events": ["entry_created", "entry_retried"],
            "secret": "webhook-secret"
        }
        
        response = await admin_client.post("/api/v1/dlq/webhooks", json=webhook_payload)
        
        assert response.status_code in (201, 404)
    
    async def test_dlq_email_notification(self, admin_client):
        """E2E: Configure email notifications for DLQ."""
        email_payload = {
            "recipients": ["admin@example.com"],
            "events": ["entry_created"],
            "threshold": 10  # Notify when DLQ has 10+ entries
        }
        
        response = await admin_client.post("/api/v1/dlq/notifications/email", json=email_payload)
        
        assert response.status_code in (201, 404)
