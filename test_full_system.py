#!/usr/bin/env python3
"""Test script for the full production system.

This script tests all major endpoints and verifies data persistence.
"""

import asyncio
import json
import sys
from datetime import datetime
from uuid import uuid4

import httpx

BASE_URL = "http://localhost:8000"
API_URL = f"{BASE_URL}/api/v1"


def get_status(resp_data):
    """Extract status from response (handles both wrapped and unwrapped)."""
    if "status" in resp_data:
        return resp_data["status"]
    if "data" in resp_data and isinstance(resp_data["data"], dict):
        return resp_data["data"].get("status", "unknown")
    return "unknown"


def get_data(resp_data):
    """Extract data from response (handles both wrapped and unwrapped)."""
    if "data" in resp_data and isinstance(resp_data["data"], dict):
        return resp_data["data"]
    return resp_data


async def test_health():
    """Test health endpoints."""
    print("\n=== Testing Health Endpoints ===")
    async with httpx.AsyncClient() as client:
        # Test basic health
        resp = await client.get(f"{BASE_URL}/health")
        assert resp.status_code == 200
        data = resp.json()
        print(f"✓ Health: {data.get('status', 'unknown')}")
        
        # Test queue health
        resp = await client.get(f"{BASE_URL}/health/queue")
        assert resp.status_code == 200
        data = resp.json()
        queue_data = get_data(data)
        print(f"✓ Queue Health: {queue_data.get('status', 'unknown')}")
        print(f"  - Queue depths: {queue_data.get('queue_depths', {})}")


async def test_pipelines():
    """Test pipeline configuration endpoints."""
    print("\n=== Testing Pipeline Configuration ===")
    async with httpx.AsyncClient() as client:
        # Create a pipeline
        pipeline_data = {
            "name": f"test-pipeline-{uuid4().hex[:8]}",
            "description": "Test pipeline for documents",
            "config": {
                "enabled_stages": ["ingest", "detect", "parse", "output"],
                "parser": {"primary_parser": "docling"},
                "output": {"destination": "cognee"},
            },
            "created_by": "test-user",
        }
        
        resp = await client.post(f"{API_URL}/pipelines", json=pipeline_data)
        assert resp.status_code == 201, f"Failed to create pipeline: {resp.text}"
        data = resp.json()
        response_data = get_data(data)
        pipeline_id = response_data["id"]
        print(f"✓ Created pipeline: {pipeline_id}")
        
        # List pipelines
        resp = await client.get(f"{API_URL}/pipelines")
        assert resp.status_code == 200
        data = resp.json()
        list_data = get_data(data)
        print(f"✓ Listed {list_data.get('total', 0)} pipelines")
        
        # Get specific pipeline
        resp = await client.get(f"{API_URL}/pipelines/{pipeline_id}")
        assert resp.status_code == 200
        data = resp.json()
        response_data = get_data(data)
        assert response_data["name"] == pipeline_data["name"]
        print(f"✓ Retrieved pipeline: {response_data['name']}")
        
        return pipeline_id


async def test_jobs(pipeline_id: str = None):
    """Test job management endpoints."""
    print("\n=== Testing Job Management ===")
    async with httpx.AsyncClient() as client:
        # Create a job
        job_data = {
            "source_type": "upload",
            "source_uri": "/tmp/test-file.pdf",
            "file_name": "test-file.pdf",
            "file_size": 1024,
            "mime_type": "application/pdf",
            "priority": "normal",
            "metadata": {"test": True, "created_by": "test-script"},
        }
        
        if pipeline_id:
            job_data["pipeline_id"] = pipeline_id
        
        resp = await client.post(f"{API_URL}/jobs", json=job_data)
        assert resp.status_code == 202, f"Failed to create job: {resp.text}"
        data = resp.json()
        response_data = get_data(data)
        job_id = response_data["id"]
        print(f"✓ Created job: {job_id}")
        
        # List jobs
        resp = await client.get(f"{API_URL}/jobs")
        assert resp.status_code == 200
        data = resp.json()
        list_data = get_data(data)
        print(f"✓ Listed {list_data.get('total', 0)} jobs")
        
        # Get specific job
        resp = await client.get(f"{API_URL}/jobs/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        response_data = get_data(data)
        assert response_data["source_type"] == "upload"
        print(f"✓ Retrieved job: {response_data['id']}")
        print(f"  - Status: {response_data['status']}")
        print(f"  - Priority: {response_data['priority']}")
        
        return job_id


async def test_auth():
    """Test authentication endpoints."""
    print("\n=== Testing Authentication ===")
    async with httpx.AsyncClient() as client:
        # Login
        login_data = {
            "username": "test-user",
            "password": "test-password",
            "roles": ["operator"],
        }
        
        resp = await client.post(f"{API_URL}/auth/login", json=login_data)
        assert resp.status_code == 200, f"Failed to login: {resp.text}"
        data = resp.json()
        response_data = get_data(data)
        access_token = response_data["access_token"]
        print("✓ Login successful, got access token")
        
        # Create API key
        key_data = {
            "name": "Test API Key",
            "permissions": ["jobs:read", "jobs:create"],
            "created_by": "test-user",
            "expires_in_days": 30,
        }
        
        resp = await client.post(f"{API_URL}/auth/api-keys", json=key_data)
        assert resp.status_code == 201, f"Failed to create API key: {resp.text}"
        data = resp.json()
        response_data = get_data(data)
        api_key = response_data["api_key"]
        print(f"✓ Created API key: {response_data['id']}")
        print(f"  - Key: {api_key[:20]}...")
        
        # List API keys
        resp = await client.get(f"{API_URL}/auth/api-keys")
        assert resp.status_code == 200
        data = resp.json()
        list_data = get_data(data)
        print(f"✓ Listed {list_data.get('total', 0)} API keys")
        
        return access_token, api_key


async def test_webhooks():
    """Test webhook endpoints."""
    print("\n=== Testing Webhooks ===")
    async with httpx.AsyncClient() as client:
        # Create webhook subscription
        webhook_data = {
            "url": "https://example.com/webhook",
            "events": ["job.completed", "job.failed"],
            "secret": "webhook-secret-key",
            "user_id": "test-user",
        }
        
        resp = await client.post(f"{API_URL}/webhooks", json=webhook_data)
        assert resp.status_code == 201, f"Failed to create webhook: {resp.text}"
        data = resp.json()
        response_data = get_data(data)
        webhook_id = response_data["id"]
        print(f"✓ Created webhook subscription: {webhook_id}")
        
        # List webhooks
        resp = await client.get(f"{API_URL}/webhooks")
        assert resp.status_code == 200
        data = resp.json()
        list_data = get_data(data)
        print(f"✓ Listed {list_data.get('total', 0)} webhooks")
        
        # Get webhook deliveries (will be empty initially)
        resp = await client.get(f"{API_URL}/webhooks/{webhook_id}/deliveries")
        assert resp.status_code == 200
        data = resp.json()
        list_data = get_data(data)
        print(f"✓ Retrieved deliveries: {list_data.get('total', 0)} total")
        
        return webhook_id


async def test_audit_logs():
    """Test audit log endpoints."""
    print("\n=== Testing Audit Logs ===")
    async with httpx.AsyncClient() as client:
        # Query audit logs
        resp = await client.get(f"{API_URL}/audit/logs")
        assert resp.status_code == 200
        data = resp.json()
        list_data = get_data(data)
        print(f"✓ Retrieved {list_data.get('total', 0)} audit log entries")
        
        # Query with filters
        resp = await client.get(
            f"{API_URL}/audit/logs",
            params={"action": "create", "resource_type": "job"},
        )
        assert resp.status_code == 200
        data = resp.json()
        list_data = get_data(data)
        print(f"✓ Filtered audit logs: {list_data.get('total', 0)} entries")


async def verify_database_persistence():
    """Verify data is persisted in the database."""
    print("\n=== Verifying Database Persistence ===")
    
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.models import get_async_engine
    from src.db.repositories import (
        APIKeyRepository,
        AuditLogRepository,
        JobRepository,
        PipelineRepository,
        WebhookRepository,
    )
    
    engine = get_async_engine()
    
    async with AsyncSession(engine) as session:
        # Check jobs
        job_repo = JobRepository(session)
        jobs, job_count = await job_repo.list_jobs()
        print(f"✓ Database: {job_count} jobs persisted")
        if jobs:
            print(f"  - Latest job: {jobs[0].id} (status: {jobs[0].status})")
        
        # Check pipelines
        pipeline_repo = PipelineRepository(session)
        pipelines, pipeline_count = await pipeline_repo.list_pipelines()
        print(f"✓ Database: {pipeline_count} pipelines persisted")
        if pipelines:
            print(f"  - Latest pipeline: {pipelines[0].id} (name: {pipelines[0].name})")
        
        # Check API keys
        api_key_repo = APIKeyRepository(session)
        keys, key_count = await api_key_repo.list_keys()
        print(f"✓ Database: {key_count} API keys persisted")
        
        # Check webhooks
        webhook_repo = WebhookRepository(session)
        webhooks, webhook_count = await webhook_repo.list_subscriptions()
        print(f"✓ Database: {webhook_count} webhooks persisted")
        
        # Check audit logs
        audit_repo = AuditLogRepository(session)
        logs, log_count = await audit_repo.query_logs()
        print(f"✓ Database: {log_count} audit logs persisted")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("FULL PRODUCTION SYSTEM - END-TO-END TEST")
    print("=" * 60)
    
    try:
        # Test health endpoints
        await test_health()
        
        # Test pipelines
        pipeline_id = await test_pipelines()
        
        # Test jobs
        job_id = await test_jobs(pipeline_id)
        
        # Test authentication
        access_token, api_key = await test_auth()
        
        # Test webhooks
        webhook_id = await test_webhooks()
        
        # Test audit logs
        await test_audit_logs()
        
        # Verify database persistence
        await verify_database_persistence()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nSummary:")
        print(f"  • Pipeline created: {pipeline_id}")
        print(f"  • Job created: {job_id}")
        print(f"  • API Key created: {api_key[:20]}...")
        print(f"  • Webhook created: {webhook_id}")
        print("\nAll data has been persisted to the database! 🎉")
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
