"""Unit tests for bulk operations API routes.

Tests the bulk operations endpoints including bulk ingest, retry, export,
and operation status queries.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException, Request, status

from src.api.models import ProcessingMode, SourceType
from src.auth.base import User


# Create a mock user with proper permissions
def create_test_user(permissions=None):
    """Create a test user with specified permissions."""
    return User(
        id=uuid4(),
        email="test@example.com",
        permissions=permissions or ["jobs:write"],
    )


@pytest.mark.unit
class TestBulkIngest:
    """Tests for POST /bulk/ingest - Bulk ingest endpoint."""

    @pytest.mark.asyncio
    async def test_bulk_ingest_success(self):
        """Test successful bulk ingest."""
        from src.api.routes.bulk import BulkIngestItem, BulkIngestRequest, bulk_ingest
        
        mock_request = MagicMock()
        
        ingest_request = BulkIngestRequest(
            items=[
                BulkIngestItem(
                    source_type=SourceType.S3,
                    source_uri="s3://bucket/file1.pdf",
                    file_name="file1.pdf",
                    external_id="ext-001",
                ),
                BulkIngestItem(
                    source_type=SourceType.AZURE_BLOB,
                    source_uri="azure://container/file2.docx",
                    file_name="file2.docx",
                    priority=8,
                ),
            ],
            mode=ProcessingMode.ASYNC,
        )
        
        user = create_test_user(["jobs:write"])
        
        response = await bulk_ingest(mock_request, ingest_request, user)
        
        assert response.success is True
        assert response.data["total_requested"] == 2
        assert response.data["total_created"] == 2
        assert response.data["total_failed"] == 0
        assert len(response.data["results"]) == 2
        assert response.data["results"][0].success is True
        assert response.data["results"][0].status == "created"

    @pytest.mark.asyncio
    async def test_bulk_ingest_single_item(self):
        """Test bulk ingest with single item."""
        from src.api.routes.bulk import BulkIngestItem, BulkIngestRequest, bulk_ingest
        
        mock_request = MagicMock()
        
        ingest_request = BulkIngestRequest(
            items=[
                BulkIngestItem(
                    source_type=SourceType.UPLOAD,
                    source_uri="/tmp/test.pdf",
                    file_name="test.pdf",
                ),
            ],
        )
        
        user = create_test_user(["jobs:write"])
        
        response = await bulk_ingest(mock_request, ingest_request, user)
        
        assert response.data["total_requested"] == 1
        assert response.data["total_created"] == 1

    @pytest.mark.asyncio
    async def test_bulk_ingest_with_callback(self):
        """Test bulk ingest with callback URL."""
        from src.api.routes.bulk import BulkIngestItem, BulkIngestRequest, bulk_ingest
        
        mock_request = MagicMock()
        
        ingest_request = BulkIngestRequest(
            items=[
                BulkIngestItem(
                    source_type=SourceType.URL,
                    source_uri="https://example.com/doc.pdf",
                ),
            ],
            callback_url="https://example.com/webhook",
        )
        
        user = create_test_user(["jobs:write"])
        
        response = await bulk_ingest(mock_request, ingest_request, user)
        
        assert response.success is True
        assert "batch_id" in response.data


@pytest.mark.unit
class TestBulkRetry:
    """Tests for POST /bulk/retry - Bulk retry endpoint."""

    @pytest.mark.asyncio
    async def test_bulk_retry_with_job_ids(self):
        """Test bulk retry with specific job IDs."""
        from src.api.routes.bulk import BulkRetryRequest, bulk_retry
        
        mock_request = MagicMock()
        
        job_ids = [uuid4(), uuid4()]
        retry_request = BulkRetryRequest(
            job_ids=job_ids,
            priority_adjustment=1,
        )
        
        user = create_test_user(["jobs:write"])
        
        response = await bulk_retry(mock_request, retry_request, user)
        
        assert response.success is True
        assert response.data["total_requested"] == 2
        assert response.data["total_retried"] == 2
        assert len(response.data["results"]) == 2

    @pytest.mark.asyncio
    async def test_bulk_retry_with_filter(self):
        """Test bulk retry with filter criteria."""
        from src.api.routes.bulk import BulkRetryFilter, BulkRetryRequest, bulk_retry
        
        mock_request = MagicMock()
        
        retry_request = BulkRetryRequest(
            filter=BulkRetryFilter(
                status=["failed"],
                source_types=[SourceType.S3],
            ),
        )
        
        user = create_test_user(["jobs:write"])
        
        # Since filter doesn't query real DB, job_ids_to_retry will be empty
        # This results in 0 retries
        response = await bulk_retry(mock_request, retry_request, user)
        
        assert response.success is True
        # No jobs matched the filter (mock has no real DB)
        assert response.data["total_requested"] == 0

    @pytest.mark.asyncio
    async def test_bulk_retry_missing_job_ids_and_filter(self):
        """Test bulk retry when neither job_ids nor filter is provided."""
        from src.api.routes.bulk import BulkRetryRequest, bulk_retry
        
        mock_request = MagicMock()
        
        retry_request = BulkRetryRequest()
        
        user = create_test_user(["jobs:write"])
        
        with pytest.raises(HTTPException) as exc_info:
            await bulk_retry(mock_request, retry_request, user)
        
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "Either job_ids or filter must be provided" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_bulk_retry_with_config(self):
        """Test bulk retry with updated configuration."""
        from src.api.routes.bulk import BulkRetryRequest, bulk_retry
        
        mock_request = MagicMock()
        
        job_ids = [uuid4()]
        retry_request = BulkRetryRequest(
            job_ids=job_ids,
            force_parser="azure_ocr",
            updated_config={"parser": {"primary_parser": "azure_ocr"}},
        )
        
        user = create_test_user(["jobs:write"])
        
        response = await bulk_retry(mock_request, retry_request, user)
        
        assert response.success is True
        assert response.data["total_retried"] == 1


@pytest.mark.unit
class TestBulkExport:
    """Tests for POST /bulk/export - Bulk export endpoint."""

    @pytest.mark.asyncio
    async def test_bulk_export_success(self):
        """Test successful bulk export creation."""
        from src.api.routes.bulk import BulkExportFilter, BulkExportRequest, bulk_export
        
        mock_request = MagicMock()
        
        export_request = BulkExportRequest(
            filter=BulkExportFilter(
                status=["completed"],
                source_types=[SourceType.S3, SourceType.AZURE_BLOB],
            ),
            format="json",
            include_metadata=True,
            include_text=False,
        )
        
        user = create_test_user(["jobs:read"])
        
        response = await bulk_export(mock_request, export_request, user)
        
        assert response.success is True
        assert response.data["status"] == "processing"
        assert response.data["format"] == "json"
        assert "export_id" in response.data
        assert response.data["message"] == "Export job created and is being processed"

    @pytest.mark.asyncio
    async def test_bulk_export_csv_format(self):
        """Test bulk export with CSV format."""
        from src.api.routes.bulk import BulkExportFilter, BulkExportRequest, bulk_export
        
        mock_request = MagicMock()
        
        export_request = BulkExportRequest(
            filter=BulkExportFilter(),
            format="csv",
        )
        
        user = create_test_user(["jobs:read"])
        
        response = await bulk_export(mock_request, export_request, user)
        
        assert response.data["format"] == "csv"

    @pytest.mark.asyncio
    async def test_bulk_export_with_callback(self):
        """Test bulk export with callback URL."""
        from src.api.routes.bulk import BulkExportFilter, BulkExportRequest, bulk_export
        
        mock_request = MagicMock()
        
        export_request = BulkExportRequest(
            filter=BulkExportFilter(),
            format="jsonl",
            callback_url="https://example.com/export-webhook",
        )
        
        user = create_test_user(["jobs:read"])
        
        response = await bulk_export(mock_request, export_request, user)
        
        assert response.success is True
        assert response.data["status"] == "processing"


@pytest.mark.unit
class TestGetBulkOperationStatus:
    """Tests for GET /bulk/status/{batch_id} - Get bulk operation status."""

    @pytest.mark.asyncio
    async def test_get_bulk_operation_status_success(self):
        """Test getting existing bulk operation status."""
        from src.api.routes.bulk import (
            BulkIngestItem,
            BulkIngestRequest,
            bulk_ingest,
            get_bulk_operation_status,
        )
        
        batch_id = uuid4()
        
        # First create a bulk operation
        mock_request = MagicMock()
        ingest_request = BulkIngestRequest(
            items=[BulkIngestItem(source_type=SourceType.S3, source_uri="s3://bucket/file.pdf")],
        )
        user = create_test_user(["jobs:write"])
        
        # Create operation
        create_response = await bulk_ingest(mock_request, ingest_request, user)
        created_batch_id = create_response.data["batch_id"]
        
        # Now get status
        user = create_test_user(["jobs:read"])
        response = await get_bulk_operation_status(created_batch_id, user)
        
        assert response.success is True
        assert response.data["batch_id"] == created_batch_id
        assert response.data["operation_type"] == "ingest"

    @pytest.mark.asyncio
    async def test_get_bulk_operation_status_not_found(self):
        """Test getting status for non-existent operation."""
        from src.api.routes.bulk import get_bulk_operation_status
        
        batch_id = uuid4()
        
        user = create_test_user(["jobs:read"])
        
        with pytest.raises(HTTPException) as exc_info:
            await get_bulk_operation_status(batch_id, user)
        
        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
        assert str(batch_id) in exc_info.value.detail


@pytest.mark.unit
class TestListBulkOperations:
    """Tests for GET /bulk/operations - List bulk operations."""

    @pytest.mark.asyncio
    async def test_list_bulk_operations_empty(self):
        """Test listing operations when none exist."""
        from src.api.routes import bulk
        from src.api.routes.bulk import list_bulk_operations
        
        # Clear operations storage first
        bulk._bulk_operations.clear()
        
        user = create_test_user(["jobs:read"])
        
        response = await list_bulk_operations(user=user)
        
        assert response.success is True
        assert response.data["operations"] == []
        assert response.data["total"] == 0
        assert response.data["page"] == 1

    @pytest.mark.asyncio
    async def test_list_bulk_operations_with_data(self):
        """Test listing operations with existing operations."""
        from src.api.routes import bulk
        from src.api.routes.bulk import list_bulk_operations
        
        # Clear and add test operations
        bulk._bulk_operations.clear()
        
        # Create a test operation
        batch_id = uuid4()
        bulk._bulk_operations[batch_id] = bulk.BulkOperationStatus(
            batch_id=batch_id,
            operation_type="ingest",
            status="completed",
            total_items=5,
            processed_items=5,
            successful_items=5,
            failed_items=0,
            created_at=datetime.utcnow(),
        )
        
        user = create_test_user(["jobs:read"])
        
        response = await list_bulk_operations(user=user)
        
        assert response.success is True
        assert response.data["total"] == 1
        assert len(response.data["operations"]) == 1
        assert response.data["operations"][0].operation_type == "ingest"
        
        # Cleanup
        bulk._bulk_operations.clear()

    @pytest.mark.asyncio
    async def test_list_bulk_operations_with_type_filter(self):
        """Test listing operations filtered by type."""
        from src.api.routes import bulk
        from src.api.routes.bulk import list_bulk_operations
        
        bulk._bulk_operations.clear()
        
        # Add operations of different types
        ingest_id = uuid4()
        export_id = uuid4()
        
        bulk._bulk_operations[ingest_id] = bulk.BulkOperationStatus(
            batch_id=ingest_id,
            operation_type="ingest",
            status="completed",
            total_items=5,
            processed_items=5,
            successful_items=5,
            failed_items=0,
            created_at=datetime.utcnow(),
        )
        
        bulk._bulk_operations[export_id] = bulk.BulkOperationStatus(
            batch_id=export_id,
            operation_type="export",
            status="processing",
            total_items=100,
            processed_items=50,
            successful_items=50,
            failed_items=0,
            created_at=datetime.utcnow(),
        )
        
        user = create_test_user(["jobs:read"])
        
        response = await list_bulk_operations(operation_type="ingest", user=user)
        
        assert response.data["total"] == 1
        assert response.data["operations"][0].operation_type == "ingest"
        
        # Cleanup
        bulk._bulk_operations.clear()

    @pytest.mark.asyncio
    async def test_list_bulk_operations_pagination(self):
        """Test listing operations with pagination."""
        from src.api.routes import bulk
        from src.api.routes.bulk import list_bulk_operations
        
        bulk._bulk_operations.clear()
        
        # Add multiple operations
        for i in range(5):
            batch_id = uuid4()
            bulk._bulk_operations[batch_id] = bulk.BulkOperationStatus(
                batch_id=batch_id,
                operation_type="ingest",
                status="completed",
                total_items=i,
                processed_items=i,
                successful_items=i,
                failed_items=0,
                created_at=datetime.utcnow(),
            )
        
        user = create_test_user(["jobs:read"])
        
        response = await list_bulk_operations(page=1, page_size=2, user=user)
        
        assert response.data["total"] == 5
        assert len(response.data["operations"]) == 2
        assert response.data["page"] == 1
        assert response.data["page_size"] == 2
        assert response.data["total_pages"] == 3
        
        # Cleanup
        bulk._bulk_operations.clear()


@pytest.mark.unit
class TestDownloadBulkExport:
    """Tests for GET /bulk/export/{export_id}/download - Download export."""

    @pytest.mark.asyncio
    async def test_download_export_not_found(self):
        """Test downloading non-existent export."""
        from src.api.routes.bulk import download_bulk_export
        
        export_id = uuid4()
        
        user = create_test_user(["jobs:read"])
        
        with pytest.raises(HTTPException) as exc_info:
            await download_bulk_export(export_id, user)
        
        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_download_export_not_an_export(self):
        """Test downloading when operation is not an export."""
        from src.api.routes import bulk
        from src.api.routes.bulk import download_bulk_export
        
        bulk._bulk_operations.clear()
        
        batch_id = uuid4()
        bulk._bulk_operations[batch_id] = bulk.BulkOperationStatus(
            batch_id=batch_id,
            operation_type="ingest",  # Not an export
            status="completed",
            total_items=5,
            processed_items=5,
            successful_items=5,
            failed_items=0,
            created_at=datetime.utcnow(),
        )
        
        user = create_test_user(["jobs:read"])
        
        with pytest.raises(HTTPException) as exc_info:
            await download_bulk_export(batch_id, user)
        
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "not an export" in exc_info.value.detail.lower()
        
        # Cleanup
        bulk._bulk_operations.clear()

    @pytest.mark.asyncio
    async def test_download_export_not_completed(self):
        """Test downloading when export is not completed."""
        from src.api.routes import bulk
        from src.api.routes.bulk import download_bulk_export
        
        bulk._bulk_operations.clear()
        
        export_id = uuid4()
        bulk._bulk_operations[export_id] = bulk.BulkOperationStatus(
            batch_id=export_id,
            operation_type="export",
            status="processing",  # Not completed
            total_items=100,
            processed_items=50,
            successful_items=50,
            failed_items=0,
            created_at=datetime.utcnow(),
        )
        
        user = create_test_user(["jobs:read"])
        
        with pytest.raises(HTTPException) as exc_info:
            await download_bulk_export(export_id, user)
        
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "not yet completed" in exc_info.value.detail.lower()
        
        # Cleanup
        bulk._bulk_operations.clear()


@pytest.mark.unit
class TestCancelBulkOperation:
    """Tests for POST /bulk/cancel/{batch_id} - Cancel bulk operation."""

    @pytest.mark.asyncio
    async def test_cancel_bulk_operation_success(self):
        """Test successful cancellation of pending operation."""
        from src.api.routes import bulk
        from src.api.routes.bulk import cancel_bulk_operation
        
        bulk._bulk_operations.clear()
        
        batch_id = uuid4()
        bulk._bulk_operations[batch_id] = bulk.BulkOperationStatus(
            batch_id=batch_id,
            operation_type="export",
            status="pending",
            total_items=100,
            processed_items=0,
            successful_items=0,
            failed_items=0,
            created_at=datetime.utcnow(),
        )
        
        user = create_test_user(["jobs:write"])
        
        response = await cancel_bulk_operation(batch_id, user)
        
        assert response.success is True
        assert response.data["status"] == "cancelled"
        assert response.data["batch_id"] == batch_id
        
        # Verify operation was updated
        assert bulk._bulk_operations[batch_id].status == "cancelled"
        
        # Cleanup
        bulk._bulk_operations.clear()

    @pytest.mark.asyncio
    async def test_cancel_bulk_operation_not_found(self):
        """Test cancelling non-existent operation."""
        from src.api.routes.bulk import cancel_bulk_operation
        
        batch_id = uuid4()
        
        user = create_test_user(["jobs:write"])
        
        with pytest.raises(HTTPException) as exc_info:
            await cancel_bulk_operation(batch_id, user)
        
        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_cancel_bulk_operation_already_completed(self):
        """Test cancelling already completed operation."""
        from src.api.routes import bulk
        from src.api.routes.bulk import cancel_bulk_operation
        
        bulk._bulk_operations.clear()
        
        batch_id = uuid4()
        bulk._bulk_operations[batch_id] = bulk.BulkOperationStatus(
            batch_id=batch_id,
            operation_type="ingest",
            status="completed",  # Already completed
            total_items=5,
            processed_items=5,
            successful_items=5,
            failed_items=0,
            created_at=datetime.utcnow(),
        )
        
        user = create_test_user(["jobs:write"])
        
        with pytest.raises(HTTPException) as exc_info:
            await cancel_bulk_operation(batch_id, user)
        
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "Cannot cancel" in exc_info.value.detail
        
        # Cleanup
        bulk._bulk_operations.clear()
