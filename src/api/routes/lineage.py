"""Data lineage API routes.

This module provides endpoints for querying data lineage information.
"""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from src.auth.base import User
from src.auth.dependencies import require_read_lineage
from src.lineage.models import LineageQuery
from src.lineage.tracker import get_lineage_tracker

router = APIRouter(prefix="/lineage", tags=["Data Lineage"])


# ============================================================================
# Response Models
# ============================================================================

class LineageRecordResponse(BaseModel):
    """Lineage record response."""
    id: str
    job_id: str
    stage: str
    step_order: int
    input_hash: str | None
    output_hash: str | None
    transformation: str
    timestamp: str
    duration_ms: int | None
    input_size_bytes: int | None
    output_size_bytes: int | None
    metadata: dict[str, Any]


class LineageNodeResponse(BaseModel):
    """Lineage node response."""
    id: str
    stage: str
    data_hash: str | None
    timestamp: str
    metadata: dict[str, Any]


class LineageEdgeResponse(BaseModel):
    """Lineage edge response."""
    from_node: str = Field(..., alias="from")
    to_node: str = Field(..., alias="to")
    transformation: str
    duration_ms: int | None
    metadata: dict[str, Any]

    class Config:
        populate_by_name = True


class LineageGraphResponse(BaseModel):
    """Lineage graph response."""
    job_id: str
    nodes: list[LineageNodeResponse]
    edges: list[LineageEdgeResponse]
    start_time: str | None
    end_time: str | None


class LineageSummaryResponse(BaseModel):
    """Lineage summary response."""
    job_id: str
    total_stages: int
    total_transformations: int
    start_time: str | None
    end_time: str | None
    total_duration_ms: int | None
    stages: list[str]


class DataIntegrityResponse(BaseModel):
    """Data integrity verification response."""
    stage: str
    verified: bool
    expected_hash: str | None
    actual_hash: str | None
    message: str


# ============================================================================
# Lineage Endpoints
# ============================================================================

@router.get("/{job_id}", response_model=list[LineageRecordResponse])
async def get_job_lineage(
    job_id: UUID,
    stage: str | None = None,
    user: User = Depends(require_read_lineage),
) -> list[LineageRecordResponse]:
    """Get lineage records for a job.
    
    Retrieves all transformation records for a specific job.
    Requires lineage:read permission.
    """
    tracker = get_lineage_tracker()

    # Query with optional stage filter
    query = LineageQuery(job_id=job_id, stage=stage)
    records = await tracker.store.query_records(query)

    return [
        LineageRecordResponse(
            id=str(record.id),
            job_id=str(record.job_id),
            stage=record.stage,
            step_order=record.step_order,
            input_hash=record.input_hash,
            output_hash=record.output_hash,
            transformation=record.transformation,
            timestamp=record.timestamp.isoformat(),
            duration_ms=record.duration_ms,
            input_size_bytes=record.input_size_bytes,
            output_size_bytes=record.output_size_bytes,
            metadata=record.metadata,
        )
        for record in records
    ]


@router.get("/{job_id}/graph", response_model=LineageGraphResponse)
async def get_lineage_graph(
    job_id: UUID,
    user: User = Depends(require_read_lineage),
) -> LineageGraphResponse:
    """Get lineage as a graph structure.
    
    Returns the lineage records as a graph of nodes and edges
    suitable for visualization.
    """
    tracker = get_lineage_tracker()

    graph = await tracker.get_lineage_graph(job_id)

    return LineageGraphResponse(
        job_id=str(graph.job_id),
        nodes=[
            LineageNodeResponse(
                id=node.id,
                stage=node.stage,
                data_hash=node.data_hash,
                timestamp=node.timestamp.isoformat(),
                metadata=node.metadata,
            )
            for node in graph.nodes
        ],
        edges=[
            LineageEdgeResponse(
                **{
                    "from": edge.from_node,
                    "to": edge.to_node,
                    "transformation": edge.transformation,
                    "duration_ms": edge.duration_ms,
                    "metadata": dict(edge.metadata),
                }  # type: ignore[arg-type]
            )
            for edge in graph.edges
        ],
        start_time=graph.start_time.isoformat() if graph.start_time else None,
        end_time=graph.end_time.isoformat() if graph.end_time else None,
    )


@router.get("/{job_id}/summary", response_model=LineageSummaryResponse)
async def get_lineage_summary(
    job_id: UUID,
    user: User = Depends(require_read_lineage),
) -> LineageSummaryResponse:
    """Get lineage summary for a job.
    
    Returns aggregated statistics about the job's processing lineage.
    """
    tracker = get_lineage_tracker()

    summary = await tracker.get_lineage_summary(job_id)

    return LineageSummaryResponse(
        job_id=str(summary.job_id),
        total_stages=summary.total_stages,
        total_transformations=summary.total_transformations,
        start_time=summary.start_time.isoformat() if summary.start_time else None,
        end_time=summary.end_time.isoformat() if summary.end_time else None,
        total_duration_ms=summary.total_duration_ms,
        stages=summary.stages,
    )


@router.post("/{job_id}/verify/{stage}", response_model=DataIntegrityResponse)
async def verify_data_integrity(
    job_id: UUID,
    stage: str,
    user: User = Depends(require_read_lineage),
) -> DataIntegrityResponse:
    """Verify data integrity for a specific stage.
    
    This endpoint would typically accept the data to verify
    in the request body. For this implementation, we return
    information about how to verify.
    """
    # In a full implementation, this would:
    # 1. Accept the current data in the request body
    # 2. Compare its hash against the stored hash
    # 3. Return verification result

    tracker = get_lineage_tracker()

    # Get lineage records for the stage
    query = LineageQuery(job_id=job_id, stage=stage)
    records = await tracker.store.query_records(query)

    if not records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No lineage records found for stage: {stage}",
        )

    record = records[0]

    return DataIntegrityResponse(
        stage=stage,
        verified=False,  # Would need actual data to verify
        expected_hash=record.output_hash,
        actual_hash=None,
        message="Data verification requires the actual data to be provided in the request body",
    )


@router.get("/{job_id}/stages")
async def get_job_stages(
    job_id: UUID,
    user: User = Depends(require_read_lineage),
) -> dict[str, Any]:
    """Get list of stages for a job."""
    tracker = get_lineage_tracker()

    summary = await tracker.get_lineage_summary(job_id)

    return {
        "job_id": str(job_id),
        "stages": summary.stages,
        "total_stages": summary.total_stages,
    }


@router.get("/{job_id}/stages/{stage}/input")
async def get_stage_input_hash(
    job_id: UUID,
    stage: str,
    user: User = Depends(require_read_lineage),
) -> dict[str, Any]:
    """Get input hash for a specific stage."""
    tracker = get_lineage_tracker()

    query = LineageQuery(job_id=job_id, stage=stage)
    records = await tracker.store.query_records(query)

    if not records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No lineage records found for stage: {stage}",
        )

    record = records[0]

    return {
        "job_id": str(job_id),
        "stage": stage,
        "input_hash": record.input_hash,
        "input_size_bytes": record.input_size_bytes,
    }


@router.get("/{job_id}/stages/{stage}/output")
async def get_stage_output_hash(
    job_id: UUID,
    stage: str,
    user: User = Depends(require_read_lineage),
) -> dict[str, Any]:
    """Get output hash for a specific stage."""
    tracker = get_lineage_tracker()

    query = LineageQuery(job_id=job_id, stage=stage)
    records = await tracker.store.query_records(query)

    if not records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No lineage records found for stage: {stage}",
        )

    # Get the last record for the stage (final output)
    record = sorted(records, key=lambda r: r.step_order)[-1]

    return {
        "job_id": str(job_id),
        "stage": stage,
        "output_hash": record.output_hash,
        "output_size_bytes": record.output_size_bytes,
        "transformation": record.transformation,
    }
