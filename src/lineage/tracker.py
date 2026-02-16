"""Data lineage tracker implementation.

This module provides the DataLineageTracker class for tracking
all transformations through the processing pipeline.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol
from uuid import UUID

from src.lineage.models import (
    DataLineageRecord,
    LineageEdge,
    LineageGraph,
    LineageNode,
    LineageQuery,
    LineageSummary,
)


class LineageStore(Protocol):
    """Protocol for lineage storage backends."""
    
    async def save_record(self, record: DataLineageRecord) -> None:
        """Save a lineage record."""
        ...
    
    async def get_records(
        self,
        job_id: UUID,
    ) -> List[DataLineageRecord]:
        """Get all records for a job."""
        ...
    
    async def query_records(
        self,
        query: LineageQuery,
    ) -> List[DataLineageRecord]:
        """Query lineage records."""
        ...


class InMemoryLineageStore:
    """In-memory lineage store for development/testing."""
    
    def __init__(self):
        """Initialize in-memory store."""
        self._records: List[DataLineageRecord] = []
        self._lock = asyncio.Lock()
    
    async def save_record(self, record: DataLineageRecord) -> None:
        """Save a lineage record."""
        async with self._lock:
            self._records.append(record)
    
    async def get_records(
        self,
        job_id: UUID,
    ) -> List[DataLineageRecord]:
        """Get all records for a job."""
        async with self._lock:
            return [r for r in self._records if r.job_id == job_id]
    
    async def query_records(
        self,
        query: LineageQuery,
    ) -> List[DataLineageRecord]:
        """Query lineage records."""
        async with self._lock:
            records = self._records
            
            if query.job_id:
                records = [r for r in records if r.job_id == query.job_id]
            
            if query.stage:
                records = [r for r in records if r.stage == query.stage]
            
            if query.start_time:
                records = [r for r in records if r.timestamp >= query.start_time]
            
            if query.end_time:
                records = [r for r in records if r.timestamp <= query.end_time]
            
            # Sort by step order
            records.sort(key=lambda r: r.step_order)
            
            return records


class DataLineageTracker:
    """Tracker for data lineage through the processing pipeline.
    
    This class tracks all transformations applied to data as it flows
    through the 7-stage pipeline, maintaining input/output hashes for
    data integrity verification.
    
    Example:
        tracker = DataLineageTracker()
        
        # Track a transformation
        await tracker.track_transformation(
            job_id=job_id,
            stage="parse",
            input_data=file_bytes,
            output_data=extracted_text.encode(),
            transformation="docling_pdf_extraction",
        )
        
        # Get lineage for a job
        records = await tracker.get_job_lineage(job_id)
        
        # Get lineage graph
        graph = await tracker.get_lineage_graph(job_id)
    """
    
    # Pipeline stages in order
    PIPELINE_STAGES = [
        "ingest",
        "detect",
        "parse",
        "enrich",
        "quality",
        "transform",
        "output",
    ]
    
    def __init__(self, store: Optional[LineageStore] = None):
        """Initialize lineage tracker.
        
        Args:
            store: Storage backend (defaults to in-memory)
        """
        self.store = store or InMemoryLineageStore()
        self._step_counters: Dict[UUID, int] = {}
    
    def _get_next_step_order(self, job_id: UUID) -> int:
        """Get the next step order for a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Next step order number
        """
        current = self._step_counters.get(job_id, 0)
        self._step_counters[job_id] = current + 1
        return current
    
    async def track_transformation(
        self,
        job_id: UUID,
        stage: str,
        transformation: str,
        input_data: Optional[bytes] = None,
        output_data: Optional[bytes] = None,
        metadata: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[int] = None,
    ) -> DataLineageRecord:
        """Track a transformation in the pipeline.
        
        Args:
            job_id: Job identifier
            stage: Pipeline stage name
            transformation: Description of transformation
            input_data: Input data bytes
            output_data: Output data bytes
            metadata: Additional metadata
            duration_ms: Processing duration in milliseconds
            
        Returns:
            Created lineage record
        """
        step_order = self._get_next_step_order(job_id)
        
        record = DataLineageRecord.from_stage(
            job_id=job_id,
            stage=stage,
            step_order=step_order,
            transformation=transformation,
            input_data=input_data,
            output_data=output_data,
            metadata=metadata or {},
            duration_ms=duration_ms,
        )
        
        await self.store.save_record(record)
        return record
    
    async def track_stage_start(
        self,
        job_id: UUID,
        stage: str,
        input_data: Optional[bytes] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DataLineageRecord:
        """Track the start of a pipeline stage.
        
        Args:
            job_id: Job identifier
            stage: Stage name
            input_data: Input data for the stage
            metadata: Additional metadata
            
        Returns:
            Created lineage record
        """
        return await self.track_transformation(
            job_id=job_id,
            stage=stage,
            transformation=f"{stage}_start",
            input_data=input_data,
            metadata={"status": "started", **(metadata or {})},
        )
    
    async def track_stage_complete(
        self,
        job_id: UUID,
        stage: str,
        output_data: Optional[bytes] = None,
        metadata: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[int] = None,
    ) -> DataLineageRecord:
        """Track the completion of a pipeline stage.
        
        Args:
            job_id: Job identifier
            stage: Stage name
            output_data: Output data from the stage
            metadata: Additional metadata
            duration_ms: Processing duration
            
        Returns:
            Created lineage record
        """
        return await self.track_transformation(
            job_id=job_id,
            stage=stage,
            transformation=f"{stage}_complete",
            output_data=output_data,
            metadata={"status": "completed", **(metadata or {})},
            duration_ms=duration_ms,
        )
    
    async def track_parsing(
        self,
        job_id: UUID,
        parser_used: str,
        input_data: bytes,
        output_data: bytes,
        page_count: Optional[int] = None,
        confidence: Optional[float] = None,
        duration_ms: Optional[int] = None,
    ) -> DataLineageRecord:
        """Track a parsing operation.
        
        Args:
            job_id: Job identifier
            parser_used: Name of parser used
            input_data: Input document bytes
            output_data: Extracted text/content bytes
            page_count: Number of pages processed
            confidence: Extraction confidence score
            duration_ms: Processing duration
            
        Returns:
            Created lineage record
        """
        return await self.track_transformation(
            job_id=job_id,
            stage="parse",
            transformation=f"parse_{parser_used}",
            input_data=input_data,
            output_data=output_data,
            metadata={
                "parser": parser_used,
                "page_count": page_count,
                "confidence": confidence,
            },
            duration_ms=duration_ms,
        )
    
    async def track_enrichment(
        self,
        job_id: UUID,
        enrichment_type: str,
        input_data: bytes,
        output_data: bytes,
        entities_extracted: Optional[int] = None,
        duration_ms: Optional[int] = None,
    ) -> DataLineageRecord:
        """Track an enrichment operation.
        
        Args:
            job_id: Job identifier
            enrichment_type: Type of enrichment applied
            input_data: Input data bytes
            output_data: Output data bytes
            entities_extracted: Number of entities extracted
            duration_ms: Processing duration
            
        Returns:
            Created lineage record
        """
        return await self.track_transformation(
            job_id=job_id,
            stage="enrich",
            transformation=f"enrich_{enrichment_type}",
            input_data=input_data,
            output_data=output_data,
            metadata={
                "enrichment_type": enrichment_type,
                "entities_extracted": entities_extracted,
            },
            duration_ms=duration_ms,
        )
    
    async def track_transformation_stage(
        self,
        job_id: UUID,
        transformation_type: str,
        input_data: bytes,
        output_data: bytes,
        chunk_count: Optional[int] = None,
        duration_ms: Optional[int] = None,
    ) -> DataLineageRecord:
        """Track the transformation stage (chunking, embeddings, etc.).
        
        Args:
            job_id: Job identifier
            transformation_type: Type of transformation
            input_data: Input data bytes
            output_data: Output data bytes
            chunk_count: Number of chunks created
            duration_ms: Processing duration
            
        Returns:
            Created lineage record
        """
        return await self.track_transformation(
            job_id=job_id,
            stage="transform",
            transformation=f"transform_{transformation_type}",
            input_data=input_data,
            output_data=output_data,
            metadata={
                "transformation_type": transformation_type,
                "chunk_count": chunk_count,
            },
            duration_ms=duration_ms,
        )
    
    async def track_output(
        self,
        job_id: UUID,
        destination_type: str,
        input_data: bytes,
        records_written: Optional[int] = None,
        duration_ms: Optional[int] = None,
    ) -> DataLineageRecord:
        """Track data output to destination.
        
        Args:
            job_id: Job identifier
            destination_type: Type of destination
            input_data: Data written
            records_written: Number of records written
            duration_ms: Processing duration
            
        Returns:
            Created lineage record
        """
        return await self.track_transformation(
            job_id=job_id,
            stage="output",
            transformation=f"output_to_{destination_type}",
            input_data=input_data,
            metadata={
                "destination_type": destination_type,
                "records_written": records_written,
            },
            duration_ms=duration_ms,
        )
    
    async def get_job_lineage(
        self,
        job_id: UUID,
    ) -> List[DataLineageRecord]:
        """Get all lineage records for a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            List of lineage records
        """
        return await self.store.get_records(job_id)
    
    async def get_lineage_graph(
        self,
        job_id: UUID,
    ) -> LineageGraph:
        """Get lineage as a graph structure.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Lineage graph
        """
        records = await self.get_job_lineage(job_id)
        
        if not records:
            return LineageGraph(job_id=job_id)
        
        # Sort by step order
        records.sort(key=lambda r: r.step_order)
        
        # Build nodes and edges
        nodes: List[LineageNode] = []
        edges: List[LineageEdge] = []
        
        prev_node_id: Optional[str] = None
        
        for record in records:
            # Create node for output of this step
            node_id = f"{record.stage}_{record.step_order}"
            
            node = LineageNode(
                id=node_id,
                job_id=job_id,
                stage=record.stage,
                data_hash=record.output_hash,
                timestamp=record.timestamp,
                metadata={
                    "transformation": record.transformation,
                    "input_hash": record.input_hash,
                    "duration_ms": record.duration_ms,
                },
            )
            nodes.append(node)
            
            # Create edge from previous node
            if prev_node_id:
                edge = LineageEdge(
                    from_node=prev_node_id,
                    to_node=node_id,
                    transformation=record.transformation,
                    duration_ms=record.duration_ms,
                )
                edges.append(edge)
            
            prev_node_id = node_id
        
        return LineageGraph(
            job_id=job_id,
            nodes=nodes,
            edges=edges,
            start_time=records[0].timestamp if records else None,
            end_time=records[-1].timestamp if records else None,
        )
    
    async def get_lineage_summary(
        self,
        job_id: UUID,
    ) -> LineageSummary:
        """Get a summary of lineage for a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Lineage summary
        """
        records = await self.get_job_lineage(job_id)
        
        if not records:
            return LineageSummary(
                job_id=job_id,
                total_stages=0,
                total_transformations=0,
                start_time=None,
                end_time=None,
                total_duration_ms=None,
                stages=[],
            )
        
        # Sort by step order
        records.sort(key=lambda r: r.step_order)
        
        stages = []
        total_duration = 0
        
        for record in records:
            if record.stage not in stages:
                stages.append(record.stage)
            if record.duration_ms:
                total_duration += record.duration_ms
        
        return LineageSummary(
            job_id=job_id,
            total_stages=len(stages),
            total_transformations=len(records),
            start_time=records[0].timestamp,
            end_time=records[-1].timestamp,
            total_duration_ms=total_duration if total_duration > 0 else None,
            stages=stages,
        )
    
    async def verify_data_integrity(
        self,
        job_id: UUID,
        stage: str,
        current_data: bytes,
    ) -> bool:
        """Verify data integrity against stored hash.
        
        Args:
            job_id: Job identifier
            stage: Stage to verify
            current_data: Current data bytes
            
        Returns:
            True if hash matches
        """
        records = await self.get_job_lineage(job_id)
        
        for record in records:
            if record.stage == stage:
                return record.verify_integrity(current_data)
        
        return False  # No record found for stage


# Global lineage tracker instance
_lineage_tracker: Optional[DataLineageTracker] = None


def get_lineage_tracker() -> DataLineageTracker:
    """Get global lineage tracker instance.
    
    Returns:
        Lineage tracker singleton
    """
    global _lineage_tracker
    if _lineage_tracker is None:
        _lineage_tracker = DataLineageTracker()
    return _lineage_tracker


def set_lineage_tracker(tracker: DataLineageTracker) -> None:
    """Set the global lineage tracker.
    
    Args:
        tracker: Lineage tracker instance
    """
    global _lineage_tracker
    _lineage_tracker = tracker
