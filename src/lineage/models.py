"""Data lineage models.

This module defines the data models for tracking data lineage
through the processing pipeline.
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class DataLineageRecord(BaseModel):
    """Record of a data transformation in the pipeline.
    
    This model tracks how data is transformed through each stage
    of the processing pipeline, including input/output hashes
    for data integrity verification.
    
    Attributes:
        id: Unique record identifier
        job_id: Associated job ID
        stage: Pipeline stage (ingest, detect, parse, enrich, etc.)
        step_order: Order of this step in the pipeline
        input_hash: Hash of input data
        output_hash: Hash of output data
        transformation: Description of transformation performed
        timestamp: When transformation occurred
        metadata: Additional metadata
        duration_ms: Processing duration in milliseconds
        input_size_bytes: Input data size
        output_size_bytes: Output data size
    """

    id: UUID = Field(default_factory=UUID)
    job_id: UUID
    stage: str
    step_order: int
    input_hash: str | None = None
    output_hash: str | None = None
    transformation: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)
    duration_ms: int | None = None
    input_size_bytes: int | None = None
    output_size_bytes: int | None = None

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary."""
        return {
            "id": str(self.id),
            "job_id": str(self.job_id),
            "stage": self.stage,
            "step_order": self.step_order,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "transformation": self.transformation,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "duration_ms": self.duration_ms,
            "input_size_bytes": self.input_size_bytes,
            "output_size_bytes": self.output_size_bytes,
        }

    @classmethod
    def from_stage(
        cls,
        job_id: UUID,
        stage: str,
        step_order: int,
        transformation: str,
        input_data: bytes | None = None,
        output_data: bytes | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> "DataLineageRecord":
        """Create a lineage record for a pipeline stage.
        
        Args:
            job_id: Job identifier
            stage: Stage name
            step_order: Step order
            transformation: Transformation description
            input_data: Input data bytes
            output_data: Output data bytes
            metadata: Additional metadata
            **kwargs: Additional fields
            
        Returns:
            Lineage record
        """
        input_hash = cls._compute_hash(input_data) if input_data else None
        output_hash = cls._compute_hash(output_data) if output_data else None

        input_size = len(input_data) if input_data else None
        output_size = len(output_data) if output_data else None

        return cls(
            job_id=job_id,
            stage=stage,
            step_order=step_order,
            input_hash=input_hash,
            output_hash=output_hash,
            transformation=transformation,
            metadata=metadata or {},
            input_size_bytes=input_size,
            output_size_bytes=output_size,
            **kwargs,
        )

    @staticmethod
    def _compute_hash(data: bytes) -> str:
        """Compute SHA-256 hash of data.
        
        Args:
            data: Data to hash
            
        Returns:
            Hex digest of hash
        """
        return hashlib.sha256(data).hexdigest()

    def verify_integrity(self, current_data: bytes) -> bool:
        """Verify data integrity against stored hash.
        
        Args:
            current_data: Current data bytes
            
        Returns:
            True if hash matches
        """
        if not self.output_hash:
            return True  # No hash to verify

        current_hash = self._compute_hash(current_data)
        return current_hash == self.output_hash


@dataclass
class LineageNode:
    """Node in the lineage graph.
    
    Represents a point in the data lineage (input, output, or intermediate).
    
    Attributes:
        id: Node identifier
        job_id: Job ID
        stage: Stage name
        data_hash: Hash of data at this node
        timestamp: Timestamp
        metadata: Node metadata
    """
    id: str
    job_id: UUID
    stage: str
    data_hash: str | None
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LineageEdge:
    """Edge in the lineage graph.
    
    Represents a transformation between two lineage nodes.
    
    Attributes:
        from_node: Source node ID
        to_node: Target node ID
        transformation: Transformation description
        duration_ms: Processing duration
        metadata: Edge metadata
    """
    from_node: str
    to_node: str
    transformation: str
    duration_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LineageGraph:
    """Graph representation of data lineage.
    
    Provides a graph view of lineage for visualization and analysis.
    
    Attributes:
        job_id: Job ID
        nodes: List of nodes
        edges: List of edges
        start_time: Pipeline start time
        end_time: Pipeline end time
    """
    job_id: UUID
    nodes: list[LineageNode] = field(default_factory=list)
    edges: list[LineageEdge] = field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert graph to dictionary."""
        return {
            "job_id": str(self.job_id),
            "nodes": [
                {
                    "id": n.id,
                    "stage": n.stage,
                    "data_hash": n.data_hash,
                    "timestamp": n.timestamp.isoformat(),
                    "metadata": n.metadata,
                }
                for n in self.nodes
            ],
            "edges": [
                {
                    "from": e.from_node,
                    "to": e.to_node,
                    "transformation": e.transformation,
                    "duration_ms": e.duration_ms,
                    "metadata": e.metadata,
                }
                for e in self.edges
            ],
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }


class LineageQuery(BaseModel):
    """Query parameters for lineage retrieval.
    
    Attributes:
        job_id: Filter by job ID
        stage: Filter by stage
        start_time: Filter by start time
        end_time: Filter by end time
        include_metadata: Include metadata in results
    """

    job_id: UUID | None = None
    stage: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    include_metadata: bool = True


class LineageSummary(BaseModel):
    """Summary of lineage for a job.
    
    Attributes:
        job_id: Job ID
        total_stages: Number of stages
        total_transformations: Number of transformations
        start_time: Pipeline start time
        end_time: Pipeline end time
        total_duration_ms: Total processing duration
        stages: List of stage names
    """

    job_id: UUID
    total_stages: int
    total_transformations: int
    start_time: datetime | None
    end_time: datetime | None
    total_duration_ms: int | None
    stages: list[str]
