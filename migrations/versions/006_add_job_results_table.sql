-- Migration: 006_add_job_results_table
-- Revision ID: 006
-- Revises: 005
-- Create Date: 2026-02-24 11:30:00.000000

-- ============================================
-- UPGRADE: Create job_results table
-- ============================================

-- Create job_results table
CREATE TABLE job_results (
    id UUID NOT NULL,
    job_id UUID NOT NULL,
    extracted_text TEXT,
    output_data JSONB,
    metadata JSONB NOT NULL DEFAULT '{}',
    quality_score FLOAT,
    processing_time_ms INTEGER,
    output_uri VARCHAR(500),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE,
    
    -- Primary key
    CONSTRAINT pk_job_results PRIMARY KEY (id),
    
    -- Foreign key to jobs table with CASCADE delete
    CONSTRAINT fk_job_results_job_id FOREIGN KEY (job_id) 
        REFERENCES jobs(id) ON DELETE CASCADE,
    
    -- Unique constraint: one result per job
    CONSTRAINT uq_job_results_job_id UNIQUE (job_id)
);

-- Create indexes for efficient lookups

-- Index on job_id for foreign key lookups and result retrieval
CREATE INDEX idx_job_results_job_id ON job_results (job_id);

-- Index on expires_at for cleanup queries
CREATE INDEX idx_job_results_expires ON job_results (expires_at);

-- Index on created_at for sorting and pagination
CREATE INDEX idx_job_results_created ON job_results (created_at);


-- ============================================
-- DOWNGRADE: Drop job_results table
-- ============================================
-- To downgrade, execute these statements in reverse order:

-- DROP INDEX idx_job_results_created;
-- DROP INDEX idx_job_results_expires;
-- DROP INDEX idx_job_results_job_id;
-- DROP TABLE job_results;


-- ============================================
-- COMMENTS
-- ============================================

COMMENT ON TABLE job_results IS 'Stores job processing results with extracted text, output data, and metadata';
COMMENT ON COLUMN job_results.id IS 'Unique identifier (UUID)';
COMMENT ON COLUMN job_results.job_id IS 'Foreign key to jobs.id (one result per job)';
COMMENT ON COLUMN job_results.extracted_text IS 'Extracted text content from the job';
COMMENT ON COLUMN job_results.output_data IS 'Structured output data as JSONB';
COMMENT ON COLUMN job_results.metadata IS 'Additional metadata as JSONB';
COMMENT ON COLUMN job_results.quality_score IS 'Quality score (0-1) for the result';
COMMENT ON COLUMN job_results.processing_time_ms IS 'Processing time in milliseconds';
COMMENT ON COLUMN job_results.output_uri IS 'URI for large results stored externally';
COMMENT ON COLUMN job_results.created_at IS 'Timestamp when the result was created';
COMMENT ON COLUMN job_results.expires_at IS 'Timestamp when the result expires (for cleanup)';
