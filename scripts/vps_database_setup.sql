-- VPS Database Setup Script for Agentic Pipeline Ingestor
-- This script ensures the database has all required extensions and structures
-- Run this before applying Alembic migrations if setting up a fresh database

-- ============================================
-- 1. EXTENSIONS
-- ============================================

-- Enable required PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";      -- For fuzzy text search
CREATE EXTENSION IF NOT EXISTS "vector";       -- For pgvector embeddings

-- Verify extensions
SELECT extname, extversion 
FROM pg_extension 
WHERE extname IN ('vector', 'pg_trgm', 'uuid-ossp');

-- ============================================
-- 2. CORE TABLES (if not using Alembic)
-- ============================================

-- Jobs table
CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status VARCHAR(20) NOT NULL DEFAULT 'created',
    source_type VARCHAR(50) NOT NULL,
    source_uri VARCHAR(500),
    file_name VARCHAR(255),
    file_size BIGINT,
    mime_type VARCHAR(100),
    priority VARCHAR(20) NOT NULL DEFAULT 'normal',
    mode VARCHAR(20) NOT NULL DEFAULT 'async',
    external_id VARCHAR(255),
    metadata_json JSONB NOT NULL DEFAULT '{}',
    error_message TEXT,
    error_code VARCHAR(50),
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 3,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    locked_by VARCHAR(255),
    locked_at TIMESTAMP WITH TIME ZONE,
    heartbeat_at TIMESTAMP WITH TIME ZONE,
    pipeline_id UUID,
    pipeline_config JSONB
);

-- Document chunks table
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_hash VARCHAR(64),
    embedding VECTOR(1536),  -- pgvector type
    chunk_metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT uq_document_chunks_job_chunk UNIQUE (job_id, chunk_index),
    CONSTRAINT fk_document_chunks_job_id 
        FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
);

-- Job results table
CREATE TABLE IF NOT EXISTS job_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL,
    extracted_text TEXT,
    output_data JSONB,
    metadata JSONB NOT NULL DEFAULT '{}',
    quality_score FLOAT,
    processing_time_ms INTEGER,
    output_uri VARCHAR(500),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    
    -- Constraints
    CONSTRAINT uq_job_results_job_id UNIQUE (job_id),
    CONSTRAINT fk_job_results_job_id 
        FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
);

-- Pipelines table
CREATE TABLE IF NOT EXISTS pipelines (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    config JSONB NOT NULL DEFAULT '{}',
    version INTEGER NOT NULL DEFAULT 1,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_by VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- ============================================
-- 3. INDEXES
-- ============================================

-- Jobs indexes
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_source_type ON jobs(source_type);
CREATE INDEX IF NOT EXISTS idx_jobs_external_id ON jobs(external_id);
CREATE INDEX IF NOT EXISTS idx_jobs_locked_by ON jobs(locked_by);

-- Document chunks indexes
CREATE INDEX IF NOT EXISTS idx_document_chunks_job_id ON document_chunks(job_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_content_hash ON document_chunks(content_hash);
CREATE INDEX IF NOT EXISTS idx_document_chunks_job_chunk ON document_chunks(job_id, chunk_index);

-- HNSW index for vector similarity search
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding_hnsw 
ON document_chunks 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- GIN index for full-text search
CREATE INDEX IF NOT EXISTS idx_document_chunks_content_tsvector 
ON document_chunks 
USING gin (to_tsvector('english', content));

-- GIN index for trigram fuzzy matching
CREATE INDEX IF NOT EXISTS idx_document_chunks_content_trgm 
ON document_chunks 
USING gin (content gin_trgm_ops);

-- Job results indexes
CREATE INDEX IF NOT EXISTS idx_job_results_job_id ON job_results(job_id);
CREATE INDEX IF NOT EXISTS idx_job_results_expires ON job_results(expires_at);
CREATE INDEX IF NOT EXISTS idx_job_results_created ON job_results(created_at);

-- ============================================
-- 4. SUPPORTING TABLES
-- ============================================

-- Content detection results
CREATE TABLE IF NOT EXISTS content_detection_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_hash VARCHAR(64) UNIQUE NOT NULL,
    file_size BIGINT NOT NULL,
    content_type VARCHAR(20) NOT NULL,
    confidence FLOAT NOT NULL,
    recommended_parser VARCHAR(50) NOT NULL,
    alternative_parsers VARCHAR[] NOT NULL DEFAULT '{}',
    text_statistics JSONB NOT NULL DEFAULT '{}',
    image_statistics JSONB NOT NULL DEFAULT '{}',
    page_results JSONB NOT NULL DEFAULT '{}',
    processing_time_ms INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    access_count INTEGER NOT NULL DEFAULT 1,
    last_accessed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Job detection results link table
CREATE TABLE IF NOT EXISTS job_detection_results (
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    detection_result_id UUID NOT NULL REFERENCES content_detection_results(id) ON DELETE CASCADE,
    PRIMARY KEY (job_id, detection_result_id)
);

-- Audit logs
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    user_id VARCHAR(255),
    api_key_id VARCHAR(255),
    action VARCHAR(50) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id VARCHAR(255),
    request_method VARCHAR(10),
    request_path VARCHAR(500),
    request_details JSONB,
    success INTEGER NOT NULL DEFAULT 1,
    error_message TEXT,
    ip_address VARCHAR(45),
    user_agent VARCHAR(500)
);

-- API keys
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key_hash VARCHAR(64) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    permissions VARCHAR[] NOT NULL DEFAULT '{}',
    is_active INTEGER NOT NULL DEFAULT 1,
    created_by VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    last_used_at TIMESTAMP WITH TIME ZONE
);

-- Webhook subscriptions
CREATE TABLE IF NOT EXISTS webhook_subscriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    url VARCHAR(500) NOT NULL,
    events VARCHAR[] NOT NULL DEFAULT '{}',
    secret VARCHAR(255),
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Webhook deliveries
CREATE TABLE IF NOT EXISTS webhook_deliveries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subscription_id UUID NOT NULL REFERENCES webhook_subscriptions(id) ON DELETE CASCADE,
    event_type VARCHAR(50) NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}',
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 5,
    http_status INTEGER,
    last_error TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    delivered_at TIMESTAMP WITH TIME ZONE,
    next_retry_at TIMESTAMP WITH TIME ZONE
);

-- ============================================
-- 5. ALEMBIC VERSION TABLE
-- ============================================

-- Create alembic version table to track migrations
CREATE TABLE IF NOT EXISTS alembic_version (
    version_num VARCHAR(32) NOT NULL PRIMARY KEY
);

-- Insert current version (update this as migrations are added)
-- This prevents alembic from trying to re-run migrations
INSERT INTO alembic_version (version_num) 
VALUES ('007') 
ON CONFLICT (version_num) DO NOTHING;

-- ============================================
-- 6. VERIFICATION QUERY
-- ============================================

-- Check all tables exist
SELECT 'Tables' as check_type, 
       COUNT(*) as count,
       (SELECT array_agg(table_name) 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name IN ('jobs', 'document_chunks', 'job_results', 'pipelines',
                           'content_detection_results', 'job_detection_results',
                           'audit_logs', 'api_keys', 'webhook_subscriptions', 
                           'webhook_deliveries')) as details
FROM information_schema.tables 
WHERE table_schema = 'public';

-- Check extensions
SELECT 'Extensions' as check_type,
       COUNT(*) as count,
       (SELECT array_agg(extname) FROM pg_extension WHERE extname IN ('vector', 'pg_trgm')) as details
FROM pg_extension 
WHERE extname IN ('vector', 'pg_trgm');
