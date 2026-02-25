-- VPS Database Fix Script
-- Run this to add missing structures without losing existing data
-- Safe to run on existing database

-- ============================================
-- 1. CREATE ALEMBIC VERSION TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS alembic_version (
    version_num VARCHAR(32) NOT NULL PRIMARY KEY
);

-- Mark all migrations as applied (since tables exist)
INSERT INTO alembic_version (version_num) VALUES ('008')
ON CONFLICT (version_num) DO NOTHING;

-- ============================================
-- 2. CREATE MISSING INDEXES (Safe)
-- ============================================

-- Content hash index (for deduplication)
CREATE INDEX IF NOT EXISTS idx_document_chunks_content_hash 
ON document_chunks (content_hash);

-- Composite index for job + chunk_index queries
CREATE INDEX IF NOT EXISTS idx_document_chunks_job_chunk 
ON document_chunks (job_id, chunk_index);

-- GIN index for full-text search
CREATE INDEX IF NOT EXISTS idx_document_chunks_content_tsvector 
ON document_chunks 
USING gin (to_tsvector('english', content));

-- GIN index for trigram fuzzy matching
CREATE INDEX IF NOT EXISTS idx_document_chunks_content_trgm 
ON document_chunks 
USING gin (content gin_trgm_ops);

-- Note: HNSW index - only create if IVFFlat doesn't exist
-- First check if IVFFlat exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes 
        WHERE indexname = 'idx_document_chunks_embedding_hnsw'
    ) AND NOT EXISTS (
        SELECT 1 FROM pg_indexes 
        WHERE indexname LIKE '%embedding%' AND indexdef LIKE '%ivfflat%'
    ) THEN
        CREATE INDEX idx_document_chunks_embedding_hnsw 
        ON document_chunks 
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    END IF;
END $$;

-- ============================================
-- 3. CREATE MISSING TABLES (Safe)
-- ============================================

-- Pipelines table
CREATE TABLE IF NOT EXISTS pipelines (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    config JSONB NOT NULL DEFAULT '{}',
    version INTEGER NOT NULL DEFAULT 1,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_by VARCHAR(255),
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pipelines_name ON pipelines (name);

-- Add foreign key from jobs to pipelines (if not exists)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'fk_jobs_pipeline_id'
    ) THEN
        ALTER TABLE jobs 
        ADD CONSTRAINT fk_jobs_pipeline_id 
        FOREIGN KEY (pipeline_id) REFERENCES pipelines(id) ON DELETE SET NULL;
    END IF;
END $$;

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
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP WITHOUT TIME ZONE,
    access_count INTEGER NOT NULL DEFAULT 1,
    last_accessed_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_detection_hash ON content_detection_results (file_hash);
CREATE INDEX IF NOT EXISTS idx_detection_type ON content_detection_results (content_type);
CREATE INDEX IF NOT EXISTS idx_detection_expires ON content_detection_results (expires_at);
CREATE INDEX IF NOT EXISTS idx_detection_created ON content_detection_results (created_at);

-- Job detection results link table
CREATE TABLE IF NOT EXISTS job_detection_results (
    job_id UUID NOT NULL,
    detection_result_id UUID NOT NULL,
    PRIMARY KEY (job_id, detection_result_id),
    CONSTRAINT fk_job_detection_results_job_id FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE,
    CONSTRAINT fk_job_detection_results_detection_id FOREIGN KEY (detection_result_id) REFERENCES content_detection_results(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_job_detection_result ON job_detection_results (detection_result_id);

-- Audit logs
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW(),
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

CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_logs (timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_user_id ON audit_logs (user_id);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_logs (action);
CREATE INDEX IF NOT EXISTS idx_audit_resource ON audit_logs (resource_type);

-- API keys
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key_hash VARCHAR(64) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    permissions VARCHAR[] NOT NULL DEFAULT '{}',
    is_active INTEGER NOT NULL DEFAULT 1,
    created_by VARCHAR(255),
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP WITHOUT TIME ZONE,
    last_used_at TIMESTAMP WITHOUT TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys (key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_created ON api_keys (created_at);

-- Webhook subscriptions
CREATE TABLE IF NOT EXISTS webhook_subscriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    url VARCHAR(500) NOT NULL,
    events VARCHAR[] NOT NULL DEFAULT '{}',
    secret VARCHAR(255),
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_webhook_subs_user ON webhook_subscriptions (user_id);
CREATE INDEX IF NOT EXISTS idx_webhook_subs_created ON webhook_subscriptions (created_at);

-- Webhook deliveries
CREATE TABLE IF NOT EXISTS webhook_deliveries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subscription_id UUID NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}',
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 5,
    http_status INTEGER,
    last_error TEXT,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW(),
    delivered_at TIMESTAMP WITHOUT TIME ZONE,
    next_retry_at TIMESTAMP WITHOUT TIME ZONE,
    CONSTRAINT fk_webhook_deliveries_subscription_id FOREIGN KEY (subscription_id) REFERENCES webhook_subscriptions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_webhook_deliv_subscription ON webhook_deliveries (subscription_id);
CREATE INDEX IF NOT EXISTS idx_webhook_deliv_event ON webhook_deliveries (event_type);
CREATE INDEX IF NOT EXISTS idx_webhook_deliv_status ON webhook_deliveries (status);
CREATE INDEX IF NOT EXISTS idx_webhook_deliv_created ON webhook_deliveries (created_at);

-- Cache tables (optional but recommended)
CREATE TABLE IF NOT EXISTS embedding_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    text_hash VARCHAR(64) NOT NULL,
    text_preview VARCHAR(200),
    model VARCHAR(100) NOT NULL,
    embedding TEXT NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW(),
    accessed_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW(),
    access_count INTEGER NOT NULL DEFAULT 1,
    UNIQUE (text_hash, model)
);

CREATE INDEX IF NOT EXISTS idx_embedding_cache_text_hash ON embedding_cache (text_hash);
CREATE INDEX IF NOT EXISTS idx_embedding_cache_model ON embedding_cache (model);

-- ============================================
-- 4. VERIFY FIXES
-- ============================================

-- Count tables
SELECT 'Tables created' as check_item, COUNT(*) as count 
FROM information_schema.tables 
WHERE table_schema = 'public';

-- Count indexes on document_chunks
SELECT 'Document chunks indexes' as check_item, COUNT(*) as count 
FROM pg_indexes 
WHERE tablename = 'document_chunks';

-- Show all tables
SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename;
