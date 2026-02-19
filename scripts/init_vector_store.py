#!/usr/bin/env python3
"""Initialize vector store database with migrations."""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


async def init_database():
    """Initialize database with pgvector extensions and tables."""
    
    # Get database URL from environment or use default
    db_url = os.getenv(
        "DB_URL",
        "postgresql+asyncpg://postgres:postgres@localhost:5432/pipeline"
    )
    
    print(f"🔄 Connecting to database: {db_url.replace('postgres:postgres', '***:***')}")
    
    engine = create_async_engine(db_url, echo=False)
    
    async with engine.begin() as conn:
        print("\n📦 Step 1: Creating pgvector extension...")
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        result = await conn.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'vector';"))
        version = result.scalar()
        print(f"   ✅ pgvector extension created (version: {version})")
        
        print("\n📦 Step 2: Creating pg_trgm extension...")
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))
        result = await conn.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'pg_trgm';"))
        version = result.scalar()
        print(f"   ✅ pg_trgm extension created (version: {version})")
        
        print("\n📦 Step 3: Creating document_chunks table...")
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                content_hash VARCHAR(64),
                embedding VECTOR(1536),
                chunk_metadata JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE (job_id, chunk_index)
            );
        """))
        print("   ✅ document_chunks table created")
        
        print("\n📦 Step 4: Creating indexes...")
        
        # HNSW index for vector similarity
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding_hnsw 
            ON document_chunks 
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """))
        print("   ✅ HNSW index created")
        
        # GIN index for full-text search
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_document_chunks_content_tsvector 
            ON document_chunks 
            USING gin (to_tsvector('english', content));
        """))
        print("   ✅ GIN full-text index created")
        
        # GIN index for trigram fuzzy matching
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_document_chunks_content_trgm 
            ON document_chunks 
            USING gin (content gin_trgm_ops);
        """))
        print("   ✅ GIN trigram index created")
        
        # Standard indexes
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_document_chunks_job_id 
            ON document_chunks (job_id);
        """))
        print("   ✅ job_id index created")
        
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_document_chunks_content_hash 
            ON document_chunks (content_hash);
        """))
        print("   ✅ content_hash index created")
        
        print("\n📦 Step 5: Verifying setup...")
        result = await conn.execute(text("""
            SELECT table_name, column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'document_chunks' 
            ORDER BY ordinal_position;
        """))
        columns = result.fetchall()
        print(f"   ✅ Table has {len(columns)} columns:")
        for col in columns:
            print(f"      - {col[1]}: {col[2]}")
        
        # Check indexes
        result = await conn.execute(text("""
            SELECT indexname, indexdef 
            FROM pg_indexes 
            WHERE tablename = 'document_chunks';
        """))
        indexes = result.fetchall()
        print(f"\n   ✅ Table has {len(indexes)} indexes:")
        for idx in indexes:
            print(f"      - {idx[0]}")
    
    await engine.dispose()
    
    print("\n" + "=" * 60)
    print("✅ DATABASE INITIALIZATION COMPLETE")
    print("=" * 60)
    print("\nVector store is ready to use!")
    print("Connection: postgresql://postgres:postgres@localhost:5432/pipeline")


if __name__ == "__main__":
    asyncio.run(init_database())
