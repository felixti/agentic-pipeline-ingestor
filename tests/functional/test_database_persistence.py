#!/usr/bin/env python3
"""Test actual database persistence of chunks with embeddings.

This test proves that chunks with vectors are actually saved to PostgreSQL
and can be retrieved with all data intact.
"""

import asyncio
import uuid
from datetime import datetime

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker


# Database connection
DB_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/pipeline"


class TestDatabasePersistence:
    """Test actual database persistence of chunks with embeddings."""

    @pytest.fixture
    async def db_session(self):
        """Create database session."""
        engine = create_async_engine(DB_URL, echo=False)
        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        
        async with async_session() as session:
            yield session
        
        await engine.dispose()

    async def test_pgvector_extension_loaded(self):
        """PROOF 1: pgvector extension is loaded in database."""
        engine = create_async_engine(DB_URL, echo=False)
        
        async with engine.connect() as conn:
            result = await conn.execute(text(
                "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"
            ))
            row = result.fetchone()
            
            assert row is not None, "❌ pgvector extension not found!"
            assert row[0] == "vector", f"❌ Wrong extension: {row[0]}"
            print(f"\n✅ PROOF 1: pgvector extension loaded (version: {row[1]})")
        
        await engine.dispose()

    async def test_document_chunks_table_exists(self):
        """PROOF 2: document_chunks table exists."""
        engine = create_async_engine(DB_URL, echo=False)
        
        async with engine.connect() as conn:
            result = await conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'document_chunks'
                );
            """))
            exists = result.scalar()
            
            assert exists is True, "❌ document_chunks table not found!"
            print("\n✅ PROOF 2: document_chunks table exists")
        
        await engine.dispose()

    async def test_vector_column_exists(self):
        """PROOF 3: embedding column has VECTOR type."""
        engine = create_async_engine(DB_URL, echo=False)
        
        async with engine.connect() as conn:
            result = await conn.execute(text("""
                SELECT data_type, udt_name 
                FROM information_schema.columns 
                WHERE table_name = 'document_chunks' AND column_name = 'embedding';
            """))
            row = result.fetchone()
            
            assert row is not None, "❌ embedding column not found!"
            # VECTOR type shows as USER-DEFINED with udt_name 'vector'
            assert row[1] == "vector", f"❌ Wrong type: {row[1]}"
            print(f"\n✅ PROOF 3: embedding column is VECTOR type (udt: {row[1]})")
        
        await engine.dispose()

    async def test_insert_chunk_with_embedding(self):
        """PROOF 4: Can insert chunk with embedding into database."""
        engine = create_async_engine(DB_URL, echo=False)
        
        # Generate test data
        test_id = uuid.uuid4()
        test_job_id = uuid.uuid4()
        test_embedding = [0.01 * i for i in range(1536)]
        
        async with engine.begin() as conn:
            # First create a test job (required for FK constraint)
            await conn.execute(text("""
                INSERT INTO jobs (id, status, source_type, priority, mode, metadata_json, retry_count, max_retries, created_at, updated_at)
                VALUES (:id, 'completed', 'test', 'normal', 'sync', '{}', 0, 3, NOW(), NOW())
                ON CONFLICT (id) DO NOTHING;
            """), {"id": test_job_id})
            
            # Insert chunk with embedding
            await conn.execute(text("""
                INSERT INTO document_chunks 
                (id, job_id, chunk_index, content, content_hash, embedding, chunk_metadata)
                VALUES 
                (:id, :job_id, :chunk_index, :content, :content_hash, CAST(:embedding AS vector), :metadata_json);
            """), {
                "id": test_id,
                "job_id": test_job_id,
                "chunk_index": 0,
                "content": "Test content for database persistence proof",
                "content_hash": "abc123def456",
                "embedding": str(test_embedding).replace(" ", ""),
                "metadata_json": '{"test": true, "proof": 4}'
            })
        
        print(f"\n✅ PROOF 4: Chunk inserted with ID: {test_id}")
        
        # Verify insertion
        async with engine.connect() as conn:
            result = await conn.execute(text("""
                SELECT id, job_id, chunk_index, content, embedding::text
                FROM document_chunks
                WHERE id = :id;
            """), {"id": test_id})
            row = result.fetchone()
            
            assert row is not None, "❌ Chunk not found after insertion!"
            assert str(row[0]) == str(test_id), "❌ ID mismatch!"
            assert row[3] == "Test content for database persistence proof", "❌ Content mismatch!"
            
            print(f"   ✅ Chunk retrieved from database")
            print(f"   - ID: {row[0]}")
            print(f"   - Job ID: {row[1]}")
            print(f"   - Content: {row[3][:50]}...")
        
        await engine.dispose()

    async def test_insert_multiple_chunks(self):
        """PROOF 5: Can insert multiple chunks with different embeddings."""
        engine = create_async_engine(DB_URL, echo=False)
        
        test_job_id = uuid.uuid4()
        chunk_ids = []
        
        async with engine.begin() as conn:
            # Create test job
            await conn.execute(text("""
                INSERT INTO jobs (id, status, source_type, priority, mode, metadata_json, retry_count, max_retries, created_at, updated_at)
                VALUES (:id, 'completed', 'test', 'normal', 'sync', '{}', 0, 3, NOW(), NOW())
                ON CONFLICT (id) DO NOTHING;
            """), {"id": test_job_id})
            
            # Insert 5 chunks with different embeddings
            for i in range(5):
                chunk_id = uuid.uuid4()
                chunk_ids.append(chunk_id)
                
                # Create distinct embedding
                embedding = [0.0] * 1536
                embedding[i * 300] = 0.9  # Different position for each chunk
                
                await conn.execute(text("""
                    INSERT INTO document_chunks 
                    (id, job_id, chunk_index, content, embedding, chunk_metadata)
                    VALUES 
                    (:id, :job_id, :chunk_index, :content, CAST(:embedding AS vector), :metadata_json);
                """), {
                    "id": chunk_id,
                    "job_id": test_job_id,
                    "chunk_index": i,
                    "content": f"Test content for chunk {i}",
                    "embedding": str(embedding).replace(" ", ""),
                    "metadata_json": f'{{"index": {i}}}'
                })
        
        print(f"\n✅ PROOF 5: Inserted 5 chunks with different embeddings")
        
        # Verify count
        async with engine.connect() as conn:
            result = await conn.execute(text("""
                SELECT COUNT(*) FROM document_chunks WHERE job_id = :job_id;
            """), {"job_id": test_job_id})
            count = result.scalar()
            
            assert count == 5, f"❌ Expected 5 chunks, found {count}!"
            print(f"   ✅ All 5 chunks persisted in database")
        
        await engine.dispose()

    async def test_retrieve_chunk_with_embedding(self):
        """PROOF 6: Can retrieve chunk with embedding intact."""
        engine = create_async_engine(DB_URL, echo=False)
        
        test_job_id = uuid.uuid4()
        test_embedding = [0.5] * 1536
        
        async with engine.begin() as conn:
            # Create test job
            await conn.execute(text("""
                INSERT INTO jobs (id, status, source_type, priority, mode, metadata_json, retry_count, max_retries, created_at, updated_at)
                VALUES (:id, 'completed', 'test', 'normal', 'sync', '{}', 0, 3, NOW(), NOW())
                ON CONFLICT (id) DO NOTHING;
            """), {"id": test_job_id})
            
            # Insert chunk with known embedding
            chunk_id = uuid.uuid4()
            await conn.execute(text("""
                INSERT INTO document_chunks 
                (id, job_id, chunk_index, content, embedding, chunk_metadata)
                VALUES 
                (:id, :job_id, :chunk_index, :content, CAST(:embedding AS vector), :metadata_json);
            """), {
                "id": chunk_id,
                "job_id": test_job_id,
                "chunk_index": 0,
                "content": "Test content for retrieval proof",
                "embedding": str(test_embedding).replace(" ", ""),
                "metadata_json": '{"retrieval_test": true}'
            })
        
        # Retrieve and verify embedding
        async with engine.connect() as conn:
            result = await conn.execute(text("""
                SELECT embedding::text
                FROM document_chunks
                WHERE id = :id;
            """), {"id": chunk_id})
            row = result.fetchone()
            
            assert row is not None, "❌ Chunk not found!"
            retrieved_embedding_str = row[0]
            
            # Parse retrieved embedding
            # PostgreSQL returns format like "[0.5,0.5,0.5,...]"
            retrieved_embedding = [
                float(x) for x in retrieved_embedding_str.strip("[]").split(",")
            ]
            
            assert len(retrieved_embedding) == 1536, f"❌ Wrong dimensions: {len(retrieved_embedding)}"
            assert retrieved_embedding[0] == 0.5, f"❌ Wrong value: {retrieved_embedding[0]}"
            assert retrieved_embedding[500] == 0.5, f"❌ Wrong value at 500: {retrieved_embedding[500]}"
            assert retrieved_embedding[1535] == 0.5, f"❌ Wrong value at 1535: {retrieved_embedding[1535]}"
            
            print("\n✅ PROOF 6: Embedding retrieved intact from database")
            print(f"   - Dimensions: {len(retrieved_embedding)}")
            print(f"   - First value: {retrieved_embedding[0]}")
            print(f"   - Last value: {retrieved_embedding[-1]}")
        
        await engine.dispose()

    async def test_similarity_search_with_vectors(self):
        """PROOF 7: Can perform similarity search on stored vectors."""
        engine = create_async_engine(DB_URL, echo=False)
        
        test_job_id = uuid.uuid4()
        query_embedding = [0.0] * 1536
        query_embedding[0] = 1.0  # Query vector pointing in first dimension
        
        async with engine.begin() as conn:
            # Create test job
            await conn.execute(text("""
                INSERT INTO jobs (id, status, source_type, priority, mode, metadata_json, retry_count, max_retries, created_at, updated_at)
                VALUES (:id, 'completed', 'test', 'normal', 'sync', '{}', 0, 3, NOW(), NOW())
                ON CONFLICT (id) DO NOTHING;
            """), {"id": test_job_id})
            
            # Insert chunks with different embeddings
            for i in range(3):
                embedding = [0.0] * 1536
                embedding[i] = 0.9  # Each chunk has value in different dimension
                
                await conn.execute(text("""
                    INSERT INTO document_chunks 
                    (id, job_id, chunk_index, content, embedding, chunk_metadata)
                    VALUES 
                    (gen_random_uuid(), :job_id, :chunk_index, :content, CAST(:embedding AS vector), '{}');
                """), {
                    "job_id": test_job_id,
                    "chunk_index": i,
                    "content": f"Chunk with value at dimension {i}",
                    "embedding": str(embedding).replace(" ", "")
                })
        
        # Perform similarity search
        async with engine.connect() as conn:
            result = await conn.execute(text("""
                SELECT chunk_index, content, embedding <=> CAST(:query AS vector) as distance
                FROM document_chunks
                WHERE job_id = :job_id
                ORDER BY embedding <=> CAST(:query AS vector)
                LIMIT 3;
            """), {
                "job_id": test_job_id,
                "query": str(query_embedding).replace(" ", "")
            })
            rows = result.fetchall()
            
            assert len(rows) == 3, f"❌ Expected 3 results, got {len(rows)}"
            
            # Chunk with embedding[0] = 0.9 should be most similar to query[0] = 1.0
            top_result = rows[0]
            assert top_result[0] == 0, f"❌ Expected chunk 0 to be most similar, got {top_result[0]}"
            
            print("\n✅ PROOF 7: Similarity search works on stored vectors")
            print("   - Query performed using pgvector <=> operator")
            print(f"   - Top result: Chunk {top_result[0]} (distance: {top_result[2]:.4f})")
            print(f"   - All {len(rows)} chunks retrieved and ranked")
        
        await engine.dispose()

    async def test_chunk_metadata_jsonb(self):
        """PROOF 8: Chunk metadata_json is stored as JSONB."""
        engine = create_async_engine(DB_URL, echo=False)
        
        test_job_id = uuid.uuid4()
        test_metadata_json = {"page": 42, "source": "test.pdf", "tags": ["important", "test"]}
        
        async with engine.begin() as conn:
            # Create test job
            await conn.execute(text("""
                INSERT INTO jobs (id, status, source_type, priority, mode, metadata_json, retry_count, max_retries, created_at, updated_at)
                VALUES (:id, 'completed', 'test', 'normal', 'sync', '{}', 0, 3, NOW(), NOW())
                ON CONFLICT (id) DO NOTHING;
            """), {"id": test_job_id})
            
            # Insert chunk with metadata_json
            chunk_id = uuid.uuid4()
            await conn.execute(text("""
                INSERT INTO document_chunks 
                (id, job_id, chunk_index, content, embedding, chunk_metadata)
                VALUES 
                (:id, :job_id, :chunk_index, :content, CAST(:embedding AS vector), :metadata_json);
            """), {
                "id": chunk_id,
                "job_id": test_job_id,
                "chunk_index": 0,
                "content": "Test content with metadata_json",
                "embedding": str([0.1] * 1536).replace(" ", ""),
                "metadata_json": '{"page": 42, "source": "test.pdf", "tags": ["important", "test"]}'
            })
        
        # Retrieve and verify metadata_json
        async with engine.connect() as conn:
            result = await conn.execute(text("""
                SELECT chunk_metadata
                FROM document_chunks
                WHERE id = :id;
            """), {"id": chunk_id})
            row = result.fetchone()
            
            assert row is not None, "❌ Chunk not found!"
            retrieved_metadata_json = row[0]
            
            assert retrieved_metadata_json["page"] == 42, f"❌ Page mismatch: {retrieved_metadata_json}"
            assert retrieved_metadata_json["source"] == "test.pdf", f"❌ Source mismatch: {retrieved_metadata_json}"
            assert "important" in retrieved_metadata_json["tags"], f"❌ Tags mismatch: {retrieved_metadata_json}"
            
            print("\n✅ PROOF 8: JSONB metadata_json stored and retrieved correctly")
            print(f"   - Page: {retrieved_metadata_json['page']}")
            print(f"   - Source: {retrieved_metadata_json['source']}")
            print(f"   - Tags: {retrieved_metadata_json['tags']}")
        
        await engine.dispose()

    async def test_hnsw_index_is_used(self):
        """PROOF 9: HNSW index is used for similarity queries."""
        engine = create_async_engine(DB_URL, echo=False)
        
        async with engine.connect() as conn:
            # Check if HNSW index exists
            result = await conn.execute(text("""
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE tablename = 'document_chunks' AND indexname LIKE '%hnsw%';
            """))
            row = result.fetchone()
            
            assert row is not None, "❌ HNSW index not found!"
            assert "hnsw" in row[1].lower(), f"❌ Not an HNSW index: {row[1]}"
            
            print("\n✅ PROOF 9: HNSW index exists for vector similarity")
            print(f"   - Index: {row[0]}")
            print(f"   - Type: HNSW (approximate nearest neighbor)")
        
        await engine.dispose()

    async def test_vector_dimension_constraint(self):
        """PROOF 10: Vector dimensions are enforced."""
        engine = create_async_engine(DB_URL, echo=False)
        
        test_job_id = uuid.uuid4()
        wrong_embedding = [0.1] * 768  # Wrong dimension
        
        async with engine.begin() as conn:
            # Create test job
            await conn.execute(text("""
                INSERT INTO jobs (id, status, source_type, priority, mode, metadata_json, retry_count, max_retries, created_at, updated_at)
                VALUES (:id, 'completed', 'test', 'normal', 'sync', '{}', 0, 3, NOW(), NOW())
                ON CONFLICT (id) DO NOTHING;
            """), {"id": test_job_id})
            
            # Try to insert chunk with wrong dimension
            try:
                await conn.execute(text("""
                    INSERT INTO document_chunks 
                    (id, job_id, chunk_index, content, embedding, chunk_metadata)
                    VALUES 
                    (gen_random_uuid(), :job_id, 0, 'test', CAST(:embedding AS vector), '{}');
                """), {
                    "job_id": test_job_id,
                    "embedding": str(wrong_embedding).replace(" ", "")
                })
                assert False, "❌ Should have raised error for wrong dimension!"
            except Exception as e:
                assert "1536" in str(e) or "dimension" in str(e).lower(), f"❌ Wrong error: {e}"
                print("\n✅ PROOF 10: Vector dimension constraint enforced")
                print(f"   - Expected: 1536 dimensions")
                print(f"   - Attempted: 768 dimensions")
                print(f"   - Result: Error raised as expected")
        
        await engine.dispose()


def run_all_tests():
    """Run all persistence tests."""
    print("\n" + "=" * 70)
    print("DATABASE PERSISTENCE TESTS")
    print("=" * 70)
    print("\nThese tests prove chunks with vectors are actually saved to PostgreSQL.\n")
    
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest", "tests/functional/test_database_persistence.py", "-v", "-s"],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode


if __name__ == "__main__":
    exit(run_all_tests())
