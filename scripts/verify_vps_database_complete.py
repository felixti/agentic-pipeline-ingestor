#!/usr/bin/env python3
"""
Complete VPS Database Verification Script

Verifies that the VPS PostgreSQL database is synchronized with all migrations,
including the Cognee pgvector schema (migration 015).

Usage:
    python3 scripts/verify_vps_database_complete.py
    python3 scripts/verify_vps_database_complete.py --db-url "postgresql+asyncpg://user:pass@host:5432/db"
"""

import argparse
import asyncio
import sys

import asyncpg


# Expected tables from all migrations 000-015
EXPECTED_TABLES = {
    # Core tables (000, 003)
    "jobs",
    "pipelines",
    "document_chunks",
    # Content detection (001)
    "content_detection_results",
    "job_detection_results",
    # Contextual retrieval (004)
    "document_hierarchy",
    # Cache tables (005)
    "embedding_cache",
    "query_cache",
    "llm_response_cache",
    # Job results (006)
    "job_results",
    # Audit and webhooks (008)
    "audit_logs",
    "api_keys",
    "webhook_subscriptions",
    "webhook_deliveries",
    # Search analytics (012)
    "search_analytics",
    "search_queries",
    # Cognee pgvector schema (015)
    "cognee_vectors",
    "cognee_documents",
    "cognee_entities",
}

# Expected extensions
EXPECTED_EXTENSIONS = {
    "vector",  # pgvector
    "pg_trgm",  # fuzzy search
    "uuid-ossp",  # UUID generation
}

# Critical indexes for performance
EXPECTED_INDEXES = {
    # Document chunks indexes
    "idx_document_chunks_embedding_hnsw",
    "idx_document_chunks_content_tsvector",
    "idx_document_chunks_content_trgm",
    "idx_document_chunks_job_id",
    # Jobs indexes
    "idx_jobs_status",
    # Cognee indexes
    "idx_cognee_vectors_embedding",
    "idx_cognee_vectors_chunk_id",
    "idx_cognee_vectors_document_id",
    "idx_cognee_vectors_dataset_id",
    "idx_cognee_documents_document_id",
    "idx_cognee_documents_dataset_id",
    "idx_cognee_entities_name",
    "idx_cognee_entities_type",
    "idx_cognee_entities_document_id",
}

# Expected columns in critical tables
EXPECTED_COLUMNS = {
    "jobs": {
        "id",
        "file_name",
        "source_uri",
        "status",
        "source_type",
        "mime_type",
        "created_at",
        "updated_at",
    },
    "document_chunks": {
        "id",
        "job_id",
        "content",
        "embedding",
        "metadata",
        "created_at",
    },
    "cognee_vectors": {
        "id",
        "chunk_id",
        "document_id",
        "dataset_id",
        "embedding",
        "metadata",
        "created_at",
        "updated_at",
    },
    "cognee_documents": {
        "id",
        "document_id",
        "dataset_id",
        "title",
        "source_path",
        "chunk_count",
        "entity_count",
        "created_at",
        "updated_at",
    },
    "cognee_entities": {
        "id",
        "entity_id",
        "name",
        "type",
        "description",
        "document_id",
        "dataset_id",
        "metadata",
        "created_at",
    },
}


async def verify_database(db_url: str) -> dict:
    """Verify database structure against expected schema."""
    results = {
        "connected": False,
        "extensions": {},
        "tables": {},
        "indexes": {},
        "columns": {},
        "alembic_version": None,
        "errors": [],
    }

    try:
        conn = await asyncpg.connect(db_url)
        results["connected"] = True
    except Exception as e:
        results["errors"].append(f"Failed to connect: {e}")
        return results

    try:
        # Check extensions
        ext_rows = await conn.fetch(
            "SELECT extname FROM pg_extension WHERE extname = ANY($1)",
            list(EXPECTED_EXTENSIONS),
        )
        found_extensions = {row["extname"] for row in ext_rows}
        for ext in EXPECTED_EXTENSIONS:
            results["extensions"][ext] = ext in found_extensions

        # Check tables
        table_rows = await conn.fetch(
            """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = ANY($1)
            """,
            list(EXPECTED_TABLES),
        )
        found_tables = {row["table_name"] for row in table_rows}
        for table in EXPECTED_TABLES:
            results["tables"][table] = table in found_tables

        # Check indexes
        index_rows = await conn.fetch(
            """
            SELECT indexname 
            FROM pg_indexes 
            WHERE schemaname = 'public' 
            AND indexname = ANY($1)
            """,
            list(EXPECTED_INDEXES),
        )
        found_indexes = {row["indexname"] for row in index_rows}
        for idx in EXPECTED_INDEXES:
            results["indexes"][idx] = idx in found_indexes

        # Check columns for critical tables
        for table, expected_cols in EXPECTED_COLUMNS.items():
            if table not in found_tables:
                results["columns"][table] = {
                    "exists": False,
                    "columns": {col: False for col in expected_cols},
                }
                continue

            col_rows = await conn.fetch(
                """
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_schema = 'public' 
                AND table_name = $1
                """,
                table,
            )
            found_cols = {row["column_name"] for row in col_rows}
            results["columns"][table] = {
                "exists": True,
                "columns": {col: col in found_cols for col in expected_cols},
            }

        # Check alembic version
        try:
            version_row = await conn.fetchrow("SELECT version_num FROM alembic_version")
            if version_row:
                results["alembic_version"] = version_row["version_num"]
        except asyncpg.UndefinedTableError:
            results["alembic_version"] = None
            results["errors"].append("alembic_version table not found - migrations not run")

    except Exception as e:
        results["errors"].append(f"Verification error: {e}")
    finally:
        await conn.close()

    return results


def print_results(results: dict) -> int:
    """Print verification results and return exit code."""
    exit_code = 0

    print("=" * 80)
    print("VPS DATABASE VERIFICATION REPORT")
    print("=" * 80)

    # Connection status
    if results["connected"]:
        print("\n✅ Database connection: SUCCESS")
    else:
        print("\n❌ Database connection: FAILED")
        for error in results["errors"]:
            print(f"   Error: {error}")
        return 1

    # Alembic version
    print(f"\n📦 Alembic Version: {results['alembic_version'] or 'Not set'}")
    if results["alembic_version"] != "015":
        print(f"   ⚠️  Expected: 015 (Cognee pgvector schema)")
        exit_code = 1
    else:
        print(f"   ✅ Up to date")

    # Extensions
    print("\n📦 PostgreSQL Extensions:")
    for ext, present in results["extensions"].items():
        status = "✅" if present else "❌"
        print(f"   {status} {ext}")
        if not present:
            exit_code = 1

    # Tables
    print("\n📋 Tables:")
    # Group tables by feature
    core_tables = {"jobs", "pipelines", "document_chunks"}
    cognee_tables = {"cognee_vectors", "cognee_documents", "cognee_entities"}
    other_tables = EXPECTED_TABLES - core_tables - cognee_tables

    print("   Core Tables:")
    for table in sorted(core_tables):
        present = results["tables"].get(table, False)
        status = "✅" if present else "❌"
        print(f"      {status} {table}")
        if not present:
            exit_code = 1

    print("   Cognee GraphRAG Tables:")
    for table in sorted(cognee_tables):
        present = results["tables"].get(table, False)
        status = "✅" if present else "❌"
        print(f"      {status} {table}")
        if not present:
            exit_code = 1

    print("   Other Tables:")
    for table in sorted(other_tables):
        present = results["tables"].get(table, False)
        status = "✅" if present else "❌"
        print(f"      {status} {table}")
        if not present:
            exit_code = 1

    # Critical indexes
    print("\n🔍 Critical Indexes:")
    chunk_indexes = {k for k in EXPECTED_INDEXES if k.startswith("idx_document_chunks")}
    cognee_idx = {k for k in EXPECTED_INDEXES if k.startswith("idx_cognee")}
    other_indexes = EXPECTED_INDEXES - chunk_indexes - cognee_idx

    print("   Document Chunk Indexes:")
    for idx in sorted(chunk_indexes):
        present = results["indexes"].get(idx, False)
        status = "✅" if present else "❌"
        print(f"      {status} {idx}")
        if not present:
            exit_code = 1

    print("   Cognee Indexes:")
    for idx in sorted(cognee_idx):
        present = results["indexes"].get(idx, False)
        status = "✅" if present else "❌"
        print(f"      {status} {idx}")
        if not present:
            exit_code = 1

    print("   Other Indexes:")
    for idx in sorted(other_indexes):
        present = results["indexes"].get(idx, False)
        status = "✅" if present else "❌"
        print(f"      {status} {idx}")

    # Column checks
    print("\n📊 Critical Column Check:")
    for table, info in results["columns"].items():
        if not info["exists"]:
            print(f"   ❌ {table}: TABLE NOT FOUND")
            exit_code = 1
            continue

        missing_cols = [col for col, present in info["columns"].items() if not present]
        if missing_cols:
            print(f"   ⚠️  {table}: Missing columns: {', '.join(missing_cols)}")
            exit_code = 1
        else:
            print(f"   ✅ {table}: All columns present")

    # Summary
    print("\n" + "=" * 80)
    if exit_code == 0:
        print("✅ ALL CHECKS PASSED - Database is up to date with migration 015")
        print("   Cognee pgvector schema is properly configured")
    else:
        print("❌ SOME CHECKS FAILED - Database needs updates")
        print("\nTo fix:")
        print("   1. Run: ./scripts/run_vps_migrations.sh")
        print("   2. Or: DB_URL=<your-db-url> alembic upgrade head")
    print("=" * 80)

    return exit_code


async def main():
    parser = argparse.ArgumentParser(
        description="Verify VPS database structure against migrations 000-015"
    )
    parser.add_argument(
        "--db-url",
        default="postgresql+asyncpg://postgres:SENHA@seu-vps:5432/pipeline",
        help="Database connection URL",
    )
    args = parser.parse_args()

    # Convert SQLAlchemy URL to asyncpg URL
    db_url = args.db_url.replace("postgresql+asyncpg://", "postgresql://")

    results = await verify_database(db_url)
    exit_code = print_results(results)
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
