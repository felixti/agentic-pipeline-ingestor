#!/usr/bin/env python3
"""Verify and report VPS database structure status.

This script checks if the VPS database has all required:
- Extensions (pgvector, pg_trgm)
- Tables (jobs, document_chunks, job_results, etc.)
- Indexes (HNSW, GIN, B-tree)
- Constraints (foreign keys, unique constraints)

Usage:
    python scripts/verify_vps_database.py [--fix]

Options:
    --fix  Generate SQL commands to fix missing structures
"""

import argparse
import asyncio
import sys
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


@dataclass
class CheckResult:
    """Result of a database check."""
    name: str
    exists: bool
    details: str = ""
    fix_sql: str = ""


class DatabaseVerifier:
    """Verifies database structure for the pipeline application."""

    REQUIRED_EXTENSIONS = ["vector", "pg_trgm"]
    
    REQUIRED_TABLES = [
        "jobs",
        "document_chunks",
        "job_results",
        "pipelines",
        "content_detection_results",
        "job_detection_results",
        "audit_logs",
        "api_keys",
        "webhook_subscriptions",
        "webhook_deliveries",
    ]
    
    REQUIRED_INDEXES = [
        # Document chunks indexes
        "idx_document_chunks_job_id",
        "idx_document_chunks_content_hash",
        "idx_document_chunks_job_chunk",
        "idx_document_chunks_embedding_hnsw",
        "idx_document_chunks_content_tsvector",
        "idx_document_chunks_content_trgm",
        # Jobs indexes
        "idx_jobs_status",
        "idx_jobs_source_type",
        "idx_jobs_external_id",
        "idx_jobs_locked_by",
        # Job results indexes
        "idx_job_results_job_id",
        "idx_job_results_expires",
        "idx_job_results_created",
    ]
    
    REQUIRED_CONSTRAINTS = [
        # Document chunks
        ("document_chunks", "uq_document_chunks_job_chunk"),
        ("document_chunks", "fk_document_chunks_job_id"),
        # Job results
        ("job_results", "uq_job_results_job_id"),
        ("job_results", "fk_job_results_job_id"),
    ]

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = create_async_engine(database_url, echo=False)
        self.results: list[CheckResult] = []

    async def verify_all(self) -> list[CheckResult]:
        """Run all verifications."""
        await self._check_extensions()
        await self._check_tables()
        await self._check_indexes()
        await self._check_constraints()
        await self._check_columns()
        return self.results

    async def _check_extensions(self) -> None:
        """Check if required PostgreSQL extensions are installed."""
        async with self.engine.connect() as conn:
            for ext in self.REQUIRED_EXTENSIONS:
                result = await conn.execute(
                    text("SELECT 1 FROM pg_extension WHERE extname = :ext"),
                    {"ext": ext}
                )
                exists = result.scalar() is not None
                
                fix_sql = f"CREATE EXTENSION IF NOT EXISTS {ext};" if not exists else ""
                
                self.results.append(CheckResult(
                    name=f"Extension: {ext}",
                    exists=exists,
                    details="Required for vector search and fuzzy matching" if ext == "vector" else "Required for fuzzy text search",
                    fix_sql=fix_sql
                ))

    async def _check_tables(self) -> None:
        """Check if required tables exist."""
        async with self.engine.connect() as conn:
            for table in self.REQUIRED_TABLES:
                result = await conn.execute(
                    text("""
                        SELECT 1 FROM information_schema.tables 
                        WHERE table_schema = 'public' AND table_name = :table
                    """),
                    {"table": table}
                )
                exists = result.scalar() is not None
                
                self.results.append(CheckResult(
                    name=f"Table: {table}",
                    exists=exists,
                    details="Core table" if table in ["jobs", "document_chunks"] else "Supporting table",
                    fix_sql="-- Run: alembic upgrade head" if not exists else ""
                ))

    async def _check_indexes(self) -> None:
        """Check if required indexes exist."""
        async with self.engine.connect() as conn:
            for index in self.REQUIRED_INDEXES:
                result = await conn.execute(
                    text("""
                        SELECT 1 FROM pg_indexes 
                        WHERE schemaname = 'public' AND indexname = :index
                    """),
                    {"index": index}
                )
                exists = result.scalar() is not None
                
                index_type = "HNSW (vector)" if "hnsw" in index else "GIN (text)" if "tsvector" in index or "trgm" in index else "B-tree"
                
                self.results.append(CheckResult(
                    name=f"Index: {index}",
                    exists=exists,
                    details=f"Type: {index_type}",
                    fix_sql="-- Run: alembic upgrade head" if not exists else ""
                ))

    async def _check_constraints(self) -> None:
        """Check if required constraints exist."""
        async with self.engine.connect() as conn:
            for table, constraint in self.REQUIRED_CONSTRAINTS:
                result = await conn.execute(
                    text("""
                        SELECT 1 FROM information_schema.table_constraints 
                        WHERE constraint_schema = 'public' 
                        AND table_name = :table 
                        AND constraint_name = :constraint
                    """),
                    {"table": table, "constraint": constraint}
                )
                exists = result.scalar() is not None
                
                constraint_type = "Unique" if "uq_" in constraint else "Foreign Key"
                
                self.results.append(CheckResult(
                    name=f"Constraint: {constraint} on {table}",
                    exists=exists,
                    details=f"Type: {constraint_type}",
                    fix_sql="-- Run: alembic upgrade head" if not exists else ""
                ))

    async def _check_columns(self) -> None:
        """Check if critical columns exist in tables."""
        async with self.engine.connect() as conn:
            # Check jobs table columns
            critical_columns = [
                ("jobs", "file_name", "VARCHAR(255)"),
                ("jobs", "source_uri", "VARCHAR(500)"),
                ("jobs", "source_type", "VARCHAR(50)"),
                ("document_chunks", "chunk_metadata", "JSONB"),
                ("document_chunks", "embedding", "VECTOR(1536)"),
            ]
            
            for table, column, expected_type in critical_columns:
                result = await conn.execute(
                    text("""
                        SELECT data_type, udt_name 
                        FROM information_schema.columns 
                        WHERE table_schema = 'public' 
                        AND table_name = :table 
                        AND column_name = :column
                    """),
                    {"table": table, "column": column}
                )
                row = result.fetchone()
                exists = row is not None
                
                actual_type = f"{row[0]}/{row[1]}" if row else "NOT FOUND"
                
                self.results.append(CheckResult(
                    name=f"Column: {table}.{column}",
                    exists=exists,
                    details=f"Expected: {expected_type}, Found: {actual_type}",
                    fix_sql="-- Run: alembic upgrade head" if not exists else ""
                ))

    def print_report(self) -> int:
        """Print verification report and return exit code."""
        print("=" * 80)
        print("VPS DATABASE STRUCTURE VERIFICATION REPORT")
        print("=" * 80)
        print()
        
        passed = sum(1 for r in self.results if r.exists)
        failed = sum(1 for r in self.results if not r.exists)
        
        # Group by category
        categories: dict[str, list[CheckResult]] = {}
        for result in self.results:
            category = result.name.split(":")[0]
            if category not in categories:
                categories[category] = []
            categories[category].append(result)
        
        for category, items in sorted(categories.items()):
            print(f"\n{category.upper()}:")
            print("-" * 80)
            for item in items:
                status = "✅" if item.exists else "❌"
                print(f"  {status} {item.name}")
                if item.details:
                    print(f"     Details: {item.details}")
        
        print()
        print("=" * 80)
        print(f"SUMMARY: {passed} passed, {failed} failed")
        print("=" * 80)
        
        return 0 if failed == 0 else 1
    
    def print_fix_sql(self) -> None:
        """Print SQL commands to fix missing structures."""
        print("\n" + "=" * 80)
        print("SQL COMMANDS TO FIX MISSING STRUCTURES")
        print("=" * 80)
        print()
        
        missing = [r for r in self.results if not r.exists and r.fix_sql]
        
        if not missing:
            print("-- All structures are in place! No fixes needed.")
            return
        
        print("-- Run these commands to fix missing structures:\n")
        
        # Group by type
        extensions = [r for r in missing if r.name.startswith("Extension:")]
        others = [r for r in missing if not r.name.startswith("Extension:")]
        
        if extensions:
            print("-- Install missing extensions:")
            for item in extensions:
                print(item.fix_sql)
            print()
        
        if others:
            print("-- For tables, indexes, and constraints, run migrations:")
            print("alembic upgrade head")
            print()
            print("-- Or manually create missing structures:")
            for item in others:
                print(f"-- {item.name}:")
                print(item.fix_sql or "-- See migration files")
                print()

    async def close(self) -> None:
        """Close database connection."""
        await self.engine.dispose()


def get_database_url() -> str:
    """Get database URL from environment or configuration."""
    import os
    
    # Try environment variable first
    db_url = os.getenv("DB_URL") or os.getenv("DATABASE_URL")
    
    if db_url:
        # Convert asyncpg to psycopg for sync operations if needed
        return db_url
    
    # Default for local development
    return "postgresql+asyncpg://postgres:postgres@localhost:5432/pipeline"


async def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify VPS database structure for Agentic Pipeline"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Generate SQL commands to fix missing structures"
    )
    parser.add_argument(
        "--db-url",
        help="Database URL (default: from DB_URL environment variable)"
    )
    
    args = parser.parse_args()
    
    db_url = args.db_url or get_database_url()
    
    print("Connecting to database...")
    
    verifier = DatabaseVerifier(db_url)
    
    try:
        await verifier.verify_all()
        exit_code = verifier.print_report()
        
        if args.fix:
            verifier.print_fix_sql()
        
        return exit_code
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        await verifier.close()


if __name__ == "__main__":
    asyncio.run(main())
