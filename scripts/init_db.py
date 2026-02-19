#!/usr/bin/env python3
"""Initialize database tables for development."""

import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy.ext.asyncio import create_async_engine

from src.db.models import Base

# Database URL from environment or default
DB_URL = os.getenv(
    "DB_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/pipeline"
)


async def init_database():
    """Create all tables."""
    print(f"Connecting to database: {DB_URL.replace('postgres:postgres', '***:***')}")
    engine = create_async_engine(DB_URL, echo=False)
    
    try:
        async with engine.begin() as conn:
            print("Creating tables...")
            await conn.run_sync(Base.metadata.create_all)
            print("✓ Tables created successfully!")
        
        # Verify connection
        from sqlalchemy import text
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT version()"))
            version = result.scalar()
            print("\n✓ Database connection verified!")
            print(f"  PostgreSQL version: {version}")
            
            # List tables
            result = await conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """))
            tables = result.fetchall()
            print(f"\n  Tables created ({len(tables)}):")
            for table in tables:
                print(f"    - {table[0]}")
                
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(init_database())
