#!/bin/bash
# Run migrations on VPS database
# Usage: ./scripts/run_vps_migrations.sh [VPS_DB_URL]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get database URL
if [ -n "$1" ]; then
    VPS_DB_URL="$1"
elif [ -n "$DB_URL" ]; then
    VPS_DB_URL="$DB_URL"
else
    echo -e "${RED}Error: Database URL not provided${NC}"
    echo "Usage: $0 [postgresql+asyncpg://user:pass@host:port/db]"
    echo "Or set DB_URL environment variable"
    exit 1
fi

echo -e "${YELLOW}=======================================${NC}"
echo -e "${YELLOW}VPS Database Migration Runner${NC}"
echo -e "${YELLOW}=======================================${NC}"
echo ""

# Verify database connection
echo "Step 1: Verifying database connection..."
if ! python3 -c "
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

async def test():
    engine = create_async_engine('$VPS_DB_URL')
    async with engine.connect() as conn:
        result = await conn.execute(text('SELECT 1'))
        assert result.scalar() == 1
    await engine.dispose()
    print('Connection successful')

asyncio.run(test())
" 2>/dev/null; then
    echo -e "${RED}Error: Cannot connect to database${NC}"
    echo "Please check the database URL and ensure the database is accessible"
    exit 1
fi

echo -e "${GREEN}✓ Database connection successful${NC}"
echo ""

# Check current migration version
echo "Step 2: Checking current migration version..."
python3 -c "
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

async def check():
    engine = create_async_engine('$VPS_DB_URL')
    async with engine.connect() as conn:
        # Check if alembic_version table exists
        result = await conn.execute(text(\"\"\"
            SELECT 1 FROM information_schema.tables 
            WHERE table_name = 'alembic_version'
        \"\"\"))
        
        if result.scalar():
            result = await conn.execute(text('SELECT version_num FROM alembic_version'))
            version = result.scalar()
            print(f'Current migration version: {version}')
        else:
            print('No migrations applied yet (alembic_version table not found)')
    await engine.dispose()

asyncio.run(check())
" 2>/dev/null || echo "Could not determine current version"
echo ""

# Run migrations
echo "Step 3: Running migrations..."
echo "----------------------------------------"

# Set the database URL for alembic
export DB_URL="$VPS_DB_URL"

# Run alembic upgrade
if alembic upgrade head; then
    echo ""
    echo -e "${GREEN}✓ Migrations completed successfully${NC}"
else
    echo ""
    echo -e "${RED}✗ Migration failed${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}=======================================${NC}"
echo -e "${GREEN}Migration process completed!${NC}"
echo -e "${YELLOW}=======================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Verify the database structure: python3 scripts/verify_vps_database.py"
echo "  2. Check application logs for any errors"
echo "  3. Test the API endpoints"
