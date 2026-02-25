#!/bin/bash
# Verify VPS database structure
# Usage: ./scripts/verify_vps.sh

VPS_IP="72.61.52.191"
VPS_USER="root"
SSH_KEY=".ssh/ralph_loop_key"

echo "============================================"
echo "VPS Database Verification"
echo "============================================"
echo ""

# Check extensions
echo "1. PostgreSQL Extensions:"
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "${VPS_USER}@${VPS_IP}" 'docker exec pipeline-postgres psql -U postgres -d pipeline -c "SELECT extname, extversion FROM pg_extension WHERE extname IN ('"'"'vector'"'"', '"'"'pg_trgm'"'"');"' 2>&1 | grep -v WARNING
echo ""

# Check tables
echo "2. Tables:"
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "${VPS_USER}@${VPS_IP}" 'docker exec pipeline-postgres psql -U postgres -d pipeline -c "SELECT tablename FROM pg_tables WHERE schemaname = '"'"'public'"'"' ORDER BY tablename;"' 2>&1 | grep -v WARNING
echo ""

# Check document_chunks indexes
echo "3. Document Chunks Indexes:"
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "${VPS_USER}@${VPS_IP}" 'docker exec pipeline-postgres psql -U postgres -d pipeline -c "SELECT indexname FROM pg_indexes WHERE tablename = '"'"'document_chunks'"'"' ORDER BY indexname;"' 2>&1 | grep -v WARNING
echo ""

# Check data counts
echo "4. Data Summary:"
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "${VPS_USER}@${VPS_IP}" 'docker exec pipeline-postgres psql -U postgres -d pipeline -c "
SELECT '"'"'Chunks'"'"' as item, COUNT(*) as count FROM document_chunks
UNION ALL
SELECT '"'"'Jobs'"'"', COUNT(*) FROM jobs
UNION ALL
SELECT '"'"'Job Results'"'"', COUNT(*) FROM job_results
UNION ALL
SELECT '"'"'Pipelines'"'"', COUNT(*) FROM pipelines;
"' 2>&1 | grep -v WARNING
echo ""

# Check alembic version
echo "5. Migration Version:"
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "${VPS_USER}@${VPS_IP}" 'docker exec pipeline-postgres psql -U postgres -d pipeline -c "SELECT version_num FROM alembic_version;"' 2>&1 | grep -v WARNING || echo "   alembic_version table not found"
echo ""

echo "============================================"
echo "Verification Complete"
echo "============================================"
