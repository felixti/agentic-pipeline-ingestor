#!/bin/bash
# Apply database fixes to VPS PostgreSQL
# Usage: ./scripts/apply_vps_fix.sh

set -e

echo "============================================"
echo "VPS Database Fix Application"
echo "============================================"
echo ""

# VPS connection details (stored in memory)
VPS_IP="72.61.52.191"
VPS_USER="root"
SSH_KEY=".ssh/ralph_loop_key"

echo "Connecting to VPS at $VPS_IP..."
echo ""

# Copy the SQL script to VPS
echo "Step 1: Copying SQL fix script to VPS..."
scp -i "$SSH_KEY" -o StrictHostKeyChecking=no scripts/vps_database_fix.sql "${VPS_USER}@${VPS_IP}:/tmp/vps_database_fix.sql"
echo "✓ Script copied"
echo ""

# Apply the fix
echo "Step 2: Applying database fixes..."
echo "This may take a moment..."
echo ""

ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "${VPS_USER}@${VPS_IP}" 'docker exec -i pipeline-postgres psql -U postgres -d pipeline < /tmp/vps_database_fix.sql' 2>&1 | grep -v "WARNING: connection is not using a post-quantum"

echo ""
echo "============================================"
echo "Fix application completed!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Verify the database: ./scripts/verify_vps_database.sh"
echo "  2. Restart the API if needed"
echo "  3. Test the search endpoints"
echo ""

# Cleanup
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "${VPS_USER}@${VPS_IP}" 'rm /tmp/vps_database_fix.sql' 2>/dev/null || true
