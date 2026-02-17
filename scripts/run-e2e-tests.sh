#!/bin/bash
# Run E2E tests locally using Docker Compose
#
# Usage:
#   ./run-e2e-tests.sh [options]
#
# Options:
#   --full          Run full test suite including performance tests
#   --quick         Run only quick smoke tests
#   --performance   Run only performance tests
#   --auth          Run only authentication tests
#   --retry         Run only retry mechanism tests
#   --dlq           Run only DLQ tests
#   --keep          Keep containers running after tests
#   --logs          Show API logs during tests
#   --help          Show this help message

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${PROJECT_ROOT}/tests/e2e/docker/docker-compose.e2e.yml"

# Default options
TEST_MARKER="e2e and not performance"
KEEP_CONTAINERS=false
SHOW_LOGS=false
GENERATE_REPORTS=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --full)
            TEST_MARKER="e2e"
            shift
            ;;
        --quick)
            TEST_MARKER="e2e and not slow"
            shift
            ;;
        --performance)
            TEST_MARKER="e2e and performance"
            shift
            ;;
        --auth)
            TEST_MARKER="e2e and auth"
            shift
            ;;
        --retry)
            TEST_MARKER="e2e and retry"
            shift
            ;;
        --dlq)
            TEST_MARKER="e2e and dlq"
            shift
            ;;
        --keep)
            KEEP_CONTAINERS=true
            shift
            ;;
        --logs)
            SHOW_LOGS=true
            shift
            ;;
        --no-reports)
            GENERATE_REPORTS=false
            shift
            ;;
        --help)
            echo "Usage: ./run-e2e-tests.sh [options]"
            echo ""
            echo "Options:"
            echo "  --full          Run full test suite including performance tests"
            echo "  --quick         Run only quick smoke tests"
            echo "  --performance   Run only performance tests"
            echo "  --auth          Run only authentication tests"
            echo "  --retry         Run only retry mechanism tests"
            echo "  --dlq           Run only DLQ tests"
            echo "  --keep          Keep containers running after tests"
            echo "  --logs          Show API logs during tests"
            echo "  --no-reports    Don't generate HTML reports"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: docker-compose is not installed${NC}"
    exit 1
fi

# Change to project root
cd "${PROJECT_ROOT}"

echo -e "${BLUE}==============================================${NC}"
echo -e "${BLUE}  E2E Test Suite - Agentic Pipeline Ingestor${NC}"
echo -e "${BLUE}==============================================${NC}"
echo ""
echo -e "${YELLOW}Test filter:${NC} ${TEST_MARKER}"
echo -e "${YELLOW}Keep containers:${NC} ${KEEP_CONTAINERS}"
echo -e "${YELLOW}Show logs:${NC} ${SHOW_LOGS}"
echo ""

# Cleanup function
cleanup() {
    if [ "$KEEP_CONTAINERS" = false ]; then
        echo ""
        echo -e "${YELLOW}Cleaning up containers...${NC}"
        docker-compose -f "${COMPOSE_FILE}" down -v --remove-orphans
        echo -e "${GREEN}Cleanup complete${NC}"
    else
        echo ""
        echo -e "${YELLOW}Containers are still running. To stop them:${NC}"
        echo "  docker-compose -f ${COMPOSE_FILE} down"
    fi
}

# Set trap for cleanup
trap cleanup EXIT

# Start E2E stack
echo -e "${BLUE}Starting E2E stack...${NC}"
docker-compose -f "${COMPOSE_FILE}" up -d postgres-e2e redis-e2e api-e2e

# Wait for services to be ready
echo -e "${BLUE}Waiting for services to be ready...${NC}"
sleep 10

# Check API health
echo -e "${BLUE}Checking API health...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:8001/health/live > /dev/null 2>&1; then
        echo -e "${GREEN}API is ready!${NC}"
        break
    fi
    echo -n "."
    sleep 2
done

# Verify API is responding
if ! curl -s http://localhost:8001/health/live > /dev/null 2>&1; then
    echo -e "${RED}Error: API failed to start${NC}"
    echo ""
    echo "API logs:"
    docker-compose -f "${COMPOSE_FILE}" logs api-e2e --tail=50
    exit 1
fi

# Show logs if requested
if [ "$SHOW_LOGS" = true ]; then
    echo -e "${BLUE}Starting log tail...${NC}"
    docker-compose -f "${COMPOSE_FILE}" logs -f api-e2e &
    LOG_PID=$!
fi

# Generate test data if needed
echo -e "${BLUE}Generating test documents...${NC}"
if [ -f "${PROJECT_ROOT}/tests/e2e/fixtures/documents/generate_test_docs.py" ]; then
    cd "${PROJECT_ROOT}/tests/e2e/fixtures/documents"
    python generate_test_docs.py 2>/dev/null || echo -e "${YELLOW}Could not generate test documents (dependencies may be missing)${NC}"
    cd "${PROJECT_ROOT}"
fi

# Create reports directory
mkdir -p "${PROJECT_ROOT}/tests/e2e/reports"

# Build test arguments
PYTEST_ARGS="-v --tb=short"

if [ "$GENERATE_REPORTS" = true ]; then
    PYTEST_ARGS="${PYTEST_ARGS} --html=/tests/reports/report.html --self-contained-html --junitxml=/tests/reports/junit.xml"
fi

# Run tests
echo ""
echo -e "${BLUE}Running E2E tests...${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

TEST_EXIT_CODE=0

# Run tests in test-runner container
docker-compose -f "${COMPOSE_FILE}" run --rm \
    -e TEST_MARKER="${TEST_MARKER}" \
    -e PYTEST_ARGS="${PYTEST_ARGS}" \
    test-runner \
    pytest /tests/e2e/scenarios \
    -m "${TEST_MARKER}" \
    ${PYTEST_ARGS} \
    || TEST_EXIT_CODE=$?

# Kill log tail if running
if [ -n "$LOG_PID" ]; then
    kill $LOG_PID 2>/dev/null || true
fi

# Display results
echo ""
echo -e "${BLUE}========================================${NC}"

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
else
    echo -e "${RED}Some tests failed (exit code: $TEST_EXIT_CODE)${NC}"
fi

# Copy reports from container if generated
if [ "$GENERATE_REPORTS" = true ]; then
    echo ""
    echo -e "${BLUE}Test Reports:${NC}"
    echo "  HTML Report: ${PROJECT_ROOT}/tests/e2e/reports/report.html"
    echo "  JUnit XML:   ${PROJECT_ROOT}/tests/e2e/reports/junit.xml"
    
    # Open report if on macOS
    if [[ "$OSTYPE" == "darwin"* ]] && [ -f "${PROJECT_ROOT}/tests/e2e/reports/report.html" ]; then
        echo ""
        echo "Opening HTML report..."
        open "${PROJECT_ROOT}/tests/e2e/reports/report.html" 2>/dev/null || true
    fi
fi

echo ""
exit $TEST_EXIT_CODE
