# Agentic Data Pipeline Ingestor - Production Docker Image
# Multi-stage build for optimized image size

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM python:3.11-slim AS builder

# Set build environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY pyproject.toml /app/
WORKDIR /app

# Install dependencies from pyproject.toml
RUN pip install --upgrade pip && \
    pip install \
    fastapi uvicorn python-multipart starlette pydantic pydantic-settings email-validator \
    sqlalchemy alembic asyncpg psycopg2-binary redis hiredis \
    litellm openai httpx aiohttp \
    prometheus-client structlog python-jose passlib python-dotenv cryptography \
    opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp \
    opentelemetry-instrumentation-fastapi opentelemetry-instrumentation-sqlalchemy \
    pymupdf pillow pdf2image pytesseract python-magic \
    python-docx openpyxl python-pptx pdfplumber pytest pytest-asyncio \
    docling azure-ai-vision-imageanalysis \
    orjson pyyaml click tenacity typing-extensions psutil greenlet \
    numpy scipy scikit-learn sentence-transformers \
    neo4j>=5.15.0 "cognee[postgres,neo4j]" \
    hipporag>=0.1.0

# ============================================================================
# Stage 2: Runtime
# ============================================================================
FROM python:3.11-slim AS runtime

# Set runtime environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/app \
    PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r pipeline && useradd -r -g pipeline pipeline

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR $APP_HOME

# Copy application code
COPY --chown=pipeline:pipeline src/ ./src/
COPY --chown=pipeline:pipeline api/ ./api/
COPY --chown=pipeline:pipeline config/ ./config/
COPY --chown=pipeline:pipeline pyproject.toml ./

# Create necessary directories
RUN mkdir -p /tmp/pipeline /var/log/pipeline /data/cognee/data /data/cognee/system /data/hipporag && \
    chown -R pipeline:pipeline /tmp/pipeline /var/log/pipeline /data $APP_HOME

# Switch to non-root user
USER pipeline

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

# Run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
