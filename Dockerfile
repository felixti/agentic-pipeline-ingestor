# Agentic Data Pipeline Ingestor - Production Docker Image
# Uses pre-built base image for faster builds

# ============================================================================
# Stage 1: Dependencies (using base image)
# ============================================================================
FROM agentic-pipeline-base:latest AS builder

# Set build environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH"

# Copy pyproject.toml to check for new dependencies
COPY pyproject.toml /app/
WORKDIR /app

# Install only the application-specific dependencies (lightweight)
# Heavy deps (torch, cognee, hipporag, docling) are already in base image
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
    orjson pyyaml click tenacity typing-extensions psutil greenlet \
    gunicorn uvloop

# ============================================================================
# Stage 2: Runtime
# ============================================================================
FROM agentic-pipeline-base:latest AS runtime

# Set runtime environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/app \
    PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN groupadd -r pipeline && useradd -r -g pipeline pipeline

# Copy virtual environment with app-specific deps from builder
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR $APP_HOME

# Copy application code (this layer changes most frequently)
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
