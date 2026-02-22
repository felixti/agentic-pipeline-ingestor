# Agentic Data Pipeline Ingestor - Production Docker Image
# Build simplificado com dependências diretas

FROM python:3.11-slim

# Set environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOME=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    libpq5 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR $APP_HOME

# Install Python dependencies diretamente (mais rápido que pip install -e .)
RUN pip install --upgrade pip && \
    pip install \
    fastapi uvicorn pydantic pydantic-settings email-validator \
    sqlalchemy alembic asyncpg psycopg2-binary \
    redis hiredis \
    litellm openai httpx aiohttp \
    pymupdf pillow pdf2image pytesseract python-magic \
    opentelemetry-api opentelemetry-sdk opentelemetry-instrumentation-fastapi \
    opentelemetry-instrumentation-sqlalchemy opentelemetry-exporter-otlp \
    prometheus-client structlog python-jose passlib python-dotenv cryptography \
    orjson pyyaml click tenacity typing-extensions pdfplumber psutil greenlet \
    numpy \
    pytest pytest-asyncio

# Copy application code
COPY src/ ./src/
COPY api/ ./api/
COPY config/ ./config/
COPY pyproject.toml ./

# Create non-root user
RUN groupadd -r pipeline && useradd -r -g pipeline pipeline && \
    mkdir -p /tmp/pipeline /var/log/pipeline && \
    chown -R pipeline:pipeline /tmp/pipeline /var/log/pipeline $APP_HOME

# Switch to non-root user
USER pipeline

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries-3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

# Run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
