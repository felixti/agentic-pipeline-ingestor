# Agentic Data Pipeline Ingestor - Makefile
# Automation for common development tasks

.PHONY: help install dev-install test lint format build up down logs clean migrate e2e sdk docs

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m # No Color

# Variables
COMPOSE_FILE := docker/docker-compose.yml
E2E_COMPOSE_FILE := tests/e2e/docker/docker-compose.e2e.yml
PROJECT_NAME := agentic-pipeline-ingestor

# =============================================================================
# Help
# =============================================================================

help: ## Show this help message
	@echo "$(BLUE)Agentic Data Pipeline Ingestor - Available Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# =============================================================================
# Installation & Setup
# =============================================================================

install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	uv pip install -e "."

dev-install: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	uv pip install -e ".[dev]"

install-all: ## Install all dependencies (dev, docling, azure, cognee)
	@echo "$(BLUE)Installing all dependencies...$(NC)"
	uv pip install -e ".[all]"

sync: ## Sync dependencies with uv.lock
	@echo "$(BLUE)Syncing dependencies...$(NC)"
	uv sync --extra dev --prerelease allow

# =============================================================================
# Docker Operations
# =============================================================================

build: ## Build Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	docker-compose -f $(COMPOSE_FILE) build

up: ## Start all services in detached mode
	@echo "$(GREEN)Starting services...$(NC)"
	docker-compose -f $(COMPOSE_FILE) up -d
	@echo "$(GREEN)Services started!$(NC)"
	@echo "  API: http://localhost:8000"
	@echo "  Docs: http://localhost:8000/docs"
	@echo "  Metrics: http://localhost:8000/metrics"

down: ## Stop all services
	@echo "$(RED)Stopping services...$(NC)"
	docker-compose -f $(COMPOSE_FILE) down

stop: ## Stop services without removing containers
	@echo "$(YELLOW)Stopping services...$(NC)"
	docker-compose -f $(COMPOSE_FILE) stop

restart: down up ## Restart all services

logs: ## Show logs from all services
	docker-compose -f $(COMPOSE_FILE) logs -f

logs-api: ## Show logs from API service only
	docker-compose -f $(COMPOSE_FILE) logs -f api

ps: ## Show running containers
	docker-compose -f $(COMPOSE_FILE) ps

# =============================================================================
# Testing
# =============================================================================

test: ## Run all tests
	@echo "$(BLUE)Running all tests...$(NC)"
	pytest

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	pytest -m unit

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	pytest -m integration

test-contract: ## Run contract tests only
	@echo "$(BLUE)Running contract tests...$(NC)"
	pytest tests/contract/

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	pytest --cov=src --cov-report=html --cov-report=term

test-coverage-open: test-coverage ## Run tests and open coverage report
	@echo "$(GREEN)Opening coverage report...$(NC)"
	python -m webbrowser htmlcov/index.html

# =============================================================================
# E2E Testing
# =============================================================================

e2e-up: ## Start E2E test environment
	@echo "$(BLUE)Starting E2E environment...$(NC)"
	docker-compose -f $(E2E_COMPOSE_FILE) up -d

e2e-down: ## Stop E2E test environment
	@echo "$(RED)Stopping E2E environment...$(NC)"
	docker-compose -f $(E2E_COMPOSE_FILE) down

e2e-test: ## Run E2E tests
	@echo "$(BLUE)Running E2E tests...$(NC)"
	./scripts/run-e2e-tests.sh

e2e-test-quick: ## Run quick E2E smoke tests
	@echo "$(BLUE)Running quick E2E tests...$(NC)"
	./scripts/run-e2e-tests.sh --quick

e2e-test-auth: ## Run E2E auth tests only
	@echo "$(BLUE)Running E2E auth tests...$(NC)"
	./scripts/run-e2e-tests.sh --auth

e2e-test-performance: ## Run E2E performance tests
	@echo "$(BLUE)Running E2E performance tests...$(NC)"
	./scripts/run-e2e-tests.sh --performance

e2e-logs: ## Show E2E test logs
	docker-compose -f $(E2E_COMPOSE_FILE) logs -f

# =============================================================================
# Code Quality
# =============================================================================

lint: ## Run linter (ruff)
	@echo "$(BLUE)Running linter...$(NC)"
	ruff check .

lint-fix: ## Run linter and fix issues
	@echo "$(BLUE)Running linter with auto-fix...$(NC)"
	ruff check . --fix

format: ## Format code with ruff
	@echo "$(BLUE)Formatting code...$(NC)"
	ruff format .

type-check: ## Run type checker (mypy)
	@echo "$(BLUE)Running type checker...$(NC)"
	mypy src/

type-check-strict: ## Run type checker in strict mode
	@echo "$(BLUE)Running strict type check...$(NC)"
	mypy src/ --strict

security: ## Run security checks
	@echo "$(BLUE)Running security checks...$(NC)"
	bandit -r src/
	safety check

quality: lint type-check security ## Run all quality checks

format-and-lint: format lint-fix ## Format code and run linter

# =============================================================================
# Database
# =============================================================================

migrate: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(NC)"
	alembic upgrade head

migrate-create: ## Create new migration (use MESSAGE="your message")
	@echo "$(BLUE)Creating migration: $(MESSAGE)$(NC)"
	alembic revision --autogenerate -m "$(MESSAGE)"

migrate-downgrade: ## Downgrade database by one revision
	@echo "$(YELLOW)Downgrading database...$(NC)"
	alembic downgrade -1

migrate-history: ## Show migration history
	alembic history --verbose

db-reset: ## Reset database (drop and recreate)
	@echo "$(RED)Resetting database...$(NC)"
	alembic downgrade base
	alembic upgrade head

# =============================================================================
# SDK Generation
# =============================================================================

sdk: ## Generate SDKs from OpenAPI spec
	@echo "$(BLUE)Generating SDKs...$(NC)"
	python sdks/generate.py

sdk-python: ## Generate Python SDK only
	@echo "$(BLUE)Generating Python SDK...$(NC)"
	python sdks/generate.py --language python

sdk-typescript: ## Generate TypeScript SDK only
	@echo "$(BLUE)Generating TypeScript SDK...$(NC)"
	python sdks/generate.py --language typescript

sdk-dry-run: ## Preview SDK generation commands
	@echo "$(BLUE)Previewing SDK generation...$(NC)"
	python sdks/generate.py --dry-run

# =============================================================================
# Documentation
# =============================================================================

docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	mkdocs build

docs-serve: ## Serve documentation locally
	@echo "$(GREEN)Starting documentation server...$(NC)"
	mkdocs serve

docs-deploy: ## Deploy documentation
	@echo "$(BLUE)Deploying documentation...$(NC)"
	mkdocs gh-deploy

# =============================================================================
# API Testing with HTTP files
# =============================================================================

api-health: ## Test health endpoints
	@echo "$(BLUE)Testing health endpoints...$(NC)"
	curl -s http://localhost:8000/health/live | jq .
	curl -s http://localhost:8000/health/ready | jq .

api-docs: ## Open API documentation in browser
	@echo "$(GREEN)Opening API documentation...$(NC)"
	python -m webbrowser http://localhost:8000/docs

api-spec: ## Download OpenAPI specification
	@echo "$(BLUE)Downloading OpenAPI spec...$(NC)"
	curl -s http://localhost:8000/api/v1/openapi.yaml -o api/openapi.yaml

# =============================================================================
# Cleanup
# =============================================================================

clean: ## Clean up build artifacts and cache
	@echo "$(YELLOW)Cleaning up...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/ 2>/dev/null || true
	@echo "$(GREEN)Cleanup complete!$(NC)"

clean-docker: ## Clean up Docker resources
	@echo "$(YELLOW)Cleaning up Docker resources...$(NC)"
	docker-compose -f $(COMPOSE_FILE) down -v --remove-orphans
	docker system prune -f
	@echo "$(GREEN)Docker cleanup complete!$(NC)"

clean-all: clean clean-docker ## Clean everything

# =============================================================================
# Development Utilities
# =============================================================================

run: ## Run the application locally (without Docker)
	@echo "$(GREEN)Starting application...$(NC)"
	uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

run-prod: ## Run the application in production mode
	@echo "$(GREEN)Starting application in production mode...$(NC)"
	uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4

shell: ## Open a shell in the API container
	docker-compose -f $(COMPOSE_FILE) exec api /bin/sh

db-shell: ## Open PostgreSQL shell
	docker-compose -f $(COMPOSE_FILE) exec postgres psql -U postgres -d pipeline

redis-cli: ## Open Redis CLI
	docker-compose -f $(COMPOSE_FILE) exec redis redis-cli

# =============================================================================
# CI/CD
# =============================================================================

ci: lint test ## Run CI checks (lint + test)

ci-full: install-all lint type-check test security ## Run full CI pipeline

# =============================================================================
# Release
# =============================================================================

version: ## Show current version
	@grep "^version" pyproject.toml | head -1

bump-patch: ## Bump patch version
	@echo "$(BLUE)Bumping patch version...$(NC)"
	bumpversion patch

bump-minor: ## Bump minor version
	@echo "$(BLUE)Bumping minor version...$(NC)"
	bumpversion minor

bump-major: ## Bump major version
	@echo "$(BLUE)Bumping major version...$(NC)"
	bumpversion major

# =============================================================================
# Information
# =============================================================================

status: ## Show project status
	@echo "$(BLUE)Project Status$(NC)"
	@echo "================"
	@echo "Git branch: $(shell git branch --show-current)"
	@echo "Git commit: $(shell git rev-parse --short HEAD)"
	@echo "Python version: $(shell python --version)"
	@echo ""
	@echo "$(BLUE)Docker Status$(NC)"
	@echo "---------------"
	@docker-compose -f $(COMPOSE_FILE) ps 2>/dev/null || echo "No containers running"

info: ## Show project information
	@echo "$(BLUE)Agentic Data Pipeline Ingestor$(NC)"
	@echo "================================"
	@echo ""
	@echo "Project: $(PROJECT_NAME)"
	@echo "Repository: https://github.com/felixti/agentic-pipeline-ingestor"
	@echo "API URL: http://localhost:8000"
	@echo "Docs URL: http://localhost:8000/docs"
	@echo ""
	@echo "$(BLUE)Quick Commands$(NC)"
	@echo "--------------"
	@echo "  make up          - Start services"
	@echo "  make down        - Stop services"
	@echo "  make test        - Run tests"
	@echo "  make logs        - View logs"
	@echo "  make help        - Show all commands"
