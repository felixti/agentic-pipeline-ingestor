# SDK Generation

This directory contains auto-generated SDKs for the Agentic Data Pipeline Ingestor API.

## Overview

SDKs are generated from the OpenAPI 3.1 specification using the [OpenAPI Generator](https://openapi-generator.tech/).

## Available SDKs

| Language | Directory | Status | Package |
|----------|-----------|--------|---------|
| Python | `python/` | Generated | `pipeline-ingestor-client` |
| TypeScript | `typescript/` | Generated | `@example/pipeline-ingestor-client` |

## Generating SDKs

### Prerequisites

Install OpenAPI Generator:

```bash
# Using npm
npm install -g @openapitools/openapi-generator-cli

# Using Homebrew
brew install openapi-generator

# Using Docker
docker pull openapitools/openapi-generator-cli:latest
```

### Generate All SDKs

```bash
python sdks/generate.py
```

### Generate Specific SDK

```bash
# Python SDK
python sdks/generate.py --language python

# TypeScript SDK
python sdks/generate.py --language typescript
```

### Dry Run

Preview commands without executing:

```bash
python sdks/generate.py --dry-run
```

## SDK Usage

### Python SDK

```python
import asyncio
from pipeline_ingestor_client import ApiClient, Configuration, JobsApi

async def main():
    config = Configuration(host="http://localhost:8000")
    async with ApiClient(config) as client:
        jobs_api = JobsApi(client)
        
        # List jobs
        jobs = await jobs_api.list_jobs()
        print(jobs)

asyncio.run(main())
```

### TypeScript SDK

```typescript
import { Configuration, JobsApi } from '@example/pipeline-ingestor-client';

const config = new Configuration({
    basePath: 'http://localhost:8000'
});

const jobsApi = new JobsApi(config);

// List jobs
const jobs = await jobsApi.listJobs();
console.log(jobs);
```

## Regeneration

SDKs should be regenerated when the OpenAPI specification changes:

1. Update `/api/openapi.yaml`
2. Run `python sdks/generate.py`
3. Review generated code
4. Update SDK versions if needed
5. Commit changes

## Customization

The generation script (`generate.py`) includes post-processing for:
- Adding package metadata files
- Updating configuration files
- Customizing output structure

Modify the post-processing functions in `generate.py` to customize SDK output.

## Manual Changes

**Important**: Do not manually edit generated SDK files. They will be overwritten on regeneration.

If you need custom functionality:
1. Extend the generated classes in your application code
2. Use the generator's configuration options
3. Submit an issue to discuss changes to the OpenAPI spec

## CI/CD Integration

Example GitHub Actions workflow for SDK generation:

```yaml
name: Generate SDKs

on:
  push:
    paths:
      - 'api/openapi.yaml'

jobs:
  generate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Java
        uses: actions/setup-java@v3
        with:
          java-version: '17'
          distribution: 'temurin'
      
      - name: Install OpenAPI Generator
        run: |
          wget https://repo1.maven.org/maven2/org/openapitools/openapi-generator-cli/7.0.1/openapi-generator-cli-7.0.1.jar -O openapi-generator-cli.jar
      
      - name: Generate SDKs
        run: |
          export PATH="$PATH:$(pwd)"
          python sdks/generate.py
      
      - name: Commit changes
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
          git add sdks/python sdks/typescript
          git diff --staged --quiet || git commit -m "Update SDKs from OpenAPI spec"
          git push
```
