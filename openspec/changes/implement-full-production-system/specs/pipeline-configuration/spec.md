# Spec: Pipeline Configuration Management

## Overview
Persist pipeline configurations to database and link them to jobs.

### Requirements

#### R1: Create Pipeline
**Given** a valid pipeline configuration
**When** POST /api/v1/pipelines is called
**Then** store configuration and return pipeline ID

#### R2: List Pipelines
**Given** pipelines exist
**When** GET /api/v1/pipelines is called
**Then** return list of all pipeline configurations

#### R3: Get Pipeline
**Given** a pipeline ID
**When** GET /api/v1/pipelines/{id} is called
**Then** return full pipeline configuration

#### R4: Update Pipeline
**Given** an existing pipeline
**When** PUT /api/v1/pipelines/{id} is called
**Then** update configuration and increment version

#### R5: Delete Pipeline
**Given** an existing pipeline not in use
**When** DELETE /api/v1/pipelines/{id} is called
**Then** mark as deleted or remove

#### R6: Link Job to Pipeline
**Given** a job is created with pipeline_id
**When** job is stored
**Then** validate pipeline exists and link it

### Scenarios

#### SC1: Create Custom Pipeline
User creates pipeline with specific stages:
```json
POST /api/v1/pipelines
{
  "name": "ocr-pipeline",
  "description": "For scanned documents",
  "parser": {"primary_parser": "azure_ocr"},
  "enabled_stages": ["ingest", "detect", "parse", "output"]
}
```
Returns pipeline ID for use in jobs.

#### SC2: Use Pipeline in Job
Create job with pipeline:
```json
POST /api/v1/jobs
{
  "source_type": "upload",
  "pipeline_id": "pipeline-uuid"
}
```
Job uses the specified pipeline configuration.
