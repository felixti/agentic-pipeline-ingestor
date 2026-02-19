# Specification: Documentation Review and Update

## Overview

Review all architecture and documentation files in the codebase and update them to reflect the current system state, particularly the pgvector vector storage integration.

## Scope

### In Scope

1. Review and update AGENTS.md
2. Review and update README.md
3. Create comprehensive ARCHITECTURE.md
4. Review and update API documentation references
5. Update any other markdown documentation

### Out of Scope

- Code changes
- New feature implementation
- Test file modifications
- Configuration file changes (except documentation)

## Requirements

### R1: Codebase Analysis

**R1.1** - Analyze src/ directory structure (118 Python files)
**R1.2** - Identify new modules and components
**R1.3** - Document service layer additions (vector search services)
**R1.4** - Document repository pattern implementations
**R1.5** - Record API endpoint changes

### R2: AGENTS.md Update

**R2.1** - Update Technology Stack section with pgvector
**R2.2** - Update Project Structure section with accurate file counts
**R2.3** - Add vector store references throughout
**R2.4** - Update API Endpoints table with new search endpoints
**R2.5** - Add vector store configuration section
**R2.6** - Update architecture diagram

### R3: README.md Update

**R3.1** - Refresh feature list with semantic search capabilities
**R3.2** - Update architecture diagram or description
**R3.3** - Add vector search to key features
**R3.4** - Update technology stack table
**R3.5** - Add vector store quick start section
**R3.6** - Update API endpoint count and descriptions

### R4: ARCHITECTURE.md Creation

**R4.1** - System Context and Overview
**R4.2** - Component Architecture (diagrams + descriptions)
**R4.3** - Data Flow Architecture
**R4.4** - API Architecture
**R4.5** - Database Architecture
**R4.6** - Security Architecture
**R4.7** - Integration Architecture
**R4.8** - Deployment Architecture
**R4.9** - Technology Stack Details

### R5: API Documentation

**R5.1** - Document search API endpoints
**R5.2** - Document chunk retrieval endpoints
**R5.3** - Add usage examples
**R5.4** - Update OpenAPI spec references

## Acceptance Criteria

- [ ] AGENTS.md updated with all pgvector references
- [ ] README.md refreshed with current features
- [ ] ARCHITECTURE.md created with comprehensive coverage
- [ ] All documentation follows consistent style
- [ ] Documentation passes qa-agent review
- [ ] No broken internal links
- [ ] Accurate file and module counts

## References

- AGENTS.md - Primary agent documentation
- README.md - Main project documentation
- openspec/changes/add-vector-store-pgvector/ - Recent feature implementation
- src/ - Source code directory
- docs/ - Existing documentation
