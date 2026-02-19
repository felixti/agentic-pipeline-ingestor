# Proposal: Review and Update Architecture Documentation

## Why

The codebase has evolved significantly with the completion of multiple OpenSpec changes, particularly the comprehensive `add-vector-store-pgvector` feature. The current architecture documentation needs to be reviewed and updated to reflect:

1. **New pgvector-based vector storage system** - A major architectural addition
2. **Current module structure** - 118 Python source files, 70 test files
3. **Updated API endpoints** - New search and chunk retrieval endpoints
4. **Service layer expansion** - Vector search, text search, hybrid search services
5. **Repository pattern implementation** - DocumentChunkRepository and others

### Problems Being Solved

1. **Documentation Drift**: AGENTS.md and README.md don't fully reflect the pgvector integration
2. **Missing Architecture Document**: No single comprehensive ARCHITECTURE.md exists
3. **Incomplete API Documentation**: New endpoints need to be documented
4. **Outdated Project Structure**: File counts and organization have changed

### Use Cases Enabled

- Better onboarding for new developers
- Clearer understanding of system capabilities
- Accurate representation of the technology stack
- Up-to-date API reference

## What Changes

### Documentation Updates Required

1. **AGENTS.md Updates**
   - Add pgvector/vector store references
   - Update project structure section
   - Add new API endpoints
   - Update technology stack

2. **README.md Updates**
   - Refresh feature list with vector search capabilities
   - Update architecture diagram
   - Add vector store configuration section
   - Update quick start guide

3. **ARCHITECTURE.md Creation** (New)
   - Comprehensive system architecture document
   - Component diagrams
   - Data flow descriptions
   - Integration points

4. **API Documentation**
   - Update OpenAPI spec with new endpoints
   - Document search API usage patterns
   - Add vector store configuration reference

## Capabilities

- [x] **Comprehensive codebase review**
  - [x] Source code structure analysis (118 files)
  - [x] Test suite review (70 files)
  - [x] Configuration files review
  - [x] Documentation audit

- [x] **AGENTS.md update**
  - [x] Add vector store technology references
  - [x] Update project structure section
  - [x] Add new API endpoints to endpoint table
  - [x] Update configuration section with vector_store.yaml

- [x] **README.md update**
  - [x] Update features list with semantic search
  - [x] Refresh architecture diagram
  - [x] Add vector search quick start
  - [x] Update technology stack table

- [x] **ARCHITECTURE.md creation**
  - [x] System overview and context
  - [x] Component architecture
  - [x] Data flow diagrams
  - [x] Technology stack details
  - [x] Integration architecture
  - [x] Security architecture
  - [x] Deployment architecture

- [x] **API documentation update**
  - [x] Document new search endpoints
  - [x] Add chunk retrieval examples
  - [x] Update OpenAPI spec references

## Impact

### Benefits

| Benefit | Description | Impact Level |
|---------|-------------|--------------|
| **Better Onboarding** | New developers can understand the system faster | High |
| **Accurate Documentation** | Documentation reflects actual system state | High |
| **Improved Maintenance** | Clear architecture guides future development | Medium |
| **Stakeholder Communication** | Clear docs for business/technical stakeholders | Medium |

### Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Missing details** | Medium | Low | Thorough review process with qa-agent |
| **Outdated quickly** | High | Low | Include documentation in change process |
| **Inconsistent style** | Low | Low | Follow existing documentation patterns |

### Breaking Changes

**None** - This is a documentation-only change with no code modifications.

### Success Metrics

- All existing documentation updated with pgvector references
- New ARCHITECTURE.md created with comprehensive coverage
- API documentation reflects all 34+ endpoints
- Documentation review passes qa-agent validation
