# Design: Review and Update Architecture Documentation

## Overview

This design document outlines the approach for reviewing and updating all architecture and documentation files in the codebase.

## Current State Analysis

### Existing Documentation

1. **AGENTS.md** (969 lines)
   - Comprehensive agent guide
   - Needs pgvector updates
   - File counts outdated

2. **README.md** (350+ lines)
   - Main project documentation
   - Missing vector search features
   - Architecture diagram outdated

3. **docs/** directory
   - content-detection.md
   - VECTOR_STORE_API_USAGE.md
   - vector_store_api.md
   - VECTOR_STORE_DEPLOYMENT_GUIDE.md

4. **Missing**
   - ARCHITECTURE.md - comprehensive system architecture
   - Central API documentation index

## Target State

### Documentation Structure

```
project/
├── AGENTS.md              # Updated with pgvector
├── README.md              # Updated with current features
├── ARCHITECTURE.md        # NEW: Comprehensive architecture doc
├── docs/
│   ├── content-detection.md
│   ├── vector_store_api.md
│   ├── VECTOR_STORE_API_USAGE.md
│   └── VECTOR_STORE_DEPLOYMENT_GUIDE.md
└── api/
    └── openapi.yaml       # Already exists
```

## Technical Approach

### Phase 1: Codebase Review

Spawn qa-agent to perform comprehensive codebase review:
- Analyze src/ directory structure
- Count files by module
- Identify new components
- Document API endpoints

### Phase 2: AGENTS.md Update

Spawn backend-developer to update AGENTS.md:
- Add pgvector to technology stack
- Update file counts
- Add vector store endpoints
- Update architecture diagram
- Add configuration section

### Phase 3: README.md Update

Spawn backend-developer to update README.md:
- Refresh feature list
- Update architecture diagram
- Add vector search features
- Update quick start

### Phase 4: ARCHITECTURE.md Creation

Spawn backend-developer to create ARCHITECTURE.md:
- System context
- Component diagrams
- Data flow
- Integration architecture

### Phase 5: Validation

Spawn qa-agent to validate all documentation:
- Check for consistency
- Verify accuracy
- Check links
- Review completeness

## Design Decisions

### D1: Documentation Format

All documentation in Markdown format following existing style.

### D2: Architecture Diagrams

Use ASCII/Markdown diagrams for portability. Consider Mermaid for complex diagrams.

### D3: File Organization

Keep ARCHITECTURE.md at project root for visibility.

### D4: Incremental Updates

Update existing docs rather than replacing to preserve history and familiar structure.

## Validation Strategy

1. **Automated Checks**
   - Markdown linting
   - Link checking
   - File count verification

2. **Manual Review**
   - Technical accuracy
   - Completeness
   - Style consistency

3. **QA Review**
   - Full documentation audit
   - Comparison with implementation
   - Acceptance criteria verification

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Missing details | Thorough review process |
| Inconsistency | Follow existing patterns |
| Rapid obsolescence | Include in change process |

## Success Criteria

- All docs updated with accurate information
- New ARCHITECTURE.md comprehensive and complete
- QA validation passes
- No broken links or references
