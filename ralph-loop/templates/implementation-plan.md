# Implementation Plan

**OpenSpec Change:** [Change Name]  
**Created:** [Date]  
**Status:** Planning → In Progress → Complete

---

## OpenSpec Context

- **Change:** openspec/changes/[change-name]/
- **Proposal:** proposal.md
- **Design:** design.md
- **Specs:** specs/[capability]/spec.md
- **OpenSpec Tasks:** tasks.md

---

## Overview

Brief summary of what needs to be built.

---

## Architecture

```
[ASCII diagram or description of components]
```

### Component Responsibilities

- **Backend**: API endpoints, business logic
- **Frontend**: UI components, user interactions
- **Database**: Data models, relationships
- **Tests**: Coverage for all components

---

## Task List

### Phase 1: Foundation
- [ ] Task 1: [Description] | Owner: db-agent | Dependencies: none
- [ ] Task 2: [Description] | Owner: backend-developer | Dependencies: Task 1

### Phase 2: Core Implementation
- [ ] Task 3: [Description] | Owner: backend-developer | Dependencies: Task 2
- [ ] Task 4: [Description] | Owner: frontend-developer | Dependencies: Task 3

### Phase 3: Testing & QA
- [ ] Task 5: [Description] | Owner: tester-agent | Dependencies: Task 3,4
- [ ] Task 6: [Description] | Owner: qa-agent | Dependencies: Task 5

---

## Dependencies

### Internal Dependencies
- Task 2 requires Task 1 completion (DB schema)
- Task 4 requires Task 3 completion (API ready)

### External Dependencies
- [ ] API contract defined
- [ ] Design mocks approved
- [ ] Third-party service access

---

## References

### OpenSpec Artifacts
- Proposal: See `proposal.md` for requirements
- Design: See `design.md` for architecture
- Specs: See `specs/[capability]/spec.md` for detailed requirements

### Shared Resources
- API Contracts: See `shared/api-contracts.json`
- Database Schema: See `shared/db-schema.md`
- QA Reports: See `shared/qa-reports/`

---

## Validation Criteria

### Acceptance Criteria
- [ ] Criterion 1: [Testable condition]
- [ ] Criterion 2: [Testable condition]
- [ ] Criterion 3: [Testable condition]

### Quality Gates
- [ ] All tests pass
- [ ] Coverage ≥ 80%
- [ ] Lint passes
- [ ] Type check passes
- [ ] QA approval

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| | | | |

---

## Iteration Log

| Iteration | Date | Agent | Task | Result |
|-----------|------|-------|------|--------|
| 1 | | db-agent | Task 1 | |
| 2 | | backend-developer | Task 2 | |
| 3 | | ... | ... | |

---

## Notes

[Any additional notes or context]
