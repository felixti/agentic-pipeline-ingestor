# Spec: [Feature Name]

**Status:** Draft  
**Created:** ${KIMI_NOW}  
**Author:** SDD Agent  
**Spec ID:** spec-YYYY-MM-DD-feature-name

---

## 1. Overview

### 1.1 Problem Statement
<!-- Describe the problem this feature solves -->

### 1.2 Solution Summary
<!-- High-level description of the approach -->

### 1.3 Success Metrics
<!-- How do we measure success? -->
- Metric 1: Target value
- Metric 2: Target value

---

## 2. Goals & Non-Goals

### 2.1 Goals
- [ ] Goal 1: Specific, measurable, achievable
- [ ] Goal 2: Another objective
- [ ] Goal 3: Another objective

### 2.2 Non-Goals (Explicitly Out of Scope)
<!-- What we are NOT doing -->
- Feature X (will be handled in separate spec)
- Optimization Y (future iteration)
- Platform Z support (not required now)

---

## 3. Technical Design

### 3.1 Architecture Overview
```
[Component A] <---> [Component B] <---> [Component C]
     |                                        |
     v                                        v
[Data Store X]                          [External API Y]
```

### 3.2 Component Breakdown

#### Component A
- **Purpose:** What it does
- **Interface:** How others interact with it
- **Implementation Notes:** Key technical decisions

#### Component B
- **Purpose:** 
- **Interface:** 
- **Implementation Notes:** 

### 3.3 Data Models

```typescript
// Entity: ExampleEntity
interface ExampleEntity {
  id: string;           // UUID
  name: string;         // Human-readable name
  status: 'active' | 'inactive';
  createdAt: Date;
  metadata?: Record<string, unknown>;
}
```

### 3.4 API Design

| Endpoint | Method | Auth | Description | Request | Response |
|----------|--------|------|-------------|---------|----------|
| /api/v1/items | GET | Yes | List items | Query: { limit, offset } | { items, total } |
| /api/v1/items | POST | Yes | Create item | ItemInput | Item |
| /api/v1/items/:id | GET | Yes | Get item | - | Item |
| /api/v1/items/:id | PUT | Yes | Update item | ItemInput | Item |
| /api/v1/items/:id | DELETE | Yes | Delete item | - | 204 |

### 3.5 State Management
<!-- How state flows through the system -->

1. User action triggers event
2. State update logic
3. Side effects
4. UI updates

### 3.6 Error Handling

| Error Scenario | HTTP Status | Error Code | User Message |
|----------------|-------------|------------|--------------|
| Invalid input | 400 | INVALID_INPUT | "Please check your input" |
| Not found | 404 | NOT_FOUND | "Item not found" |
| Unauthorized | 401 | UNAUTHORIZED | "Please sign in" |
| Server error | 500 | INTERNAL_ERROR | "Something went wrong" |

### 3.7 Security Considerations
- Authentication method
- Authorization rules
- Data validation
- Rate limiting

---

## 4. Implementation Phases

### Phase 1: Foundation
**Goal:** Basic structure and core functionality
- [ ] Task 1: Set up project structure
- [ ] Task 2: Implement data models
- [ ] Task 3: Basic CRUD operations
- [ ] Task 4: Write unit tests

**Exit Criteria:**
- All tests pass
- Basic functionality works

### Phase 2: Core Feature
**Goal:** Complete feature implementation
- [ ] Task 5: Implement business logic
- [ ] Task 6: Add validation
- [ ] Task 7: Error handling
- [ ] Task 8: Integration tests

**Exit Criteria:**
- Feature works end-to-end
- Edge cases handled

### Phase 3: Polish
**Goal:** Production readiness
- [ ] Task 9: Performance optimization
- [ ] Task 10: Documentation
- [ ] Task 11: Final testing
- [ ] Task 12: Monitoring/logging

**Exit Criteria:**
- Performance targets met
- All acceptance criteria pass

---

## 5. Acceptance Criteria

- [ ] **AC1:** Given [context], when [action], then [expected result]
- [ ] **AC2:** Performance: Handles X requests/second
- [ ] **AC3:** Error case: When [error condition], then [expected behavior]
- [ ] **AC4:** UI/UX: [Specific UI requirement]
- [ ] **AC5:** Compatibility: Works with [platform/version]

---

## 6. Dependencies

### 6.1 Technical Dependencies
- [ ] Dependency A (version X+)
- [ ] Dependency B (already available)

### 6.2 External Dependencies
- [ ] External API access
- [ ] Infrastructure requirement

### 6.3 Blockers
- [ ] Decision needed on [topic]
- [ ] Waiting for [other team/feature]

---

## 7. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation Strategy |
|------|------------|--------|---------------------|
| Technical complexity | Medium | High | Prototype first, spike solution |
| Performance issues | Low | High | Benchmark early, optimize Phase 3 |
| Integration problems | Medium | Medium | Test with mocks, staging environment |
| Scope creep | High | Medium | Strict phase gates, spec change process |

---

## 8. Open Questions

1. Question 1?
2. Question 2?

---

## 9. Notes & References

- Related specs: [link]
- Design docs: [link]
- Reference implementation: [link]
- External resources: [link]
