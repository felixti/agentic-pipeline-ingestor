# Ralph Loop Example Usage

Example of implementing an OpenSpec change using Ralph Loop.

## Scenario: User Authentication

### Step 1: OpenSpec Change Ready

Ensure OpenSpec artifacts are complete:

```bash
python .agents/openspec_kit.py status "user-authentication"
```

Expected output:
```
Change: user-authentication
Schema: spec-driven
Status: 4/4 artifacts complete
isApplyReady: true
```

Artifacts:
- `openspec/changes/user-authentication/proposal.md` ✓
- `openspec/changes/user-authentication/specs/*/spec.md` ✓
- `openspec/changes/user-authentication/design.md` ✓
- `openspec/changes/user-authentication/tasks.md` ✓

### Step 2: Start Ralph Loop

```
"Use Ralph Loop to implement user-authentication"
```

### Step 3: IMPLEMENTATION_PLAN.md Generated

Ralph Loop creates `openspec/changes/user-authentication/IMPLEMENTATION_PLAN.md`:

```markdown
# Implementation Plan: User Authentication

## OpenSpec Context
- Change: user-authentication
- Specs: specs/user-registration/, specs/user-login/, etc.

## Task List

### Phase 1: Database
- [ ] Task 1: Create users table | Owner: db-agent | Dependencies: none
- [ ] Task 2: Create refresh_tokens table | Owner: db-agent | Dependencies: Task 1

### Phase 2: Backend APIs
- [ ] Task 3: Registration endpoint | Owner: backend-developer | Dependencies: Task 1
- [ ] Task 4: Login endpoint | Owner: backend-developer | Dependencies: Task 2
- [ ] Task 5: Auth middleware | Owner: backend-developer | Dependencies: Task 4

### Phase 3: Frontend
- [ ] Task 6: Registration page | Owner: frontend-developer | Dependencies: Task 3
- [ ] Task 7: Login page | Owner: frontend-developer | Dependencies: Task 4

### Phase 4: Testing
- [ ] Task 8: Unit tests | Owner: tester-agent | Dependencies: Task 3-5
- [ ] Task 9: E2E tests | Owner: tester-agent | Dependencies: Task 6-7

### Phase 5: QA
- [ ] Task 10: Validation | Owner: qa-agent | Dependencies: Task 8-9

## Iteration Log
| Iteration | Date | Agent | Task | Result |
|-----------|------|-------|------|--------|
```

### Step 4: Ralph Loop Iterations

**Iteration 1: Database Schema**
```
Task: Create users and tokens tables
Agent: db-agent
Status: ✓ Complete
Files: migrations/versions/001_*.py, src/db/models.py
```

**Iteration 2: Backend APIs (Sequential)**
```
Task: Registration endpoint
Agent: backend-developer
Status: ✓ Complete
Files: src/api/auth.py
```

**Iteration 3: Backend APIs**
```
Task: Login endpoint
Agent: backend-developer
Status: ✓ Complete
Files: src/api/auth.py
```

**Iteration 4: Frontend (Parallel)**
```
Task: Registration page + Login page
Agents: frontend-developer (parallel spawn)
Status: ✓ Complete
Files: frontend/pages/Register.tsx, frontend/pages/Login.tsx
```

**Iteration 5: Testing**
```
Task: Unit tests + E2E tests
Agents: tester-agent
Status: ✓ Complete
Coverage: 87%
```

**Iteration 6: QA**
```
Task: Final validation
Agent: qa-agent
Status: ✓ Passed
```

### Step 5: Completion

**IMPLEMENTATION_PLAN.md Updated:**
```markdown
## Iteration Log
| Iteration | Date | Agent | Task | Result |
|-----------|------|-------|------|--------|
| 1 | 2026-02-18 | db-agent | Task 1-2 | ✓ Complete |
| 2 | 2026-02-18 | backend-developer | Task 3 | ✓ Complete |
| 3 | 2026-02-18 | backend-developer | Task 4 | ✓ Complete |
| 4 | 2026-02-18 | frontend-developer | Task 6-7 | ✓ Complete |
| 5 | 2026-02-18 | tester-agent | Task 8-9 | ✓ Complete |
| 6 | 2026-02-18 | qa-agent | Task 10 | ✓ Passed |
```

**tasks.md Updated:**
```markdown
# Tasks

- [x] Create database schema
- [x] Implement registration API
- [x] Implement login API
- [x] Create registration UI
- [x] Create login UI
- [x] Write tests
- [x] QA validation
```

### Step 6: Verify and Archive

```
"Verify user-authentication"
"Archive user-authentication"
```

## Key Takeaways

1. **OpenSpec First** - Complete artifacts before Ralph Loop
2. **Fresh Context** - Each agent gets clean context
3. **Parallel Where Possible** - Frontend tasks run together
4. **Backpressure** - Tests validate each iteration
5. **Proper Completion** - Updates both plan and tasks.md

## Token Efficiency

| Metric | Apply-Change | Ralph Loop |
|--------|--------------|------------|
| Context | Accumulates 100K+ | Fresh ~15K |
| Total Work | 1 long session | 6 iterations |
| Overflow Risk | High | None |
| Speed | 1x | 1.5-2x (parallel) |
