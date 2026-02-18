---
name: openspec-ralph-loop
description: Implement OpenSpec change using Ralph Loop - parallel specialized agents with fresh context per iteration. Use when user wants faster implementation with domain experts working in parallel, or when the change spans multiple domains (backend, frontend, database).
---

Implement an OpenSpec change using Ralph Loop methodology - parallel specialized agents with fresh context per iteration.

**When to Use**
- User says "use ralph loop", "parallel agents", "spawn agents"
- Change spans multiple domains (backend + frontend + database)
- Want faster implementation through parallelization
- Want domain experts (backend-dev, frontend-dev, db-agent) working simultaneously
- Current `openspec-apply-change` is too slow for complex changes

**Input**: Optionally specify a change name. If omitted, check conversation context or list available changes.

**How Ralph Loop Differs from Apply-Change**

| Aspect | Apply-Change | Ralph Loop |
|--------|--------------|------------|
| Execution | Sequential tasks | Parallel domain agents |
| Context | Persistent (accumulates) | Fresh per iteration |
| Specialization | Single agent | Domain experts (backend, frontend, db, qa) |
| Speed | Linear | Parallel where possible |
| Best for | Simple changes | Complex, multi-domain changes |

**Prerequisites**
- OpenSpec change exists with complete artifacts
- Required: proposal.md, specs/, design.md, tasks.md
- Ralph Loop system installed at `ralph-loop/`

**Steps**

1. **Select the change**

   If a name is provided, use it. Otherwise:
   - Infer from conversation context
   - List changes and let user select:
     ```bash
     python .agents/openspec_kit.py list
     ```

2. **Verify OpenSpec artifacts are complete**

   ```bash
   python .agents/openspec_kit.py status "<name>"
   ```
   
   Parse JSON:
   - Check `isApplyReady`: must be true
   - Check `applyRequires` artifacts are done

   **If not ready:**
   - Show missing artifacts
   - Suggest using openspec-continue-change or openspec-ff-change
   - STOP

3. **Read all OpenSpec artifacts**

   Read for context:
   - `openspec/changes/<name>/proposal.md`
   - `openspec/changes/<name>/specs/**/*.md`
   - `openspec/changes/<name>/design.md`
   - `openspec/changes/<name>/tasks.md`

4. **Check for existing IMPLEMENTATION_PLAN.md**

   Check if `openspec/changes/<name>/IMPLEMENTATION_PLAN.md` exists.

   **If not exists:** Generate it from tasks.md and specs
   
   **If exists:** Read it and continue from current state

5. **Initialize Ralph Loop context**

   Load the Ralph Loop orchestrator configuration:
   - `ralph-loop/ralph-orchestrator.yaml`
   - Subagent definitions from `ralph-loop/subagents/`

6. **Execute Ralph Loop iterations**

   For each uncompleted task in IMPLEMENTATION_PLAN.md:

   a. **Identify next unblocked task(s)**
      - Check dependencies
      - Find tasks with no incomplete dependencies

   b. **Determine agent type from task owner**
      - `db-agent` → Database work
      - `backend-developer` → API/services
      - `frontend-developer` → UI/components  
      - `tester-agent` → Tests
      - `qa-agent` → Validation

   c. **Spawn agent using Task tool**
      
      Use the subagent's system prompt from `ralph-loop/system-prompts/`
      
      ```python
      Task(
          description="<task description>",
          subagent_name="coder",
          prompt="""You are the <agent-type>.
          
          OpenSpec Change: openspec/changes/<name>/
          Spec: <relevant spec path>
          Plan: openspec/changes/<name>/IMPLEMENTATION_PLAN.md
          
          Task: <description>
          
          Acceptance Criteria:
          - <criterion 1>
          - <criterion 2>
          
          Run validation after implementation."""
      )
      ```

   d. **Wait for completion**

   e. **Run backpressure gates**
      ```bash
      make test      # or project-specific test command
      make lint      # or project-specific lint command
      ```

   f. **Update status**
      - Mark task complete in IMPLEMENTATION_PLAN.md
      - Mark task complete in tasks.md
      - Log iteration in plan's iteration log

   g. **Check for parallel opportunities**
      - If multiple unblocked tasks exist, spawn agents in parallel
      - Wait for all to complete before continuing

7. **On completion or pause, show status**

   Display:
   - Tasks completed this session
   - Overall progress
   - Remaining tasks
   - Any blockers or issues

8. **Final QA (if all tasks complete)**

   Spawn qa-agent for final validation.

**Output During Implementation**

```
## Ralph Loop: <change-name>

Iteration 1:
├─ Task: Create database schema
├─ Agent: db-agent
└─ Status: ✓ Complete

Iteration 2:
├─ Task: Implement auth API
├─ Agent: backend-developer  
└─ Status: ✓ Complete

Iteration 3:
├─ Task: Implement login UI
├─ Task: Write tests
├─ Agents: frontend-developer, tester-agent (parallel)
└─ Status: ✓ Complete

Progress: 7/7 tasks complete ✓
```

**Output On Completion**

```
## Ralph Loop Complete: <change-name>

**Change:** <change-name>
**Schema:** <schema-name>
**Progress:** 7/7 tasks complete ✓
**Iterations:** 7
**Agents Spawned:** 7

All tasks complete! Ready for verification and archive.
```

**Guardrails**
- Always verify OpenSpec artifacts are complete before starting
- One agent per task - don't overload
- Run validation gates after each task
- Update both IMPLEMENTATION_PLAN.md and tasks.md
- Stop on errors or blockers
- Fresh context for each iteration (don't accumulate)

**Integration with OpenSpec**

Ralph Loop extends OpenSpec's `apply` phase:

```
OpenSpec Workflow:
  new → continue → [Ralph Loop] → verify → archive
                     ↑
              replaces apply-change
              adds parallel agents
              adds fresh context
```

After Ralph Loop completes:
1. All tasks in tasks.md should be checked
2. IMPLEMENTATION_PLAN.md shows complete iteration log
3. Proceed to openspec-verify-change
4. Then openspec-archive-change
