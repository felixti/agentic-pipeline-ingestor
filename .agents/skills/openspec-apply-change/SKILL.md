---
name: openspec-apply-change
description: Implement tasks from an OpenSpec change. Use when the user wants to start implementing, continue implementation, or work through tasks.
---

Implement tasks from an OpenSpec change.

**When to Use**
- User says "implement", "apply", "let's code", "start coding"
- User wants to work on the tasks from a change
- User has completed artifacts and is ready to implement

**Input**: Optionally specify a change name. If omitted, check if it can be inferred from conversation context. If vague or ambiguous you MUST prompt for available changes.

**Steps**

1. **Select the change**

   If a name is provided, use it. Otherwise:
   - Infer from conversation context if the user mentioned a change
   - List changes and let user select:
     ```bash
     python .agents/openspec_kit.py list
     ```

   Always announce: "Using change: <name>"

2. **Check status to understand the schema**
   ```bash
   python .agents/openspec_kit.py status "<name>"
   ```
   
   Parse the JSON to understand:
   - `schemaName`: The workflow being used
   - `isApplyReady`: Whether required artifacts are complete
   - `applyRequires`: Artifacts needed before implementation

3. **Check if ready to apply**

   If `isApplyReady` is false:
   - Show which required artifacts are missing
   - Suggest using openspec-continue-change to create them
   - STOP

4. **Read context files**

   Read all artifact files for context:
   - `openspec/changes/<name>/proposal.md`
   - `openspec/changes/<name>/specs/**/*.md` (if exists)
   - `openspec/changes/<name>/design.md` (if exists)
   - `openspec/changes/<name>/tasks.md`

5. **Show current progress**

   Display:
   - Schema being used
   - Parse tasks.md for checkboxes
   - Show: "N/M tasks complete"
   - List remaining tasks

6. **Implement tasks (loop until done or blocked)**

   For each pending task (checkbox `- [ ]`):
   - Show which task is being worked on
   - Make the code changes required
   - Keep changes minimal and focused
   - Mark task complete in tasks.md: `- [ ]` → `- [x]`
   - Continue to next task

   **Pause if:**
   - Task is unclear → ask for clarification
   - Implementation reveals a design issue → suggest updating artifacts
   - Error or blocker encountered → report and wait for guidance
   - User interrupts

7. **On completion or pause, show status**

   Display:
   - Tasks completed this session
   - Overall progress: "N/M tasks complete"
   - If all done: suggest archive
   - If paused: explain why and wait for guidance

**Output During Implementation**

```
## Implementing: <change-name> (schema: <schema-name>)

Working on task 3/7: <task description>
[...implementation happening...]
✓ Task complete

Working on task 4/7: <task description>
[...implementation happening...]
✓ Task complete
```

**Output On Completion**

```
## Implementation Complete

**Change:** <change-name>
**Schema:** <schema-name>
**Progress:** 7/7 tasks complete ✓

All tasks complete! Ready to archive this change.
```

**Guardrails**
- Keep going through tasks until done or blocked
- Always read context files before starting
- If task is ambiguous, pause and ask before implementing
- If implementation reveals issues, pause and suggest artifact updates
- Keep code changes minimal and scoped to each task
- Update task checkbox immediately after completing each task
- Pause on errors, blockers, or unclear requirements - don't guess
