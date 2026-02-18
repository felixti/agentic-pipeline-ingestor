---
name: openspec-archive-change
description: Archive a completed change in the experimental workflow. Use when the user wants to finalize and archive a change after implementation is complete.
---

Archive a completed change in the experimental workflow.

**When to Use**
- User says "archive", "complete", "finish", "we're done"
- Implementation is finished
- User wants to clean up completed work

**Input**: Optionally specify a change name. If omitted, check if it can be inferred from conversation context. If vague or ambiguous you MUST prompt for available changes.

**Steps**

1. **If no change name provided, prompt for selection**

   ```bash
   python .agents/openspec_kit.py list
   ```
   
   Show only active changes (not already archived).

   **IMPORTANT**: Do NOT guess or auto-select. Always let the user choose.

2. **Check artifact completion status**

   ```bash
   python .agents/openspec_kit.py status "<name>"
   ```
   
   Parse JSON:
   - `artifacts`: List with status (`done` or other)
   - `isComplete`: Boolean

   **If not complete:**
   - Display warning
   - Ask user if they want to proceed anyway
   - Proceed only if confirmed

3. **Check task completion status**

   Read `openspec/changes/<name>/tasks.md`
   
   Count: `- [ ]` (incomplete) vs `- [x]` (complete)

   **If incomplete tasks:**
   - Display warning
   - Ask user if they want to proceed
   - Proceed only if confirmed

4. **Assess delta spec sync state**

   Check for delta specs: `openspec/changes/<name>/specs/`
   
   Compare with main specs: `openspec/specs/<capability>/spec.md`
   
   Show summary of what would be synced.

   **Prompt options:**
   - If changes needed: "Sync now (recommended)", "Archive without syncing"
   - If synced: "Archive now", "Sync anyway", "Cancel"

   If user chooses sync, use openspec-sync-specs logic first.

5. **Perform the archive**

   ```bash
   python .agents/openspec_kit.py archive "<name>"
   ```

6. **Display summary**

   Show:
   - Change name
   - Schema used
   - Archive location
   - Whether specs were synced
   - Any warnings acknowledged

**Output On Success**

```
## Archive Complete

**Change:** <change-name>
**Schema:** <schema-name>
**Archived to:** openspec/changes/archive/YYYY-MM-DD-<name>/
**Specs:** ✓ Synced (or "No delta specs" or "Sync skipped")

All artifacts complete. All tasks complete.
```

**Guardrails**
- Always prompt for change selection if not provided
- Don't block on warnings - just inform and confirm
- Preserve .openspec.yaml when archiving
- Show clear summary
