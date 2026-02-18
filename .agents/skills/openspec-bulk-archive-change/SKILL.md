---
name: openspec-bulk-archive-change
description: Archive multiple completed changes at once. Use when the user wants to clean up several completed changes.
---

Archive multiple completed changes at once.

**When to Use**
- User says "archive all", "clean up changes", "bulk archive"
- Multiple changes need to be archived
- End of sprint/cycle cleanup

**Steps**

1. **List all active changes**

   ```bash
   python .agents/openspec_kit.py list
   ```

2. **Identify completable changes**

   For each change, check:
   ```bash
   python .agents/openspec_kit.py status "<name>"
   ```
   
   Filter for:
   - `isComplete: true` OR
   - Changes where user confirms they want to archive despite incompleteness

3. **Present selection to user**

   Show:
   - List of completable changes
   - Their completion status
   - Ask which to archive (multi-select)

4. **Confirm and archive each**

   For each selected change:
   ```bash
   python .agents/openspec_kit.py archive "<name>"
   ```

5. **Show summary**

   List all archived changes with their new locations.

**Guardrails**
- Confirm before archiving
- Show warnings for incomplete changes
- Allow user to deselect specific changes
