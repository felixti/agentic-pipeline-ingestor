---
name: openspec-sync-specs
description: Sync delta specs from a change to main specs. Use when the user wants to update main specs with changes from a delta spec, without archiving the change.
---

Sync delta specs from a change to main specs.

**When to Use**
- User says "sync specs", "update main specs"
- Before archiving to ensure main specs are current
- When delta specs have evolved and need to be merged

**Input**: Optionally specify a change name. If omitted, prompt for selection.

**Steps**

1. **If no change name, prompt for selection**

   ```bash
   python .agents/openspec_kit.py list
   ```

2. **Find delta specs**

   Check: `openspec/changes/<name>/specs/`
   
   If no delta specs:
   - Inform user
   - STOP

3. **Compare each delta spec with main**

   For each capability spec in delta:
   - Read delta: `openspec/changes/<name>/specs/<capability>/spec.md`
   - Check main: `openspec/specs/<capability>/spec.md`
   - Determine diff (adds, modifications, removals)

4. **Show sync preview**

   Display:
   ```
   ## Sync Preview: <change-name>

   | Capability | Action | Changes |
   |------------|--------|---------|
   | auth       | Modify | 3 sections updated |
   | billing    | Add    | New spec |
   ```

5. **Apply sync**

   For each capability:
   - Ensure directory exists: `openspec/specs/<capability>/`
   - Write/overwrite: `openspec/specs/<capability>/spec.md`
   - Preserve delta in change (don't delete)

6. **Show completion**

   List synced capabilities and their locations in main specs.

**Guardrails**
- Always show preview before applying
- Preserve delta specs in change directory
- Create capability directories as needed
- Inform user of any conflicts
