---
name: openspec-verify-change
description: Verify implementation matches change artifacts. Use when the user wants to validate that implementation is complete, correct, and coherent before archiving.
---

Verify that an implementation matches the change artifacts (specs, tasks, design).

**When to Use**
- User says "verify", "check implementation", "did we do this right?"
- Before archiving a change
- When user wants to validate completeness

**Input**: Optionally specify a change name. If omitted, check if it can be inferred from conversation context. If vague or ambiguous you MUST prompt for available changes.

**Steps**

1. **If no change name provided, prompt for selection**

   ```bash
   python .agents/openspec_kit.py list
   ```
   
   Show changes with implementation tasks. Mark incomplete as "(In Progress)".

2. **Check status**
   ```bash
   python .agents/openspec_kit.py status "<name>"
   ```
   
   Parse to understand:
   - `schemaName`: The workflow being used
   - Which artifacts exist

3. **Read all artifacts**

   Read available artifacts from `openspec/changes/<name>/`:
   - `proposal.md`
   - `specs/**/*.md`
   - `design.md`
   - `tasks.md`

4. **Initialize verification report structure**

   Three dimensions:
   - **Completeness**: Tasks and spec coverage
   - **Correctness**: Requirement implementation and scenario coverage
   - **Coherence**: Design adherence and pattern consistency

5. **Verify Completeness**

   **Task Completion**:
   - Parse tasks.md checkboxes: `- [ ]` vs `- [x]`
   - Count complete vs total
   - CRITICAL issue for each incomplete task

   **Spec Coverage**:
   - For each delta spec, extract requirements
   - Search codebase for keywords related to requirements
   - CRITICAL if requirements appear unimplemented

6. **Verify Correctness**

   **Requirement Implementation**:
   - For each requirement, search codebase
   - If found, note file paths
   - WARNING if divergence detected

   **Scenario Coverage**:
   - For each scenario, check if handled
   - Check if tests exist
   - WARNING if uncovered

7. **Verify Coherence**

   **Design Adherence**:
   - If design.md exists, extract key decisions
   - Verify implementation follows decisions
   - WARNING if contradiction

   **Code Pattern Consistency**:
   - Review for consistency with project patterns
   - SUGGESTION if deviations found

8. **Generate Verification Report**

   ```
   ## Verification Report: <change-name>

   ### Summary
   | Dimension    | Status           |
   |--------------|------------------|
   | Completeness | X/Y tasks        |
   | Correctness  | M/N reqs covered |
   | Coherence    | Followed/Issues  |

   ### Issues

   **CRITICAL** (Must fix):
   - Issue 1

   **WARNING** (Should fix):
   - Issue 1

   **SUGGESTION** (Nice to fix):
   - Issue 1

   ### Assessment
   [Ready/Not ready for archive]
   ```

**Verification Heuristics**

- **Completeness**: Focus on objective checklist items
- **Correctness**: Use keyword search, reasonable inference
- **Coherence**: Look for glaring inconsistencies
- **False Positives**: Prefer SUGGESTION over WARNING over CRITICAL
- **Actionability**: Every issue must have a specific recommendation

**Graceful Degradation**

- If only tasks.md exists: verify tasks only
- If tasks + specs exist: verify completeness and correctness
- If full artifacts: verify all three dimensions
- Always note which checks were skipped
