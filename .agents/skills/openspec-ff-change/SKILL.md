---
name: openspec-ff-change
description: Fast-forward through OpenSpec artifact creation. Use when the user wants to quickly create all artifacts needed for implementation without stepping through each one individually.
---

Fast-forward through artifact creation - generate everything needed to start implementation in one go.

**When to Use**
- User says "fast forward", "create all artifacts", "skip to implementation"
- User has a clear idea and wants to move quickly
- User wants to batch-create proposal, specs, design, and tasks

**Input**: The user's request should include a change name (kebab-case) OR a description of what they want to build.

**Steps**

1. **If no clear input provided, ask what they want to build**

   Ask: "What change do you want to work on? Describe what you want to build or fix."

   From their description, derive a kebab-case name.

   **IMPORTANT**: Do NOT proceed without understanding what the user wants to build.

2. **Validate name and check for existing change**
   - Validate kebab-case
   - Check if exists: `ls openspec/changes/`
   - If exists, suggest continuing instead

3. **Create the change directory**
   ```bash
   python .agents/openspec_kit.py create "<name>"
   ```

4. **Get the artifact build order**
   ```bash
   python .agents/openspec_kit.py status "<name>"
   ```
   
   Parse the JSON to get:
   - `applyRequires`: array of artifact IDs needed before implementation
   - `artifacts`: list of all artifacts with their status

5. **Create artifacts in sequence until apply-ready**

   Use TodoWrite to track progress through the artifacts.

   Loop through artifacts in dependency order:

   a. **For each artifact that is `ready`**:
      - Get instructions:
        ```bash
        python .agents/openspec_kit.py instructions <artifact-id> "<name>"
        ```
      - The instructions JSON includes:
        - `context`: Background from dependencies (read for context)
        - `template`: The structure to use
        - `instruction`: Schema-specific guidance
        - `outputPath`: Where to write
        - `dependencies`: Files to read for context
      - Read any completed dependency files
      - Create the artifact using `template` as structure
      - Show brief progress: "✓ Created <artifact-id>"

   b. **Continue until all `applyRequires` artifacts are complete**
      - After creating each artifact, re-run status
      - Check if every artifact ID in `applyRequires` has `status: "done"`
      - Stop when all `applyRequires` artifacts are done

   c. **If unclear context**: Ask user briefly, then continue

6. **Show final status**
   ```bash
   python .agents/openspec_kit.py status "<name>"
   ```

**Output**

After completing all artifacts, summarize:
- Change name and location
- List of artifacts created
- What's ready: "All artifacts created! Ready for implementation."
- Prompt: "Want to implement? Just ask me to apply the change."

**Artifact Creation Guidelines**

- Follow the schema's template for each artifact
- Read dependency artifacts for context before creating new ones
- Use `template` as the structure - fill in its sections
- Make reasonable decisions to keep momentum
- **IMPORTANT**: `context` is for YOU, not content for the file

**Guardrails**
- Create ALL artifacts needed for implementation
- Always read dependency artifacts before creating a new one
- If context is critically unclear, ask briefly
- If change exists, suggest continuing instead
- Verify each artifact file exists after writing
