---
name: openspec-new-change
description: Start a new OpenSpec change using the artifact workflow. Creates a structured change with proposal, specs, design, and tasks. Use when the user wants to create a new feature, fix, or modification with a structured step-by-step approach.
---

Start a new change using the artifact-driven OpenSpec approach.

**When to Use**
- User wants to build a new feature
- User wants to fix a bug
- User wants to make a structured modification
- User says things like "I want to...", "Let's add...", "We need to..."

**Input**: The user's request should include a change name (kebab-case) OR a description of what they want to build.

**Steps**

1. **If no clear input provided, ask what they want to build**

   Ask the user: "What change do you want to work on? Describe what you want to build or fix."

   From their description, derive a kebab-case name (e.g., "add user authentication" → `add-user-auth`).

   **IMPORTANT**: Do NOT proceed without understanding what the user wants to build.

2. **Validate the change name**
   - Must be kebab-case (lowercase, hyphens between words)
   - Examples: `user-auth`, `fix-login-bug`, `refactor-api`
   - If invalid, ask for a valid name

3. **Check if change already exists**
   
   Run: `ls -la openspec/changes/` (if directory exists)
   
   If a change with that name exists, suggest continuing it instead of creating new.

4. **Create the change directory**

   Use the Python kit:
   ```bash
   python .agents/openspec_kit.py create "<name>" [schema]
   ```
   
   Omit the schema parameter unless the user explicitly requests a different workflow.
   
   This creates a scaffolded change at `openspec/changes/<name>/`.

5. **Show the artifact status**
   
   ```bash
   python .agents/openspec_kit.py status "<name>"
   ```
   
   This shows which artifacts need to be created and which are ready.

6. **Get instructions for the first artifact**
   
   The first artifact is typically `proposal`. Get its instructions:
   ```bash
   python .agents/openspec_kit.py instructions proposal "<name>"
   ```

7. **STOP and wait for user direction**

**Output**

After completing the steps, summarize:
- Change name and location (`openspec/changes/<name>/`)
- Schema/workflow being used and its artifact sequence
- Current status (0/N artifacts complete)
- The template for the first artifact
- Prompt: "Ready to create the first artifact? Just describe what this change is about and I'll draft it, or ask me to continue."

**Guardrails**
- Do NOT create any artifacts yet - just show the instructions
- Do NOT advance beyond showing the first artifact template
- If the name is invalid (not kebab-case), ask for a valid name
- If a change with that name already exists, suggest continuing that change instead
