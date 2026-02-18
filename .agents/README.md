# OpenSpec for Kimi Code

An adaptation of the OpenSpec (Antigravity) workflow for Kimi Code CLI.

## Overview

This directory contains a complete OpenSpec workflow implementation that works natively with Kimi Code's skill system. It replaces the `openspec` CLI tool with a lightweight Python module (`openspec_kit.py`) while preserving the same artifact-driven development workflow.

## Architecture

```
.agents/
├── openspec_kit.py              # State manager (replaces openspec CLI)
├── skills/                      # Kimi Code skills
│   ├── openspec-new-change/     # Start a new change
│   ├── openspec-continue-change/# Continue/create artifacts
│   ├── openspec-apply-change/   # Implement tasks
│   ├── openspec-verify-change/  # Verify implementation
│   ├── openspec-archive-change/ # Archive completed changes
│   ├── openspec-ff-change/      # Fast-forward all artifacts
│   ├── openspec-explore/        # Explore mode
│   ├── openspec-bulk-archive/   # Archive multiple changes
│   ├── openspec-sync-specs/     # Sync specs to main
│   └── openspec-onboard/        # Guided onboarding
└── README.md                    # This file
```

## How It Works

### Comparison: Antigravity vs Kimi Code

| Aspect | Antigravity | Kimi Code (This Adaptation) |
|--------|-------------|------------------------------|
| **Invocation** | `/opsx:new <name>` | Natural language: "Create a new change for..." |
| **State Check** | `openspec status --change <name>` | `python .agents/openspec_kit.py status "<name>"` |
| **Create Artifact** | `openspec instructions <id> --change <name>` | `python .agents/openspec_kit.py instructions <id> "<name>"` |
| **List Changes** | `openspec list` | `python .agents/openspec_kit.py list` |
| **Archive** | `openspec archive` | `python .agents/openspec_kit.py archive "<name>"` |

### Directory Structure

```
openspec/
├── changes/
│   ├── <change-name>/           # Active change
│   │   ├── .openspec.yaml       # Metadata
│   │   ├── proposal.md          # Artifact: Proposal
│   │   ├── specs/               # Artifact: Specs
│   │   │   └── <capability>/
│   │   │       └── spec.md
│   │   ├── design.md            # Artifact: Design
│   │   └── tasks.md             # Artifact: Tasks
│   └── archive/                 # Archived changes
│       └── YYYY-MM-DD-<name>/
└── specs/                       # Main specs (synced from changes)
    └── <capability>/
        └── spec.md
```

## Usage

### Starting a New Change

**User says:**
> "I want to add user authentication to the API"

**What happens:**
1. Kimi Code detects `openspec-new-change` skill
2. Asks for clarification if needed
3. Creates change: `python .agents/openspec_kit.py create "user-auth"`
4. Shows first artifact template
5. Waits for user direction

### Continuing a Change

**User says:**
> "Continue with the user-auth change"

**What happens:**
1. Kimi Code detects `openspec-continue-change` skill
2. Checks status: `python .agents/openspec_kit.py status "user-auth"`
3. Finds next ready artifact
4. Gets instructions: `python .agents/openspec_kit.py instructions <id> "user-auth"`
5. Creates the artifact
6. Shows progress

### Fast-Forward (Create All Artifacts)

**User says:**
> "Fast forward the user-auth change"

**What happens:**
1. Kimi Code detects `openspec-ff-change` skill
2. Creates all artifacts in sequence
3. Stops when ready for implementation
4. Shows summary

### Implementing Tasks

**User says:**
> "Apply the user-auth change" or "Implement user-auth"

**What happens:**
1. Kimi Code detects `openspec-apply-change` skill
2. Reads all context files
3. Works through tasks one by one
4. Marks tasks complete as they're done

### Verifying Implementation

**User says:**
> "Verify the user-auth change"

**What happens:**
1. Kimi Code detects `openspec-verify-change` skill
2. Checks tasks, specs, and design
3. Generates verification report
4. Shows completeness, correctness, coherence

### Archiving

**User says:**
> "Archive the user-auth change"

**What happens:**
1. Kimi Code detects `openspec-archive-change` skill
2. Checks completion status
3. Optionally syncs specs
4. Moves to archive: `python .agents/openspec_kit.py archive "user-auth"`

## The `openspec_kit.py` Module

### Commands

```bash
# Create a new change
python .agents/openspec_kit.py create "change-name" [schema]

# Get status
python .agents/openspec_kit.py status "change-name"

# Get instructions for artifact
python .agents/openspec_kit.py instructions <artifact-id> "change-name"

# List all changes
python .agents/openspec_kit.py list

# Archive a change
python .agents/openspec_kit.py archive "change-name"
```

### JSON Output

All commands output JSON for easy parsing by skills:

```json
// Status output
{
  "name": "user-auth",
  "schemaName": "spec-driven",
  "path": "openspec/changes/user-auth",
  "artifacts": [
    {"id": "proposal", "status": "done", ...},
    {"id": "spec/auth-flow", "status": "ready", ...},
    {"id": "design", "status": "blocked", ...}
  ],
  "isComplete": false,
  "isApplyReady": false
}

// Instructions output
{
  "artifactId": "proposal",
  "template": "# Proposal: user-auth\n\n## Why\n...",
  "outputPath": "openspec/changes/user-auth/proposal.md",
  "context": "...",
  "dependencies": []
}
```

## Schemas

The default schema is `spec-driven` with this artifact sequence:

```
proposal → spec → design → tasks
   ↓          ↓        ↓
(required) (multiple) (required) (required)
```

### Artifacts

| Artifact | Purpose | Dependencies |
|----------|---------|--------------|
| **proposal** | Why, what, capabilities | None |
| **spec** | Detailed requirements | proposal |
| **design** | Architecture decisions | spec |
| **tasks** | Implementation checklist | design |

## Skills Reference

| Skill | Purpose | When to Use |
|-------|---------|-------------|
| `openspec-new-change` | Start a new change | "Create a change for..." |
| `openspec-continue-change` | Create next artifact | "Continue", "Next artifact" |
| `openspec-apply-change` | Implement tasks | "Implement", "Apply", "Code" |
| `openspec-verify-change` | Verify work | "Verify", "Check", "Did we" |
| `openspec-archive-change` | Archive completed | "Archive", "Done", "Finish" |
| `openspec-ff-change` | Create all artifacts | "Fast forward", "All artifacts" |
| `openspec-explore` | Think/explore | "Explore", "Think about", "Help me understand" |
| `openspec-bulk-archive` | Archive multiple | "Archive all", "Clean up" |
| `openspec-sync-specs` | Sync to main specs | "Sync specs", "Update main" |
| `openspec-onboard` | Learn the workflow | "Onboard", "How does this work" |

## Example Workflow

```
User: I want to add OAuth authentication

Kimi: I'll create a new change for OAuth authentication.
      ✓ Created change: add-oauth-auth
      First artifact: proposal.md
      Ready to create it? Tell me about this change...

User: We need to support Google and GitHub OAuth

Kimi: ✓ Created proposal.md with capabilities
      Next: specs for oauth-google and oauth-github
      Continue?

User: Yes

Kimi: ✓ Created specs/oauth-google/spec.md
      ✓ Created specs/oauth-github/spec.md
      Next: design.md
      Continue?

User: Yes

Kimi: ✓ Created design.md
      Next: tasks.md
      Continue?

User: Yes

Kimi: ✓ Created tasks.md
      All artifacts complete! Ready to implement.
      Run apply?

User: Apply the change

Kimi: ## Implementing: add-oauth-auth
      Working on task 1/5: Set up OAuth library...
      [code changes]
      ✓ Task complete
      ...
      All tasks complete! Ready to archive.

User: Archive it

Kimi: ✓ Archived to openspec/changes/archive/2024-01-15-add-oauth-auth/
```

## Migration from Antigravity

If you have existing Antigravity/OpenSpec changes:

1. **Directory structure is the same** - `openspec/changes/<name>/`
2. **Metadata file changes**:
   - Antigravity: `.openspec.json` or internal state
   - This: `.openspec.yaml` (created automatically)
3. **Artifacts are compatible** - proposal.md, specs/, design.md, tasks.md

To migrate an existing change:
```bash
# Create metadata file
python -c "
from agents.openspec_kit import get_changes_dir, validate_name
from pathlib import Path
import yaml

name = 'your-change-name'
change_dir = get_changes_dir() / name
metadata = {
    'name': name,
    'schema': 'spec-driven',
    'created': '2024-01-01T00:00:00',
    'modified': '2024-01-01T00:00:00'
}
with open(change_dir / '.openspec.yaml', 'w') as f:
    yaml.dump(metadata, f)
"
```

## Extending

### Adding a New Schema

1. Edit `.agents/openspec_kit.py`
2. Add to `SCHEMAS` dictionary:

```python
SCHEMAS = {
    "spec-driven": { ... },
    "my-schema": {
        "name": "my-schema",
        "artifacts": [
            {
                "id": "concept",
                "name": "Concept",
                "outputPath": "concept.md",
                "dependsOn": [],
                "template": "# Concept\n\n..."
            },
            # ... more artifacts
        ],
        "applyRequires": ["tasks"]
    }
}
```

### Customizing Templates

Edit the `template` strings in `openspec_kit.py` `SCHEMAS` definitions.

## Troubleshooting

### "Change not found"
- Check: `python .agents/openspec_kit.py list`
- Verify the change directory exists: `ls openspec/changes/`

### "Invalid change name"
- Names must be kebab-case: `my-change-name`
- No spaces, no underscores, lowercase only

### Skills not detected
- Ensure `.agents/skills/` is in Kimi Code's skill path
- Restart Kimi Code after adding skills

## License

MIT - Same as original OpenSpec/Antigravity
