# Ralph Loop + OpenSpec Integration

Complete integration guide for using Ralph Loop with OpenSpec.

## Overview

Ralph Loop is a **parallel implementation engine** for OpenSpec changes. It replaces sequential `apply-change` with domain-specialized agents working concurrently.

## Where Ralph Loop Fits

```
OpenSpec Workflow:
┌─────────┐   ┌──────────┐   ┌──────────────┐   ┌─────────┐   ┌──────────┐
│   New   │ → │ Continue │ → │   Apply      │ → │ Verify  │ → │ Archive  │
│ Change  │   │ (artifacts)│  │ (implement)  │   │         │   │          │
└─────────┘   └──────────┘   └──────────────┘   └─────────┘   └──────────┘
                                  ↑
                    ┌─────────────┴─────────────┐
                    │    apply-change           │  Sequential, single agent
                    │    OR                     │
                    │    ralph-loop             │  Parallel, domain agents
                    └───────────────────────────┘
```

## Integration Points

### 1. OpenSpec → Ralph Loop (Input)

Ralph Loop reads from OpenSpec change:

| OpenSpec File | Purpose |
|--------------|---------|
| `proposal.md` | Requirements, scope |
| `specs/*/spec.md` | Detailed specifications |
| `design.md` | Architecture decisions |
| `tasks.md` | Task list (checkboxes) |

### 2. Ralph Loop Generated Files

Ralph Loop creates in change folder:

| File | Purpose |
|------|---------|
| `IMPLEMENTATION_PLAN.md` | Task breakdown with agent assignments |

### 3. Ralph Loop → OpenSpec (Output)

Ralph Loop updates:

| File | Update |
|------|--------|
| `tasks.md` | Mark tasks complete (`- [ ]` → `- [x]`) |
| `IMPLEMENTATION_PLAN.md` | Log iterations |

## Usage

### 1. Create OpenSpec Change

```bash
python .agents/openspec_kit.py create "my-feature"
```

### 2. Complete OpenSpec Artifacts

Use OpenSpec skills:
- `openspec-continue-change` - Create proposal, specs, design, tasks
- `openspec-ff-change` - Fast-forward all artifacts

Verify ready:
```bash
python .agents/openspec_kit.py status "my-feature"
# isApplyReady: true
```

### 3. Use Ralph Loop

```
"Use Ralph Loop to implement my-feature"
```

This triggers `openspec-ralph-loop` skill.

### 4. Complete OpenSpec Workflow

After Ralph Loop:
```
"Verify my-feature"
"Archive my-feature"
```

## Skill Reference

| Skill | Purpose | When to Use |
|-------|---------|-------------|
| `openspec-new-change` | Create change | Starting new work |
| `openspec-continue-change` | Create artifacts | Building specs/plan |
| `openspec-ff-change` | Fast-forward | Skip to implementation |
| `openspec-ralph-loop` | **Parallel implementation** | **Complex/multi-domain** |
| `openspec-verify-change` | Verify work | Before archive |
| `openspec-archive-change` | Archive | Done |

## File Structure

```
project/
├── openspec/                      # OpenSpec root
│   ├── changes/
│   │   └── [change]/             # Change folder
│   │       ├── .openspec.yaml
│   │       ├── proposal.md
│   │       ├── specs/
│   │       ├── design.md
│   │       ├── tasks.md          # ← Updated by Ralph Loop
│   │       └── IMPLEMENTATION_PLAN.md  # ← Created by Ralph Loop
│   ├── specs/                     # Main specs
│   └── changes/archive/          # Archived
├── ralph-loop/                    # Ralph Loop system
│   ├── ralph-orchestrator.yaml
│   ├── subagents/
│   ├── system-prompts/
│   └── templates/
└── .agents/
    └── skills/
        ├── openspec-*/           # OpenSpec skills
        └── openspec-ralph-loop/  # ← Ralph Loop skill
            └── SKILL.md
```

## Configuration

Both systems configured via YAML:

**OpenSpec** (`openspec/config.yaml`):
```yaml
schema: spec-driven
context: |
  Tech stack: Python, FastAPI
```

**Ralph Loop** (`ralph-loop/ralph-orchestrator.yaml`):
```yaml
system_prompt_args:
  OPEN_SPEC_DIR: "openspec"
  CHANGES_DIR: "openspec/changes"
  MAX_ITERATIONS: "50"
```

## Best Practices

1. **Complete OpenSpec First** - Ralph Loop needs complete artifacts
2. **One Change at a Time** - Ralph Loop operates on single change
3. **Don't Modify Specs During** - Stop, update OpenSpec, restart
4. **Update tasks.md** - Ralph Loop should mark tasks complete
5. **Verify Then Archive** - Use OpenSpec verify before archive

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "isApplyReady: false" | Complete OpenSpec artifacts first |
| "No IMPLEMENTATION_PLAN.md" | Ralph Loop will generate it |
| Context overflow | Ralph Loop handles fresh context per iteration |
| Task dependencies | Ralph Loop identifies and orders tasks |

## Comparison

| Aspect | Apply-Change | Ralph Loop |
|--------|--------------|------------|
| Speed | 1x (sequential) | 1.5-2x (parallel) |
| Context | Accumulates | Fresh per iteration |
| Agents | Single | Domain specialists |
| Complexity | Simple changes | Multi-domain |
| Setup | None | Ralph Loop files |

## Summary

Ralph Loop **extends** OpenSpec, replacing the apply phase with parallel agents:

- Same OpenSpec artifacts (proposal, specs, design, tasks)
- Same workflow (new → continue → **ralph-loop** → verify → archive)
- Faster execution through parallelization
- Better context management (fresh per iteration)

Use **apply-change** for simple, single-domain changes.
Use **ralph-loop** for complex, multi-domain changes.
