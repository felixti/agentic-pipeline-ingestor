# Ralph Loop Agent Structure

Architecture documentation for the Ralph Loop multi-agent system.

## Directory Structure

```
ralph-loop/
├── ralph-orchestrator.yaml          # Main orchestrator configuration
├── subagents/                        # Agent definitions (YAML)
├── system-prompts/                   # Agent behaviors (Markdown)
└── templates/                        # Reusable templates
```

## Agent Hierarchy

```
┌────────────────────────────────────────────────────────────────┐
│                    RALPH LOOP ORCHESTRATOR                      │
│  (Coordinates OpenSpec change implementation)                   │
└──────────────────────────┬─────────────────────────────────────┘
                           │ spawns
           ┌───────────────┼───────────────┬───────────────┐
           ▼               ▼               ▼               ▼
┌─────────────────┐ ┌─────────────┐ ┌──────────┐ ┌─────────────┐
│  BACKEND DEV    │ │ FRONTEND DEV│ │  DB AGENT│ │ TESTER AGENT│
│ APIs, Services  │ │ UI, Pages   │ │ Schema   │ │ Tests      │
└─────────────────┘ └─────────────┘ └──────────┘ └─────────────┘
                                                           │
                           ┌─────────────────────────────────┘
                           ▼
                   ┌───────────────┐
                   │   QA AGENT    │
                   │ Validation    │
                   └───────────────┘
```

## YAML Configuration

### Orchestrator (`ralph-orchestrator.yaml`)

```yaml
version: 1
agent:
  name: ralph-loop-orchestrator
  system_prompt_path: ./system-prompts/ralph-orchestrator.md
  tools: [...]                       # Available tools
  subagents:                         # Child agents
    [agent-name]:
      path: ./subagents/[name].yaml
      description: "When to spawn"
  system_prompt_args:                # Template variables
    OPEN_SPEC_DIR: "openspec"
    CHANGES_DIR: "openspec/changes"
```

### Subagent (`subagents/*.yaml`)

```yaml
version: 1
agent:
  name: [agent-name]
  system_prompt_path: ../system-prompts/[name].md
  tools: [...]                       # Subset of tools
  system_prompt_args:
    VAR_NAME: "value"
```

## Communication Flow

```
┌─────────────┐    Task Tool     ┌─────────────┐
│ Orchestrator│ ───────────────▶ │  Subagent   │
│             │                  │ (fresh ctx) │
│             │ ◀─────────────── │             │
└─────────────┘    Results       └─────────────┘
        │                               │
        │ Read/Write                    │ Implements
        ▼                               ▼
┌─────────────────────────────────────────────┐
│  openspec/changes/[change]/                 │
│  ├── proposal.md                            │
│  ├── specs/                                 │
│  ├── design.md                              │
│  ├── tasks.md         ◀── Updated          │
│  └── IMPLEMENTATION_PLAN.md  ◀── Updated   │
└─────────────────────────────────────────────┘
```

## Tool Permissions

| Tool | Orchestrator | Subagents |
|------|:------------:|:---------:|
| Task (spawn) | ✅ | ❌ |
| SetTodoList | ✅ | ✅ |
| ReadFile | ✅ | ✅ |
| WriteFile | ✅ | ✅ |
| Shell | ✅ | ✅ |
| SearchWeb | ✅ | Some |

## System Prompt Structure

Each agent prompt follows this structure:

1. **Identity** - Role and expertise
2. **Goal** - Mission statement
3. **Process** - Step-by-step workflow
4. **Standards** - Code/style guidelines
5. **Deliverables** - Expected outputs
6. **Constraints** - What not to do
7. **Context** - Variables (paths, config)

## Integration Points

### OpenSpec → Ralph Loop

1. Change selected from `openspec/changes/`
2. Artifacts read (proposal, specs, design, tasks)
3. IMPLEMENTATION_PLAN.md generated

### Ralph Loop → OpenSpec

1. tasks.md updated (checkboxes marked)
2. IMPLEMENTATION_PLAN.md iteration log updated
3. On completion: verify → archive

## Extension

### Adding a New Agent

1. Create `subagents/[name].yaml`
2. Create `system-prompts/[name].md`
3. Register in `ralph-orchestrator.yaml` subagents
4. Update orchestrator to spawn for relevant tasks

### Customizing Paths

Edit `system_prompt_args` in `ralph-orchestrator.yaml`:

```yaml
system_prompt_args:
  OPEN_SPEC_DIR: "custom/openspec"
  MAX_ITERATIONS: "100"
```
