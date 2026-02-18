# Ralph Loop - Multi-Agent Implementation System

Parallel specialized agents for OpenSpec changes. An alternative to sequential `apply-change` that uses domain experts with fresh context per iteration.

## Overview

Ralph Loop implements the **Ralph Wiggum technique**: continuous iteration with fresh context until tasks are verifiably complete. It extends OpenSpec by adding parallel agent execution.

**Use when:**
- Change spans multiple domains (backend + frontend + database)
- You want faster implementation through parallelization
- You want domain experts working simultaneously

## Quick Start

```bash
# 1. Ensure OpenSpec change is ready (proposal, specs, design, tasks)
python .agents/openspec_kit.py status "my-feature"

# 2. Use Ralph Loop skill
"Use Ralph Loop to implement my-feature"
```

## How It Works

```
┌────────────────────────────────────────────────────────────────┐
│ 1. Read OpenSpec Change                                        │
│    └── openspec/changes/[change]/                              │
│        ├── proposal.md                                         │
│        ├── specs/                                              │
│        ├── design.md                                           │
│        └── tasks.md                                            │
├────────────────────────────────────────────────────────────────┤
│ 2. Generate/Read IMPLEMENTATION_PLAN.md                        │
│    └── Creates task breakdown with agent assignments           │
├────────────────────────────────────────────────────────────────┤
│ 3. Execute Ralph Loop Iterations                               │
│    ├── Identify unblocked tasks                                │
│    ├── Spawn domain specialist(s)                              │
│    │   ├── db-agent → Schema                                   │
│    │   ├── backend-developer → APIs                            │
│    │   ├── frontend-developer → UI                             │
│    │   ├── tester-agent → Tests                                │
│    │   └── qa-agent → Validation                               │
│    ├── Run validation gates                                    │
│    └── Update plan & tasks.md                                  │
├────────────────────────────────────────────────────────────────┤
│ 4. Complete                                                    │
│    └── All tasks done → Verify → Archive                       │
└────────────────────────────────────────────────────────────────┘
```

## The Ralph Loop Cycle

1. **Fresh Context** - Each iteration starts with ~176K tokens
2. **One Task Per Iteration** - Keep context lean
3. **Parallel Agents** - Domain experts work concurrently
4. **Backpressure** - Tests/linting reject bad outputs
5. **Disposable Plans** - Regenerate when stale

## Agent Team

| Agent | Specialty | When Spawned |
|-------|-----------|--------------|
| `db-agent` | Schema, migrations, queries | Database work |
| `backend-developer` | APIs, services, business logic | Backend features |
| `frontend-developer` | UI components, pages | Frontend work |
| `tester-agent` | Unit, integration, E2E tests | Test coverage |
| `qa-agent` | Validation, compliance | Final review |

## Comparison

| Aspect | Apply-Change | Ralph Loop |
|--------|--------------|------------|
| Execution | Sequential tasks | Parallel agents |
| Context | Accumulates | Fresh per iteration |
| Agents | Single | Domain specialists |
| Speed | Linear | Parallel |
| Best for | Simple changes | Complex, multi-domain |

## File Structure

```
ralph-loop/
├── ralph-orchestrator.yaml          # Main orchestrator config
├── subagents/                        # Agent definitions
│   ├── backend-developer.yaml
│   ├── frontend-developer.yaml
│   ├── db-agent.yaml
│   ├── tester-agent.yaml
│   └── qa-agent.yaml
├── system-prompts/                   # Agent behaviors
│   ├── ralph-orchestrator.md
│   ├── backend-developer.md
│   ├── frontend-developer.md
│   ├── db-agent.md
│   ├── tester-agent.md
│   └── qa-agent.md
├── templates/
│   └── implementation-plan.md       # Plan template
└── README.md                         # This file
```

## Integration with OpenSpec

Ralph Loop extends the OpenSpec workflow:

```
OpenSpec:   new → continue → [apply-change OR ralph-loop] → verify → archive
                              ↑
                       ralph-loop: parallel agents, fresh context
```

**Skill**: `.agents/skills/openspec-ralph-loop/SKILL.md`

## Configuration

Edit `ralph-orchestrator.yaml`:

```yaml
system_prompt_args:
  OPEN_SPEC_DIR: "openspec"
  CHANGES_DIR: "openspec/changes"
  MAIN_SPECS_DIR: "openspec/specs"
  MAX_ITERATIONS: "50"
```

## Key Principles

1. **Context is Scarce** - ~176K usable tokens from 200K window
2. **Plans are Disposable** - Regenerate when stale
3. **Backpressure Converges** - Wrong outputs get rejected
4. **One Task Per Iteration** - Context stays lean
5. **Fresh Context Each Loop** - Spawn subagents for exploration

## Resources

- [Ralph Wiggum Playbook](https://paddo.dev/blog/ralph-wiggum-playbook/)
- [Claude Code Agent Teams](https://code.claude.com/docs/en/agent-teams)
- OpenSpec: `.agents/README.md`

## License

MIT
