# Spec-Driven Development (SDD) Agent System

A custom agent system for Kimi Code CLI that enforces **Spec-Driven Development** workflow, inspired by Kiro's approach.

> **"No code without spec. No merge without validation."**

## 📁 Structure

```
sdd-agents/
├── sdd-orchestrator.yaml          # Main orchestrator agent
├── system-prompts/
│   ├── orchestrator.md            # Main agent instructions
│   ├── spec-writer.md             # Spec writer subagent instructions
│   ├── implementer.md             # Implementer subagent instructions
│   └── reviewer.md                # Reviewer subagent instructions
├── subagents/
│   ├── spec-writer.yaml           # Spec writer subagent config
│   ├── implementer.yaml           # Implementer subagent config
│   └── reviewer.yaml              # Reviewer subagent config
└── templates/
    └── spec-template.md           # Spec document template
```

## 🚀 Usage

### 1. Start the SDD Orchestrator

```bash
kimi --agent-file /path/to/sdd-agents/sdd-orchestrator.yaml
```

### 2. Describe Your Feature

Tell the orchestrator what you want to build:

> "I want to build a user authentication system with JWT tokens, supporting login, register, and password reset."

### 3. The 4-Phase SDD Workflow

The orchestrator will automatically:

#### Phase 1: Discovery
- Clarify requirements
- Identify scope and constraints

#### Phase 2: Specification ⭐ **MANDATORY**
- Delegate to `spec-writer` subagent
- Creates comprehensive spec in `./specs/`
- You review and approve
- **BLOCKING**: No code until spec is approved

#### Phase 3: Implementation
- Delegate to `implementer` subagent
- Follows spec phases exactly
- Tracks progress via todo list

#### Phase 4: Validation
- Delegate to `reviewer` subagent
- Validates spec compliance
- Approves or rejects with specific feedback

## 📝 Spec Template

Specs are saved to `./specs/spec-YYYY-MM-DD-feature-name.md` with sections:

1. **Overview** - Problem statement and solution
2. **Goals & Non-Goals** - Scope definition
3. **Technical Design** - Architecture, models, APIs
4. **Implementation Phases** - Broken down milestones
5. **Acceptance Criteria** - Testable conditions
6. **Dependencies** - Technical and external
7. **Risks & Mitigations** - Risk assessment
8. **Open Questions** - Pending decisions
9. **Notes & References** - Resources

## 🔄 Example Workflow

```
User: "Build a task management API"

Orchestrator:
  → Delegates to spec-writer
    ← Creates spec-2026-02-16-task-management-api.md
  → Shows spec for approval
  ← User approves
  → Delegates to implementer with spec path
    ← Implements Phase 1, 2, 3
  → Delegates to reviewer
    ← Returns: "✅ APPROVED - All acceptance criteria met"
  → Reports completion to user
```

## ⚙️ Customization

### Change Spec Directory

Edit `sdd-orchestrator.yaml`:
```yaml
system_prompt_args:
  SPEC_DIR: "./docs/specs"  # Change from default ./specs
```

### Change Implementation Directory
```yaml
system_prompt_args:
  IMPLEMENTATION_DIR: "./src"  # Change from default .
```

### Add Custom Variables

Add to any system prompt:
```yaml
system_prompt_args:
  PROJECT_NAME: "MyProject"
  TECH_STACK: "React, Node.js, PostgreSQL"
```

Then use in prompts: `${PROJECT_NAME}`, `${TECH_STACK}`

## 📋 Best Practices

1. **Never skip the spec phase** - The whole point of SDD
2. **Be specific in requirements** - Vague requirements = vague specs
3. **Review specs carefully** - They're contracts
4. **One spec per feature** - Don't bundle unrelated changes
5. **Iterate on rejected reviews** - Fix issues, don't override

## 🎯 Benefits of SDD

- **Clear requirements** before coding starts
- **Reduced rework** from misunderstood requirements
- **Better estimates** with detailed phases
- **Objective acceptance** via defined criteria
- **Documentation** as a byproduct
- **Stakeholder alignment** through spec review

## 🛠️ Customizing the Agents

### Modify Spec Writer
Edit `system-prompts/spec-writer.md` to:
- Add company-specific sections
- Change output format
- Add domain-specific guidelines

### Modify Implementer
Edit `system-prompts/implementer.md` to:
- Enforce coding standards
- Add testing requirements
- Change code quality rules

### Modify Reviewer
Edit `system-prompts/reviewer.md` to:
- Add compliance checks
- Change review criteria
- Adjust strictness level

## 📝 Tips

- Keep specs in version control
- Review specs with stakeholders
- Update specs when requirements change (don't hack around them)
- Use the reviewer for code reviews too
- Archive old specs in `./specs/archive/`
