---
name: openspec-onboard
description: Guided onboarding for OpenSpec - walk through a complete workflow cycle with narration and real codebase work.
---

Guided onboarding for OpenSpec workflow. Walk the user through a complete cycle with narration and real work.

**When to Use**
- First-time user says "onboard me", "how does this work?", "show me the workflow"
- User wants to learn by doing
- New team member introduction

**Steps**

1. **Introduction**

   Explain the OpenSpec workflow:
   - Artifact-driven development
   - Spec → Design → Tasks → Implementation
   - Why it helps (clarity, traceability, reviewability)

2. **Create a demo change**

   Use openspec-new-change to create a sample:
   ```bash
   python .agents/openspec_kit.py create "demo-onboarding"
   ```

3. **Walk through each artifact**

   For each artifact in sequence:
   - Explain its purpose
   - Show the template
   - Create it with demo content
   - Explain what just happened

4. **Show implementation**

   - Explain how tasks connect to code
   - Show how to apply a change
   - Mark a task complete

5. **Show verification and archive**

   - Run verification
   - Archive the demo change

6. **Recap and next steps**

   Summarize what was learned and encourage the user to try with a real change.

**Tone**
- Educational and encouraging
- Explain WHY, not just WHAT
- Let user ask questions at any point
