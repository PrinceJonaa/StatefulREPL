---
agent: agent
tools: ['agent', 'read', 'search', 'web', 'todo']
description: "Orient phase — gather context and map the landscape before planning"
---

# Orient Phase (O-P-W-T-R)

You are in the **Orient** phase. Your job is to understand the full context before any planning or building begins.

## Steps

1. **Read .loom/ state files:**
   - Check `artifact-registry.md` — what exists already?
   - Check `claim-ledger.md` — what has been claimed?
   - Check `trace-wisdom-log.md` — what lessons apply?
   - Check `paradox-queue.md` — any open contradictions?

2. **Map the codebase:**
   - Identify relevant source files and their relationships
   - Note the project structure and conventions
   - Find existing tests and their coverage

3. **Identify stakeholders and constraints:**
   - What does the user want?
   - What are the technical constraints?
   - What resources/tools are available?

4. **Produce an Orient Report:**

```
ORIENT REPORT
=============
Goal: <what we're trying to achieve>
Artifacts available: <list with IDs if in registry>
Key constraints: <technical and scope limits>
Relevant wisdom: <any SCAR entries that apply>
Open paradoxes: <any PARADOX entries that apply>
Missing context: <what we still need to learn>
Recommended next step: <specific Plan phase action>
```

## Rules
- Do NOT start building during Orient
- Do NOT skip reading .loom/ files
- If critical context is missing, say so and ask the user
