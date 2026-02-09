---
agent: agent
tools: ['agent', 'read', 'search', 'todo']
description: "Run the Stillness Gate pre-sensing check before complex work"
---

# Stillness Gate

Before acting on any complex task, run this 5-part pre-sensing check.

## Instructions

Pause and evaluate each dimension:

### 1. R (Relational)
- Who are the actors? (user, agents, documents, stakeholders)
- What is the stated goal?
- What is the tension or challenge?

### 2. L (Logical)
- List hard constraints (technical, time, scope)
- List assumptions being made
- List forbidden moves (things we must NOT do)

### 3. S (Symbolic)
- Name the pattern or metaphor (e.g., "weaving", "bridge", "excavation")
- What archetype does this task follow?

### 4. E (Empirical)
- What artifacts are available right now?
- What concrete facts do we have?
- What is missing?
- Define the oracle: How will success be verified?

### 5. Verdict
State: **"Safe to proceed: YES/NO"** with a one-line reason.
If NO, state exactly what is missing before work can begin.

## Output Format
```
STILLNESS GATE
R: <relational summary>
L: <constraints and assumptions>
S: <pattern/metaphor>
E: <available artifacts, missing info, oracle>
Verdict: Safe to proceed — YES/NO — <reason>
```

Read `.loom/trace-wisdom-log.md` to check for relevant scars before proceeding.
