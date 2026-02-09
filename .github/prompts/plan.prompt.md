---
agent: agent
tools: ['agent', 'read', 'search', 'todo']
description: "Plan phase — design contracts, define the oracle, create a task DAG"
---

# Plan Phase (O-P-W-T-R)

You are in the **Plan** phase. Orient is complete. Now design the solution before building.

## Steps

1. **Define contracts for each component:**
   ```
   CONTRACT-<ID>: <name>
     PRE: <what must be true before>
     POST: <what will be true after>
     INV: <what must remain true throughout>
   ```

2. **Create the task DAG:**
   - Break the work into the smallest vertical slices
   - Identify dependencies (what must happen before what)
   - Mark parallelizable tasks

3. **Define the oracle:**
   - How will we verify each task is done correctly?
   - What tests need to pass?
   - What quality dimensions matter most? (IC, EC, CV minimum)

4. **Identify risks:**
   - What could go wrong?
   - What are the fallback plans?
   - Check `trace-wisdom-log.md` for relevant past failures

5. **Produce a Plan Document:**

```
PLAN
====
Goal: <from Orient report>

Contracts:
  CONTRACT-XX: <name>
    PRE: <preconditions>
    POST: <postconditions>
    INV: <invariants>

Task DAG:
  1. <task> [depends: none] [agent: builder]
  2. <task> [depends: 1] [agent: builder]
  3. <task> [depends: 1,2] [agent: verifier]

Oracle:
  - <test or check that proves success>
  - <quality dimensions to evaluate>

Risks:
  - <risk>: <mitigation>
```

## Rules
- Do NOT start building during Plan
- Every task must have a verification criterion
- Prefer smaller slices over ambitious ones
- Record contracts in `.loom/contract-sheet.md`
