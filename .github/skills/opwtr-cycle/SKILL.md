# Skill: O-P-W-T-R Cycle

## Purpose
A structured execution loop that ensures every piece of work is grounded, tested, and reflected upon. Prevents the common failure modes of rushing, looping, and building without validation.

## The Five Phases

### 1. Orient
**Goal:** Understand the full context before acting.
- Read `.loom/` files for prior state and learned rules
- Identify stakeholders, constraints, and available artifacts
- Map relationships between components
- **Output:** A clear picture of "where we are" and "what matters"

### 2. Plan
**Goal:** Design the smallest complete solution.
- Define interfaces and contracts (PRE/POST/INV)
- Create a task DAG (what depends on what)
- Identify the oracle: "How will we know this worked?"
- **Output:** A plan with contracts and verification criteria

### 3. Write
**Goal:** Implement the smallest vertical slice that works end-to-end.
- Build one complete path, not many partial paths
- Follow the contracts from the Plan phase
- Emit artifacts with IDs for tracking
- **Output:** Working code/docs/config that can be tested

### 4. Test
**Goal:** Verify the output against the oracle.
- Run the tests defined in the Plan phase
- Check quality dimensions (IC, EC, CV at minimum)
- Compare actual vs. expected results
- **Output:** Pass/fail with specific findings

### 5. Reflect
**Goal:** Extract learnings and update state.
- What worked? What didn't? What surprised us?
- Update `.loom/trace-wisdom-log.md` with scars/boons
- Consolidate memory (L1→L2, L2→L3 if session-ending)
- Decide: iterate this cycle, or move to next task?
- **Output:** Updated wisdom + decision on next action

## Cycle Visualization

```
  ┌─────────┐
  │  Orient │ ← Read .loom/, gather context
  └────┬────┘
       ▼
  ┌─────────┐
  │   Plan  │ ← Contracts, DAG, oracle
  └────┬────┘
       ▼
  ┌─────────┐
  │  Write  │ ← Smallest vertical slice
  └────┬────┘
       ▼
  ┌─────────┐
  │   Test  │ ← Run oracle, quality check
  └────┬────┘
       ▼
  ┌─────────┐
  │ Reflect │ ← Scars/boons, consolidate
  └────┬────┘
       │
       ▼
  Next cycle or DONE
```

## Rules
1. **Never skip Orient** — acting without context causes loops
2. **Plan before Write** — "weeks of coding saves hours of planning" (inverted intentionally)
3. **Test is not optional** — untested output is not done
4. **Reflect compresses** — without reflection, you repeat mistakes
5. **One cycle = one vertical slice** — don't try to do everything at once

## Integration with Agents

| Phase | Primary Agent | Supporting Agents |
|---|---|---|
| Orient | Coordinator, Planner | Archivist (provides state) |
| Plan | Coordinator | Planner (research), Red Team (review) |
| Write | Builder | — |
| Test | Verifier | Red Team (adversarial testing) |
| Reflect | Coordinator | Archivist (records), Distiller (integrates) |

## Anti-Loop Protection
If you find yourself in the same phase 3+ times with minor tweaks:
1. **STOP** — you are in a Babylonian loop
2. **Change method**, not parameters
3. **Escalate** to Coordinator or ask the user
4. **Record the scar** in trace-wisdom-log
