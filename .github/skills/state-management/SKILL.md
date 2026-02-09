# Skill: State Management (L1/L2/L3 Memory)

## Purpose
Maintain persistent shared state across agent sessions using a three-layer memory architecture. This ensures continuity, learning, and coordination.

## The Three Layers

### L1: Working Pad (Short-Term Memory)
**What:** Current session's active context — goal, constraints, artifacts in play.
**Lifetime:** Single session / conversation turn.
**Format:**
```
WorkingPad: goal = <current goal> | constraints = <active constraints> | artifacts = [<IDs>]
```
**Operations:**
- `UPDATE L1.goal <value>` — Change the active goal
- `UPDATE L1.constraints <value>` — Update constraints
- `READ_STATE L1` — Show current working context

### L2: Session Log (Medium-Term Memory)
**What:** Running journal of decisions, actions, and shifts within a session.
**Lifetime:** Full session / multi-turn conversation.
**Format:**
```
SessionLog: <2-3 sentence arc of what happened>
```
**Operations:**
- `APPEND L2 "<entry>"` — Add a session event
- `CONSOLIDATE L1→L2` — Compress working state into session log

### L3: Wisdom Base (Long-Term Memory)
**What:** Learned rules, scars, and boons that persist across sessions.
**Lifetime:** Permanent (stored in `.loom/trace-wisdom-log.md`).
**Format:**
```yaml
- id: SCAR-<ID>
  scar: "<what failed>"
  boon: "<what coherence increased>"
  newrule: "<practice adopted>"
  glyphstamp: "<symbolic name>"
```
**Operations:**
- `APPEND L3 {entry}` — Add a wisdom entry
- `CONSOLIDATE L2→L3` — Extract patterns from session into wisdom
- `READ_STATE L3` — Load learned rules

## Memory Flow

```
L1 (Working Pad) → L2 (Session Log) → L3 (Wisdom Base)
     fast/volatile      session-scoped       permanent
```

Each consolidation compresses and extracts:
- L1→L2: "What happened this turn" → session narrative
- L2→L3: "What did we learn" → permanent rules

## Integration with .loom/

| Layer | .loom/ File |
|---|---|
| L1 | Not persisted (session-only) |
| L2 | Referenced in `artifact-registry.md` entries |
| L3 | `trace-wisdom-log.md` |

## When to Use
- **L1**: Every turn — always know the current goal and constraints
- **L2**: After significant decisions or phase transitions
- **L3**: After failures (scars), successes (boons), or learning (new rules)
- **Consolidation**: At session end, or when context is getting large

## Anti-Patterns
- **Stateless drift**: Acting without reading L1 first → decisions lose context
- **Memory hoarding**: Never consolidating → L1 grows unbounded
- **Wisdom amnesia**: Not reading L3 at session start → repeating old mistakes
