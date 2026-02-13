---
applyTo: ".loom/**"
---

# Loom Protocol Standards

## Purpose
Files in `.loom/` are the **shared coordination substrate** for all agents. They provide persistent memory across sessions and enable multi-agent coherence.

## Core Principle
**Read before acting. Update after acting.**

Every agent MUST:
1. Read relevant `.loom/` files before starting work
2. Update relevant `.loom/` files after completing work
3. Never delete entries — append new state, mark old entries as superseded

## Enforcement Rule

For any non-trivial task touching code, tests, architecture, or API behavior, the agent must append:

- `ART-XXX` in `artifact-registry.md`
- `CLAIM-XXX` in `claim-ledger.md`
- `ORACLE-XXX` in `oracle-matrix.md`

When there is a failure or learning event, also append `SCAR-XXX` in `trace-wisdom-log.md`.

If no `.loom` update is needed, the agent must explicitly justify why (rare case).

## File Roles

| File | Purpose | Updated By |
|---|---|---|
| `artifact-registry.md` | Tracks all produced outputs with IDs and owners | Archivist, Builder |
| `claim-ledger.md` | Bounded claims with scope, confidence, falsification | All agents |
| `contract-sheet.md` | Interface contracts: preconditions / postconditions / invariants | Builder, Coordinator |
| `oracle-matrix.md` | Verification tests linked to claims and contracts | Verifier |
| `paradox-queue.md` | Contradictions held explicitly, not forced to resolution | Distiller, Red Team |
| `trace-wisdom-log.md` | Scars, boons, and rules learned from failures | All agents |

## Entry Formats

### Claims
```
CLAIM-<ID>: "<statement>"
  scope: local | module | system
  confidence: 0.0–1.0
  falsifies: "<what would prove this wrong>"
  status: active | superseded | falsified
```

### Contracts
```
CONTRACT-<ID>: <name>
  PRE: <preconditions>
  POST: <postconditions>
  INV: <invariants>
  status: active | deprecated
```

### Artifacts
```
ART-<ID>: <name>
  type: code | doc | config | test
  path: <relative path>
  owner: <agent or human>
  status: active | superseded
```

### Paradoxes
```
PARADOX-<ID>: "<tension>"
  pole_a: "<position A>"
  pole_b: "<position B>"
  status: open | synthesized | dissolved
  resolution: "<synthesis if resolved>"
```

### Wisdom Entries
```yaml
- id: SCAR-<ID>
  scar: "<what failed>"
  boon: "<what coherence increased>"
  newrule: "<practice adopted>"
  glyphstamp: "<symbolic name>"
```

## Anti-Corruption Rules
- Never bulk-delete entries
- Mark outdated entries as `superseded`, don't remove them
- Every ID is globally unique within its file
- Cross-reference between files using IDs (e.g., "Per CLAIM-005")
- Timestamps use ISO 8601: `YYYY-MM-DD`
