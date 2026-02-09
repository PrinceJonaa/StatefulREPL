---
agent: agent
tools: ['agent', 'read', 'search', 'edit', 'todo']
description: "Consolidate memory — compress L1→L2→L3 and update .loom/ files"
---

# Memory Consolidation

Compress and persist the current session's knowledge into long-term memory.

## Steps

### 1. Consolidate L1 → L2 (Working Pad → Session Log)
Summarize the current session's arc:
- What was the goal?
- What decisions were made?
- What was built or changed?
- What remains open?

### 2. Consolidate L2 → L3 (Session Log → Wisdom Base)
Extract learnable patterns:
- **Scars**: What went wrong or caused friction?
- **Boons**: What worked well or increased coherence?
- **New Rules**: What practice should be adopted going forward?

Format each entry:
```yaml
- id: SCAR-<next_id>
  scar: "<what failed>"
  boon: "<what coherence increased>"
  newrule: "<practice adopted>"
  glyphstamp: "<symbolic name>"
```

### 3. Update .loom/ Files
- Append new wisdom to `trace-wisdom-log.md`
- Update `artifact-registry.md` with new artifacts produced
- Update `claim-ledger.md` with new or revised claims
- Close resolved entries in `paradox-queue.md`

### 4. Produce a Consolidation Summary

```
CONSOLIDATION
=============
Session arc: <2-3 sentence summary>

New artifacts registered: <ART-IDs or "none">
Claims updated: <CLAIM-IDs or "none">
Paradoxes resolved: <PARADOX-IDs or "none">

Wisdom entries added:
  SCAR-XX: <glyphstamp> — <one-line summary>

State: <clean / has open items>
Next session should start by: <specific action>
```

## Rules
- Never skip wisdom extraction after failures
- Always read existing .loom/ state before appending
- Use sequential IDs (check current max ID before adding)
