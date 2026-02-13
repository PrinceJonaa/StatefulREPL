# Paradox Queue

> Contradictions held explicitly, not forced to premature resolution.
> A paradox is two positions that both seem true but conflict.

## Format

```
PARADOX-<ID>: "<tension in one sentence>"
  pole_a: "<position A and why it matters>"
  pole_b: "<position B and why it matters>"
  status: open | synthesized | dissolved
  date_opened: <YYYY-MM-DD>
  date_resolved: <YYYY-MM-DD or blank>
  resolution: "<synthesis or reason for dissolution, if resolved>"
  related: <CLAIM-IDs, CONTRACT-IDs, or "none">
```

## Queue

<!-- Add new entries below. Use sequential IDs. -->

PARADOX-001: "Fast implementation flow vs mandatory Loom write-back discipline"
  pole_a: "Move fast on code delivery to keep momentum and reduce friction"
  pole_b: "Enforce append-only `.loom` updates every non-trivial task for persistent coherence"
  status: open
  date_opened: 2026-02-13
  date_resolved: 
  resolution: ""
  related: CLAIM-004
