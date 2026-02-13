# Contract Sheet

> Interface contracts defining preconditions, postconditions, and invariants.
> Contracts are agreements between components — they define what must be true.

## Format

```
CONTRACT-<ID>: <name>
  PRE: <what must be true before invocation>
  POST: <what will be true after successful completion>
  INV: <what must remain true throughout>
  status: active | deprecated
  owner: <agent or module>
  date: <YYYY-MM-DD>
```

## Contracts

<!-- Add new entries below. Use sequential IDs. -->

CONTRACT-001: ContextCompressionAPI
  PRE: Request includes either non-empty text or use_state=true; target_ratio in [0.05, 1.0]
  POST: Response includes compressed_text, token counts, and compression_ratio in [0, 1]
  INV: compressed_tokens <= original_tokens except empty-input edge case normalization
  status: active
  owner: src/stateful_repl/server.py
  date: 2026-02-13

CONTRACT-002: PrefetchPredictionAPI
  PRE: record key is non-empty; predict limit >= 1
  POST: Predict returns ranked candidate list (possibly empty) with bounded scores
  INV: Engine history remains bounded by configured window_size
  status: active
  owner: src/stateful_repl/prefetch.py
  date: 2026-02-13

CONTRACT-003: CalibrationFitAPI
  PRE: At least 3 (predicted, observed) samples; predicted in [0, 1], observed in {0,1}
  POST: Returns report with brier_before/brier_after and persisted offset/scale
  INV: Calibrated probabilities are clamped to [0, 1]
  status: active
  owner: src/stateful_repl/calibration.py
  date: 2026-02-13
