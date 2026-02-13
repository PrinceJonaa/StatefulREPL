# Artifact Registry

> Tracks all produced outputs with unique IDs and ownership.
> **Read before building. Update after building.**

## Format

```
ART-<ID>: <name>
  type: code | doc | config | test | spec
  path: <relative path from project root>
  owner: <agent name or "human">
  created: <YYYY-MM-DD>
  status: active | superseded | archived
  notes: <optional context>
```

## Registry

<!-- Add new entries below. Use sequential IDs. -->

ART-001: Phase 4 compression module
  type: code
  path: src/stateful_repl/compression.py
  owner: human+copilot
  created: 2026-02-13
  status: active
  notes: Extractive compression for text/state context.

ART-002: Phase 4 prefetch module
  type: code
  path: src/stateful_repl/prefetch.py
  owner: human+copilot
  created: 2026-02-13
  status: active
  notes: Transition+recency predictive prefetch engine.

ART-003: Phase 4 calibration module
  type: code
  path: src/stateful_repl/calibration.py
  owner: human+copilot
  created: 2026-02-13
  status: active
  notes: Affine calibration learning and persisted weights.

ART-004: Phase 4 API endpoints
  type: code
  path: src/stateful_repl/server.py
  owner: human+copilot
  created: 2026-02-13
  status: active
  notes: Adds /context/compress, /prefetch/*, /calibration/*, /events/replay.

ART-005: Phase 4 test suite
  type: test
  path: tests/test_phase4.py
  owner: human+copilot
  created: 2026-02-13
  status: active
  notes: Feature + benchmark coverage for phase 4 slice.

ART-006: Phase 4 measurable quality gates
  type: code
  path: src/stateful_repl/compression.py, src/stateful_repl/prefetch.py, src/stateful_repl/calibration.py, src/stateful_repl/server.py
  owner: human+copilot
  created: 2026-02-13
  status: active
  notes: Adds retention metrics, prefetch hit-rate@k/MRR, and calibration holdout Brier metrics with API endpoints.

ART-007: Runtime Loom write-back automation
  type: code
  path: src/stateful_repl/loom_writeback.py, src/stateful_repl/server.py, src/stateful_repl/cli.py
  owner: human+copilot
  created: 2026-02-13
  status: active
  notes: Introduces packet-based runtime append API and CLI command for ART/CLAIM/ORACLE/SCAR writes.

ART-008: CI Loom guard hardening
  type: config
  path: .github/workflows/loom-writeback-guard.yml
  owner: human+copilot
  created: 2026-02-13
  status: active
  notes: Reduces false positives, adds actionable failure guidance, and supports justified loom:skip override token.
