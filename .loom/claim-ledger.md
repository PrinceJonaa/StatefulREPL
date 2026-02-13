# Claim Ledger

> Bounded claims with scope, confidence, and falsification criteria.
> Every important architectural or factual claim must be recorded here.

## Format

```
CLAIM-<ID>: "<statement>"
  scope: local | module | system
  confidence: 0.0–1.0
  falsifies: "<what would prove this wrong>"
  status: active | superseded | falsified
  source: <agent, human, or artifact ID>
  date: <YYYY-MM-DD>
```

## Claims

<!-- Add new entries below. Use sequential IDs. -->

CLAIM-001: "Extractive compression can reduce context size while preserving core task signal for orchestration prompts."
  scope: module
  confidence: 0.82
  falsifies: "Compression outputs repeatedly omit goal/critical constraints in regression tests"
  status: active
  source: ART-001
  date: 2026-02-13

CLAIM-002: "A transition+recency model provides useful first-pass prefetch candidates for session workflows."
  scope: module
  confidence: 0.77
  falsifies: "Top-3 prediction recall remains below 0.4 on representative traces"
  status: active
  source: ART-002
  date: 2026-02-13

CLAIM-003: "Affine post-hoc calibration reduces or maintains Brier score versus uncalibrated confidence."
  scope: module
  confidence: 0.79
  falsifies: "Brier-after is consistently worse than Brier-before on holdout samples"
  status: active
  source: ART-003
  date: 2026-02-13

CLAIM-004: "Event-store replay endpoint can reconstruct meaningful prior state for time-travel debugging."
  scope: system
  confidence: 0.74
  falsifies: "Replay state diverges from expected L1/L2/L3 outcomes in endpoint tests"
  status: active
  source: ART-004
  date: 2026-02-13

CLAIM-005: "Phase 4 quality-gate metrics (retention score, hit-rate@k/MRR, holdout Brier delta) provide actionable pass/fail signals for optimization decisions."
  scope: system
  confidence: 0.81
  falsifies: "Metrics remain uncorrelated with observed task quality/regression outcomes across repeated runs"
  status: active
  source: ART-006
  date: 2026-02-13

CLAIM-006: "Runtime packetized write-back in server/CLI reduces omission risk versus manual `.loom` editing."
  scope: system
  confidence: 0.84
  falsifies: "Writeback endpoint/CLI usage still leaves frequent missing ART/CLAIM/ORACLE updates in merged changes"
  status: active
  source: ART-007
  date: 2026-02-13

CLAIM-007: "A scoped substantive-file matcher plus explicit bypass token reduces Loom-guard false positives without weakening enforcement intent."
  scope: system
  confidence: 0.80
  falsifies: "Guard continues to block routine non-substantive changes at similar rate or allows substantive changes without Loom updates"
  status: active
  source: ART-008
  date: 2026-02-13

CLAIM-008: "Loom guard + runtime writeback are operational in current workspace."
  scope: system
  confidence: 0.90
  falsifies: "Writeback IDs are not appended or guard workflow is missing from repo."
  status: active
  source: ART-009
  date: 2026-02-13

CLAIM-009: "Scheduled watchdog checks provide ongoing 24x7 enforcement signal for Loom ledger health in Copilot/GitHub workflows."
  scope: system
  confidence: 0.82
  falsifies: "Watchdog fails to detect missing/stale Loom entries or does not alert via issue workflow"
  status: active
  source: ART-010
  date: 2026-02-13
