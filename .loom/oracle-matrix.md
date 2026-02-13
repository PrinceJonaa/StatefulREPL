# Oracle Matrix

> Verification tests linked to claims and contracts.
> An oracle defines how to know if something is true or working.

## Format

```
ORACLE-<ID>: <name>
  verifies: <CLAIM-ID or CONTRACT-ID>
  method: test | manual-check | consistency-check | benchmark
  command: <how to run it, if applicable>
  expected: <what the passing result looks like>
  last_run: <YYYY-MM-DD or "never">
  result: pass | fail | not-run
  notes: <optional context>
```

## Oracles

<!-- Add new entries below. Use sequential IDs. -->

ORACLE-001: Compression goal retention
  verifies: CLAIM-001
  method: test
  command: pytest tests/test_phase4.py::TestCompression::test_compress_state_includes_goal_context -q
  expected: test passes and compressed output contains goal context
  last_run: 2026-02-13
  result: pass
  notes: Confirms safety check that re-injects goal when omitted.

ORACLE-002: Prefetch prediction quality
  verifies: CLAIM-002
  method: test
  command: pytest tests/test_phase4.py::TestPrefetch::test_predict_next_from_sequence -q
  expected: top prediction for anchor 'goal' is 'constraints'
  last_run: 2026-02-13
  result: pass
  notes: Validates transition-learning behavior.

ORACLE-003: Calibration brier sanity
  verifies: CLAIM-003
  method: test
  command: pytest tests/test_phase4.py::TestCalibration::test_fit_updates_parameters_and_improves_brier -q
  expected: report generated with finite brier metrics and persisted params
  last_run: 2026-02-13
  result: pass
  notes: Loader hardened for empty/invalid calibration file.

ORACLE-004: Time-travel replay endpoint
  verifies: CLAIM-004
  method: test
  command: pytest tests/test_phase4.py::TestPhase4ServerEndpoints::test_events_replay -q
  expected: endpoint returns reconstructed state payload with L1/L2/L3 shape
  last_run: 2026-02-13
  result: pass
  notes: Server now records state mutations into event store.

ORACLE-005: Full regression suite
  verifies: CONTRACT-001
  method: test
  command: pytest tests/ -q
  expected: complete suite passes
  last_run: 2026-02-13
  result: pass
  notes: 241 tests passing.
