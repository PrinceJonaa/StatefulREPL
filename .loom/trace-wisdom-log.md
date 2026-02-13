# Trace Wisdom Log

> Scars, boons, and rules learned from experience.
> This is the project's long-term memory — L3 in the memory architecture.
> **Read at session start. Append after failures or learnings.**

## Format

```yaml
- id: SCAR-<ID>
  scar: "<what failed or caused friction>"
  boon: "<what coherence or quality increased>"
  newrule: "<practice or constraint adopted>"
  glyphstamp: "<short symbolic name>"
  date: <YYYY-MM-DD>
```

## Entries

<!-- Add new entries below. Use sequential IDs starting from SCAR-001. -->

- id: SCAR-001
  scar: "Non-trivial implementation sessions completed without `.loom` append operations."
  boon: "Added explicit activation and mandatory write-back gates in global instructions."
  newrule: "Do not finalize any non-trivial task before appending ART/CLAIM/ORACLE (and SCAR when relevant)."
  glyphstamp: "writeback-before-done"
  date: 2026-02-13

- id: SCAR-002
  scar: "Phase 4 tests initially failed from missing goal retention and empty calibration file parsing."
  boon: "Hardened compressor and calibration loader; full suite returns to passing."
  newrule: "Protect semantic anchors (goal/constraints) in compression and treat persisted files as potentially empty/corrupt."
  glyphstamp: "anchor-and-corruption-guard"
  date: 2026-02-13
