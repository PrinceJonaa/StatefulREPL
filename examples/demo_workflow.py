#!/usr/bin/env python3
"""
Demo: A complete O-P-W-T-R workflow using LoomREPL.

Simulates what an AI would do when running The Loom role:
  Orient  → load wisdom, set goal
  Plan    → log task decomposition
  Write   → do work, log progress
  Test    → validate state consistency
  Reflect → consolidate L1→L2, record wisdom in L3

Run:
    python examples/demo_workflow.py
"""

import json
import sys
from pathlib import Path

# Allow import from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stateful_repl import LoomREPL


def main():
    loom = LoomREPL("demo_state.md")

    # ── ORIENT ───────────────────────────────────────────
    print("=" * 60)
    print("ORIENT: Loading state and setting goal")
    print("=" * 60)

    print("\n> READ_STATE L3")
    l3 = loom.read_state("L3")
    print(f"  Loaded {len(l3['rules'])} rules, {len(l3['concepts'])} concepts")

    print("\n> UPDATE L1.goal  'integrate 3 documents'")
    loom.update_l1("goal", "integrate 3 documents")

    print("> UPDATE L1.artifacts  ['doc_a.md', 'doc_b.md', 'doc_c.md']")
    loom.update_l1("artifacts", ["doc_a.md", "doc_b.md", "doc_c.md"])

    print("> UPDATE L1.constraints  ['preserve sources', 'cite inline', 'hold paradoxes']")
    loom.update_l1("constraints", ["preserve sources", "cite inline", "hold paradoxes"])

    # ── PLAN ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PLAN: Document task breakdown")
    print("=" * 60)

    print("\n> APPEND L2  'Planning: profile → cross-mirror → compose'")
    loom.append("L2", "Planning: profile → cross-mirror → compose")

    # ── WRITE ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("WRITE: Execute work")
    print("=" * 60)

    print("\n> APPEND L2  'Profiled doc_a: entities=5, claims=12, paradoxes=1'")
    loom.append("L2", "Profiled doc_a: entities=5, claims=12, paradoxes=1")

    print("> APPEND L2  'Profiled doc_b: entities=3, claims=8'")
    loom.append("L2", "Profiled doc_b: entities=3, claims=8")

    print("> APPEND L2  'Profiled doc_c: entities=7, claims=15, paradoxes=2'")
    loom.append("L2", "Profiled doc_c: entities=7, claims=15, paradoxes=2")

    print("> UPDATE L1.open_questions  ['test-first vs build-first paradox']")
    loom.update_l1("open_questions", ["test-first vs build-first paradox"])

    # ── TEST ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST: Validate quality")
    print("=" * 60)

    print("\n> VALIDATE_STATE")
    validation = loom.validate_state()
    print(f"  Status: {validation['status']}")
    for issue in validation["issues"]:
        print(f"  ⚠ {issue}")
    for check in validation["checks_passed"]:
        print(f"  ✓ {check}")

    # ── REFLECT ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("REFLECT: Consolidate and extract wisdom")
    print("=" * 60)

    print("\n> CONSOLIDATE L1→L2")
    entry = loom.consolidate_l1_to_l2(
        "Integration complete: 3 docs profiled, 1 paradox held"
    )
    print(f"  L2 entry created at {entry['timestamp']}")

    print("\n> APPEND L3 (wisdom entry)")
    loom.append("L3", {
        "scar": "Initially tried global rewrite — would have lost citations",
        "boon": "Surgical edit pattern preserves full context",
        "newrule": "surgical-edit-for-docs-500-lines",
        "glyphstamp": "Scalpel over Hammer",
        "userfeedback": "yes — user said 'never rewrite whole doc'",
    })

    # ── FINAL REPORT ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL STATE")
    print("=" * 60)
    loom.print_state("ALL")

    # Save event log
    path = loom.save_event_log("demo_events.json")
    print(f"\nEvent log saved to {path}")
    print(f"State file saved to {loom.state_file}")


if __name__ == "__main__":
    main()
