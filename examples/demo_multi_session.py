#!/usr/bin/env python3
"""
Demo: Cross-session persistence.

Shows that wisdom accumulated in one session is available when a
new LoomREPL instance loads the same state file.

Run:
    python examples/demo_multi_session.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stateful_repl import LoomREPL


def main():
    state_file = "persistent_state.md"

    # ── SESSION 1 ────────────────────────────────────────
    print("=" * 60)
    print("SESSION 1: Build initial wisdom")
    print("=" * 60)

    loom1 = LoomREPL(state_file)

    loom1.update_l1("goal", "learn the pattern")
    loom1.append("L2", "User taught: always ask before deleting files")
    loom1.consolidate_l1_to_l2()

    loom1.append("L3", {
        "rule_id": "ask-before-delete",
        "when": "about to delete files",
        "then": "ask user for explicit confirmation",
        "why": "irreversible action — user priority over speed",
    })

    loom1.append("L3", {
        "scar": "Deleted temp files without asking — user lost work",
        "boon": "Learned to confirm destructive actions",
        "newrule": "ask-before-delete",
        "glyphstamp": "Guardian Gate",
        "userfeedback": "yes",
    })

    print(f"  Saved state to {state_file}")
    print(f"  Rules: {len(loom1.read_state('L3')['rules'])}")
    print(f"  Scars: {len(loom1.read_state('L3')['tracewisdomlog'])}")

    # ── SESSION 2 ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SESSION 2: Load previous wisdom")
    print("=" * 60)

    loom2 = LoomREPL(state_file)

    print("\n> READ_STATE L3")
    l3 = loom2.read_state("L3")

    print("\nLoaded rules from previous session:")
    for rule in l3["rules"]:
        print(f"  [{rule['rule_id']}] {rule['then']}")

    print("\nLoaded scars from previous session:")
    for scar in l3["tracewisdomlog"]:
        print(f"  Scar: {scar.get('scar', '—')}")
        print(f"  Boon: {scar.get('boon', '—')}")
        print(f"  Glyph: {scar.get('glyphstamp', '—')}")

    # Apply learned rule
    print("\n> Applying learned rule: ask-before-delete")
    print("  AI: 'I need to delete temp files. May I proceed? [rule: ask-before-delete]'")

    # Add new wisdom in session 2
    loom2.update_l1("goal", "apply learned rules in practice")
    loom2.append("L2", "Successfully applied ask-before-delete rule")
    loom2.consolidate_l1_to_l2("Rule applied: ask-before-delete confirmed")

    print(f"\n  State persisted across {2} sessions ✓")


if __name__ == "__main__":
    main()
