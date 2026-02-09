"""
LoomREPL: Stateful memory system with L1/L2/L3 layering.

Phase 1 MVP — File-based persistence.

Memory Architecture:
  L1 (Working Pad) — Current turn: goal, constraints, artifacts, open questions.
  L2 (Session Log) — Conversation arc: timestamped entries, consolidation snapshots.
  L3 (Wisdom Base) — Cross-session: rules, concepts, tracewisdomlog (scar/boon/newrule).

Consolidation Flow:
  L1 → L2: Compress working context into session history, then clear L1.
  L2 → L3: Extract recurring patterns into permanent rules, archive L2.

Persistence:
  State is rendered to Markdown (loom_state.md) for human readability.
  Events are logged to JSON for audit trail and future replay.
"""

import json
import os
import yaml
from datetime import datetime
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────

@dataclass
class StateEvent:
    """Immutable record of a state mutation. Forms the append-only event log."""

    timestamp: str
    layer: str        # "L1", "L2", "L3"
    operation: str    # "update.goal", "append", "consolidate_from_l1", etc.
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _empty_l1() -> Dict[str, Any]:
    return {
        "goal": "",
        "constraints": [],
        "artifacts": [],
        "open_questions": [],
    }


def _empty_state(session_id: str) -> Dict[str, Any]:
    now = datetime.now().isoformat()
    return {
        "metadata": {
            "created": now,
            "last_modified": now,
            "session_id": session_id,
        },
        "L1": _empty_l1(),
        "L2": [],
        "L3": {
            "rules": [],
            "concepts": [],
            "tracewisdomlog": [],
        },
    }


# ─────────────────────────────────────────────────────────
# Core REPL
# ─────────────────────────────────────────────────────────

class LoomREPL:
    """
    Stateful REPL for AI memory management.

    Implements the L1 (working) → L2 (session) → L3 (wisdom) consolidation
    pipeline that The Loom role prompt describes.
    """

    def __init__(
        self,
        state_file: str = "loom_state.md",
        enable_events: bool = True,
    ):
        self.state_file = Path(state_file)
        self.event_log: List[StateEvent] = []
        self.enable_events = enable_events

        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.state = _empty_state(session_id)

        if self.state_file.exists():
            self.load_state()
        else:
            self.save_state()

    # ── helpers ──────────────────────────────────────────

    def _emit(self, layer: str, operation: str, data: Dict[str, Any]) -> None:
        if not self.enable_events:
            return
        self.event_log.append(
            StateEvent(
                timestamp=datetime.now().isoformat(),
                layer=layer,
                operation=operation,
                data=data,
            )
        )

    # ── READ ─────────────────────────────────────────────

    def read_state(self, layer: Optional[str] = None) -> Dict[str, Any]:
        """
        READ_STATE <layer>

        Returns the requested memory layer, or the full state if layer is
        None / "ALL".

        Valid layers: L1, L2, L3, METADATA, ALL (or None).
        """
        if layer is None or layer == "ALL":
            return self.state
        if layer == "L1":
            return self.state["L1"]
        if layer == "L2":
            return {"L2": self.state["L2"]}
        if layer == "L3":
            return self.state["L3"]
        if layer == "METADATA":
            return self.state["metadata"]
        raise ValueError(
            f"Invalid layer: {layer!r}. Must be L1, L2, L3, METADATA, or ALL."
        )

    def print_state(self, layer: Optional[str] = None) -> None:
        """Pretty-print the requested layer to stdout."""
        print(json.dumps(self.read_state(layer), indent=2, default=str))

    # ── WRITE ────────────────────────────────────────────

    def update_l1(self, field_name: str, value: Any) -> None:
        """
        UPDATE L1.<field> <value>

        Accepted fields: goal, constraints, artifacts, open_questions.
        """
        if field_name not in self.state["L1"]:
            raise ValueError(
                f"Invalid L1 field: {field_name!r}. "
                f"Must be one of: {list(self.state['L1'].keys())}"
            )
        old = self.state["L1"][field_name]
        self.state["L1"][field_name] = value
        self._emit("L1", f"update.{field_name}", {"old": old, "new": value})
        self.save_state()

    def append(self, layer: str, content: Any) -> None:
        """
        APPEND <layer> <content>

        L2 accepts any JSON-serialisable value.
        L3 accepts a dict with one of these key sets:
          - rule_id / when / then / why   → appended to L3.rules
          - concept / definition          → appended to L3.concepts
          - scar / boon                   → appended to L3.tracewisdomlog
        """
        if layer == "L1":
            raise ValueError(
                "L1 requires UPDATE, not APPEND. Use update_l1() instead."
            )

        if layer == "L2":
            entry = {
                "timestamp": datetime.now().isoformat(),
                "content": content,
            }
            self.state["L2"].append(entry)
            self._emit("L2", "append", entry)

        elif layer == "L3":
            if not isinstance(content, dict):
                raise TypeError("L3 content must be a dict.")
            if "rule_id" in content:
                content.setdefault("created", datetime.now().isoformat())
                self.state["L3"]["rules"].append(content)
            elif "concept" in content:
                self.state["L3"]["concepts"].append(content)
            elif "scar" in content or "boon" in content:
                content.setdefault("timestamp", datetime.now().isoformat())
                self.state["L3"]["tracewisdomlog"].append(content)
            else:
                raise ValueError(
                    "L3 dict must contain 'rule_id', 'concept', or 'scar'/'boon'."
                )
            self._emit("L3", "append", content)

        else:
            raise ValueError(f"Invalid layer: {layer!r}")

        self.save_state()

    def clear_l1(self) -> None:
        """Reset L1 working pad to blank state."""
        self.state["L1"] = _empty_l1()
        self._emit("L1", "clear", {})
        self.save_state()

    # ── CONSOLIDATION ────────────────────────────────────

    def consolidate_l1_to_l2(self, summary: Optional[str] = None) -> dict:
        """
        CONSOLIDATE L1→L2

        Snapshots L1 into L2 session log, then clears L1.
        Returns the L2 entry that was created.
        """
        if summary is None:
            summary = self._summarize_l1()

        entry = {
            "timestamp": datetime.now().isoformat(),
            "phase": "consolidation",
            "summary": summary,
            "l1_snapshot": {k: v for k, v in self.state["L1"].items()},
        }
        self.state["L2"].append(entry)
        self._emit("L2", "consolidate_from_l1", entry)
        self.clear_l1()
        return entry

    def consolidate_l2_to_l3(self, extract_rules: bool = True) -> dict:
        """
        CONSOLIDATE L2→L3

        Optionally extracts recurring patterns from L2 into L3 rules,
        then archives L2 to a JSON file and clears it.
        """
        if not self.state["L2"]:
            return {"message": "L2 is empty, nothing to consolidate."}

        extracted = {"rules_created": 0, "concepts_created": 0}

        if extract_rules:
            for pattern in self._extract_patterns():
                rule = {
                    "rule_id": pattern["rule_id"],
                    "when": pattern["trigger"],
                    "then": pattern["action"],
                    "why": pattern["reason"],
                    "created": datetime.now().isoformat(),
                }
                self.state["L3"]["rules"].append(rule)
                extracted["rules_created"] += 1

        self._archive_l2()
        self._emit("L3", "consolidate_from_l2", extracted)
        self.save_state()
        return extracted

    def _summarize_l1(self) -> str:
        parts: List[str] = []
        if self.state["L1"]["goal"]:
            parts.append(f"Goal: {self.state['L1']['goal']}")
        n_art = len(self.state["L1"]["artifacts"])
        if n_art:
            parts.append(f"Artifacts: {n_art}")
        n_q = len(self.state["L1"]["open_questions"])
        if n_q:
            parts.append(f"Open questions: {n_q}")
        return " | ".join(parts) if parts else "Work completed"

    def _extract_patterns(self) -> List[Dict[str, Any]]:
        """Detect repeated actions in L2 (simplified heuristic)."""
        counts: Dict[str, int] = {}
        for entry in self.state["L2"]:
            key = str(entry.get("summary") or entry.get("content", ""))
            counts[key] = counts.get(key, 0) + 1
        return [
            {
                "rule_id": f"pattern_{i}",
                "trigger": "repeated_action",
                "action": action,
                "reason": f"Occurred {count} times in session",
            }
            for i, (action, count) in enumerate(counts.items(), 1)
            if count >= 3
        ]

    def _archive_l2(self) -> None:
        archive_dir = self.state_file.parent / "loom_archives"
        archive_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_file = archive_dir / f"L2_{ts}.json"
        with open(archive_file, "w", encoding="utf-8") as f:
            json.dump(self.state["L2"], f, indent=2, default=str)
        self.state["L2"] = []

    # ── VALIDATION ───────────────────────────────────────

    def validate_state(self) -> Dict[str, Any]:
        """
        VALIDATE_STATE

        Runs internal consistency checks and returns a report.
        """
        issues: List[str] = []

        # L1: goal should exist if other fields are populated
        l1 = self.state["L1"]
        if not l1["goal"] and any(
            [l1["constraints"], l1["artifacts"], l1["open_questions"]]
        ):
            issues.append("L1: Has constraints/artifacts/questions but no goal.")

        # L2: warn if too many un-consolidated entries
        n_l2 = len(self.state["L2"])
        if n_l2 > 20:
            issues.append(
                f"L2: {n_l2} entries — consider consolidating to L3."
            )

        # L3: rules should have justifications
        for rule in self.state["L3"]["rules"]:
            if not rule.get("why"):
                issues.append(
                    f"L3: Rule '{rule.get('rule_id')}' has no 'why' justification."
                )

        # L3: scars referencing non-existent rules
        scar_rules = {
            s.get("newrule")
            for s in self.state["L3"]["tracewisdomlog"]
            if s.get("newrule")
        }
        actual_rules = {r.get("rule_id") for r in self.state["L3"]["rules"]}
        orphaned = scar_rules - actual_rules
        if orphaned:
            issues.append(f"L3: Scars reference non-existent rules: {orphaned}")

        return {
            "status": "VALID" if not issues else "WARNINGS",
            "issues": issues,
            "checks_passed": [
                "L1 structure intact",
                "L2 is list of dicts",
                "L3 sections exist",
            ],
        }

    # ── PERSISTENCE ──────────────────────────────────────

    def save_state(self) -> None:
        """Render state to Markdown and write to disk."""
        self.state["metadata"]["last_modified"] = datetime.now().isoformat()
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as f:
            f.write(self._render_markdown())

    def load_state(self) -> None:
        """Parse Markdown state file back into dict."""
        with open(self.state_file, "r", encoding="utf-8") as f:
            self.state = self._parse_markdown(f.read())

    def _render_markdown(self) -> str:
        meta = self.state["metadata"]
        l1 = self.state["L1"]
        l2 = self.state["L2"]
        l3 = self.state["L3"]

        lines = [
            f"# Loom State — {meta['session_id']}",
            "",
            f"**Created:** {meta['created']}  ",
            f"**Last Modified:** {meta['last_modified']}",
            "",
            "---",
            "",
            "## L1: Working Pad",
            "",
            f"**Goal:** {l1['goal']}  ",
            f"**Constraints:** {json.dumps(l1['constraints'])}  ",
            f"**Artifacts:** {json.dumps(l1['artifacts'])}  ",
            f"**Open Questions:** {json.dumps(l1['open_questions'])}",
            "",
            "---",
            "",
            "## L2: Session Log",
            "",
        ]

        if not l2:
            lines.append("*No entries yet.*")
            lines.append("")
        else:
            for entry in l2:
                lines.append(f"### {entry.get('timestamp', 'unknown')}")
                if "phase" in entry:
                    lines.append(f"**Phase:** {entry['phase']}")
                if "summary" in entry:
                    lines.append(f"**Summary:** {entry['summary']}")
                if "content" in entry:
                    lines.append(f"**Content:** {entry['content']}")
                lines.append("")

        lines += [
            "---",
            "",
            "## L3: Wisdom Base",
            "",
            "### Rules",
            "",
        ]

        if not l3["rules"]:
            lines.append("*No rules yet.*")
            lines.append("")
        else:
            for rule in l3["rules"]:
                lines.append(f"- **{rule.get('rule_id', 'unnamed')}**")
                lines.append(f"  - When: {rule.get('when', '')}")
                lines.append(f"  - Then: {rule.get('then', '')}")
                lines.append(f"  - Why: {rule.get('why', '')}")
                lines.append("")

        lines += [
            "### Concepts",
            "",
        ]

        if not l3["concepts"]:
            lines.append("*No concepts yet.*")
            lines.append("")
        else:
            for concept in l3["concepts"]:
                lines.append(f"- **{concept.get('concept', 'Unnamed')}**")
                lines.append(f"  - Definition: {concept.get('definition', '')}")
                lines.append(f"  - Relations: {concept.get('relations', '')}")
                lines.append(f"  - Evidence: {concept.get('evidence', '')}")
                lines.append("")

        lines += [
            "### Tracewisdomlog",
            "",
            "```yaml",
            yaml.dump(
                l3["tracewisdomlog"],
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            ).rstrip(),
            "```",
            "",
        ]

        return "\n".join(lines)

    def _parse_markdown(self, content: str) -> Dict[str, Any]:
        """
        Simple section-based Markdown parser.

        Recovers L1 fields, L2 entries (basic), and L3 tracewisdomlog from
        the YAML block. Production version should use a proper AST parser.
        """
        state = _empty_state(session_id="recovered")

        current_section: Optional[str] = None
        yaml_buffer: List[str] = []
        in_yaml = False

        # Recover metadata from header
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("# Loom State"):
                parts = stripped.split("—", 1)
                if len(parts) > 1:
                    state["metadata"]["session_id"] = parts[1].strip()
            elif stripped.startswith("**Created:**"):
                state["metadata"]["created"] = (
                    stripped.split("**Created:**")[1].strip().rstrip(" ")
                )
            elif stripped.startswith("**Last Modified:**"):
                state["metadata"]["last_modified"] = (
                    stripped.split("**Last Modified:**")[1].strip()
                )

        # Parse sections
        for line in content.split("\n"):
            stripped = line.strip()

            # YAML block handling
            if stripped == "```yaml":
                in_yaml = True
                yaml_buffer = []
                continue
            if stripped == "```" and in_yaml:
                in_yaml = False
                raw = "\n".join(yaml_buffer)
                if raw.strip():
                    parsed = yaml.safe_load(raw)
                    if isinstance(parsed, list):
                        state["L3"]["tracewisdomlog"] = parsed
                yaml_buffer = []
                continue
            if in_yaml:
                yaml_buffer.append(line)
                continue

            # Section detection
            if stripped.startswith("## L1"):
                current_section = "L1"
                continue
            if stripped.startswith("## L2"):
                current_section = "L2"
                continue
            if stripped.startswith("## L3"):
                current_section = "L3"
                continue

            # L1 field parsing
            if current_section == "L1":
                for key in ("Goal", "Constraints", "Artifacts", "Open Questions"):
                    field_name = key.lower().replace(" ", "_")
                    prefix = f"**{key}:**"
                    if stripped.startswith(prefix):
                        raw_val = stripped[len(prefix):].strip()
                        if field_name == "goal":
                            state["L1"]["goal"] = raw_val
                        else:
                            try:
                                state["L1"][field_name] = json.loads(raw_val)
                            except (json.JSONDecodeError, ValueError):
                                pass

            # L3 rule parsing
            if current_section == "L3" and stripped.startswith("- **"):
                # Lightweight rule detection (full parser in Phase 2)
                pass

        return state

    # ── EVENT LOG ────────────────────────────────────────

    def get_event_log(self) -> List[dict]:
        """Return all events from the current session as dicts."""
        return [e.to_dict() for e in self.event_log]

    def save_event_log(self, filepath: Optional[str] = None) -> str:
        """Write event log to JSON; returns the path used."""
        if filepath is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"loom_events_{ts}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.get_event_log(), f, indent=2, default=str)
        return filepath
