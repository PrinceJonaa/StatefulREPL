#!/usr/bin/env python3
"""Loom watchdog.

Runs lightweight continuous checks so the project can enforce Loom
principles on a schedule (e.g., GitHub Actions cron).
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path


REQUIRED_FILES = [
    ".loom/artifact-registry.md",
    ".loom/claim-ledger.md",
    ".loom/oracle-matrix.md",
    ".loom/trace-wisdom-log.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Loom coordination health")
    parser.add_argument("--root", default=".", help="Workspace root")
    parser.add_argument(
        "--max-stale-days",
        type=int,
        default=7,
        help="Fail if latest .loom entry is older than this many days",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()

    errors: list[str] = []
    for rel in REQUIRED_FILES:
        path = root / rel
        if not path.exists():
            errors.append(f"Missing required Loom file: {rel}")

    if errors:
        return fail(errors)

    art = read_text(root / ".loom/artifact-registry.md")
    claim = read_text(root / ".loom/claim-ledger.md")
    oracle = read_text(root / ".loom/oracle-matrix.md")
    wisdom = read_text(root / ".loom/trace-wisdom-log.md")

    counts = {
        "ART": len(re.findall(r"\bART-\d{3}\b", art)),
        "CLAIM": len(re.findall(r"\bCLAIM-\d{3}\b", claim)),
        "ORACLE": len(re.findall(r"\bORACLE-\d{3}\b", oracle)),
        "SCAR": len(re.findall(r"\bSCAR-\d{3}\b", wisdom)),
    }

    for key, count in counts.items():
        if count < 1:
            errors.append(f"No {key} entries found")

    latest = latest_date([art, claim, oracle, wisdom])
    if latest is None:
        errors.append("No ISO dates found in Loom ledgers")
    else:
        stale_after = date.today() - timedelta(days=args.max_stale_days)
        if latest < stale_after:
            errors.append(
                f"Loom entries stale: latest={latest.isoformat()} exceeds max-stale-days={args.max_stale_days}"
            )

    if errors:
        return fail(errors)

    print("✅ Loom watchdog healthy")
    print(
        f"ART={counts['ART']} CLAIM={counts['CLAIM']} "
        f"ORACLE={counts['ORACLE']} SCAR={counts['SCAR']}"
    )
    print(f"Latest date: {latest.isoformat() if latest else 'n/a'}")
    return 0


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def latest_date(texts: list[str]) -> date | None:
    found: list[date] = []
    for text in texts:
        for token in re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text):
            try:
                found.append(datetime.strptime(token, "%Y-%m-%d").date())
            except ValueError:
                continue
    return max(found) if found else None


def fail(errors: list[str]) -> int:
    print("❌ Loom watchdog failed")
    for err in errors:
        print(f" - {err}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
