"""Tests for runtime Loom write-back utilities."""

from __future__ import annotations

from pathlib import Path

import pytest


def _seed_loom(tmp_path: Path) -> None:
    loom = tmp_path / ".loom"
    loom.mkdir(parents=True, exist_ok=True)
    (loom / "artifact-registry.md").write_text("# Artifact Registry\n\n## Registry\n", encoding="utf-8")
    (loom / "claim-ledger.md").write_text("# Claim Ledger\n\n## Claims\n", encoding="utf-8")
    (loom / "oracle-matrix.md").write_text("# Oracle Matrix\n\n## Oracles\n", encoding="utf-8")
    (loom / "trace-wisdom-log.md").write_text("# Trace Wisdom Log\n\n## Entries\n", encoding="utf-8")


class TestLoomWriteback:
    """Behavior tests for automatic `.loom` append operations."""

    def test_append_minimum_creates_required_ids(self, tmp_path: Path):
        """append_minimum should generate ART/CLAIM/ORACLE IDs."""
        from stateful_repl.loom_writeback import LoomWriteback, WritebackPacket

        _seed_loom(tmp_path)
        writer = LoomWriteback(str(tmp_path))

        packet = WritebackPacket(
            artifact_name="A",
            artifact_type="code",
            artifact_path="src/a.py",
            claim_statement="A is valid",
            claim_scope="module",
            claim_confidence=0.8,
            claim_falsifies="A fails",
            oracle_name="A oracle",
            oracle_method="test",
            oracle_command="pytest -q",
            oracle_expected="pass",
        )

        ids = writer.append_minimum(packet, owner="test")
        assert ids["artifact_id"] == "ART-001"
        assert ids["claim_id"] == "CLAIM-001"
        assert ids["oracle_id"] == "ORACLE-001"

    def test_append_minimum_with_scar_creates_scar(self, tmp_path: Path):
        """When scar/boon exists, SCAR ID should be created."""
        from stateful_repl.loom_writeback import LoomWriteback, WritebackPacket

        _seed_loom(tmp_path)
        writer = LoomWriteback(str(tmp_path))

        packet = WritebackPacket(
            artifact_name="B",
            artifact_type="test",
            artifact_path="tests/b.py",
            claim_statement="B is valid",
            claim_scope="module",
            claim_confidence=0.9,
            claim_falsifies="B fails",
            oracle_name="B oracle",
            oracle_method="test",
            oracle_command="pytest tests/b.py -q",
            oracle_expected="pass",
            scar="minor failure",
            boon="learned",
            glyphstamp="scar-test",
        )

        ids = writer.append_minimum(packet, owner="test")
        assert ids["scar_id"] == "SCAR-001"

    def test_rejects_invalid_confidence(self, tmp_path: Path):
        """Invalid confidence outside [0,1] should raise ValueError."""
        from stateful_repl.loom_writeback import LoomWriteback, WritebackPacket

        _seed_loom(tmp_path)
        writer = LoomWriteback(str(tmp_path))

        packet = WritebackPacket(
            artifact_name="C",
            artifact_type="code",
            artifact_path="src/c.py",
            claim_statement="C",
            claim_scope="module",
            claim_confidence=1.5,
            claim_falsifies="fail",
            oracle_name="C oracle",
            oracle_method="test",
            oracle_command="pytest -q",
            oracle_expected="pass",
        )

        with pytest.raises(ValueError, match="claim_confidence"):
            writer.append_minimum(packet)

    def test_missing_loom_dir_raises(self, tmp_path: Path):
        """Missing `.loom` directory should raise FileNotFoundError."""
        from stateful_repl.loom_writeback import LoomWriteback, WritebackPacket

        writer = LoomWriteback(str(tmp_path))
        packet = WritebackPacket(
            artifact_name="D",
            artifact_type="code",
            artifact_path="src/d.py",
            claim_statement="D",
            claim_scope="module",
            claim_confidence=0.7,
            claim_falsifies="fail",
            oracle_name="D oracle",
            oracle_method="test",
            oracle_command="pytest -q",
            oracle_expected="pass",
        )

        with pytest.raises(FileNotFoundError, match=".loom directory"):
            writer.append_minimum(packet)
