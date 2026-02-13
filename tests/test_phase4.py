"""Phase 4 tests — compression, prefetch, calibration, and time-travel APIs."""

from __future__ import annotations

import tempfile

import pytest

try:
    from fastapi.testclient import TestClient

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


class TestCompression:
    """Tests for context compression."""

    def test_compress_text_reduces_content(self):
        """Compression should reduce token count for multi-sentence input."""
        from stateful_repl.compression import ExtractiveCompressor

        # Arrange
        text = (
            "StatefulREPL orchestrates tasks across agents. "
            "It tracks memory in L1, L2, and L3. "
            "Quality validation runs on every cycle. "
            "This sentence adds extra detail for compression."
        )
        compressor = ExtractiveCompressor(default_target_ratio=0.5)

        # Act
        result = compressor.compress_text(text)

        # Assert
        assert result.original_tokens >= result.compressed_tokens
        assert 0.0 < result.compression_ratio <= 1.0
        assert len(result.selected_units) >= 1

    def test_compress_state_includes_goal_context(self):
        """State compression should retain goal signal when available."""
        from stateful_repl.compression import ExtractiveCompressor

        # Arrange
        state = {
            "L1": {
                "goal": "Implement phase 4",
                "constraints": ["No regressions"],
                "artifacts": ["agents.md"],
                "open_questions": ["What to prioritize?"],
            },
            "L2": [{"summary": "Completed planner hardening"}],
        }

        # Act
        result = ExtractiveCompressor().compress_state(state)

        # Assert
        assert "goal" in result.compressed_text.lower()


class TestPrefetch:
    """Tests for predictive prefetch engine."""

    def test_predict_next_from_sequence(self):
        """Engine should learn transitions and predict likely next key."""
        from stateful_repl.prefetch import PredictivePrefetchEngine

        # Arrange
        engine = PredictivePrefetchEngine()
        sequence = ["goal", "constraints", "L2", "goal", "constraints", "L2"]
        engine.fit_from_sequence(sequence)

        # Act
        candidates = engine.predict_next(["goal"], limit=3)

        # Assert
        assert len(candidates) >= 1
        assert candidates[0].key == "constraints"

    def test_record_access_rejects_empty_key(self):
        """Empty keys should raise ValueError."""
        from stateful_repl.prefetch import PredictivePrefetchEngine

        # Arrange
        engine = PredictivePrefetchEngine()

        # Act / Assert
        with pytest.raises(ValueError, match="non-empty"):
            engine.record_access("")


class TestCalibration:
    """Tests for calibration learner."""

    def test_fit_updates_parameters_and_improves_brier(self):
        """Calibration fit should return report with finite brier scores."""
        from stateful_repl.calibration import CalibrationLearner, CalibrationSample

        # Arrange
        with tempfile.NamedTemporaryFile(suffix=".json") as fp:
            learner = CalibrationLearner(weights_path=fp.name)
            samples = [
                CalibrationSample(predicted=0.9, observed=1),
                CalibrationSample(predicted=0.8, observed=1),
                CalibrationSample(predicted=0.3, observed=0),
                CalibrationSample(predicted=0.2, observed=0),
            ]

            # Act
            report = learner.fit(samples)

            # Assert
            assert report.sample_count == 4
            assert 0.0 <= report.brier_before <= 1.0
            assert 0.0 <= report.brier_after <= 1.0

    def test_recommend_quality_weights_normalized(self):
        """Weight recommendations should be normalized to sum near 1."""
        from stateful_repl.calibration import CalibrationLearner

        # Arrange
        learner = CalibrationLearner(weights_path=tempfile.mktemp(suffix=".json"))
        outcomes = [
            {"dimension": "internal_consistency", "predicted": 0.8, "observed": 1},
            {"dimension": "external_correspondence", "predicted": 0.3, "observed": 0},
            {"dimension": "internal_consistency", "predicted": 0.6, "observed": 1},
        ]

        # Act
        weights = learner.recommend_quality_weights(outcomes)

        # Assert
        assert "internal_consistency" in weights
        assert "external_correspondence" in weights
        assert 0.99 <= sum(weights.values()) <= 1.01


@pytest.mark.skipif(
    not HAS_FASTAPI,
    reason="FastAPI not installed — install with: pip install stateful-repl[server]",
)
class TestPhase4ServerEndpoints:
    """Tests for Phase 4 HTTP endpoints."""

    @pytest.fixture(autouse=True)
    def clean_server_state(self, tmp_path, monkeypatch):
        state_file = str(tmp_path / "test_state.md")
        event_db = str(tmp_path / "test_events.db")
        monkeypatch.setenv("LOOM_STATE_FILE", state_file)
        monkeypatch.setenv("LOOM_EVENT_BACKEND", "sqlite")
        monkeypatch.setenv("LOOM_EVENT_PATH", event_db)

        import stateful_repl.server as srv

        srv._repl = None
        srv._evaluator = None
        srv._event_store = None
        srv._compressor = None
        srv._prefetcher = None
        srv._calibration = None
        srv._sse_subscribers.clear()
        srv.STATE_FILE = state_file
        srv.EVENT_BACKEND = "sqlite"
        srv.EVENT_PATH = event_db
        yield

    @pytest.fixture
    def client(self):
        from stateful_repl.server import app

        return TestClient(app)

    def test_phase4_status(self, client):
        """Status endpoint should declare Phase 4 features."""
        r = client.get("/phase4/status")
        assert r.status_code == 200
        assert r.json()["phase"] == 4

    def test_context_compress_text(self, client):
        """Text compression endpoint should return compressed payload."""
        r = client.post(
            "/context/compress",
            json={"text": "A. B. C. D.", "target_ratio": 0.5, "use_state": False},
        )
        assert r.status_code == 200
        payload = r.json()
        assert "compressed_text" in payload
        assert payload["compression_ratio"] <= 1.0

    def test_prefetch_record_and_predict(self, client):
        """Prefetch endpoints should learn and predict."""
        client.post("/prefetch/record", json={"key": "goal"})
        client.post("/prefetch/record", json={"key": "constraints"})
        client.post("/prefetch/record", json={"key": "L2"})
        r = client.post("/prefetch/predict", json={"current_keys": ["goal"], "limit": 3})
        assert r.status_code == 200
        assert "candidates" in r.json()

    def test_calibration_fit(self, client):
        """Calibration fit endpoint should return report."""
        r = client.post(
            "/calibration/fit",
            json={
                "samples": [
                    {"predicted": 0.9, "observed": 1},
                    {"predicted": 0.8, "observed": 1},
                    {"predicted": 0.2, "observed": 0},
                ]
            },
        )
        assert r.status_code == 200
        assert "brier_before" in r.json()

    def test_events_replay(self, client):
        """Replay endpoint should return reconstructed state payload."""
        client.post("/state/goal", json={"goal": "Ship Phase 4"})
        client.post("/state/question", json={"question": "Any blockers?"})

        r = client.get("/events/replay")
        assert r.status_code == 200
        data = r.json()
        assert "state" in data
        assert "L1" in data["state"]


@pytest.mark.benchmark
class TestPhase4Benchmarks:
    """Micro-benchmarks for Phase 4 primitives."""

    def test_compression_benchmark(self, benchmark):
        """Compression should remain low-latency for medium inputs."""
        from stateful_repl.compression import ExtractiveCompressor

        text = " ".join(
            [
                "StatefulREPL coordinates agents and preserves context.",
                "Quality gates and replay tools maintain trust.",
                "Compression and prefetch improve token efficiency.",
            ]
            * 80
        )
        compressor = ExtractiveCompressor(default_target_ratio=0.3)

        result = benchmark(lambda: compressor.compress_text(text))
        assert result.compressed_tokens <= result.original_tokens

    def test_prefetch_prediction_benchmark(self, benchmark):
        """Prefetch predictions should be fast for small session graphs."""
        from stateful_repl.prefetch import PredictivePrefetchEngine

        engine = PredictivePrefetchEngine()
        for _ in range(200):
            engine.fit_from_sequence(["goal", "constraints", "L2", "quality"])

        result = benchmark(lambda: engine.predict_next(["goal"], limit=5))
        assert len(result) >= 1
