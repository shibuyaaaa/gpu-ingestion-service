import json
import subprocess
from pathlib import Path

import app.models.allinone as allinone_module
from app.models.allinone import AllInOneRuntime


class FakeAllInOne:
    def __init__(self):
        self.kwargs = None

    def analyze(self, **kwargs):
        self.kwargs = kwargs
        output_dir = Path(kwargs["out_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "result.json").write_text(json.dumps({"segments": []}), encoding="utf-8")
        return "ok"


def test_allinone_analysis_uses_fresh_byproducts_and_overwrite(tmp_path, monkeypatch):
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(b"audio")
    output_dir = tmp_path / "analysis"
    stale_file = output_dir / "byproducts" / "spec" / "stale.npy"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_bytes(b"partial")

    runtime = AllInOneRuntime(model_name="harmonix-fold0", device="cuda:0")
    fake = FakeAllInOne()
    runtime._allin1 = fake
    monkeypatch.setattr(runtime, "_ensure_static_models_link", lambda byproduct_root: None)

    runtime._analyze_sync(audio_path, output_dir)

    assert fake.kwargs is not None
    assert fake.kwargs["overwrite"] is True
    assert fake.kwargs["keep_byproducts"] is False
    assert not stale_file.exists()


def test_memory_bounded_demix_adds_segment_limit(tmp_path, monkeypatch):
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(b"audio")
    demix_dir = tmp_path / "demix"
    captured = {}

    def fake_run(cmd, check):
        captured["cmd"] = cmd
        captured["check"] = check
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(allinone_module.subprocess, "run", fake_run)

    demix_paths = AllInOneRuntime._memory_bounded_demix([audio_path], demix_dir, "cuda:0")

    assert demix_paths == [demix_dir / "htdemucs" / "input"]
    assert captured["check"] is True
    assert "--segment" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--segment") + 1] == "5"
    assert "--jobs" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--jobs") + 1] == "0"
