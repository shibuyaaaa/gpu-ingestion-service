import json
import subprocess
import threading
import time
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
        demix_dir = Path(kwargs["demix_dir"]) / "htdemucs_ft" / "input"
        demix_dir.mkdir(parents=True, exist_ok=True)
        for stem in ("bass", "drums", "other", "vocals"):
            (demix_dir / f"{stem}.wav").write_bytes(stem.encode())
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
    monkeypatch.setenv("ALL_IN_ONE_DEMUCS_MODEL", "htdemucs_ft")
    demix_dir = output_dir / "byproducts" / "demix" / "htdemucs_ft" / "input"

    result = runtime._analyze_sync(audio_path, output_dir)

    assert fake.kwargs is not None
    assert fake.kwargs["overwrite"] is True
    assert fake.kwargs["keep_byproducts"] is True
    assert not stale_file.exists()
    assert result["stem_paths"] == {
        "bass": str(demix_dir / "bass.wav"),
        "drums": str(demix_dir / "drums.wav"),
        "other": str(demix_dir / "other.wav"),
        "vocals": str(demix_dir / "vocals.wav"),
    }
    assert result["timings"]["dry_run"] is False
    assert result["timings"]["allin1_analyze_seconds"] >= 0
    assert result["timings"]["stem_count"] == 4
    assert runtime.status()["last_timings"] == result["timings"]


def test_find_demix_stems_returns_only_existing_stems(tmp_path, monkeypatch):
    monkeypatch.setenv("ALL_IN_ONE_DEMUCS_MODEL", "htdemucs_ft")
    byproducts = tmp_path / "byproducts"
    stem_dir = byproducts / "demix" / "htdemucs_ft" / "song"
    stem_dir.mkdir(parents=True)
    (stem_dir / "other.wav").write_bytes(b"other")
    (stem_dir / "vocals.wav").write_bytes(b"vocals")

    stems = AllInOneRuntime._find_demix_stems(byproducts, tmp_path / "song.wav")

    assert stems == {
        "other": str(stem_dir / "other.wav"),
        "vocals": str(stem_dir / "vocals.wav"),
    }


def test_memory_bounded_demix_adds_segment_limit(tmp_path, monkeypatch):
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(b"audio")
    demix_dir = tmp_path / "demix"
    captured = {}
    monkeypatch.setenv("ALL_IN_ONE_DEMUCS_MODEL", "htdemucs_ft")
    monkeypatch.setenv("ALL_IN_ONE_DEMUCS_BACKEND", "cli")

    def fake_run(cmd, check):
        captured["cmd"] = cmd
        captured["check"] = check
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(allinone_module.subprocess, "run", fake_run)
    allinone_module._ANALYZE_LOCAL.timings = {}

    try:
        demix_paths = AllInOneRuntime._memory_bounded_demix([audio_path], demix_dir, "cuda:0")
        timings = dict(allinone_module._ANALYZE_LOCAL.timings)
    finally:
        allinone_module._ANALYZE_LOCAL.timings = None

    assert demix_paths == [demix_dir / "htdemucs_ft" / "input"]
    assert captured["check"] is True
    assert captured["cmd"][captured["cmd"].index("--name") + 1] == "htdemucs_ft"
    assert captured["cmd"][captured["cmd"].index("-n") + 1] == "htdemucs_ft"
    assert "--repo" not in captured["cmd"]
    assert "--segment" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--segment") + 1] == "7.5"
    assert "--jobs" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--jobs") + 1] == "0"
    assert timings["demix_backend"] == "cli"
    assert timings["demix_pending_tracks"] == 1
    assert timings["demix_cli_seconds"] >= 0


def test_memory_bounded_demix_allows_native_segment_default(tmp_path, monkeypatch):
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(b"audio")
    demix_dir = tmp_path / "demix"
    captured = {}
    monkeypatch.setenv("ALL_IN_ONE_DEMUCS_MODEL", "htdemucs_ft")
    monkeypatch.setenv("ALL_IN_ONE_DEMUCS_BACKEND", "cli")
    monkeypatch.setenv("ALL_IN_ONE_DEMUCS_SEGMENT_SECONDS", "")

    def fake_run(cmd, check):
        captured["cmd"] = cmd
        captured["check"] = check
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(allinone_module.subprocess, "run", fake_run)
    allinone_module._ANALYZE_LOCAL.timings = {}

    try:
        AllInOneRuntime._memory_bounded_demix([audio_path], demix_dir, "cuda:0")
    finally:
        allinone_module._ANALYZE_LOCAL.timings = None

    assert captured["check"] is True
    assert "--segment" not in captured["cmd"]


def test_memory_bounded_demix_clamps_unsafe_segment_limit(tmp_path, monkeypatch):
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(b"audio")
    demix_dir = tmp_path / "demix"
    captured = {}
    monkeypatch.setenv("ALL_IN_ONE_DEMUCS_MODEL", "htdemucs_ft")
    monkeypatch.setenv("ALL_IN_ONE_DEMUCS_BACKEND", "cli")
    monkeypatch.setenv("ALL_IN_ONE_DEMUCS_SEGMENT_SECONDS", "15")

    def fake_run(cmd, check):
        captured["cmd"] = cmd
        captured["check"] = check
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(allinone_module.subprocess, "run", fake_run)
    allinone_module._ANALYZE_LOCAL.timings = {}

    try:
        AllInOneRuntime._memory_bounded_demix([audio_path], demix_dir, "cuda:0")
        timings = dict(allinone_module._ANALYZE_LOCAL.timings)
    finally:
        allinone_module._ANALYZE_LOCAL.timings = None

    assert captured["check"] is True
    assert captured["cmd"][captured["cmd"].index("--segment") + 1] == "7.5"
    assert timings["demix_segment_seconds"] == 7.5
    assert timings["demix_segment_configured_seconds"] == 15.0
    assert timings["demix_segment_max_seconds"] == 7.5
    assert timings["demix_segment_clamped"] is True


def test_memory_bounded_demix_uses_static_repo_when_model_yaml_exists(tmp_path, monkeypatch):
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(b"audio")
    demix_dir = tmp_path / "demix"
    static_models = tmp_path / "static_models"
    static_models.mkdir()
    (static_models / "htdemucs.yaml").write_text("model: htdemucs", encoding="utf-8")
    captured = {}
    monkeypatch.setenv("ALL_IN_ONE_DEMUCS_MODEL", "htdemucs")
    monkeypatch.setenv("ALL_IN_ONE_DEMUCS_BACKEND", "cli")

    def fake_run(cmd, check):
        captured["cmd"] = cmd
        captured["check"] = check
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(allinone_module.subprocess, "run", fake_run)

    AllInOneRuntime._memory_bounded_demix([audio_path], demix_dir, "cuda:0")

    assert "--repo" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--repo") + 1] == str(static_models.resolve())


def test_memory_bounded_demix_uses_resident_backend_by_default(tmp_path, monkeypatch):
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(b"audio")
    demix_dir = tmp_path / "demix"
    calls = {}
    monkeypatch.delenv("ALL_IN_ONE_DEMUCS_BACKEND", raising=False)
    monkeypatch.setenv("ALL_IN_ONE_DEMUCS_MODEL", "htdemucs_ft")

    def fake_resident(paths, output_dir, device, demucs_model, static_models_dir):
        calls["paths"] = paths
        calls["output_dir"] = output_dir
        calls["device"] = device
        calls["demucs_model"] = demucs_model
        calls["static_models_dir"] = static_models_dir

    monkeypatch.setattr(allinone_module, "_run_demucs_resident", fake_resident)

    demix_paths = AllInOneRuntime._memory_bounded_demix([audio_path], demix_dir, "cuda:0")

    assert demix_paths == [demix_dir / "htdemucs_ft" / "input"]
    assert calls["paths"] == [audio_path]
    assert calls["output_dir"] == demix_dir
    assert calls["device"] == "cuda:0"
    assert calls["demucs_model"] == "htdemucs_ft"


def test_preload_resident_demucs_uses_static_model_dir(tmp_path, monkeypatch):
    calls = {}
    static_models = tmp_path / "static_models"

    def fake_resident_model(model_name, device, static_models_dir):
        calls["model_name"] = model_name
        calls["device"] = device
        calls["static_models_dir"] = static_models_dir
        return object()

    monkeypatch.setenv("ALL_IN_ONE_DEMUCS_MODEL", "htdemucs_ft")
    monkeypatch.setenv("ALL_IN_ONE_STATIC_MODELS_DIR", str(static_models))
    monkeypatch.setattr(allinone_module, "_resident_demucs_model", fake_resident_model)

    runtime = AllInOneRuntime(model_name="harmonix-fold0", device="cuda:0")
    runtime._preload_resident_demucs_sync()

    assert calls == {
        "model_name": "htdemucs_ft",
        "device": "cuda:0",
        "static_models_dir": static_models,
    }
    assert runtime.status()["demucs"]["preloaded"] is True


def test_load_preloads_resident_demucs_when_enabled(monkeypatch):
    class FakeAllInOneModule:
        def analyze(self):
            return None

    calls = {"preload": 0}

    def fake_preload(self):
        calls["preload"] += 1
        self._demucs_preloaded = True

    monkeypatch.setitem(allinone_module.sys.modules, "allin1", FakeAllInOneModule())
    monkeypatch.delenv("ALL_IN_ONE_DEMUCS_BACKEND", raising=False)
    monkeypatch.delenv("ALL_IN_ONE_DEMUCS_PRELOAD", raising=False)
    monkeypatch.setattr(AllInOneRuntime, "_preload_resident_demucs_sync", fake_preload)

    runtime = AllInOneRuntime(model_name="harmonix-fold0", device="cuda:0")
    runtime._load_sync()

    assert calls["preload"] == 1
    assert runtime.loaded is True
    assert runtime.status()["demucs"]["preload_enabled"] is True
    assert runtime.status()["demucs"]["preloaded"] is True
    assert runtime.status()["last_timings"]["demucs_preload_seconds"] >= 0


def test_load_can_skip_resident_demucs_preload(monkeypatch):
    class FakeAllInOneModule:
        def analyze(self):
            return None

    calls = {"preload": 0}

    def fake_preload(self):
        calls["preload"] += 1

    monkeypatch.setitem(allinone_module.sys.modules, "allin1", FakeAllInOneModule())
    monkeypatch.setenv("ALL_IN_ONE_DEMUCS_PRELOAD", "false")
    monkeypatch.setattr(AllInOneRuntime, "_preload_resident_demucs_sync", fake_preload)

    runtime = AllInOneRuntime(model_name="harmonix-fold0", device="cuda:0")
    runtime._load_sync()

    assert calls["preload"] == 0
    assert runtime.loaded is True
    assert runtime.status()["demucs"]["preload_enabled"] is False
    assert runtime.status()["demucs"]["preloaded"] is False


def test_save_demucs_sources_saves_all_sources(tmp_path):
    calls = []

    def fake_save_audio(source, target, *, samplerate):
        calls.append((source, target, samplerate))

    allinone_module._save_demucs_sources(
        sources=[FakeSource("bass"), FakeSource("drums")],
        names=["bass", "drums"],
        out_dir=tmp_path,
        samplerate=44100,
        save_audio_fn=fake_save_audio,
        workers=1,
    )

    assert calls == [
        ("cpu:bass", str(tmp_path / "bass.wav"), 44100),
        ("cpu:drums", str(tmp_path / "drums.wav"), 44100),
    ]


def test_save_demucs_sources_can_use_multiple_workers(tmp_path):
    thread_names = set()
    calls = []
    lock = threading.Lock()

    def fake_save_audio(source, target, *, samplerate):
        time.sleep(0.01)
        with lock:
            thread_names.add(threading.current_thread().name)
            calls.append((source, Path(target).name, samplerate))

    allinone_module._save_demucs_sources(
        sources=[FakeSource("bass"), FakeSource("drums"), FakeSource("other"), FakeSource("vocals")],
        names=["bass", "drums", "other", "vocals"],
        out_dir=tmp_path,
        samplerate=44100,
        save_audio_fn=fake_save_audio,
        workers=2,
    )

    assert sorted(calls) == [
        ("cpu:bass", "bass.wav", 44100),
        ("cpu:drums", "drums.wav", 44100),
        ("cpu:other", "other.wav", 44100),
        ("cpu:vocals", "vocals.wav", 44100),
    ]
    assert any(name.startswith("demucs-save") for name in thread_names)


def test_demucs_save_workers_defaults_to_two(monkeypatch):
    monkeypatch.delenv("ALL_IN_ONE_DEMUCS_SAVE_WORKERS", raising=False)

    assert allinone_module._demucs_save_workers() == 2


class FakeSource:
    def __init__(self, name: str):
        self.name = name

    def cpu(self):
        return f"cpu:{self.name}"
