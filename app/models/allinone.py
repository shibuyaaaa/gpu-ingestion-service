import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


class AllInOneRuntime:
    """Resident wrapper for all-in-one/Harmonix analysis."""

    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        dry_run: bool = False,
        cuda_graphs_enabled: bool = False,
        cuda_graph_audio_seconds: float = 30.0,
    ):
        self.model_name = model_name
        self.device = device
        self.dry_run = dry_run
        self.cuda_graphs_enabled = cuda_graphs_enabled
        self.cuda_graph_audio_seconds = cuda_graph_audio_seconds
        self.loaded = False
        self._allin1: Any = None
        self._last_timings: dict[str, Any] = {}

    async def load(self) -> None:
        if self.loaded:
            return
        if self.dry_run:
            self.loaded = True
            return
        await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        import allin1

        self._allin1 = allin1
        self._patch_allin1_demix()
        self.loaded = True

    async def analyze(self, audio_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
        await self.load()
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if self.dry_run:
            return await asyncio.to_thread(self._dry_run_analysis, Path(audio_path), output)
        return await asyncio.to_thread(self._analyze_sync, Path(audio_path), output)

    def _dry_run_analysis(self, audio_path: Path, output_dir: Path) -> dict[str, Any]:
        analysis = {
            "duration": 180.0,
            "bpm": 120.0,
            "segments": [
                {"label": "intro", "start": 0.0, "end": 16.0},
                {"label": "verse", "start": 16.0, "end": 48.0},
                {"label": "chorus", "start": 48.0, "end": 80.0},
            ],
            "source": str(audio_path),
        }
        target = output_dir / "analyzer_result.json"
        target.write_text(json.dumps(analysis), encoding="utf-8")
        timings = {"allin1_analyze_seconds": 0.0, "dry_run": True}
        self._last_timings = timings
        return {"analysis": analysis, "analyzer_result_path": str(target), "timings": timings}

    def _analyze_sync(self, audio_path: Path, output_dir: Path) -> dict[str, Any]:
        if self._allin1 is None:
            raise RuntimeError("all-in-one runtime is not loaded")
        timings: dict[str, Any] = {"dry_run": False}
        _ANALYZE_LOCAL.timings = timings
        byproduct_root = output_dir / "byproducts"
        try:
            started = time.perf_counter()
            if byproduct_root.exists():
                shutil.rmtree(byproduct_root)
            byproduct_root.mkdir(parents=True, exist_ok=True)
            self._ensure_static_models_link(byproduct_root)
            timings["byproduct_setup_seconds"] = _elapsed(started)

            started = time.perf_counter()
            result = self._allin1.analyze(
                paths=[str(audio_path)],
                out_dir=str(output_dir),
                model=self.model_name,
                device=self.device,
                visualize=False,
                sonify=False,
                include_activations=False,
                include_embeddings=False,
                demix_dir=str(byproduct_root / "demix"),
                spec_dir=str(byproduct_root / "spec"),
                keep_byproducts=True,
                overwrite=True,
            )
            timings["allin1_analyze_seconds"] = _elapsed(started)

            started = time.perf_counter()
            analysis_path = self._find_analysis_json(output_dir)
            timings["find_analysis_json_seconds"] = _elapsed(started)

            started = time.perf_counter()
            analysis: dict[str, Any] = {}
            if analysis_path:
                analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
            timings["load_analysis_json_seconds"] = _elapsed(started)

            started = time.perf_counter()
            stem_paths = self._find_demix_stems(byproduct_root, audio_path)
            timings["find_demix_stems_seconds"] = _elapsed(started)
            timings["stem_count"] = len(stem_paths)
            self._last_timings = dict(timings)
            return {
                "analysis": analysis,
                "analyzer_result_path": str(analysis_path) if analysis_path else None,
                "stem_paths": stem_paths,
                "raw_result": str(result),
                "timings": dict(timings),
            }
        finally:
            _ANALYZE_LOCAL.timings = None

    def _patch_allin1_demix(self) -> None:
        if self._allin1 is None:
            return
        analyze_fn = getattr(self._allin1, "analyze", None)
        globals_dict = getattr(analyze_fn, "__globals__", None)
        if isinstance(globals_dict, dict):
            globals_dict["demix"] = self._memory_bounded_demix

    @staticmethod
    def _memory_bounded_demix(paths: list[Path], demix_dir: Path, device: Any) -> list[Path]:
        """Drop-in all-in-one Demucs hook with resident weights for L4 VMs."""
        timings = getattr(_ANALYZE_LOCAL, "timings", None)
        todos = []
        demix_paths = []
        for path in paths:
            path = Path(path)
            demucs_model = _demucs_model_name()
            out_dir = demix_dir / demucs_model / path.stem
            demix_paths.append(out_dir)
            if out_dir.is_dir() and all((out_dir / f"{stem}.wav").is_file() for stem in _DEMUCS_STEMS):
                continue
            todos.append(path)

        existing = len(paths) - len(todos)
        print(f"=> Found {existing} tracks already demixed, {len(todos)} to demix.")
        if isinstance(timings, dict):
            timings["demix_existing_tracks"] = existing
            timings["demix_pending_tracks"] = len(todos)
        if not todos:
            return demix_paths

        demucs_model = _demucs_model_name()
        static_models_dir = (demix_dir.parent / "static_models").resolve()
        started = time.perf_counter()
        if _demucs_backend() == "cli":
            demix_timings = _run_demucs_cli(todos, demix_dir, device, demucs_model, static_models_dir)
            backend = "cli"
            if isinstance(timings, dict):
                timings["demix_total_seconds"] = _elapsed(started)
                timings["demix_backend"] = backend
                if isinstance(demix_timings, dict):
                    timings.update(demix_timings)
            return demix_paths

        demix_timings = _run_demucs_resident(todos, demix_dir, device, demucs_model, static_models_dir)
        if isinstance(timings, dict):
            timings["demix_total_seconds"] = _elapsed(started)
            timings["demix_backend"] = "resident"
            if isinstance(demix_timings, dict):
                timings.update(demix_timings)
        return demix_paths

    @staticmethod
    def _ensure_static_models_link(byproduct_root: Path) -> None:
        """all-in-one resolves Demucs models as demix_dir.parent/static_models."""
        link_path = byproduct_root / "static_models"
        if link_path.exists():
            return
        source = Path("/opt/all-in-one-audio/static_models")
        if not source.is_dir():
            raise RuntimeError(f"all-in-one static model directory is missing: {source}")
        os.symlink(source, link_path, target_is_directory=True)

    @staticmethod
    def _find_analysis_json(output_dir: Path) -> Path | None:
        candidates = sorted(output_dir.rglob("*.json"))
        for candidate in candidates:
            if candidate.name in {"result.json", "analyzer_result.json"}:
                return candidate
        return candidates[0] if candidates else None

    @staticmethod
    def _find_demix_stems(byproduct_root: Path, audio_path: Path) -> dict[str, str]:
        stem_dir = byproduct_root / "demix" / _demucs_model_name() / audio_path.stem
        stems: dict[str, str] = {}
        for stem in _DEMUCS_STEMS:
            path = stem_dir / f"{stem}.wav"
            if path.is_file():
                stems[stem] = str(path)
        return stems

    def status(self) -> dict[str, Any]:
        return {
            "name": "all-in-one",
            "model_name": self.model_name,
            "device": self.device,
            "loaded": self.loaded,
            "dry_run": self.dry_run,
            "resident_policy": "load-package-once-process-resident",
            "cuda_graphs": {
                "enabled": self.cuda_graphs_enabled,
                "captured": False,
                "reason": (
                    "all-in-one analysis currently uses dynamic file-length IO and package-level orchestration; "
                    "capture only after isolating a fixed-shape pure tensor forward"
                ),
                "target_audio_seconds": self.cuda_graph_audio_seconds,
            },
            "demucs": {
                "model": _demucs_model_name(),
                "backend": _demucs_backend(),
                "segment_seconds": _demucs_segment_seconds_text(),
                "jobs": os.getenv("ALL_IN_ONE_DEMUCS_JOBS", "0"),
                "save_workers": _demucs_save_workers(),
                "resident_cache_keys": list(_DEMUCS_MODEL_CACHE.keys()),
            },
            "last_timings": self._last_timings,
        }


def _run_demucs_cli(paths: list[Path], demix_dir: Path, device: Any, demucs_model: str, static_models_dir: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "demucs.separate",
        "--out",
        demix_dir.as_posix(),
        "--name",
        demucs_model,
        "--device",
        str(device),
        "-n",
        demucs_model,
    ]
    if (static_models_dir / f"{demucs_model}.yaml").is_file():
        cmd.extend(["--repo", static_models_dir.as_posix()])
    segment_seconds = _demucs_segment_seconds_text()
    if segment_seconds:
        cmd.extend(["--segment", segment_seconds])
    jobs = os.getenv("ALL_IN_ONE_DEMUCS_JOBS", "0").strip()
    if jobs:
        cmd.extend(["--jobs", jobs])
    cmd.extend(path.as_posix() for path in paths)
    started = time.perf_counter()
    subprocess.run(cmd, check=True)
    return {
        "demix_cli_seconds": _elapsed(started),
        "demix_track_count": len(paths),
    }


def _run_demucs_resident(paths: list[Path], demix_dir: Path, device: Any, demucs_model: str, static_models_dir: Path) -> dict[str, Any]:
    import torch
    from demucs.apply import apply_model
    from demucs.audio import AudioFile, save_audio

    timings: dict[str, Any] = {
        "demix_track_count": len(paths),
        "demix_audio_read_seconds": 0.0,
        "demix_normalize_seconds": 0.0,
        "demix_pin_seconds": 0.0,
        "demix_transfer_seconds": 0.0,
        "demix_apply_seconds": 0.0,
        "demix_save_seconds": 0.0,
    }
    started = time.perf_counter()
    model = _resident_demucs_model(demucs_model, device, static_models_dir)
    timings["demix_model_ready_seconds"] = _elapsed(started)
    torch_device = torch.device(device)
    segment = _demucs_segment_seconds()
    jobs = _int_env("ALL_IN_ONE_DEMUCS_JOBS", 0)
    for path in paths:
        path = Path(path)
        out_dir = demix_dir / demucs_model / path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        wav = AudioFile(str(path)).read(
            streams=0,
            samplerate=model.samplerate,
            channels=model.audio_channels,
        )
        timings["demix_audio_read_seconds"] += _elapsed(started)
        started = time.perf_counter()
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        batch = wav[None]
        timings["demix_normalize_seconds"] += _elapsed(started)
        if _bool_env("ALL_IN_ONE_DEMUCS_PIN_MEMORY", True):
            started = time.perf_counter()
            batch = _maybe_pin(batch)
            timings["demix_pin_seconds"] += _elapsed(started)
        with torch.inference_mode():
            started = time.perf_counter()
            device_batch = _transfer_to_device(batch, torch_device)
            timings["demix_transfer_seconds"] += _elapsed(started)
            started = time.perf_counter()
            sources = apply_model(
                model,
                device_batch,
                device=torch_device,
                split=True,
                overlap=_float_env("ALL_IN_ONE_DEMUCS_OVERLAP", 0.25),
                segment=segment,
                num_workers=jobs,
                progress=False,
            )[0]
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
            timings["demix_apply_seconds"] += _elapsed(started)
            sources = sources * ref.std() + ref.mean()
        started = time.perf_counter()
        _save_demucs_sources(
            sources=sources,
            names=model.sources,
            out_dir=out_dir,
            samplerate=model.samplerate,
            save_audio_fn=save_audio,
            workers=_demucs_save_workers(),
        )
        timings["demix_save_seconds"] += _elapsed(started)
    return {key: _round_timing(value) if isinstance(value, float) else value for key, value in timings.items()}


def _save_demucs_sources(
    *,
    sources: Any,
    names: Any,
    out_dir: Path,
    samplerate: int,
    save_audio_fn: Any,
    workers: int,
) -> None:
    items = [(source.cpu(), str(out_dir / f"{name}.wav")) for source, name in zip(sources, names)]
    if not items:
        return
    worker_count = min(max(1, workers), len(items))
    if worker_count == 1:
        for source, target in items:
            save_audio_fn(source, target, samplerate=samplerate)
        return
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="demucs-save") as executor:
        futures = [
            executor.submit(save_audio_fn, source, target, samplerate=samplerate)
            for source, target in items
        ]
        for future in futures:
            future.result()


def _resident_demucs_model(model_name: str, device: Any, static_models_dir: Path) -> Any:
    import torch
    from demucs.pretrained import get_model

    repo = static_models_dir if (static_models_dir / f"{model_name}.yaml").is_file() else None
    cache_key = f"{model_name}:{device}:{repo or 'default'}"
    with _DEMUCS_MODEL_LOCK:
        cached = _DEMUCS_MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached
        if repo is not None:
            try:
                model = get_model(model_name, repo=repo)
            except TypeError:
                model = get_model(model_name)
        else:
            model = get_model(model_name)
        model.to(torch.device(device))
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        if torch.cuda.is_available():
            torch.cuda.synchronize(torch.device(device))
        _DEMUCS_MODEL_CACHE[cache_key] = model
        return model


_DEMUCS_STEMS = ("bass", "drums", "other", "vocals")
_DEMUCS_MODEL_CACHE: dict[str, Any] = {}
_DEMUCS_MODEL_LOCK = threading.RLock()
_ANALYZE_LOCAL = threading.local()


def _elapsed(started: float) -> float:
    return _round_timing(time.perf_counter() - started)


def _round_timing(value: float) -> float:
    return round(float(value), 6)


def _demucs_model_name() -> str:
    return os.getenv("ALL_IN_ONE_DEMUCS_MODEL", os.getenv("HTDEMUCS_MODEL", "htdemucs_ft")).strip() or "htdemucs_ft"


def _demucs_backend() -> str:
    value = os.getenv("ALL_IN_ONE_DEMUCS_BACKEND", "resident").strip().lower()
    return value if value in {"resident", "cli"} else "resident"


def _demucs_save_workers() -> int:
    return max(1, _int_env("ALL_IN_ONE_DEMUCS_SAVE_WORKERS", 2))


def _demucs_segment_seconds_text() -> str:
    return os.getenv("ALL_IN_ONE_DEMUCS_SEGMENT_SECONDS", "7.5").strip()


def _demucs_segment_seconds() -> float | None:
    value = _demucs_segment_seconds_text()
    if value == "":
        return None
    return float(value)


def _transfer_to_device(batch: Any, device: Any) -> Any:
    import torch

    if getattr(device, "type", None) != "cuda":
        return batch.to(device)
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        device_batch = batch.to(device, non_blocking=True)
    stream.synchronize()
    return device_batch


def _maybe_pin(tensor: Any) -> Any:
    try:
        return tensor.pin_memory()
    except RuntimeError:
        return tensor


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _optional_float_env(name: str) -> float | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return float(value)


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)
