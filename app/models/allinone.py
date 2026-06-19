import asyncio
import json
import os
import shutil
import subprocess
import sys
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
        return {"analysis": analysis, "analyzer_result_path": str(target)}

    def _analyze_sync(self, audio_path: Path, output_dir: Path) -> dict[str, Any]:
        if self._allin1 is None:
            raise RuntimeError("all-in-one runtime is not loaded")
        byproduct_root = output_dir / "byproducts"
        if byproduct_root.exists():
            shutil.rmtree(byproduct_root)
        byproduct_root.mkdir(parents=True, exist_ok=True)
        self._ensure_static_models_link(byproduct_root)
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
            keep_byproducts=False,
            overwrite=True,
        )
        analysis_path = self._find_analysis_json(output_dir)
        analysis: dict[str, Any] = {}
        if analysis_path:
            analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        return {
            "analysis": analysis,
            "analyzer_result_path": str(analysis_path) if analysis_path else None,
            "raw_result": str(result),
        }

    def _patch_allin1_demix(self) -> None:
        if self._allin1 is None:
            return
        analyze_fn = getattr(self._allin1, "analyze", None)
        globals_dict = getattr(analyze_fn, "__globals__", None)
        if isinstance(globals_dict, dict):
            globals_dict["demix"] = self._memory_bounded_demix

    @staticmethod
    def _memory_bounded_demix(paths: list[Path], demix_dir: Path, device: Any) -> list[Path]:
        """Drop-in all-in-one Demucs hook with bounded segment size for 16GB L4 VMs."""
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
        if not todos:
            return demix_paths

        demucs_model = _demucs_model_name()
        static_models_dir = (demix_dir.parent / "static_models").resolve()
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
        segment_seconds = os.getenv("ALL_IN_ONE_DEMUCS_SEGMENT_SECONDS", "5").strip()
        if segment_seconds:
            cmd.extend(["--segment", segment_seconds])
        jobs = os.getenv("ALL_IN_ONE_DEMUCS_JOBS", "0").strip()
        if jobs:
            cmd.extend(["--jobs", jobs])
        cmd.extend(path.as_posix() for path in todos)
        subprocess.run(cmd, check=True)
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
                "segment_seconds": os.getenv("ALL_IN_ONE_DEMUCS_SEGMENT_SECONDS", "5"),
                "jobs": os.getenv("ALL_IN_ONE_DEMUCS_JOBS", "0"),
            },
        }


_DEMUCS_STEMS = ("bass", "drums", "other", "vocals")


def _demucs_model_name() -> str:
    return os.getenv("ALL_IN_ONE_DEMUCS_MODEL", os.getenv("HTDEMUCS_MODEL", "htdemucs_ft")).strip() or "htdemucs_ft"
