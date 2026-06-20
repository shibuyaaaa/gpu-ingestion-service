import json
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class GPUState:
    available: bool
    name: str | None = None
    utilization_pct: float | None = None
    memory_total_mb: int | None = None
    memory_used_mb: int | None = None
    memory_free_mb: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "name": self.name,
            "utilization_pct": self.utilization_pct,
            "memory_total_mb": self.memory_total_mb,
            "memory_used_mb": self.memory_used_mb,
            "memory_free_mb": self.memory_free_mb,
            "error": self.error,
        }


class GPUProbe:
    """Lightweight `nvidia-smi` wrapper for ops endpoints."""

    def __init__(self, *, cache_seconds: float = 1.0):
        self.cache_seconds = max(0.0, float(cache_seconds))
        self._cached_state: GPUState | None = None
        self._cached_at: float = 0.0

    def snapshot(self) -> GPUState:
        now = time.monotonic()
        if self.cache_seconds > 0 and self._cached_state is not None and now - self._cached_at <= self.cache_seconds:
            return self._cached_state
        state = self.snapshot_uncached()
        self._cached_state = state
        self._cached_at = now
        return state

    def snapshot_uncached(self) -> GPUState:
        return self._snapshot_uncached()

    def _snapshot_uncached(self) -> GPUState:
        if not shutil.which("nvidia-smi"):
            return GPUState(available=False, error="nvidia-smi not found")
        query = (
            "name,utilization.gpu,memory.total,memory.used,memory.free"
        )
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu={query}",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            first = result.stdout.strip().splitlines()[0]
            name, util, total, used, free = [part.strip() for part in first.split(",")]
            return GPUState(
                available=True,
                name=name,
                utilization_pct=float(util),
                memory_total_mb=int(total),
                memory_used_mb=int(used),
                memory_free_mb=int(free),
            )
        except Exception as exc:
            return GPUState(available=False, error=str(exc))


class GPUUsageSampler:
    """Samples nvidia-smi in a background thread during blocking GPU work.

    The values are coarse device-level utilization snapshots, not true model
    FLOP utilization. They are still useful for spotting idle gaps, memory
    pressure, and queue/CPU bottlenecks around a GPU-locked job.
    """

    def __init__(self, probe: GPUProbe, *, interval_seconds: float = 0.5):
        self.probe = probe
        self.interval_seconds = max(0.0, float(interval_seconds))
        self.enabled = self.interval_seconds > 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[GPUState] = []
        self._started_at: float | None = None
        self._stopped_at: float | None = None

    def __enter__(self) -> "GPUUsageSampler":
        self._started_at = time.monotonic()
        self._record_sample()
        if self.enabled:
            self._thread = threading.Thread(target=self._run, name="gpu-usage-sampler", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 2))
        self._record_sample()
        self._stopped_at = time.monotonic()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._record_sample()

    def _record_sample(self) -> None:
        self._samples.append(self.probe.snapshot_uncached())

    def summary(self) -> dict[str, Any]:
        elapsed = None
        if self._started_at is not None:
            elapsed = (self._stopped_at or time.monotonic()) - self._started_at
        available = [sample for sample in self._samples if sample.available]
        summary: dict[str, Any] = {
            "gpu_sample_enabled": self.enabled,
            "gpu_sample_count": len(self._samples),
            "gpu_sample_available_count": len(available),
            "gpu_sample_elapsed_seconds": _round(elapsed),
        }
        if not available:
            first_error = next((sample.error for sample in self._samples if sample.error), None)
            if first_error:
                summary["gpu_sample_error"] = first_error
            return summary

        utils = [sample.utilization_pct for sample in available if sample.utilization_pct is not None]
        used = [sample.memory_used_mb for sample in available if sample.memory_used_mb is not None]
        free = [sample.memory_free_mb for sample in available if sample.memory_free_mb is not None]
        if utils:
            summary["gpu_utilization_avg_pct"] = _round(sum(utils) / len(utils))
            summary["gpu_utilization_max_pct"] = _round(max(utils))
        if used:
            summary["gpu_memory_used_avg_mb"] = _round(sum(used) / len(used))
            summary["gpu_memory_used_max_mb"] = max(used)
            summary["gpu_memory_used_start_mb"] = used[0]
            summary["gpu_memory_used_end_mb"] = used[-1]
        if free:
            summary["gpu_memory_free_min_mb"] = min(free)
        return summary


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def json_safe_gpu_state(state: GPUState) -> str:
    return json.dumps(state.to_dict(), sort_keys=True)
