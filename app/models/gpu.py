import json
import shutil
import subprocess
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
        state = self._snapshot_uncached()
        self._cached_state = state
        self._cached_at = now
        return state

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


def json_safe_gpu_state(state: GPUState) -> str:
    return json.dumps(state.to_dict(), sort_keys=True)
