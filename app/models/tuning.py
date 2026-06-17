from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PinnedMemoryPolicy:
    enabled: bool
    max_bytes: int
    slots: int
    seconds: float
    channels: int
    sample_rate: int
    bytes_per_sample: int = 4

    @classmethod
    def from_settings(cls, settings: Settings) -> "PinnedMemoryPolicy":
        max_bytes = int(
            settings.pinned_audio_seconds
            * settings.pinned_audio_sample_rate
            * settings.pinned_audio_channels
            * 4
            * max(1, settings.pinned_audio_slots)
        )
        return cls(
            enabled=settings.pinned_audio_staging,
            max_bytes=max_bytes,
            slots=max(1, settings.pinned_audio_slots),
            seconds=settings.pinned_audio_seconds,
            channels=settings.pinned_audio_channels,
            sample_rate=settings.pinned_audio_sample_rate,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_bytes": self.max_bytes,
            "max_mib": round(self.max_bytes / 1024 / 1024, 1),
            "slots": self.slots,
            "seconds": self.seconds,
            "channels": self.channels,
            "sample_rate": self.sample_rate,
            "bytes_per_sample": self.bytes_per_sample,
        }


class PinnedAudioStager:
    """Pinned host-memory staging for CPU audio tensors.

    This is intentionally per-transfer instead of a large preallocated pool.
    Demucs jobs are long and GPU concurrency is 1 on the initial L4, so the
    useful win is avoiding pageable H2D copies without permanently locking
    excessive host RAM.
    """

    def __init__(self, policy: PinnedMemoryPolicy):
        self.policy = policy
        self.pinned_transfers = 0
        self.pageable_transfers = 0
        self.skipped_oversize = 0
        self.last_transfer_bytes = 0

    def stage(self, tensor: Any) -> Any:
        if not self.policy.enabled:
            self.pageable_transfers += 1
            self.last_transfer_bytes = _tensor_nbytes(tensor)
            return tensor
        size = _tensor_nbytes(tensor)
        self.last_transfer_bytes = size
        if size > self.policy.max_bytes:
            self.skipped_oversize += 1
            logger.warning(
                "audio tensor %.1f MiB exceeds pinned-memory budget %.1f MiB; using pageable transfer",
                size / 1024 / 1024,
                self.policy.max_bytes / 1024 / 1024,
            )
            return tensor
        try:
            staged = tensor.pin_memory()
            self.pinned_transfers += 1
            return staged
        except RuntimeError as exc:
            self.pageable_transfers += 1
            logger.warning("pin_memory failed; using pageable transfer: %s", exc)
            return tensor

    def status(self) -> dict[str, Any]:
        return {
            "policy": self.policy.to_dict(),
            "pinned_transfers": self.pinned_transfers,
            "pageable_transfers": self.pageable_transfers,
            "skipped_oversize": self.skipped_oversize,
            "last_transfer_mib": round(self.last_transfer_bytes / 1024 / 1024, 2),
        }


def configure_torch_runtime(settings: Settings) -> dict[str, Any]:
    """Apply conservative CUDA runtime knobs before model warmup."""
    if not settings.cuda_runtime_tuning:
        return {"enabled": False}
    try:
        import torch

        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(settings.cuda_matmul_precision)
        torch.backends.cuda.matmul.allow_tf32 = settings.cuda_allow_tf32
        torch.backends.cudnn.allow_tf32 = settings.cuda_allow_tf32
        torch.backends.cudnn.benchmark = settings.cuda_cudnn_benchmark
        return {
            "enabled": True,
            "allow_tf32": settings.cuda_allow_tf32,
            "cudnn_benchmark": settings.cuda_cudnn_benchmark,
            "matmul_precision": settings.cuda_matmul_precision,
            "allocator_conf": _cuda_alloc_conf(),
        }
    except Exception as exc:
        logger.warning("CUDA runtime tuning could not be applied: %s", exc)
        return {"enabled": False, "error": str(exc)}


def cuda_memory_snapshot(device: str) -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}
        dev = torch.device(device)
        return {
            "available": True,
            "allocated_mib": round(torch.cuda.memory_allocated(dev) / 1024 / 1024, 1),
            "reserved_mib": round(torch.cuda.memory_reserved(dev) / 1024 / 1024, 1),
            "max_allocated_mib": round(torch.cuda.max_memory_allocated(dev) / 1024 / 1024, 1),
            "max_reserved_mib": round(torch.cuda.max_memory_reserved(dev) / 1024 / 1024, 1),
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def maybe_empty_cuda_cache(enabled: bool) -> None:
    if not enabled:
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        logger.exception("torch.cuda.empty_cache failed")


def _tensor_nbytes(tensor: Any) -> int:
    if hasattr(tensor, "nelement") and hasattr(tensor, "element_size"):
        return int(tensor.nelement() * tensor.element_size())
    return 0


def _cuda_alloc_conf() -> str | None:
    import os

    return os.getenv("PYTORCH_CUDA_ALLOC_CONF") or os.getenv("PYTORCH_ALLOC_CONF")

