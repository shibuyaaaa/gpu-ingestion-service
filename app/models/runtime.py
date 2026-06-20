import asyncio
from dataclasses import dataclass

from app.config import Settings
from app.models.allinone import AllInOneRuntime
from app.models.cloud_run_allinone import CloudRunAllInOneRuntime
from app.models.gpu import GPUProbe
from app.models.htdemucs import HTDemucsRuntime
from app.models.tuning import (
    PinnedAudioStager,
    PinnedMemoryPolicy,
    configure_torch_runtime,
    cuda_memory_snapshot,
)


@dataclass
class ModelRuntimeBundle:
    all_in_one: AllInOneRuntime
    htdemucs: HTDemucsRuntime
    gpu_probe: GPUProbe
    gpu_lock: asyncio.Lock
    settings: Settings
    torch_runtime_status: dict
    pinned_audio_stager: PinnedAudioStager
    cuda_graph_policy: dict
    active_gpu_job_id: str | None = None
    active_gpu_model: str | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> "ModelRuntimeBundle":
        pinned_policy = PinnedMemoryPolicy.from_settings(settings)
        pinned_stager = PinnedAudioStager(pinned_policy)
        if settings.model_backend in {"cloud_run", "cloud_run_fallback", "remote_gpu"}:
            cloud_runtime = CloudRunAllInOneRuntime(
                url=settings.cloud_run_fallback_url,
                model_name=settings.all_in_one_model,
                audio_separator_model=settings.all_in_one_audio_separator_model,
                auth=settings.all_in_one_auth,
                api_key=settings.all_in_one_api_key,
                timeout_seconds=settings.all_in_one_timeout_seconds,
                id_token_audience=settings.all_in_one_id_token_audience,
                upload_transcode_enabled=settings.cloud_run_upload_transcode_enabled,
                max_upload_bytes=settings.cloud_run_max_upload_bytes,
                upload_bitrate=settings.cloud_run_upload_bitrate,
            )
            return cls(
                all_in_one=cloud_runtime,
                htdemucs=cloud_runtime,
                gpu_probe=GPUProbe(cache_seconds=settings.gpu_probe_cache_seconds),
                gpu_lock=asyncio.Lock(),
                settings=settings,
                torch_runtime_status={"configured": False, "enabled": False, "reason": "remote_cloud_run_backend"},
                pinned_audio_stager=pinned_stager,
                cuda_graph_policy={
                    "enabled": False,
                    "target_audio_seconds": settings.cuda_graph_audio_seconds,
                    "status": "remote_cloud_run_backend",
                    "notes": ["CUDA tuning applies only to local resident model backends."],
                },
            )
        return cls(
            all_in_one=AllInOneRuntime(
                model_name=settings.all_in_one_model,
                device=settings.gpu_device,
                dry_run=settings.dry_run_mode,
                cuda_graphs_enabled=settings.cuda_graphs_enabled,
                cuda_graph_audio_seconds=settings.cuda_graph_audio_seconds,
            ),
            htdemucs=HTDemucsRuntime(
                model_name=settings.htdemucs_model,
                device=settings.gpu_device,
                dry_run=settings.dry_run_mode,
                stager=pinned_stager,
                empty_cache_after_job=settings.cuda_empty_cache_after_job,
            ),
            gpu_probe=GPUProbe(cache_seconds=settings.gpu_probe_cache_seconds),
            gpu_lock=asyncio.Lock(),
            settings=settings,
            torch_runtime_status={"configured": False},
            pinned_audio_stager=pinned_stager,
            cuda_graph_policy={
                "enabled": settings.cuda_graphs_enabled,
                "target_audio_seconds": settings.cuda_graph_audio_seconds,
                "status": "disabled" if not settings.cuda_graphs_enabled else "benchmark_required",
                "notes": [
                    "CUDA Graph replay needs static shapes and stable memory addresses.",
                    "HTDemucs split mode and all-in-one orchestration are dynamic for full songs.",
                    "Candidate path is a fixed-duration quick-dissect segment runner after L4 benchmarking.",
                ],
            },
        )

    async def warmup(self) -> None:
        # Runtime tuning must happen before model warmup and before any real CUDA allocations.
        if self.torch_runtime_status == {"configured": False}:
            self.torch_runtime_status = configure_torch_runtime(self.settings)
        if self.all_in_one.dry_run and self.htdemucs.dry_run:
            await asyncio.gather(self.all_in_one.load(), self.htdemucs.load())
            return
        async with self.gpu_lock:
            await self.htdemucs.load()
            await self.all_in_one.load()

    def status(self) -> dict:
        gpu = self.gpu_probe.snapshot().to_dict()
        cuda_device = getattr(self.htdemucs, "device", None)
        return {
            "gpu": gpu,
            "active_gpu_job_id": self.active_gpu_job_id,
            "active_gpu_model": self.active_gpu_model,
            "torch_runtime": self.torch_runtime_status,
            "cuda_memory": cuda_memory_snapshot(cuda_device) if cuda_device else {"available": False, "reason": "remote_backend"},
            "pinned_audio": self.pinned_audio_stager.status(),
            "cuda_graph_policy": self.cuda_graph_policy,
            "models": {
                "htdemucs": self.htdemucs.status(),
                "all_in_one": self.all_in_one.status(),
            },
        }
