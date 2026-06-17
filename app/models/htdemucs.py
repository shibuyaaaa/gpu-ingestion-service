import asyncio
import shutil
import subprocess
from pathlib import Path
from typing import Any

from app.models.tuning import PinnedAudioStager, maybe_empty_cuda_cache


class HTDemucsRuntime:
    """Resident HTDemucs runtime.

    In production this loads the Demucs model once and keeps it in process.
    `dry_run=True` creates deterministic artifact files so the orchestration
    layer can be tested without CUDA/model packages.
    """

    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        dry_run: bool = False,
        stager: PinnedAudioStager | None = None,
        empty_cache_after_job: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.dry_run = dry_run
        self.stager = stager
        self.empty_cache_after_job = empty_cache_after_job
        self.loaded = False
        self._model: Any = None

    async def load(self) -> None:
        if self.loaded:
            return
        if self.dry_run:
            self.loaded = True
            return
        await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        import torch
        from demucs.pretrained import get_model

        model = get_model(self.model_name)
        model.to(torch.device(self.device))
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self._model = model
        if torch.cuda.is_available():
            torch.cuda.synchronize(torch.device(self.device))
        self.loaded = True

    async def separate(self, audio_path: str | Path, output_dir: str | Path) -> dict[str, str]:
        await self.load()
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if self.dry_run:
            return await asyncio.to_thread(self._dry_run_outputs, Path(audio_path), output)
        return await asyncio.to_thread(self._separate_sync, Path(audio_path), output)

    def _dry_run_outputs(self, audio_path: Path, output_dir: Path) -> dict[str, str]:
        stems = {}
        for stem in ("vocals", "drums", "bass", "other"):
            target = output_dir / f"{stem}.wav"
            if shutil.which("ffmpeg"):
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(audio_path),
                        "-t",
                        "1",
                        str(target),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                )
            if not target.exists():
                target.write_bytes(b"dry-run-stem")
            stems[stem] = str(target)
        return stems

    def _separate_sync(self, audio_path: Path, output_dir: Path) -> dict[str, str]:
        import torch
        from demucs.apply import apply_model
        from demucs.audio import AudioFile, save_audio

        if self._model is None:
            raise RuntimeError("HTDemucs model is not loaded")

        model = self._model
        device = torch.device(self.device)
        wav = AudioFile(str(audio_path)).read(
            streams=0,
            samplerate=model.samplerate,
            channels=model.audio_channels,
        )
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        batch = wav[None]
        if self.stager is not None:
            batch = self.stager.stage(batch)

        try:
            with torch.inference_mode():
                device_batch = self._transfer_to_device(batch, device)
                sources = apply_model(
                    model,
                    device_batch,
                    device=device,
                    split=True,
                    overlap=0.25,
                    progress=False,
                )[0]
                sources = sources * ref.std() + ref.mean()

            outputs: dict[str, str] = {}
            for source, name in zip(sources, model.sources):
                target = output_dir / f"{name}.wav"
                save_audio(source.cpu(), str(target), samplerate=model.samplerate)
                outputs[name] = str(target)
            return outputs
        finally:
            maybe_empty_cuda_cache(self.empty_cache_after_job)

    @staticmethod
    def _transfer_to_device(batch: Any, device: Any) -> Any:
        import torch

        if getattr(device, "type", None) != "cuda":
            return batch.to(device)
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            device_batch = batch.to(device, non_blocking=True)
        stream.synchronize()
        return device_batch

    def status(self) -> dict[str, Any]:
        return {
            "name": "htdemucs",
            "model_name": self.model_name,
            "device": self.device,
            "loaded": self.loaded,
            "dry_run": self.dry_run,
            "resident_policy": "load-once-process-resident",
            "pinned_audio_staging": self.stager.status() if self.stager else None,
            "empty_cache_after_job": self.empty_cache_after_job,
        }
