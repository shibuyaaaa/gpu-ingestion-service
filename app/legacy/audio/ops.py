import asyncio
import os
import shutil
from pathlib import Path

_FFMPEG_SEMAPHORES: dict[tuple[int, int], asyncio.Semaphore] = {}


class AudioOps:
    """Small ffmpeg helpers used by local job adapters."""

    @staticmethod
    async def convert_to_wav(input_path: str | Path, output_path: str | Path) -> str:
        await AudioOps._run_ffmpeg(
            [
                *AudioOps._base_ffmpeg_cmd(),
                "-i",
                str(input_path),
                "-ac",
                "2",
                "-ar",
                "44100",
                str(output_path),
            ]
        )
        return str(output_path)

    @staticmethod
    async def convert_to_mp3(input_path: str | Path, output_path: str | Path) -> str:
        await AudioOps._run_ffmpeg(
            [
                *AudioOps._base_ffmpeg_cmd(),
                "-i",
                str(input_path),
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "320k",
                str(output_path),
            ]
        )
        return str(output_path)

    @staticmethod
    async def extract_segment(
        input_path: str | Path,
        output_path: str | Path,
        *,
        start: float,
        duration: float,
    ) -> str:
        await AudioOps._run_ffmpeg(
            [
                *AudioOps._base_ffmpeg_cmd(),
                "-ss",
                str(start),
                "-i",
                str(input_path),
                "-t",
                str(duration),
                "-codec:a",
                "libmp3lame",
                "-q:a",
                "2",
                str(output_path),
            ]
        )
        return str(output_path)

    @staticmethod
    async def mix(inputs: list[str | Path], output_path: str | Path) -> str:
        existing = [Path(item) for item in inputs if Path(item).exists()]
        if not existing:
            raise RuntimeError("no input stems exist for mix")
        if len(existing) == 1:
            shutil.copyfile(existing[0], output_path)
            return str(output_path)
        cmd = AudioOps._base_ffmpeg_cmd()
        for item in existing:
            cmd.extend(["-i", str(item)])
        cmd.extend(
            [
                "-filter_complex",
                f"amix=inputs={len(existing)}:duration=longest:normalize=0",
                "-ac",
                "2",
                str(output_path),
            ]
        )
        await AudioOps._run_ffmpeg(cmd)
        return str(output_path)

    @staticmethod
    async def _run_ffmpeg(cmd: list[str]) -> None:
        semaphore = AudioOps._ffmpeg_semaphore()
        if semaphore is None:
            await AudioOps._run_ffmpeg_unbounded(cmd)
            return
        async with semaphore:
            await AudioOps._run_ffmpeg_unbounded(cmd)

    @staticmethod
    async def _run_ffmpeg_unbounded(cmd: list[str]) -> None:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(stderr.decode("utf-8", errors="replace")[-1000:])

    @staticmethod
    def _base_ffmpeg_cmd() -> list[str]:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        threads = os.getenv("FFMPEG_THREADS", "1").strip()
        if threads and threads != "0":
            cmd.extend(["-threads", str(max(1, int(threads)))])
        return cmd

    @staticmethod
    def _ffmpeg_semaphore() -> asyncio.Semaphore | None:
        raw_limit = os.getenv("FFMPEG_CONCURRENCY", "3").strip()
        if raw_limit in {"", "0"}:
            return None
        limit = max(1, int(raw_limit))
        key = (id(asyncio.get_running_loop()), limit)
        semaphore = _FFMPEG_SEMAPHORES.get(key)
        if semaphore is None:
            semaphore = asyncio.Semaphore(limit)
            _FFMPEG_SEMAPHORES[key] = semaphore
        return semaphore
