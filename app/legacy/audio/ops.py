import asyncio
import os
import shutil
from pathlib import Path
from typing import Any

_FFMPEG_SEMAPHORES: dict[tuple[int, int], asyncio.Semaphore] = {}
_FFMPEG_STATS: dict[str, float | int | None] = {
    "configured_concurrency": None,
    "active": 0,
    "max_active": 0,
    "total_calls": 0,
    "total_wait_seconds": 0.0,
    "total_run_seconds": 0.0,
    "last_wait_seconds": 0.0,
    "last_run_seconds": 0.0,
}


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
            AudioOps._set_configured_concurrency()
            await AudioOps._run_ffmpeg_tracked(cmd, wait_seconds=0.0)
            return
        AudioOps._set_configured_concurrency()
        wait_started = asyncio.get_running_loop().time()
        async with semaphore:
            wait_seconds = asyncio.get_running_loop().time() - wait_started
            await AudioOps._run_ffmpeg_tracked(cmd, wait_seconds=wait_seconds)

    @staticmethod
    async def _run_ffmpeg_tracked(cmd: list[str], *, wait_seconds: float) -> None:
        _FFMPEG_STATS["total_calls"] = int(_FFMPEG_STATS["total_calls"] or 0) + 1
        _FFMPEG_STATS["total_wait_seconds"] = float(_FFMPEG_STATS["total_wait_seconds"] or 0.0) + wait_seconds
        _FFMPEG_STATS["last_wait_seconds"] = wait_seconds
        _FFMPEG_STATS["active"] = int(_FFMPEG_STATS["active"] or 0) + 1
        _FFMPEG_STATS["max_active"] = max(int(_FFMPEG_STATS["max_active"] or 0), int(_FFMPEG_STATS["active"] or 0))
        run_started = asyncio.get_running_loop().time()
        try:
            await AudioOps._run_ffmpeg_unbounded(cmd)
        finally:
            run_seconds = asyncio.get_running_loop().time() - run_started
            _FFMPEG_STATS["total_run_seconds"] = float(_FFMPEG_STATS["total_run_seconds"] or 0.0) + run_seconds
            _FFMPEG_STATS["last_run_seconds"] = run_seconds
            _FFMPEG_STATS["active"] = max(0, int(_FFMPEG_STATS["active"] or 0) - 1)

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
        limit = AudioOps._configured_ffmpeg_concurrency()
        if limit is None:
            return None
        key = (id(asyncio.get_running_loop()), limit)
        semaphore = _FFMPEG_SEMAPHORES.get(key)
        if semaphore is None:
            semaphore = asyncio.Semaphore(limit)
            _FFMPEG_SEMAPHORES[key] = semaphore
        return semaphore

    @staticmethod
    def _configured_ffmpeg_concurrency() -> int | None:
        raw_limit = os.getenv("FFMPEG_CONCURRENCY", "4").strip()
        if raw_limit in {"", "0"}:
            return None
        return max(1, int(raw_limit))

    @staticmethod
    def _set_configured_concurrency() -> None:
        configured = AudioOps._configured_ffmpeg_concurrency()
        if configured is None:
            _FFMPEG_STATS["configured_concurrency"] = None
            return
        _FFMPEG_STATS["configured_concurrency"] = configured

    @staticmethod
    def runtime_status() -> dict[str, Any]:
        AudioOps._set_configured_concurrency()
        total_calls = int(_FFMPEG_STATS["total_calls"] or 0)
        total_wait = float(_FFMPEG_STATS["total_wait_seconds"] or 0.0)
        total_run = float(_FFMPEG_STATS["total_run_seconds"] or 0.0)
        return {
            "configured_concurrency": _FFMPEG_STATS["configured_concurrency"],
            "active": int(_FFMPEG_STATS["active"] or 0),
            "max_active": int(_FFMPEG_STATS["max_active"] or 0),
            "total_calls": total_calls,
            "total_wait_seconds": round(total_wait, 6),
            "total_run_seconds": round(total_run, 6),
            "avg_wait_seconds": round(total_wait / total_calls, 6) if total_calls else 0.0,
            "avg_run_seconds": round(total_run / total_calls, 6) if total_calls else 0.0,
            "last_wait_seconds": round(float(_FFMPEG_STATS["last_wait_seconds"] or 0.0), 6),
            "last_run_seconds": round(float(_FFMPEG_STATS["last_run_seconds"] or 0.0), 6),
        }

    @staticmethod
    def reset_runtime_status() -> None:
        _FFMPEG_STATS.update(
            {
                "configured_concurrency": None,
                "active": 0,
                "max_active": 0,
                "total_calls": 0,
                "total_wait_seconds": 0.0,
                "total_run_seconds": 0.0,
                "last_wait_seconds": 0.0,
                "last_run_seconds": 0.0,
            }
        )
