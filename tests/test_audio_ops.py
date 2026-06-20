import asyncio

import pytest

from app.legacy.audio import AudioOps


def test_ffmpeg_base_command_caps_threads_by_default(monkeypatch):
    monkeypatch.delenv("FFMPEG_THREADS", raising=False)

    cmd = AudioOps._base_ffmpeg_cmd()

    assert cmd[:5] == ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    assert cmd[cmd.index("-threads") + 1] == "1"


def test_ffmpeg_base_command_allows_explicit_thread_count(monkeypatch):
    monkeypatch.setenv("FFMPEG_THREADS", "2")

    cmd = AudioOps._base_ffmpeg_cmd()

    assert cmd[cmd.index("-threads") + 1] == "2"


def test_ffmpeg_base_command_can_leave_ffmpeg_default_threads(monkeypatch):
    monkeypatch.setenv("FFMPEG_THREADS", "0")

    cmd = AudioOps._base_ffmpeg_cmd()

    assert "-threads" not in cmd


@pytest.mark.asyncio
async def test_extract_segment_uses_thread_cap(monkeypatch, tmp_path):
    captured = {}

    class FakeProc:
        returncode = 0

        async def communicate(self):
            return b"", b""

    async def fake_create_subprocess_exec(*cmd, stdout, stderr):
        captured["cmd"] = list(cmd)
        return FakeProc()

    monkeypatch.setenv("FFMPEG_THREADS", "1")
    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_create_subprocess_exec)

    await AudioOps.extract_segment(
        tmp_path / "input.wav",
        tmp_path / "output.mp3",
        start=1.25,
        duration=8.5,
    )

    cmd = captured["cmd"]
    assert cmd[cmd.index("-threads") + 1] == "1"
    assert cmd[cmd.index("-ss") + 1] == "1.25"
    assert cmd[cmd.index("-t") + 1] == "8.5"


@pytest.mark.asyncio
async def test_ffmpeg_concurrency_limits_active_subprocesses(monkeypatch, tmp_path):
    active = 0
    max_active = 0

    class FakeProc:
        returncode = 0

        async def communicate(self):
            nonlocal active
            await asyncio.sleep(0.01)
            active -= 1
            return b"", b""

    async def fake_create_subprocess_exec(*cmd, stdout, stderr):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        return FakeProc()

    monkeypatch.setenv("FFMPEG_CONCURRENCY", "2")
    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_create_subprocess_exec)

    await asyncio.gather(
        *(
            AudioOps.convert_to_mp3(tmp_path / f"input-{index}.wav", tmp_path / f"output-{index}.mp3")
            for index in range(5)
        )
    )

    assert max_active == 2


@pytest.mark.asyncio
async def test_ffmpeg_runtime_status_tracks_wait_and_active_cap(monkeypatch, tmp_path):
    AudioOps.reset_runtime_status()

    class FakeProc:
        returncode = 0

        async def communicate(self):
            await asyncio.sleep(0.01)
            return b"", b""

    async def fake_create_subprocess_exec(*cmd, stdout, stderr):
        return FakeProc()

    monkeypatch.setenv("FFMPEG_CONCURRENCY", "2")
    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_create_subprocess_exec)

    await asyncio.gather(
        *(
            AudioOps.convert_to_mp3(tmp_path / f"input-{index}.wav", tmp_path / f"output-{index}.mp3")
            for index in range(5)
        )
    )

    status = AudioOps.runtime_status()
    assert status["configured_concurrency"] == 2
    assert status["active"] == 0
    assert status["max_active"] == 2
    assert status["total_calls"] == 5
    assert status["total_run_seconds"] > 0
    assert status["total_wait_seconds"] > 0
