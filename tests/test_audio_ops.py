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
