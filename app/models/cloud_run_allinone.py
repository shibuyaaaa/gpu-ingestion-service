import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import httpx


class CloudRunAllInOneRuntime:
    """Cloud Run wrapper for the legacy all-in-one analyzer + Demucs service."""

    def __init__(
        self,
        *,
        url: str,
        model_name: str,
        audio_separator_model: str,
        auth: str = "none",
        api_key: str = "",
        timeout_seconds: float = 1800.0,
        id_token_audience: str = "",
        upload_transcode_enabled: bool = True,
        max_upload_bytes: int = 24_000_000,
        upload_bitrate: str = "320k",
    ):
        self.url = url.rstrip("/")
        self.model_name = model_name
        self.audio_separator_model = audio_separator_model
        self.auth = auth.strip().lower()
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.id_token_audience = id_token_audience.strip()
        self.upload_transcode_enabled = upload_transcode_enabled
        self.max_upload_bytes = max_upload_bytes
        self.upload_bitrate = upload_bitrate
        self.loaded = False
        self.dry_run = False

    async def load(self) -> None:
        if not self.url:
            raise RuntimeError("ALL_IN_ONE_GCP_URL is required for Cloud Run model backend")
        self.loaded = True

    async def analyze(self, audio_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
        await self.load()
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        response = await self._predict(audio_path)

        analyzer_url = response.get("analyzer_result")
        analysis: dict[str, Any] = {}
        analysis_path: Path | None = None
        if isinstance(analyzer_url, str) and analyzer_url:
            analysis = await self._download_json(analyzer_url)
            analysis_path = output / "analyzer_result.json"
            analysis_path.write_text(json.dumps(analysis), encoding="utf-8")

        return {
            "analysis": analysis,
            "analyzer_result_path": str(analysis_path) if analysis_path else None,
            "analyzer_result_url": analyzer_url,
            "stem_urls": self._stem_urls(response),
            "raw_response": response,
        }

    async def separate(self, audio_path: str | Path, output_dir: str | Path) -> dict[str, str]:
        await self.load()
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        response = await self._predict(audio_path)
        stems: dict[str, str] = {}
        for stem, url in self._stem_urls(response).items():
            suffix = Path(url).suffix or ".wav"
            target = output / f"{stem}{suffix}"
            await self._download_file(url, target)
            stems[stem] = str(target)
        if not stems:
            raise RuntimeError("Cloud Run all-in-one response did not include demucs stem URLs")
        return stems

    async def _predict(self, audio_path: str | Path) -> dict[str, Any]:
        headers = await asyncio.to_thread(self._headers)
        data = {
            "audioSeparator": "false",
            "audioSeparatorModel": self.audio_separator_model,
            "model": self.model_name,
            "visualize": "false",
            "sonify": "false",
            "include_embeddings": "false",
            "include_activations": "false",
        }
        source_path = Path(audio_path)
        upload_path, cleanup_path = await self._upload_path(source_path)
        try:
            with upload_path.open("rb") as audio_file:
                files = {
                    "music_input": (
                        upload_path.name,
                        audio_file,
                        self._content_type(upload_path),
                    )
                }
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.post(f"{self.url}/predict", data=data, files=files, headers=headers)
        finally:
            if cleanup_path is not None:
                cleanup_path.unlink(missing_ok=True)
        if response.status_code >= 400:
            raise RuntimeError(f"Cloud Run all-in-one failed: HTTP {response.status_code}: {response.text[:1000]}")
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Cloud Run all-in-one returned a non-object response")
        return payload

    async def _upload_path(self, path: Path) -> tuple[Path, Path | None]:
        if not self.upload_transcode_enabled:
            return path, None
        if path.stat().st_size <= self.max_upload_bytes:
            return path, None
        target = path.with_name(f"{path.stem}.cloudrun-upload.mp3")
        await self._transcode_for_upload(path, target)
        return target, target

    async def _transcode_for_upload(self, source: Path, target: Path) -> None:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-codec:a",
            "libmp3lame",
            "-b:a",
            self.upload_bitrate,
            str(target),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(stderr.decode("utf-8", errors="replace")[-1000:])

        # Keep a final guard because Cloud Run rejects oversized multipart bodies before app code runs.
        if target.stat().st_size > self.max_upload_bytes:
            lower_bitrate = os.getenv("CLOUD_RUN_UPLOAD_FALLBACK_BITRATE", "192k").strip() or "192k"
            if lower_bitrate == self.upload_bitrate:
                raise RuntimeError(f"transcoded upload is still too large: {target.stat().st_size} bytes")
            cmd[cmd.index(self.upload_bitrate)] = lower_bitrate
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(stderr.decode("utf-8", errors="replace")[-1000:])
            if target.stat().st_size > self.max_upload_bytes:
                raise RuntimeError(f"transcoded upload is still too large: {target.stat().st_size} bytes")

    @staticmethod
    def _content_type(path: Path) -> str:
        if path.suffix.lower() == ".mp3":
            return "audio/mpeg"
        if path.suffix.lower() in {".wav", ".wave"}:
            return "audio/wav"
        return "application/octet-stream"

    def _headers(self) -> dict[str, str]:
        if self.auth in {"", "none"}:
            return {}
        if self.auth == "api_key":
            if not self.api_key:
                raise RuntimeError("ALL_IN_ONE_AUTH=api_key requires ALL_IN_ONE_API_KEY")
            return {"Authorization": f"Bearer {self.api_key}"}
        if self.auth == "google_id_token":
            from google.auth.transport.requests import Request as GoogleAuthRequest
            from google.oauth2 import id_token

            audience = self.id_token_audience or self.url
            token = id_token.fetch_id_token(GoogleAuthRequest(), audience)
            return {"Authorization": f"Bearer {token}"}
        if self.auth == "gcloud_identity_token":
            cmd = ["gcloud", "auth", "print-identity-token"]
            if self.id_token_audience:
                cmd.extend(["--audiences", self.id_token_audience])
            token = subprocess.check_output(cmd, text=True).strip()
            return {"Authorization": f"Bearer {token}"}
        raise RuntimeError(f"Unsupported ALL_IN_ONE_AUTH value: {self.auth}")

    @staticmethod
    def _stem_urls(response: dict[str, Any]) -> dict[str, str]:
        mapping = {
            "vocals": "demucs_vocals",
            "drums": "demucs_drums",
            "bass": "demucs_bass",
            "other": "demucs_other",
        }
        stems = {}
        for stem, key in mapping.items():
            value = response.get(key)
            if isinstance(value, str) and value:
                stems[stem] = value
        return stems

    @staticmethod
    async def _download_json(url: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.get(url)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"analyzer result is not a JSON object: {url}")
        return payload

    @staticmethod
    async def _download_file(url: str, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
            response = await client.get(url)
        response.raise_for_status()
        target.write_bytes(response.content)

    def status(self) -> dict[str, Any]:
        return {
            "name": "all-in-one-cloud-run",
            "model_name": self.model_name,
            "url": self.url,
            "loaded": self.loaded,
            "dry_run": False,
            "resident_policy": "remote-cloud-run-gpu",
            "audio_separator_model": self.audio_separator_model,
            "auth": self.auth,
            "upload": {
                "transcode_enabled": self.upload_transcode_enabled,
                "max_bytes": self.max_upload_bytes,
                "bitrate": self.upload_bitrate,
            },
        }
