import asyncio
import mimetypes
import shutil
import subprocess
from pathlib import Path
from typing import Any


class GCSClient:
    def __init__(self, *, project_id: str, bucket_name: str, cdn_base_url: str):
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.cdn_base_url = cdn_base_url.rstrip("/")
        self._client: Any | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            from google.cloud import storage

            self._client = storage.Client(project=self.project_id)
        return self._client

    def health(self) -> bool:
        return self.client.bucket(self.bucket_name).exists()

    async def exists(self, gcs_path: str, *, bucket_name: str | None = None) -> bool:
        return await asyncio.to_thread(self._exists_sync, gcs_path, bucket_name)

    def _exists_sync(self, gcs_path: str, bucket_name: str | None) -> bool:
        resolved_bucket = bucket_name or self.bucket_name
        try:
            bucket = self.client.bucket(resolved_bucket)
            return bool(bucket.blob(gcs_path).exists())
        except Exception:
            return self._gcloud_storage_exists(f"gs://{resolved_bucket}/{gcs_path}")

    async def download(self, gcs_path: str, local_path: str | Path, *, bucket_name: str | None = None) -> str:
        return await asyncio.to_thread(self._download_sync, gcs_path, Path(local_path), bucket_name)

    def _download_sync(self, gcs_path: str, local_path: Path, bucket_name: str | None) -> str:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_bucket = bucket_name or self.bucket_name
        try:
            bucket = self.client.bucket(resolved_bucket)
            bucket.blob(gcs_path).download_to_filename(str(local_path))
        except Exception:
            self._gcloud_storage_cp(f"gs://{resolved_bucket}/{gcs_path}", str(local_path))
        return str(local_path)

    async def upload(
        self,
        local_path: str | Path,
        gcs_path: str,
        *,
        bucket_name: str | None = None,
        content_type: str | None = None,
    ) -> str:
        return await asyncio.to_thread(
            self._upload_sync,
            Path(local_path),
            gcs_path,
            bucket_name,
            content_type,
        )

    def _upload_sync(
        self,
        local_path: Path,
        gcs_path: str,
        bucket_name: str | None,
        content_type: str | None,
    ) -> str:
        resolved_bucket = bucket_name or self.bucket_name
        resolved_type = content_type or mimetypes.guess_type(local_path.name)[0] or "application/octet-stream"
        try:
            bucket = self.client.bucket(resolved_bucket)
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(str(local_path), content_type=resolved_type)
        except Exception:
            self._gcloud_storage_cp(str(local_path), f"gs://{resolved_bucket}/{gcs_path}", content_type=resolved_type)
        return f"{self.cdn_base_url}/{gcs_path}"

    @staticmethod
    def _gcloud_storage_cp(source: str, target: str, *, content_type: str | None = None) -> None:
        if not shutil.which("gcloud"):
            raise RuntimeError("google-cloud-storage failed and gcloud CLI is not installed")
        cmd = ["gcloud", "storage", "cp"]
        if content_type:
            cmd.extend(["--content-type", content_type])
        cmd.extend([source, target])
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

    @staticmethod
    def _gcloud_storage_exists(target: str) -> bool:
        if not shutil.which("gcloud"):
            return False
        proc = subprocess.run(
            ["gcloud", "storage", "ls", target],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return proc.returncode == 0
