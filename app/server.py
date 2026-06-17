import logging
import shutil
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response

from app.config import settings
from app.jobs import UnsupportedJobType, build_default_registry
from app.jobs.context import JobContext
from app.legacy.db import DBClient
from app.legacy.utils import GCSClient, parse_pubsub_envelope
from app.models import ModelRuntimeBundle
from app.queue import JobStore, QueueFull
from app.workers import WorkerManager

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

store = JobStore(settings.queue_db_path, max_depth=settings.max_total_queue_depth)
models = ModelRuntimeBundle.from_settings(settings)
gcs = GCSClient(
    project_id=settings.gcp_project_id,
    bucket_name=settings.gcp_bucket_name,
    cdn_base_url=settings.cdn_base_url,
)
db = DBClient()
context = JobContext(settings=settings, store=store, models=models, gcs=gcs, db=db)
registry = build_default_registry()
workers = WorkerManager(context=context, registry=registry)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.start_workers:
        await workers.start()
    try:
        yield
    finally:
        await workers.stop()


app = FastAPI(title="GPU Ingestion Service", lifespan=lifespan)


@app.post("/pubsub")
async def handle_pubsub(request: Request) -> Response:
    if not workers.accepting:
        return Response(content="draining", status_code=503)
    try:
        envelope = await request.json()
        payload, message_id = parse_pubsub_envelope(envelope)
        _validate_supported_job(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        job, created = store.enqueue(payload, source_message_id=message_id)
    except QueueFull as exc:
        return Response(content=str(exc), status_code=429)

    logger.info("accepted pubsub job id=%s type=%s created=%s", job.id, job.job_type, created)
    return Response(content="OK", status_code=200)


@app.post("/jobs")
async def enqueue_manual_job(payload: dict[str, Any]) -> dict[str, Any]:
    if not workers.accepting:
        raise HTTPException(status_code=503, detail="service is draining")
    try:
        _validate_supported_job(payload)
    except (UnsupportedJobType, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        job, created = store.enqueue(payload)
    except QueueFull as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    return {
        "job_id": job.id,
        "job_type": job.job_type.value,
        "created": created,
        "stage": job.stage.value,
        "status": job.status.value,
        "priority": job.priority,
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    queue_state = store.stats()
    external: dict[str, Any] = {"gcs": "skipped", "db": "skipped"}
    if not settings.dry_run_mode:
        external["gcs"] = await _safe_async_bool(lambda: gcs.health())
        external["db"] = await db.health()
    model_state = models.status()
    degraded = queue_state["backpressure"] or (
        not settings.dry_run_mode and (external["gcs"] is False or external["db"] is False)
    )
    return {
        "status": "degraded" if degraded else "healthy",
        "service": settings.service_name,
        "dry_run_mode": settings.dry_run_mode,
        "ingress": {"mode": "http", "queue": "sqlite"},
        "queue": queue_state,
        "workers": workers.state(),
        "external": external,
        "models": model_state["models"],
        "gpu": model_state["gpu"],
    }


@app.get("/ops/state")
async def ops_state() -> dict[str, Any]:
    return {
        "queue": store.stats(),
        "workers": workers.state(),
        "models": models.status(),
        "ingress": {"mode": "http", "queue": "sqlite"},
    }


@app.get("/ops/readiness")
async def ops_readiness() -> dict[str, Any]:
    checks = {
        "queue_db_parent_writable": _path_writable(settings.queue_db_path.parent),
        "work_dir_writable": _path_writable(settings.work_dir),
        "ffmpeg_available": shutil.which("ffmpeg") is not None,
        "gpu_visible": models.status()["gpu"]["available"],
        "models_loaded": {
            "htdemucs": models.htdemucs.loaded,
            "all_in_one": models.all_in_one.loaded,
        },
    }
    required = [
        checks["queue_db_parent_writable"],
        checks["work_dir_writable"],
        checks["ffmpeg_available"],
    ]
    if not settings.dry_run_mode:
        required.append(checks["gpu_visible"])
        required.extend(checks["models_loaded"].values())
    ready = all(required)
    return {"ready": ready, "dry_run_mode": settings.dry_run_mode, "checks": checks}


@app.get("/ops/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return {
        "id": job.id,
        "job_type": job.job_type.value,
        "stage": job.stage.value,
        "status": job.status.value,
        "payload": job.payload,
        "artifacts": job.artifacts,
        "attempts": job.attempts,
        "max_attempts": job.max_attempts,
        "priority": job.priority,
        "worker_id": job.worker_id,
        "error": job.error,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "events": store.recent_events(job.id),
    }


@app.get("/ops/gpu")
async def ops_gpu() -> dict[str, Any]:
    return models.status()


@app.post("/ops/jobs/{job_id}/retry")
async def retry_job(job_id: str) -> dict[str, Any]:
    try:
        job = store.retry(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc
    return {"job_id": job.id, "stage": job.stage.value, "status": job.status.value}


@app.post("/ops/drain")
async def drain() -> dict[str, Any]:
    workers.drain()
    return {"accepting": False, "message": "service is draining; existing queued work will continue"}


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": settings.service_name, "status": "running"}


async def _safe_async_bool(fn) -> bool:
    import asyncio

    try:
        return bool(await asyncio.to_thread(fn))
    except Exception:
        return False


def _path_writable(path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _validate_supported_job(payload: dict[str, Any]) -> None:
    registry.get(str(payload.get("job_type") or "unknown"))
    source = (
        payload.get("source")
        or payload.get("youtube_url")
        or payload.get("spotify_source")
        or payload.get("spotify_url")
        or payload.get("spotify_query")
    )
    if not source or not str(source).strip():
        raise ValueError("job payload must include source, youtube_url, spotify_source, spotify_url, or spotify_query")
