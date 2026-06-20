# GPU Ingestion Service

Standalone local-first ingestion worker for a single GCE L4 GPU server.

This repo intentionally does not import or mutate the legacy `ingestion-api`
worker. It copies/adapts the small reusable pieces needed for a local pipeline:
durable queueing, source download, GCS IO, model runtimes, post-processing, and
ops visibility.

## Runtime Shape

```text
local HTTP job
  -> FastAPI
  -> SQLite/WAL durable local queue
  -> download workers
  -> analyze worker
       all-in-one/Harmonix structure analysis
  -> process workers
       quick dissect: HTDemucs on chorus, then enqueue bulk continuation
       bulk dissect: HTDemucs one segment per queue claim
  -> GCS + DB updates
```

`POST /jobs` is the VM-local ingress path used by the crawler and manual
replay. `POST /pubsub` remains available only as a compatibility receiver for
legacy/shadow traffic. Once a request is committed to SQLite, the local worker
scheduler owns the job lifecycle.

For production smoke tests, include `skip_library_precheck: true` and
`skip_library_write: true` in the job payload. That forces real processing while
preventing Aime library rows from being created or updated.

The only canonical job types are:

```text
quick_dissect
bulk_dissect
```

Each job must include one source field containing a Spotify track link or song
search string:

```json
{"job_type": "quick_dissect", "source": "https://open.spotify.com/track/..."}
{"job_type": "bulk_dissect", "source": "Daft Punk One More Time"}
```

## Local Development

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
DRY_RUN_MODE=true pytest
DRY_RUN_MODE=true uvicorn app.server:app --reload --port 8080
```

`DRY_RUN_MODE=true` makes model stages create deterministic local artifacts
without requiring CUDA, HTDemucs, or all-in-one model packages. Production should
run with `DRY_RUN_MODE=false`.

## Important Environment

- `QUEUE_DB_PATH`: SQLite queue path. Default: `data/queue.sqlite3`.
- `WORK_DIR`: local temp/artifact working directory. Default: `tmp`.
- `MAX_TOTAL_QUEUE_DEPTH`: backpressure limit. Default: `1000` for crawler fanout.
- `DOWNLOAD_WORKERS`, `DOWNLOAD_BATCH_SIZE`: local source/download workers.
- `ANALYZE_BATCH_SIZE`: analyze-stage jobs per GPU loop. Default: `1` for one L4.
- `PROCESS_WORKERS`, `PROCESS_BATCH_SIZE`: process-stage segment workers. Default: `4` workers with `FFMPEG_THREADS=1` to hide GCS/DB/ffmpeg wait on the 3-core ingestion cpuset without increasing GPU concurrency.
- `WORKER_POLL_SECONDS`: idle queue polling interval. Default: `0.10` to keep fanout and quick-job handoff latency low.
- `FFMPEG_THREADS`: per-ffmpeg subprocess thread cap for CPU segment slicing/transcoding. Default: `1` to avoid oversubscribing a 4-vCPU L4 VM with multiple process workers.
- `JOB_LEASE_TIMEOUT_SECONDS`: startup recovery timeout for crashed workers.
- `WORK_DIR_CLEANUP_*`: terminal job workdir cleanup. The service removes only
  workdirs no non-terminal job still references.
- `MODEL_BACKEND`: `local`, `remote_gpu`, or `cloud_run_fallback`. Default: `local`.
- `ALL_IN_ONE_GCP_URL`: Cloud Run all-in-one `/predict` service URL for remote GPU mode.
- `ALL_IN_ONE_AUTH`: `none`, `api_key`, `google_id_token`, or `gcloud_identity_token`.
- `ALL_IN_ONE_TIMEOUT_SECONDS`: remote GPU request timeout. Default: `1800`.
- `ALL_IN_ONE_DEMUCS_SEGMENT_SECONDS`: memory-bounded Demucs segment size used inside all-in-one analysis. Default: `7.5`, measured faster than `5` on the L4 while staying below the unsafe `15` second HTDemucs shape boundary.
- `ALL_IN_ONE_DEMUCS_MAX_SEGMENT_SECONDS`: safety clamp for the effective Demucs segment size. Default: `7.5`; raise only for measured experiments.
- `ALL_IN_ONE_DEMUCS_OVERLAP`: Demucs split overlap inside all-in-one analysis. Default: `0.10` for faster L4 throughput.
- `ALL_IN_ONE_DEMUCS_JOBS`: Demucs worker count used inside all-in-one analysis. Default: `0`.
- `ALL_IN_ONE_DEMUCS_SAVE_WORKERS`: CPU workers for saving Demucs stems after inference. Default: `2`.
- `GPU_HEALTH_RESTART_*`: restart the service after repeated container GPU/NVML failures.
- `GPU_PROBE_CACHE_SECONDS`: short cache for `nvidia-smi` ops probes. Default: `1.0`.
- `GPU_JOB_SAMPLE_INTERVAL_SECONDS`: per-GPU-job `nvidia-smi` sampling interval for `/ops/timings`. Default: `0.5`; set `0` to disable. This is a coarse utilization proxy, not true FLOP MFU.
- `DRY_RUN_MODE`: test/dev mode. Default: `false`.
- `GCP_PROJECT_ID`, `GCP_BUCKET_NAME`, `CDN_BASE_URL`: GCP output config.
- `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`: required for Spotify link/name resolution.
- `CRAWLER_INGESTION_URL`: VM-local ingestion API URL. Default: `http://127.0.0.1:8080`.
- `GPU_DEVICE`: CUDA device string. Default: `cuda:0`.

## Queue Choice

The first production target is one GPU VM, so the service uses SQLite/WAL as the
single local durable queue and job state table. The crawler submits to the local
FastAPI service on the same VM; after the service accepts a request, SQLite
tracks stage, status, retries, artifacts, events, and ops visibility.

Workers claim queued jobs by priority first, then FIFO within the same priority.
Explicit payload `priority` wins. If omitted, job-type defaults are applied:
`quick_dissect` is highest priority and `bulk_dissect` is lower priority.

This avoids running a second local queue system on the same machine and keeps
the failure boundary simple.

## Production Target

Start with a GCE `g2-standard-4` VM:

- 1 NVIDIA L4
- 4 vCPU
- 16 GiB RAM
- persistent disk for queue, temp files, model cache

Run with Docker or systemd using the files in `deploy/`.

See `deploy/README.md` for the GCE setup sequence.
See `docs/gpu-runtime.md` for pinned-memory, CUDA, and model-residency policy.

Use `/ops/readiness` before sending real jobs. In production mode it requires
queue/work directories, ffmpeg, GPU visibility, and loaded local models.
