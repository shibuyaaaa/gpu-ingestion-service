import os
from dataclasses import dataclass
from pathlib import Path


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


@dataclass(frozen=True)
class Settings:
    service_name: str = os.getenv("SERVICE_NAME", "gpu-ingestion-service")
    queue_db_path: Path = Path(os.getenv("QUEUE_DB_PATH", "data/queue.sqlite3"))
    work_dir: Path = Path(os.getenv("WORK_DIR", "tmp"))
    max_total_queue_depth: int = _int_env("MAX_TOTAL_QUEUE_DEPTH", 200)
    download_workers: int = _int_env("DOWNLOAD_WORKERS", _int_env("PREP_WORKERS", 4))
    process_workers: int = _int_env("PROCESS_WORKERS", _int_env("POSTPROCESS_WORKERS", 1))
    download_batch_size: int = _int_env("DOWNLOAD_BATCH_SIZE", _int_env("PREP_BATCH_SIZE", 2))
    process_batch_size: int = _int_env("PROCESS_BATCH_SIZE", _int_env("POSTPROCESS_BATCH_SIZE", 1))
    analyze_batch_size: int = _int_env("ANALYZE_BATCH_SIZE", _int_env("GPU_BATCH_SIZE", 1))
    worker_poll_seconds: float = float(os.getenv("WORKER_POLL_SECONDS", "0.25"))
    job_lease_timeout_seconds: int = _int_env("JOB_LEASE_TIMEOUT_SECONDS", 1800)
    start_workers: bool = _bool_env("START_WORKERS", True)
    dry_run_mode: bool = _bool_env("DRY_RUN_MODE", False)
    model_backend: str = os.getenv("MODEL_BACKEND", "local").strip().lower()
    gpu_device: str = os.getenv("GPU_DEVICE", "cuda:0")
    htdemucs_model: str = os.getenv("HTDEMUCS_MODEL", "htdemucs")
    all_in_one_model: str = os.getenv("ALL_IN_ONE_MODEL", "harmonix-fold0")
    all_in_one_audio_separator_model: str = os.getenv("ALL_IN_ONE_AUDIO_SEPARATOR_MODEL", "Kim_Vocal_2.onnx")
    all_in_one_auth: str = os.getenv("ALL_IN_ONE_AUTH", "none").strip().lower()
    all_in_one_api_key: str = os.getenv("ALL_IN_ONE_API_KEY", "").strip()
    all_in_one_timeout_seconds: float = _float_env("ALL_IN_ONE_TIMEOUT_SECONDS", 1800.0)
    all_in_one_id_token_audience: str = os.getenv("ALL_IN_ONE_ID_TOKEN_AUDIENCE", "").strip()
    cuda_runtime_tuning: bool = _bool_env("CUDA_RUNTIME_TUNING", True)
    cuda_allow_tf32: bool = _bool_env("CUDA_ALLOW_TF32", True)
    cuda_cudnn_benchmark: bool = _bool_env("CUDA_CUDNN_BENCHMARK", True)
    cuda_matmul_precision: str = os.getenv("CUDA_MATMUL_PRECISION", "high")
    cuda_empty_cache_after_job: bool = _bool_env("CUDA_EMPTY_CACHE_AFTER_JOB", False)
    pinned_audio_staging: bool = _bool_env("PINNED_AUDIO_STAGING", True)
    pinned_audio_seconds: float = _float_env("PINNED_AUDIO_SECONDS", 600.0)
    pinned_audio_channels: int = _int_env("PINNED_AUDIO_CHANNELS", 2)
    pinned_audio_sample_rate: int = _int_env("PINNED_AUDIO_SAMPLE_RATE", 44100)
    pinned_audio_slots: int = _int_env("PINNED_AUDIO_SLOTS", 2)
    cuda_graphs_enabled: bool = _bool_env("CUDA_GRAPHS_ENABLED", False)
    cuda_graph_audio_seconds: float = _float_env("CUDA_GRAPH_AUDIO_SECONDS", 30.0)
    gcp_project_id: str = os.getenv("GCP_PROJECT_ID", "imposing-kayak-422917-b0")
    gcp_bucket_name: str = os.getenv("GCP_BUCKET_NAME", "shibuya-assets")
    cdn_base_url: str = os.getenv("CDN_BASE_URL", "https://cdn.shibuyaaa.com").rstrip("/")
    cloud_run_fallback_url: str = os.getenv("ALL_IN_ONE_GCP_URL", "").rstrip("/")


settings = Settings()
