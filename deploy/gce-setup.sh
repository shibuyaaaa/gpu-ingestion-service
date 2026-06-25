#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sudo sh
fi

distribution=$(. /etc/os-release; echo "$ID$VERSION_ID")
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

sudo mkdir -p /var/lib/gpu-ingestion/tmp
sudo chmod 0775 /var/lib/gpu-ingestion

if [ ! -f /etc/gpu-ingestion.env ]; then
  sudo tee /etc/gpu-ingestion.env >/dev/null <<'ENV'
SERVICE_NAME=gpu-ingestion-service
QUEUE_DB_PATH=/var/lib/gpu-ingestion/queue.sqlite3
WORK_DIR=/var/lib/gpu-ingestion/tmp
INGESTION_CPUSET=0-2
START_WORKERS=true
DRY_RUN_MODE=false
MODEL_BACKEND=local
GPU_DEVICE=cuda:0
HTDEMUCS_MODEL=htdemucs_ft
ALL_IN_ONE_DEMUCS_MODEL=htdemucs_ft
ALL_IN_ONE_DEMUCS_SEGMENT_SECONDS=7.5
ALL_IN_ONE_DEMUCS_MAX_SEGMENT_SECONDS=7.5
ALL_IN_ONE_DEMUCS_OVERLAP=0.05
ALL_IN_ONE_DEMUCS_JOBS=0
ALL_IN_ONE_DEMUCS_SAVE_WORKERS=2
ALL_IN_ONE_DEMUCS_PRELOAD=true
FFMPEG_THREADS=1
YTDLP_DOWNLOAD_ATTEMPTS=3
YTDLP_RETRY_DELAY_SECONDS=2
ALL_IN_ONE_MODEL=harmonix-fold0
SEGMENT_MANIFEST_UPLOAD_ENABLED=false
CUDA_RUNTIME_TUNING=true
CUDA_ALLOW_TF32=true
CUDA_CUDNN_BENCHMARK=true
CUDA_MATMUL_PRECISION=high
CUDA_EMPTY_CACHE_AFTER_JOB=false
PINNED_AUDIO_STAGING=true
PINNED_AUDIO_SECONDS=600
PINNED_AUDIO_CHANNELS=2
PINNED_AUDIO_SAMPLE_RATE=44100
PINNED_AUDIO_SLOTS=2
CUDA_GRAPHS_ENABLED=false
CUDA_GRAPH_AUDIO_SECONDS=30
GPU_HEALTH_RESTART_ENABLED=true
GPU_HEALTH_CHECK_INTERVAL_SECONDS=60
GPU_HEALTH_RESTART_FAILURES=2
GPU_PROBE_CACHE_SECONDS=1.0
GPU_JOB_SAMPLE_INTERVAL_SECONDS=0.5
MAX_TOTAL_QUEUE_DEPTH=1000
DOWNLOAD_WORKERS=2
DOWNLOAD_BATCH_SIZE=1
PROCESS_WORKERS=6
PROCESS_BATCH_SIZE=1
ANALYZE_BATCH_SIZE=1
FFMPEG_CONCURRENCY=4
WORKER_POLL_SECONDS=0.10
DEFAULT_RETRY_DELAY_SECONDS=30
DOWNLOAD_RETRY_DELAY_SECONDS=5
JOB_LEASE_TIMEOUT_SECONDS=1800
WORK_DIR_CLEANUP_ENABLED=true
WORK_DIR_CLEANUP_INTERVAL_SECONDS=300
WORK_DIR_CLEANUP_MIN_AGE_SECONDS=900
WORK_DIR_CLEANUP_MAX_DIRS_PER_RUN=100
SOURCE_AUDIO_CACHE_ENABLED=true
SOURCE_AUDIO_CACHE_MAX_ENTRIES=100
SOURCE_AUDIO_CACHE_MAX_BYTES=8gb
SOURCE_AUDIO_UPLOAD_ENABLED=true
ANALYZER_RESULT_UPLOAD_ENABLED=true
ANALYSIS_CACHE_ENABLED=true
ANALYSIS_CACHE_MAX_ENTRIES=4
ANALYSIS_CACHE_MAX_BYTES=2gb
SEGMENT_STEM_CACHE_ENABLED=true
SEGMENT_STEM_CACHE_MAX_ENTRIES=1000
SEGMENT_STEM_CACHE_MAX_BYTES=8gb
GCS_SEGMENT_UPLOAD_CACHE_ENABLED=true
GCS_SEGMENT_UPLOAD_URL_CACHE_MAX_ENTRIES=10000
GCS_SEGMENT_UPLOAD_DISK_CACHE_ENABLED=true
GCP_PROJECT_ID=imposing-kayak-422917-b0
GCP_BUCKET_NAME=shibuya-assets
CDN_BASE_URL=https://cdn.shibuyaaa.com
DATABASE_URL=
DB_POOL_MIN_SIZE=1
DB_POOL_MAX_SIZE=5
LIBRARY_PRECHECK_ENABLED=true
LIBRARY_CACHE_IDLE_TTL_SECONDS=600
LIBRARY_CACHE_MAX_AGE_SECONDS=300
SPOTIFY_CLIENT_ID=
SPOTIFY_CLIENT_SECRET=
CRAWLER_ENABLED=false
CRAWLER_BATCH_SIZE=50
CRAWLER_POLL_SECONDS=60
CRAWLER_CPUSET=3
CRAWLER_SPOTIFY_PLAYLIST_URLS=
CRAWLER_KWORB_CHART_URLS=https://kworb.net/spotify/country/us_daily_totals.html,https://kworb.net/spotify/country/ca_daily_totals.html,https://kworb.net/spotify/country/global_daily_totals.html,https://kworb.net/spotify/country/us_daily.html,https://kworb.net/spotify/country/ca_daily.html,https://kworb.net/spotify/country/global_daily.html
CRAWLER_INGESTION_URL=http://127.0.0.1:8080
CRAWLER_SESSION_DB_PATH=/var/lib/gpu-ingestion/crawler.sqlite3
CRAWLER_MAX_CANDIDATE_PAGES=10
CRAWLER_OPS_BASE_URL=http://127.0.0.1:8080
ENV
fi

sudo cp deploy/gpu-ingestion.service /etc/systemd/system/gpu-ingestion.service
sudo cp deploy/gpu-ingestion-crawler.service /etc/systemd/system/gpu-ingestion-crawler.service
sudo systemctl daemon-reload
echo "Edit /etc/gpu-ingestion.env, then run: sudo systemctl enable --now gpu-ingestion"
echo "For autonomous crawling, set CRAWLER_ENABLED=true and CRAWLER_KWORB_CHART_URLS, then run: sudo systemctl enable --now gpu-ingestion-crawler"
