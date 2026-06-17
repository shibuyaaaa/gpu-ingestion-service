# YouTube-First Pipeline Validation

Date: 2026-06-17  
Repo: `/Users/ishaankundesu/Documents/shibuya/gpu-ingestion-service`

## Summary

The local queue/worker service can now start from a known-good YouTube URL without Spotify credentials. The tested runtime used the local SQLite queue and local workers, with GPU work delegated to the existing Cloud Run `all-in-one-audio` service. That Cloud Run service returns both Harmonix analysis and Demucs stems in the same `/predict` response.

Validated:

- `youtube_url` job input bypasses Spotify resolution.
- `quick_dissect` completes the first selected segment, uploads stems, then creates a low-priority `bulk_dissect` continuation.
- `bulk_dissect` processes one segment per claim and requeues itself until complete.
- Full-stem outputs from Cloud Run are downloaded once and reused for segment extraction.
- Final segment MP3 outputs upload to GCS/CDN.
- Queue backpressure works at the configured max depth.

## Runtime Under Test

Input video:

```text
https://www.youtube.com/watch?v=8M6PhD0v9TE
```

Video metadata resolved by `yt-dlp`:

```text
title: Inspiring Ambient | Background Music for Video | 20 Sec #musicvideo #royaltyfreemusic
channel: CircleNoteMusic - Royalty Free Music
duration: 22 seconds
```

Model backend:

```text
MODEL_BACKEND=remote_gpu
ALL_IN_ONE_GCP_URL=https://all-in-one-audio-ijvhsb7soq-uc.a.run.app
ALL_IN_ONE_AUTH=gcloud_identity_token
```

Important local auth note: local `GOOGLE_APPLICATION_CREDENTIALS` pointed at an invalid old service-account key. The GCS helper now falls back to `gcloud storage cp` if the Python storage client fails, which allowed validation with the currently working `gcloud` user login.

## Code Changes Made

- Added YouTube URL input support in:
  - `app/legacy/utils/source.py`
  - `app/jobs/adapters.py`
  - `app/server.py`
- Added Cloud Run all-in-one runtime:
  - `app/models/cloud_run_allinone.py`
- Wired remote GPU backend in:
  - `app/models/runtime.py`
- Reused remote full-song Demucs stems for segment processing:
  - `app/jobs/adapters.py`
- Added GCS CLI fallback:
  - `app/legacy/utils/gcs.py`
- Added tests for YouTube URL validation/resolution.

## Test Gates

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m compileall -q app tests
passed

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q
30 passed in 1.62s
```

## Live Quick Dissect Result

Run directory:

```text
/tmp/gpu-ingestion-live-quick-1781718149
```

Result:

```text
status: completed
attempts: 0
total wall time: 46.90s
quick job time: 26.09s
bulk continuation time: 20.67s
queue final state: completed=2, active_depth=0, failed=0
```

Stage wall-clock timing from queue events. For segment rows, this is processing time, not segment audio duration; it includes local stem extraction and four GCS/CDN uploads.

| Stage | Approx Time |
| --- | ---: |
| download | 3.81s |
| analyze + remote all-in-one/Demucs | 12.16s |
| quick process/upload selected segment | 9.97s |
| continuation segment 1 | 11.15s |
| continuation segment 2 | 9.52s |

Quick behavior:

```text
quick_dissect priority: 100
segments found: 3
quick processed: seg-0
bulk continuation id: live-quick-1:bulk
bulk continuation priority: 10
bulk skipped: seg-0
bulk processed: seg-1, seg-2
```

Uploaded quick artifact verified:

```text
https://cdn.shibuyaaa.com/gpu-ingestion/live-quick-1/segments/seg-0/vocals.mp3
HTTP 200, content-length 14912
```

## Live Bulk Dissect Result

Run directory:

```text
/tmp/gpu-ingestion-live-bulk-1781718224
```

Result:

```text
status: completed
attempts: 0
total wall time: 52.44s
job duration: 52.31s
queue final state: completed=1, active_depth=0, failed=0
```

Stage timing from queue events:

| Stage | Approx Time |
| --- | ---: |
| download | 3.53s |
| analyze + remote all-in-one/Demucs | 13.64s |
| process segment 0 | 11.61s |
| process segment 1 | 10.99s |
| process segment 2 | 12.46s |

Bulk behavior:

```text
bulk_dissect priority: 10
segments found: 3
processed one segment per process claim
processed segments: seg-0, seg-1, seg-2
skipped segments: none
```

Uploaded bulk artifact verified:

```text
https://cdn.shibuyaaa.com/gpu-ingestion/live-bulk-1/segments/seg-2/other.mp3
HTTP 200, content-length 76849
```

## Queue Load Results

Dry-run load test used YouTube-backed payloads and local workers. This intentionally avoided repeated paid Cloud Run GPU calls while validating queue/dequeue behavior.

Run directory:

```text
/tmp/gpu-ingestion-load-1781718307
```

Worker config:

```text
download_workers=4
download_batch_size=2
analyze_batch_size=1
process_workers=4
process_batch_size=4
```

Results:

| Jobs | Accepted | Rejected | Drain Time | Throughput | Max Active Depth | Max Processing |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 10 | 1 | 5.29s | 1.89 jobs/s | 10 | 10 |
| 25 | 25 | 1 | 15.16s | 1.65 jobs/s | 25 | 15 |
| 50 | 50 | 1 | 30.45s | 1.64 jobs/s | 50 | 16 |

Default queue-depth check:

```text
max_total_queue_depth=200
accepted: 200
rejected: 1
active_depth: 200
backpressure: true
```

So the current default maximum active queue depth is 200 jobs, and the store rejects the next job with `QueueFull`.

## Late Quick Job During Bulk Backlog

Additional priority tests were run after the initial pipeline validation.

Deterministic queue test:

```text
test_quick_process_job_preempts_bulk_process_backlog_after_current_claim
passed
```

The test creates five `bulk_dissect` jobs already queued in `process`, claims one bulk job as already running, then inserts a late `quick_dissect` job directly into `process`. The next queue claim is the quick job, not the older queued bulk jobs.

Worker-level dry-run process simulation:

```text
run_dir: /tmp/gpu-ingestion-priority-sim-1781719001
claim order:
1. bulk-0 process
2. quick-late process
3. bulk-0 process
quick-late status: completed
quick-late priority: 100
```

Full three-stage dry-run simulation:

```text
run_dir: /tmp/gpu-ingestion-full-priority-sim-1781719030
quick-late entered at download after bulk work had already reached process
quick-late status: completed
quick-late priority: 100
```

Relevant claim order excerpt:

```text
bulk-0 process
quick-late download
bulk-0 process
quick-late analyze
bulk-0 process
quick-late process
bulk-1 process
```

Interpretation:

- A quick job does not interrupt a segment that is already actively processing.
- Once a worker returns to the queue, priority ordering works: `quick_dissect` priority `100` beats `bulk_dissect` priority `10`.
- To keep this responsive, process-stage defaults were changed to `PROCESS_WORKERS=1` and `PROCESS_BATCH_SIZE=1`. This prevents a worker from preclaiming several bulk segments before a late quick job arrives.

## What The Runs Prove

- The local service can run without Spotify credentials when jobs include a `youtube_url`.
- The Cloud Run all-in-one service already provides Demucs stems; no separate HTDemucs endpoint is needed for this hybrid test mode.
- Quick jobs correctly finish user-facing work first and convert to low-priority bulk continuation.
- Bulk jobs do not monopolize process forever; they process one segment per queue claim.
- Queue priority and max-depth backpressure are working.
- GCS/CDN output path is functional after local auth fallback.

## Limits Of This Validation

- This was not a resident local L4 model test. GPU work still happened in Cloud Run.
- The source was a short 22-second YouTube video. Longer songs will scale mainly with Cloud Run all-in-one/Demucs time and number of segments.
- Dry-run load throughput is orchestration throughput, not real GPU throughput.
- Local Application Default Credentials are stale; use `gcloud auth application-default login` or deploy with a valid service account for the normal Python GCS client path.

## Recommended Next Step

Run a small live batch of 3 to 5 real YouTube song URLs through `remote_gpu` mode to get more realistic timing distributions, then deploy the same code onto the stopped L4 VM only after that. For the VM target, switch `MODEL_BACKEND=local` and validate resident model load once.

## Appendix: Live Segment And Stem URLs

The service stores segment metadata in each job's `artifacts.segments` and stores processed segment outputs in `artifacts.processed_segments[segment_id].outputs`.

### Quick Run: `live-quick-1`

Analyzer result:

```text
https://cdn.shibuyaaa.com/all-in-one-audio/79566431-bc61-46c6-839b-b9908576ddbb/analyzer_result.json
```

Full-song Demucs stems from Cloud Run:

```text
vocals: https://cdn.shibuyaaa.com/all-in-one-audio/79566431-bc61-46c6-839b-b9908576ddbb/demucs_vocals.wav
drums:  https://cdn.shibuyaaa.com/all-in-one-audio/79566431-bc61-46c6-839b-b9908576ddbb/demucs_drums.wav
bass:   https://cdn.shibuyaaa.com/all-in-one-audio/79566431-bc61-46c6-839b-b9908576ddbb/demucs_bass.wav
other:  https://cdn.shibuyaaa.com/all-in-one-audio/79566431-bc61-46c6-839b-b9908576ddbb/demucs_other.wav
```

Segments:

| Segment | Start | End | Label | Processed By |
| --- | ---: | ---: | --- | --- |
| `seg-0` | 0.00 | 1.12 | start | quick job |
| `seg-1` | 1.12 | 19.76 | intro | bulk continuation |
| `seg-2` | 19.76 | 21.46 | start | bulk continuation |

Processed stem outputs:

| Job | Segment | Vocals | Drums | Bass | Other |
| --- | --- | --- | --- | --- | --- |
| `live-quick-1` | `seg-0` | https://cdn.shibuyaaa.com/gpu-ingestion/live-quick-1/segments/seg-0/vocals.mp3 | https://cdn.shibuyaaa.com/gpu-ingestion/live-quick-1/segments/seg-0/drums.mp3 | https://cdn.shibuyaaa.com/gpu-ingestion/live-quick-1/segments/seg-0/bass.mp3 | https://cdn.shibuyaaa.com/gpu-ingestion/live-quick-1/segments/seg-0/other.mp3 |
| `live-quick-1:bulk` | `seg-1` | https://cdn.shibuyaaa.com/gpu-ingestion/live-quick-1:bulk/segments/seg-1/vocals.mp3 | https://cdn.shibuyaaa.com/gpu-ingestion/live-quick-1:bulk/segments/seg-1/drums.mp3 | https://cdn.shibuyaaa.com/gpu-ingestion/live-quick-1:bulk/segments/seg-1/bass.mp3 | https://cdn.shibuyaaa.com/gpu-ingestion/live-quick-1:bulk/segments/seg-1/other.mp3 |
| `live-quick-1:bulk` | `seg-2` | https://cdn.shibuyaaa.com/gpu-ingestion/live-quick-1:bulk/segments/seg-2/vocals.mp3 | https://cdn.shibuyaaa.com/gpu-ingestion/live-quick-1:bulk/segments/seg-2/drums.mp3 | https://cdn.shibuyaaa.com/gpu-ingestion/live-quick-1:bulk/segments/seg-2/bass.mp3 | https://cdn.shibuyaaa.com/gpu-ingestion/live-quick-1:bulk/segments/seg-2/other.mp3 |

### Bulk Run: `live-bulk-1`

Analyzer result:

```text
https://cdn.shibuyaaa.com/all-in-one-audio/1aa71eab-ce37-4b3e-ad50-50a04e32f8a9/analyzer_result.json
```

Full-song Demucs stems from Cloud Run:

```text
vocals: https://cdn.shibuyaaa.com/all-in-one-audio/1aa71eab-ce37-4b3e-ad50-50a04e32f8a9/demucs_vocals.wav
drums:  https://cdn.shibuyaaa.com/all-in-one-audio/1aa71eab-ce37-4b3e-ad50-50a04e32f8a9/demucs_drums.wav
bass:   https://cdn.shibuyaaa.com/all-in-one-audio/1aa71eab-ce37-4b3e-ad50-50a04e32f8a9/demucs_bass.wav
other:  https://cdn.shibuyaaa.com/all-in-one-audio/1aa71eab-ce37-4b3e-ad50-50a04e32f8a9/demucs_other.wav
```

Segments:

| Segment | Start | End | Label |
| --- | ---: | ---: | --- |
| `seg-0` | 0.00 | 1.12 | start |
| `seg-1` | 1.12 | 17.65 | intro |
| `seg-2` | 17.65 | 21.46 | start |

Segment durations:

| Segment | Audio Duration |
| --- | ---: |
| `seg-0` | 1.12s |
| `seg-1` | 16.53s |
| `seg-2` | 3.81s |

Processed stem outputs:

| Segment | Vocals | Drums | Bass | Other |
| --- | --- | --- | --- | --- |
| `seg-0` | https://cdn.shibuyaaa.com/gpu-ingestion/live-bulk-1/segments/seg-0/vocals.mp3 | https://cdn.shibuyaaa.com/gpu-ingestion/live-bulk-1/segments/seg-0/drums.mp3 | https://cdn.shibuyaaa.com/gpu-ingestion/live-bulk-1/segments/seg-0/bass.mp3 | https://cdn.shibuyaaa.com/gpu-ingestion/live-bulk-1/segments/seg-0/other.mp3 |
| `seg-1` | https://cdn.shibuyaaa.com/gpu-ingestion/live-bulk-1/segments/seg-1/vocals.mp3 | https://cdn.shibuyaaa.com/gpu-ingestion/live-bulk-1/segments/seg-1/drums.mp3 | https://cdn.shibuyaaa.com/gpu-ingestion/live-bulk-1/segments/seg-1/bass.mp3 | https://cdn.shibuyaaa.com/gpu-ingestion/live-bulk-1/segments/seg-1/other.mp3 |
| `seg-2` | https://cdn.shibuyaaa.com/gpu-ingestion/live-bulk-1/segments/seg-2/vocals.mp3 | https://cdn.shibuyaaa.com/gpu-ingestion/live-bulk-1/segments/seg-2/drums.mp3 | https://cdn.shibuyaaa.com/gpu-ingestion/live-bulk-1/segments/seg-2/bass.mp3 | https://cdn.shibuyaaa.com/gpu-ingestion/live-bulk-1/segments/seg-2/other.mp3 |
