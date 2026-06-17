# GPU Runtime Policy

This service is optimized for one GCE L4 GPU first. The policy is conservative:
keep models resident, avoid pageable host-to-device transfers where practical,
prioritize HTDemucs, and do not enable CUDA Graphs or low-level L2 cache
controls until an L4 benchmark proves the target shapes are stable.

## Model Residency

- `HTDemucsRuntime.load()` loads the model once, moves it to `GPU_DEVICE`, sets
  eval mode, disables gradients, and keeps the module object alive for the
  process lifetime.
- `AllInOneRuntime.load()` imports and holds the all-in-one package once.
- The worker calls `ModelRuntimeBundle.warmup()` before starting queues.
- `CUDA_EMPTY_CACHE_AFTER_JOB=false` by default. Emptying the cache after each
  job would fight PyTorch's caching allocator and can reintroduce allocation
  overhead.

## Pinned Host Memory

Pinned memory helps CPU to GPU transfer only when the tensor is actually copied
with `non_blocking=True` to CUDA. The service stages HTDemucs audio batches
through pinned memory when the batch is under budget.

Default budget:

```text
PINNED_AUDIO_SECONDS=600
PINNED_AUDIO_SAMPLE_RATE=44100
PINNED_AUDIO_CHANNELS=2
PINNED_AUDIO_SLOTS=2
bytes_per_sample=4  # float32

600 * 44100 * 2 * 2 * 4 = 423,360,000 bytes ~= 403.7 MiB
```

That is enough for two 10-minute stereo float32 staging buffers on a 16 GiB
host. It is deliberately not a giant pinned pool because page-locked memory can
hurt the OS if overused.

For the first L4 VM:

```text
PINNED_AUDIO_STAGING=true
PINNED_AUDIO_SECONDS=600
PINNED_AUDIO_SLOTS=2
ANALYZE_BATCH_SIZE=1
```

If songs above 10 minutes are common, increase `PINNED_AUDIO_SECONDS`; oversized
tensors automatically fall back to pageable transfer and emit a warning.

## CUDA Runtime Knobs

Configured before model warmup:

```text
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,garbage_collection_threshold:0.8
CUDA_RUNTIME_TUNING=true
CUDA_ALLOW_TF32=true
CUDA_CUDNN_BENCHMARK=true
CUDA_MATMUL_PRECISION=high
```

Why:

- TF32 is usually a good inference tradeoff on NVIDIA Tensor Core GPUs.
- cuDNN benchmark can select faster kernels after seeing stable shapes.
- allocator split/GC settings reduce fragmentation risk in long-lived workers.

## L1/L2 Cache Policy

Do not set custom L1/L2 cache policy in the Python service yet.

NVIDIA exposes L2 persistence through CUDA stream access-policy windows and
CUDA graph kernel nodes, but PyTorch module execution does not give us a clean,
stable pointer/window for the library kernels used by Demucs/all-in-one. Applying
this blindly would be fragile and may hurt streaming workloads.

The correct next step is an Nsight Systems/Compute run on the L4. Only consider
custom L2 persistence if a fixed tensor forward repeatedly reads the same
global-memory region and the access window is known.

## CUDA Graphs

`CUDA_GRAPHS_ENABLED=false` by default.

CUDA Graphs need static shapes and stable memory addresses. Full-song HTDemucs
has variable input lengths, split-mode chunk counts, overlap-add behavior, and
file IO around model execution. The all-in-one path has package-level dynamic
orchestration.

Candidate path for later:

```text
quick dissect only
-> fixed 30s or 45s segment
-> preallocated input/output tensors
-> isolate pure HTDemucs forward
-> capture/replay benchmark
```

Do not enable CUDA Graphs for full-song jobs without that benchmark.

## Stage Routing

HTDemucs is the bottleneck, but the queue policy is job priority first and FIFO
second. That means an urgent quick-dissect job can jump ahead of bulk work, and
two jobs with the same priority run in arrival order even if their current GPU
stages differ.

Bulk dissect:

```text
download -> analyze -> process one segment -> requeue process until complete
```

Quick dissect:

```text
download -> analyze -> process chorus -> complete quick job -> enqueue bulk continuation
```

The continuation starts in `process`, skips the already-processed chorus segment,
and uses lower `bulk_dissect` priority. Bulk work processes one segment per queue
claim so new quick jobs can jump ahead between bulk segments.

## Tomorrow's L4 Validation Checklist

1. `GET /ops/readiness` after service start.
2. Confirm `models.htdemucs.loaded=true` and `models.all_in_one.loaded=true`.
3. Run one dry job, then one real 30s quick-dissect segment.
4. Check `/ops/gpu`:
   - GPU visible
   - VRAM reserved/allocated
   - pinned transfer count increasing
   - no model reload between jobs
5. Run 5 back-to-back jobs and confirm:
   - no increasing VRAM leak
   - no repeated model load logs
   - quick jobs claim ahead of bulk continuations
   - bulk continuations skip the quick chorus segment
