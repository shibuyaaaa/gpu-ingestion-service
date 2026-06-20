import asyncio
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from app.jobs.base import JobAdapter, StageResult
from app.jobs.context import JobContext
from app.library_membership import LibraryLookupResult
from app.legacy.audio import AudioOps
from app.legacy.utils.source import (
    download_youtube_audio,
    extract_youtube_video_id,
    resolve_source_metadata,
    resolve_youtube_match,
)
from app.queue import JobRecord, JobStage, JobType


PROCESS_PRIORITY_QUICK_CHORD = 400
PROCESS_PRIORITY_BULK_CHORD = 300
PROCESS_PRIORITY_QUICK_OTHER = 200
PROCESS_PRIORITY_BULK_OTHER = 100

PROCESS_MODE_SEGMENT_CHORD = "segment_chord"
PROCESS_MODE_SEGMENT_OTHER = "segment_other"
_SOURCE_AUDIO_CACHE_LOCKS: dict[tuple[int, str], asyncio.Lock] = {}
_ANALYSIS_CACHE_LOCKS: dict[tuple[int, str], asyncio.Lock] = {}
_SEGMENT_STEM_CACHE_LOCKS: dict[tuple[int, str], asyncio.Lock] = {}
_MAX_CACHE_LOCKS = 2048


class DissectAdapter(JobAdapter):
    job_types: set[str] = set()

    async def download(self, job: JobRecord, context: JobContext) -> StageResult:
        stage_started = time.perf_counter()
        timings: dict[str, float] = {}
        source = self._source_from_payload(job.payload)
        work_dir = context.job_work_dir(job)
        work_dir.mkdir(parents=True, exist_ok=True)
        wav_path = work_dir / "input.wav"

        if context.settings.dry_run_mode:
            started = time.perf_counter()
            source_path = work_dir / "source.wav"
            source_path.write_bytes(b"dry-run-audio")
            timings["dry_run_source_write_seconds"] = round(time.perf_counter() - started, 6)
            resolved = {
                "source": source,
                "youtube_url": "dry-run://youtube",
                "spotify_metadata": {"title": source, "artist": "", "query_only": True},
                "youtube_match": {"url": "dry-run://youtube"},
                "audio_path": str(source_path),
            }
        else:
            payload_metadata = job.payload.get("spotify_metadata")
            skip_library_precheck = _truthy_payload_flag(job, "skip_library_precheck")
            if isinstance(payload_metadata, dict) and payload_metadata.get("title"):
                timings["source_metadata_seconds"] = 0.0
                resolved = {
                    "source": source,
                    "spotify_metadata": payload_metadata,
                }
            elif skip_library_precheck and extract_youtube_video_id(source):
                timings["source_metadata_seconds"] = 0.0
                resolved = _minimal_youtube_resolved(source)
            else:
                started = time.perf_counter()
                resolved = await resolve_source_metadata(source)
                timings["source_metadata_seconds"] = round(time.perf_counter() - started, 6)
            if skip_library_precheck:
                timings["library_precheck_seconds"] = 0.0
                library_result = LibraryLookupResult(
                    checked=False,
                    exists=False,
                    source="skipped_by_job",
                )
            else:
                started = time.perf_counter()
                library_result = await context.library.lookup(resolved.get("spotify_metadata"))
                timings["library_precheck_seconds"] = round(time.perf_counter() - started, 6)
            if library_result.exists:
                timings["download_total_seconds"] = round(time.perf_counter() - stage_started, 6)
                return StageResult(
                    next_stage=None,
                    artifacts={
                        "source": source,
                        "youtube_url": resolved.get("youtube_url"),
                        "spotify_metadata": resolved.get("spotify_metadata"),
                        "youtube_match": resolved.get("youtube_match"),
                        "library_precheck": library_result.to_dict(),
                        "download_timings": timings,
                        "final_outputs": {
                            "job_type": job.job_type.value,
                            "source": source,
                            "status": "skipped_existing_library_song",
                            "library_song": library_result.song.to_dict() if library_result.song else None,
                            "library_precheck_source": library_result.source,
                            "dry_run": context.settings.dry_run_mode,
                        },
                    },
                )
            if resolved.get("youtube_url") and resolved.get("youtube_match"):
                timings["youtube_match_seconds"] = 0.0
            else:
                started = time.perf_counter()
                resolved = await resolve_youtube_match(resolved)
                timings["youtube_match_seconds"] = round(time.perf_counter() - started, 6)
            if await self._restore_or_create_cached_wav(
                youtube_url=resolved["youtube_url"],
                wav_path=wav_path,
                context=context,
                timings=timings,
            ):
                source_path = wav_path
            else:
                started = time.perf_counter()
                audio_path = await download_youtube_audio(resolved["youtube_url"], work_dir)
                timings["youtube_download_seconds"] = round(time.perf_counter() - started, 6)
                source_path = Path(audio_path)

        if source_path.suffix.lower() in {".wav", ".wave"}:
            if source_path != wav_path:
                started = time.perf_counter()
                shutil.copyfile(source_path, wav_path)
                timings["wav_copy_seconds"] = round(time.perf_counter() - started, 6)
            else:
                timings["wav_copy_seconds"] = 0.0
        else:
            started = time.perf_counter()
            await AudioOps.convert_to_wav(source_path, wav_path)
            timings["wav_convert_seconds"] = round(time.perf_counter() - started, 6)
        timings["download_total_seconds"] = round(time.perf_counter() - stage_started, 6)

        return StageResult(
            next_stage=JobStage.ANALYZE,
            artifacts={
                "work_dir": str(work_dir),
                "source": source,
                "source_path": str(source_path),
                "audio_path": str(wav_path),
                "youtube_url": resolved.get("youtube_url"),
                "spotify_metadata": resolved.get("spotify_metadata"),
                "youtube_match": resolved.get("youtube_match"),
                "library_precheck": library_result.to_dict() if not context.settings.dry_run_mode else None,
                "download_timings": timings,
                "skip_library_precheck": _truthy_payload_flag(job, "skip_library_precheck"),
                "skip_library_write": _truthy_payload_flag(job, "skip_library_write"),
            },
        )

    async def _restore_or_create_cached_wav(
        self,
        *,
        youtube_url: str,
        wav_path: Path,
        context: JobContext,
        timings: dict[str, float],
    ) -> bool:
        if not context.settings.source_audio_cache_enabled:
            return False
        video_id = extract_youtube_video_id(youtube_url)
        if not video_id:
            return False

        cache_dir = context.settings.work_dir / "source-cache"
        cache_path = cache_dir / f"{_safe_job_part(video_id)}.wav"
        lock = _source_audio_cache_lock(video_id)
        wait_started = time.perf_counter()
        async with lock:
            timings["source_audio_cache_wait_seconds"] = round(time.perf_counter() - wait_started, 6)
            cache_dir.mkdir(parents=True, exist_ok=True)
            if cache_path.is_file():
                started = time.perf_counter()
                _link_or_copy(cache_path, wav_path)
                restore_seconds = round(time.perf_counter() - started, 6)
                timings["source_audio_cache_restore_seconds"] = restore_seconds
                timings["source_audio_cache_copy_seconds"] = restore_seconds
                timings["youtube_download_seconds"] = 0.0
                timings["wav_copy_seconds"] = 0.0
                return True

            tmp_dir = cache_dir / ".tmp" / _safe_job_part(video_id)
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            tmp_dir.mkdir(parents=True, exist_ok=True)
            try:
                started = time.perf_counter()
                audio_path = Path(await download_youtube_audio(youtube_url, tmp_dir))
                timings["youtube_download_seconds"] = round(time.perf_counter() - started, 6)

                tmp_wav = tmp_dir / "cached.wav"
                if audio_path.suffix.lower() in {".wav", ".wave"}:
                    started = time.perf_counter()
                    shutil.copyfile(audio_path, tmp_wav)
                    timings["source_audio_cache_build_copy_seconds"] = round(time.perf_counter() - started, 6)
                else:
                    started = time.perf_counter()
                    await AudioOps.convert_to_wav(audio_path, tmp_wav)
                    timings["source_audio_cache_build_convert_seconds"] = round(time.perf_counter() - started, 6)

                started = time.perf_counter()
                os.replace(tmp_wav, cache_path)
                _link_or_copy(cache_path, wav_path)
                timings["source_audio_cache_store_seconds"] = round(time.perf_counter() - started, 6)
                timings["wav_copy_seconds"] = 0.0
                await _prune_source_audio_cache(
                    cache_dir,
                    max_entries=context.settings.source_audio_cache_max_entries,
                    max_bytes=context.settings.source_audio_cache_max_bytes,
                )
                return True
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    async def _restore_cached_analysis(
        self,
        *,
        cache_key: str | None,
        output_dir: Path,
        context: JobContext,
    ) -> dict[str, Any] | None:
        if not context.settings.analysis_cache_enabled or not cache_key:
            return None
        cache_dir = context.settings.work_dir / "analysis-cache" / _safe_job_part(cache_key)
        lock = _analysis_cache_lock(cache_key)
        wait_started = time.perf_counter()
        async with lock:
            wait_seconds = round(time.perf_counter() - wait_started, 6)
            if not _analysis_cache_entry_complete(cache_dir):
                return None
            started = time.perf_counter()
            restored = await asyncio.to_thread(_restore_analysis_cache_sync, cache_dir, output_dir)
            timings = {
                "analysis_cache_hit": 1,
                "analysis_cache_wait_seconds": wait_seconds,
                "analysis_cache_restore_seconds": round(time.perf_counter() - started, 6),
                "allin1_analyze_seconds": 0.0,
                "demix_total_seconds": 0.0,
                "demix_apply_seconds": 0.0,
                "stem_count": len(restored["stem_paths"]),
            }
            return {
                "analysis": restored["analysis"],
                "analyzer_result_path": restored["analyzer_result_path"],
                "stem_paths": restored["stem_paths"],
                "timings": timings,
            }

    async def _store_cached_analysis(
        self,
        *,
        cache_key: str | None,
        result: dict[str, Any],
        context: JobContext,
        timings: dict[str, Any],
    ) -> None:
        if not context.settings.analysis_cache_enabled or not cache_key:
            return
        stem_paths = result.get("stem_paths") or result.get("full_stem_paths") or {}
        analyzer_result_path = result.get("analyzer_result_path")
        if not analyzer_result_path or not _has_required_stems(stem_paths):
            return

        cache_root = context.settings.work_dir / "analysis-cache"
        cache_dir = cache_root / _safe_job_part(cache_key)
        lock = _analysis_cache_lock(cache_key)
        wait_started = time.perf_counter()
        async with lock:
            wait_seconds = round(time.perf_counter() - wait_started, 6)
            if _analysis_cache_entry_complete(cache_dir):
                timings["analysis_cache_store_wait_seconds"] = wait_seconds
                timings["analysis_cache_store_seconds"] = 0.0
                return
            started = time.perf_counter()
            await asyncio.to_thread(
                _store_analysis_cache_sync,
                cache_root,
                cache_dir,
                analyzer_result_path,
                stem_paths,
                result.get("analysis") or {},
            )
            timings["analysis_cache_store_wait_seconds"] = wait_seconds
            timings["analysis_cache_store_seconds"] = round(time.perf_counter() - started, 6)
            timings["analysis_cache_hit"] = 0
            await _prune_analysis_cache(
                cache_root,
                max_entries=context.settings.analysis_cache_max_entries,
                max_bytes=context.settings.analysis_cache_max_bytes,
            )

    async def analyze(self, job: JobRecord, context: JobContext) -> StageResult:
        audio_path = job.artifacts["audio_path"]
        output_dir = Path(job.artifacts["work_dir"]) / "analysis"
        cache_key = _analysis_cache_key(job.artifacts)
        result = await self._restore_cached_analysis(
            cache_key=cache_key,
            output_dir=output_dir,
            context=context,
        )
        if result is None:
            async with context.models.gpu_lock:
                with context.models.track_gpu_work(job_id=job.id, model_name="all-in-one") as gpu_usage:
                    result = await context.models.all_in_one.analyze(audio_path, output_dir)
            analysis_timings = dict(result.get("timings") or {})
            analysis_timings.update(gpu_usage.summary())
            await self._store_cached_analysis(
                cache_key=cache_key,
                result=result,
                context=context,
                timings=analysis_timings,
            )
        else:
            analysis_timings = dict(result.get("timings") or {})
        analysis = result.get("analysis", {})
        segments = self._normalize_segments(analysis)
        chorus_segment = self._find_chorus_segment(segments)
        full_stem_urls = result.get("stem_urls") or {}
        full_stem_paths = result.get("stem_paths") or result.get("full_stem_paths") or {}
        if not full_stem_paths and full_stem_urls and not context.settings.dry_run_mode:
            full_stem_paths = await self._download_full_stems(full_stem_urls, Path(job.artifacts["work_dir"]))
        artifacts = {
            "analysis": analysis,
            "analysis_timings": analysis_timings,
            "analyzer_result_path": result.get("analyzer_result_path"),
            "analyzer_result_url": result.get("analyzer_result_url"),
            "full_stem_urls": full_stem_urls,
            "full_stem_paths": full_stem_paths,
            "segments": segments,
            "chorus_segment": chorus_segment,
            "skip_segment_ids": job.artifacts.get("skip_segment_ids", []),
        }
        fanout = self._enqueue_analyzed_process_jobs(job, context, artifacts)
        return StageResult(
            next_stage=None,
            artifacts={
                **artifacts,
                "fanout": fanout,
                "final_outputs": {
                    "job_type": job.job_type.value,
                    "source": job.artifacts.get("source"),
                    "status": "process_fanout_enqueued",
                    "fanout": fanout,
                    "dry_run": context.settings.dry_run_mode,
                },
            },
        )

    async def process(self, job: JobRecord, context: JobContext) -> StageResult:
        raise NotImplementedError

    def _enqueue_analyzed_process_jobs(
        self,
        job: JobRecord,
        context: JobContext,
        artifacts: dict[str, Any],
    ) -> dict[str, Any]:
        segments = artifacts.get("segments") or []
        chorus_segment = artifacts.get("chorus_segment") or {}
        chorus_segment_id = str(chorus_segment.get("id") or "")
        skipped = {str(segment_id) for segment_id in artifacts.get("skip_segment_ids", [])}
        root_job_id = str(job.payload.get("root_job_id") or job.id)
        child_specs = []

        for segment in segments:
            segment_id = str(segment["id"])
            if segment_id in skipped:
                continue
            is_quick_chorus = job.job_type == JobType.QUICK_DISSECT and segment_id == chorus_segment_id
            child_job_type = JobType.QUICK_DISSECT if is_quick_chorus else JobType.BULK_DISSECT
            process_group = "quick" if is_quick_chorus else "bulk"
            priority = PROCESS_PRIORITY_QUICK_CHORD if is_quick_chorus else PROCESS_PRIORITY_BULK_CHORD
            child_id = f"{root_job_id}:{process_group}:{_safe_job_part(segment_id)}:chord"
            child_specs.append(
                {
                    "child_id": child_id,
                    "job_type": child_job_type,
                    "priority": priority,
                    "segment_id": segment_id,
                    "process_group": process_group,
                    "payload": {
                        "source": job.artifacts.get("source") or job.payload.get("source"),
                        "root_job_id": root_job_id,
                        "process_mode": PROCESS_MODE_SEGMENT_CHORD,
                        "process_group": process_group,
                        "segment_id": segment_id,
                        "skip_library_write": _truthy_payload_flag(job, "skip_library_write"),
                    },
                    "artifacts": {
                        **self._base_process_artifacts(job, artifacts),
                        "process_mode": PROCESS_MODE_SEGMENT_CHORD,
                        "process_group": process_group,
                        "root_job_id": root_job_id,
                        "parent_job_id": job.id,
                        "segment_id": segment_id,
                        "segment": segment,
                        "requires_gpu": not bool(artifacts.get("full_stem_paths")),
                        "other_stems_priority": (
                            PROCESS_PRIORITY_QUICK_OTHER if is_quick_chorus else PROCESS_PRIORITY_BULK_OTHER
                        ),
                    },
                }
            )

        enqueued_children = context.store.enqueue_process_children(parent_job=job, children=child_specs)
        children = [
            {
                "job_id": child.id,
                "segment_id": str(spec["segment_id"]),
                "process_group": str(spec["process_group"]),
                "process_mode": PROCESS_MODE_SEGMENT_CHORD,
                "priority": child.priority,
            }
            for spec, child in zip(child_specs, enqueued_children, strict=True)
        ]

        return {
            "root_job_id": root_job_id,
            "strategy": "segment_chord_jobs_enqueue_other_stem_jobs",
            "children": children,
            "child_count": len(children),
            "priority_order": [
                {"name": "quick_chord", "priority": PROCESS_PRIORITY_QUICK_CHORD},
                {"name": "bulk_chord", "priority": PROCESS_PRIORITY_BULK_CHORD},
                {"name": "quick_other", "priority": PROCESS_PRIORITY_QUICK_OTHER},
                {"name": "bulk_other", "priority": PROCESS_PRIORITY_BULK_OTHER},
            ],
        }

    @staticmethod
    def _base_process_artifacts(job: JobRecord, artifacts: dict[str, Any]) -> dict[str, Any]:
        return {
            "work_dir": job.artifacts["work_dir"],
            "source": job.artifacts.get("source"),
            "source_path": job.artifacts.get("source_path"),
            "audio_path": job.artifacts["audio_path"],
            "youtube_url": job.artifacts.get("youtube_url"),
            "spotify_metadata": job.artifacts.get("spotify_metadata"),
            "youtube_match": job.artifacts.get("youtube_match"),
            "analyzer_result_path": artifacts.get("analyzer_result_path"),
            "analyzer_result_url": artifacts.get("analyzer_result_url"),
            "full_stem_paths": artifacts.get("full_stem_paths", {}),
            "skip_segment_ids": artifacts.get("skip_segment_ids", []),
            "skip_library_precheck": job.artifacts.get("skip_library_precheck")
            or job.payload.get("skip_library_precheck"),
            "skip_library_write": job.artifacts.get("skip_library_write")
            or job.payload.get("skip_library_write"),
        }

    async def _process_segment(
        self,
        job: JobRecord,
        context: JobContext,
        segment: dict[str, Any],
        *,
        output_prefix: str,
        stem_group: str = "all",
    ) -> dict[str, Any]:
        work_dir = Path(job.artifacts["work_dir"])
        timings: dict[str, float] = {}
        segment_id = str(segment["id"])
        segment_dir = work_dir / output_prefix / segment_id
        segment_dir.mkdir(parents=True, exist_ok=True)
        stems_dir = segment_dir / "stems"
        existing_stem_paths = job.artifacts.get("stem_paths") or {}
        full_stem_paths = job.artifacts.get("full_stem_paths") or {}
        can_slice_full_stems = bool(full_stem_paths) and not context.settings.dry_run_mode
        needs_source_segment = not existing_stem_paths and not can_slice_full_stems
        existing_segment_path = job.artifacts.get("segment_path")
        segment_path = Path(existing_segment_path) if existing_segment_path else None
        if needs_source_segment:
            segment_path = Path(existing_segment_path) if existing_segment_path else segment_dir / "input.mp3"
            if existing_segment_path and segment_path.exists():
                pass
            elif context.settings.dry_run_mode:
                shutil.copyfile(job.artifacts["audio_path"], segment_path)
            else:
                started = time.perf_counter()
                await AudioOps.extract_segment(
                    job.artifacts["audio_path"],
                    segment_path,
                    start=float(segment["start"]),
                    duration=max(0.1, float(segment["end"]) - float(segment["start"])),
                )
                timings["source_segment_extract_seconds"] = round(time.perf_counter() - started, 6)

        if existing_stem_paths:
            stems = dict(existing_stem_paths)
            stem_source = "inherited_segment_stems"
        else:
            if can_slice_full_stems:
                stems_to_extract = dict(_filter_stems(full_stem_paths, stem_group))
                segment_cache_key = _segment_stem_cache_key(job.artifacts, segment)
                started = time.perf_counter()
                stems = await _restore_cached_segment_stems(
                    cache_key=segment_cache_key,
                    requested_stems=stems_to_extract.keys(),
                    output_dir=stems_dir,
                    context=context,
                )
                if stems is not None:
                    timings["segment_stem_cache_hit"] = 1
                    timings["segment_stem_cache_restore_seconds"] = round(time.perf_counter() - started, 6)
                    timings["stem_segment_extract_seconds"] = 0.0
                    stem_source = "segment_stem_cache"
                else:
                    timings["segment_stem_cache_hit"] = 0
                    started = time.perf_counter()
                    stems = await self._extract_stem_segments(stems_to_extract, stems_dir, segment)
                    timings["stem_segment_extract_seconds"] = round(time.perf_counter() - started, 6)
                    store_started = time.perf_counter()
                    await _store_cached_segment_stems(
                        cache_key=segment_cache_key,
                        stems=stems,
                        context=context,
                    )
                    timings["segment_stem_cache_store_seconds"] = round(time.perf_counter() - store_started, 6)
                    stem_source = "all_in_one_full_song_stems"
            else:
                if segment_path is None:
                    raise RuntimeError("segment audio path was not prepared for HTDemucs processing")
                started = time.perf_counter()
                async with context.models.gpu_lock:
                    with context.models.track_gpu_work(job_id=job.id, model_name="htdemucs") as gpu_usage:
                        stems = await context.models.htdemucs.separate(segment_path, stems_dir)
                timings["htdemucs_segment_seconds"] = round(time.perf_counter() - started, 6)
                timings.update(gpu_usage.summary())
                stem_source = "segment_htdemucs"

        selected_stems = dict(_filter_stems(stems, stem_group))

        if context.settings.dry_run_mode:
            uploaded = {stem: path for stem, path in selected_stems.items()}
            early_library_publish = None
            manifest_url = None
        else:
            uploaded = {}
            chord_published = False
            early_library_publish = None
            upload_started = time.perf_counter()
            ordered_stems = _ordered_upload_stems(selected_stems)
            chord_items = [(stem, path) for stem, path in ordered_stems if _canonical_stem_type(stem) == "chord"]
            other_items = [(stem, path) for stem, path in ordered_stems if _canonical_stem_type(stem) != "chord"]
            for stem, path in chord_items:
                uploaded[stem] = await self._prepare_and_upload_stem(
                    stem=stem,
                    path=path,
                    job_id=job.id,
                    segment_id=segment_id,
                    context=context,
                )
                if not chord_published:
                    chord_result = {
                        "segment": segment,
                        "segment_path": str(segment_path) if segment_path else None,
                        "stem_paths": stems,
                        "outputs": dict(uploaded),
                    }
                    chord_publish = await context.library_writer.publish_segment(
                        job=job,
                        segment=segment,
                        segment_result=chord_result,
                        status="partial",
                    )
                    early_library_publish = chord_publish.to_dict()
                    _ensure_library_publish_ok(early_library_publish)
                    chord_published = True
            other_uploads = await asyncio.gather(
                *(
                    self._prepare_and_upload_stem(
                        stem=stem,
                        path=path,
                        job_id=job.id,
                        segment_id=segment_id,
                        context=context,
                    )
                    for stem, path in other_items
                )
            )
            uploaded.update((stem, url) for (stem, _), url in zip(other_items, other_uploads, strict=True))
            timings["upload_and_publish_seconds"] = round(time.perf_counter() - upload_started, 6)
            manifest_url = None
            if context.settings.segment_manifest_upload_enabled:
                manifest_url = await _upload_segment_manifest(
                    job=job,
                    context=context,
                    segment_dir=segment_dir,
                    segment=segment,
                    segment_id=segment_id,
                    stem_group=stem_group,
                    outputs=uploaded,
                    library_publish=early_library_publish,
                )

        return {
            "segment": segment,
            "segment_path": str(segment_path) if segment_path else None,
            "stem_paths": stems,
            "stem_source": stem_source,
            "timings": timings,
            "outputs": uploaded,
            "early_library_publish": early_library_publish if not context.settings.dry_run_mode else None,
            "manifest_url": manifest_url,
        }

    @staticmethod
    async def _prepare_and_upload_stem(
        *,
        stem: str,
        path: str,
        job_id: str,
        segment_id: str,
        context: JobContext,
    ) -> str:
        local_path = Path(path)
        if local_path.suffix.lower() != ".mp3":
            mp3_path = local_path.with_suffix(".mp3")
            await AudioOps.convert_to_mp3(local_path, mp3_path)
            local_path = mp3_path
        gcs_path = f"gpu-ingestion/{job_id}/segments/{segment_id}/{stem}.mp3"
        return await context.gcs.upload(local_path, gcs_path, content_type="audio/mpeg")

    async def _process_fanout_job(self, job: JobRecord, context: JobContext) -> StageResult:
        process_mode = job.artifacts.get("process_mode")
        segment = job.artifacts.get("segment")
        if not segment:
            raise RuntimeError("fanout process job is missing segment metadata")

        process_group = str(job.artifacts.get("process_group") or "bulk")
        output_prefix = f"{process_group}_segments"
        if _should_skip_segment_processing(segment):
            final_outputs = {
                "job_type": job.job_type.value,
                "source": job.artifacts.get("source"),
                "process_mode": process_mode,
                "process_group": process_group,
                "segment_id": str(segment["id"]),
                "status": "skipped_tiny_or_boundary_segment",
                "dry_run": context.settings.dry_run_mode,
            }
            return StageResult(next_stage=None, artifacts={"final_outputs": final_outputs})

        if process_mode == PROCESS_MODE_SEGMENT_CHORD:
            result = await self._process_segment(
                job,
                context,
                segment,
                output_prefix=output_prefix,
                stem_group="chord",
            )
            other_job = self._enqueue_other_stems_job(job, context, segment, result)
            library_complete = None
            fanout_maybe_complete = False
            if other_job is None and not context.settings.dry_run_mode:
                root_job_id = str(job.artifacts.get("root_job_id") or job.payload.get("root_job_id") or job.id)
                sibling_summary = context.store.child_summary(root_job_id, exclude_job_id=job.id)
                fanout_maybe_complete = sibling_summary["active"] == 0 and sibling_summary["failed"] == 0
                if fanout_maybe_complete:
                    library_complete = (await context.library_writer.mark_complete(job=job)).to_dict()
                    _ensure_library_publish_ok(library_complete)
            final_outputs = {
                "job_type": job.job_type.value,
                "source": job.artifacts.get("source"),
                "process_mode": process_mode,
                "process_group": process_group,
                "segment_id": str(segment["id"]),
                "chord_outputs": result.get("outputs", {}),
                "other_stems_job_id": other_job.id if other_job else None,
                "quick_dissect_confirmation": process_group == "quick",
                "library_complete": library_complete,
                "dry_run": context.settings.dry_run_mode,
            }
            return StageResult(
                next_stage=None,
                artifacts={
                    "segment_result": result,
                    "other_stems_job_id": other_job.id if other_job else None,
                    "library_complete": library_complete,
                    "fanout_maybe_complete": fanout_maybe_complete,
                    "final_outputs": final_outputs,
                },
            )

        if process_mode == PROCESS_MODE_SEGMENT_OTHER:
            result = await self._process_segment(
                job,
                context,
                segment,
                output_prefix=output_prefix,
                stem_group="other",
            )
            library_publish = None
            library_complete = None
            fanout_maybe_complete = False
            if not context.settings.dry_run_mode:
                library_publish = (
                    await context.library_writer.publish_segment(
                        job=job,
                        segment=segment,
                        segment_result=result,
                        status="partial",
                    )
                ).to_dict()
                _ensure_library_publish_ok(library_publish)
                result["library_publish"] = library_publish
                root_job_id = str(job.artifacts.get("root_job_id") or job.payload.get("root_job_id") or job.id)
                sibling_summary = context.store.child_summary(root_job_id, exclude_job_id=job.id)
                fanout_maybe_complete = sibling_summary["active"] == 0 and sibling_summary["failed"] == 0
                if fanout_maybe_complete:
                    library_complete = (await context.library_writer.mark_complete(job=job)).to_dict()
                    _ensure_library_publish_ok(library_complete)

            final_outputs = {
                "job_type": job.job_type.value,
                "source": job.artifacts.get("source"),
                "process_mode": process_mode,
                "process_group": process_group,
                "segment_id": str(segment["id"]),
                "stem_outputs": result.get("outputs", {}),
                "library_publish": library_publish,
                "library_complete": library_complete,
                "dry_run": context.settings.dry_run_mode,
            }
            return StageResult(
                next_stage=None,
                artifacts={
                    "segment_result": result,
                    "library_publish": library_publish,
                    "library_complete": library_complete,
                    "fanout_maybe_complete": fanout_maybe_complete,
                    "final_outputs": final_outputs,
                },
            )

        raise RuntimeError(f"unknown process_mode for fanout job: {process_mode}")

    def _enqueue_other_stems_job(
        self,
        job: JobRecord,
        context: JobContext,
        segment: dict[str, Any],
        chord_result: dict[str, Any],
    ) -> JobRecord | None:
        other_stems = dict(_filter_stems(chord_result.get("stem_paths") or {}, "other"))
        use_full_stems = False
        if not other_stems and job.artifacts.get("full_stem_paths") and not context.settings.dry_run_mode:
            other_stems = dict(_filter_stems(job.artifacts.get("full_stem_paths") or {}, "other"))
            use_full_stems = True
        if not other_stems:
            return None

        root_job_id = str(job.artifacts.get("root_job_id") or job.payload.get("root_job_id") or job.id)
        process_group = str(job.artifacts.get("process_group") or "bulk")
        priority = int(
            job.artifacts.get(
                "other_stems_priority",
                PROCESS_PRIORITY_QUICK_OTHER if process_group == "quick" else PROCESS_PRIORITY_BULK_OTHER,
            )
        )
        child_id = f"{root_job_id}:{process_group}:{_safe_job_part(str(segment['id']))}:other"
        return context.store.enqueue_process_child(
            parent_job=job,
            child_id=child_id,
            job_type=job.job_type,
            priority=priority,
            payload={
                "source": job.artifacts.get("source") or job.payload.get("source"),
                "root_job_id": root_job_id,
                "process_mode": PROCESS_MODE_SEGMENT_OTHER,
                "process_group": process_group,
                "segment_id": str(segment["id"]),
                "skip_library_write": _truthy_payload_flag(job, "skip_library_write"),
            },
            artifacts={
                **self._base_process_artifacts(job, job.artifacts),
                "process_mode": PROCESS_MODE_SEGMENT_OTHER,
                "process_group": process_group,
                "root_job_id": root_job_id,
                "parent_job_id": job.id,
                "segment_id": str(segment["id"]),
                "segment": segment,
                "segment_path": chord_result.get("segment_path"),
                "stem_paths": {} if use_full_stems else chord_result.get("stem_paths") or {},
                "chord_outputs": chord_result.get("outputs") or {},
                "requires_gpu": False,
            },
        )

    @staticmethod
    async def _download_full_stems(stem_urls: dict[str, str], work_dir: Path) -> dict[str, str]:
        target_dir = work_dir / "full_stems"
        target_dir.mkdir(parents=True, exist_ok=True)

        async def download_one(client: httpx.AsyncClient, stem: str, url: str) -> tuple[str, str]:
            suffix = Path(url.split("?", 1)[0]).suffix or ".wav"
            target = target_dir / f"{stem}{suffix}"
            response = await client.get(url)
            response.raise_for_status()
            target.write_bytes(response.content)
            return stem, str(target)

        async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
            pairs = await asyncio.gather(
                *(download_one(client, stem, url) for stem, url in stem_urls.items() if isinstance(url, str) and url)
            )
        return dict(pairs)

    @staticmethod
    async def _extract_stem_segments(
        full_stem_paths: dict[str, str],
        output_dir: Path,
        segment: dict[str, Any],
    ) -> dict[str, str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        duration = max(0.1, float(segment["end"]) - float(segment["start"]))
        ordered_stems = list(full_stem_paths.items())

        async def extract_one(stem: str, full_path: str) -> tuple[str, str]:
            target = output_dir / f"{stem}.mp3"
            await AudioOps.extract_segment(full_path, target, start=float(segment["start"]), duration=duration)
            return stem, str(target)

        pairs = await asyncio.gather(*(extract_one(stem, full_path) for stem, full_path in ordered_stems))
        return dict(pairs)

    @staticmethod
    def _source_from_payload(payload: dict) -> str:
        source = (
            payload.get("source")
            or payload.get("youtube_url")
            or payload.get("spotify_source")
            or payload.get("spotify_url")
            or payload.get("spotify_query")
        )
        if not source or not str(source).strip():
            raise RuntimeError("job payload must include source, youtube_url, spotify_source, spotify_url, or spotify_query")
        return str(source).strip()

    @staticmethod
    def _normalize_segments(analysis: dict[str, Any]) -> list[dict[str, Any]]:
        raw_segments = analysis.get("segments") or []
        duration = float(analysis.get("duration") or 180.0)
        normalized = []
        for index, segment in enumerate(raw_segments):
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", start + 30.0))
            if end <= start:
                end = start + 30.0
            normalized.append(
                {
                    "id": str(segment.get("id") or f"seg-{index}"),
                    "start": start,
                    "end": end,
                    "label": str(segment.get("label") or f"segment-{index}"),
                }
            )
        meaningful = [
            segment
            for segment in normalized
            if (float(segment["end"]) - float(segment["start"])) >= 5.0
            and segment["label"].lower() not in {"start", "end", "silence"}
        ]
        if meaningful:
            return meaningful
        if normalized:
            return normalized
        return [{"id": "seg-0", "start": 0.0, "end": min(duration, 30.0), "label": "segment"}]

    @staticmethod
    def _find_chorus_segment(segments: list[dict[str, Any]]) -> dict[str, Any]:
        choruses = [segment for segment in segments if "chorus" in segment["label"].lower()]
        if choruses:
            return choruses[2] if len(choruses) >= 3 else choruses[1] if len(choruses) >= 2 else choruses[0]

        def duration(segment: dict[str, Any]) -> float:
            return float(segment["end"]) - float(segment["start"])

        useful = [
            segment
            for segment in segments
            if duration(segment) >= 5.0
            and segment["label"].lower() not in {"start", "end", "outro", "silence"}
        ]
        return max(useful or segments, key=duration)


class BulkDissectAdapter(DissectAdapter):
    job_types = {JobType.BULK_DISSECT.value}

    async def process(self, job: JobRecord, context: JobContext) -> StageResult:
        if job.artifacts.get("process_mode"):
            return await self._process_fanout_job(job, context)

        segments = job.artifacts.get("segments") or []
        processed = dict(job.artifacts.get("processed_segments") or {})
        skipped = set(job.artifacts.get("skip_segment_ids") or [])

        next_segment = None
        for segment in segments:
            segment_id = str(segment["id"])
            if segment_id not in processed and segment_id not in skipped:
                next_segment = segment
                break

        if next_segment is None:
            library_publish = None
            if not context.settings.dry_run_mode:
                library_publish = (
                    await context.library_writer.mark_complete(job=job)
                ).to_dict()
                _ensure_library_publish_ok(library_publish)
            final_outputs = {
                "job_type": job.job_type.value,
                "source": job.artifacts.get("source"),
                "processed_segment_ids": sorted(processed),
                "skipped_segment_ids": sorted(skipped),
                "dry_run": context.settings.dry_run_mode,
                "library_publish": library_publish,
            }
            result_url = None
            if not context.settings.dry_run_mode:
                metadata_path = Path(job.artifacts["work_dir"]) / "bulk_result.json"
                metadata_path.write_text(json.dumps(final_outputs, indent=2), encoding="utf-8")
                result_url = await context.gcs.upload(
                    metadata_path,
                    f"gpu-ingestion/{job.id}/bulk_result.json",
                    content_type="application/json",
                )
                final_outputs["result_url"] = result_url
            return StageResult(next_stage=None, artifacts={"final_outputs": final_outputs, "result_url": result_url})

        result = await self._process_segment(job, context, next_segment, output_prefix="bulk_segments")
        next_processed = {**processed, str(next_segment["id"]): result}
        remaining = [
            segment
            for segment in segments
            if str(segment["id"]) not in next_processed and str(segment["id"]) not in skipped
        ]
        library_publish = None
        if not context.settings.dry_run_mode:
            library_publish = (
                await context.library_writer.publish_segment(
                    job=job,
                    segment=next_segment,
                    segment_result=result,
                    status="partial" if remaining else "complete",
                )
            ).to_dict()
            _ensure_library_publish_ok(library_publish)
            result["library_publish"] = library_publish
        return StageResult(
            next_stage=JobStage.PROCESS if remaining else None,
            artifacts={
                "processed_segments": next_processed,
                "last_processed_segment_id": str(next_segment["id"]),
                "library_publish": library_publish,
                "final_outputs": {
                    "job_type": job.job_type.value,
                    "source": job.artifacts.get("source"),
                    "processed_segment_ids": sorted(next_processed),
                    "skipped_segment_ids": sorted(skipped),
                    "dry_run": context.settings.dry_run_mode,
                    "library_publish": library_publish,
                },
            },
        )


class QuickDissectAdapter(DissectAdapter):
    job_types = {JobType.QUICK_DISSECT.value}

    async def process(self, job: JobRecord, context: JobContext) -> StageResult:
        if job.artifacts.get("process_mode"):
            return await self._process_fanout_job(job, context)

        chorus_segment = job.artifacts.get("chorus_segment")
        if not chorus_segment:
            raise RuntimeError("quick_dissect job is missing chorus_segment")

        chorus_result = await self._process_segment(job, context, chorus_segment, output_prefix="quick_chorus")
        library_publish = None
        if not context.settings.dry_run_mode:
            library_publish = (
                await context.library_writer.publish_segment(
                    job=job,
                    segment=chorus_segment,
                    segment_result=chorus_result,
                    status="partial",
                )
            ).to_dict()
            _ensure_library_publish_ok(library_publish)
            chorus_result["library_publish"] = library_publish
        continuation = context.store.enqueue_continuation(
            parent_job=job,
            payload={
                "source": job.artifacts.get("source"),
                "parent_job_id": job.id,
            },
            artifacts={
                "work_dir": job.artifacts["work_dir"],
                "source": job.artifacts.get("source"),
                "source_path": job.artifacts.get("source_path"),
                "audio_path": job.artifacts["audio_path"],
                "youtube_url": job.artifacts.get("youtube_url"),
                "spotify_metadata": job.artifacts.get("spotify_metadata"),
                "youtube_match": job.artifacts.get("youtube_match"),
                "analysis": job.artifacts.get("analysis", {}),
                "analyzer_result_path": job.artifacts.get("analyzer_result_path"),
                "analyzer_result_url": job.artifacts.get("analyzer_result_url"),
                "full_stem_urls": job.artifacts.get("full_stem_urls", {}),
                "full_stem_paths": job.artifacts.get("full_stem_paths", {}),
                "segments": job.artifacts.get("segments", []),
                "chorus_segment": chorus_segment,
                "skip_segment_ids": [str(chorus_segment["id"])],
                "processed_segments": {},
                "skip_library_precheck": job.artifacts.get("skip_library_precheck")
                or job.payload.get("skip_library_precheck"),
                "skip_library_write": job.artifacts.get("skip_library_write")
                or job.payload.get("skip_library_write"),
            },
        )

        final_outputs = {
            "job_type": job.job_type.value,
            "quick_dissect": True,
            "source": job.artifacts.get("source"),
            "youtube_url": job.artifacts.get("youtube_url"),
            "spotify_metadata": job.artifacts.get("spotify_metadata"),
            "chorus_segment": chorus_segment,
            "chorus_outputs": chorus_result.get("outputs", {}),
            "dry_run": context.settings.dry_run_mode,
            "bulk_continuation_job_id": continuation.id,
            "library_publish": library_publish,
        }
        return StageResult(
            next_stage=None,
            artifacts={
                "quick_chorus_result": chorus_result,
                "processed_segments": {str(chorus_segment["id"]): chorus_result},
                "skip_segment_ids": [str(chorus_segment["id"])],
                "bulk_continuation_job_id": continuation.id,
                "library_publish": library_publish,
                "final_outputs": final_outputs,
            },
        )


def _canonical_stem_type(stem: str) -> str:
    mapping = {
        "vocals": "voice",
        "vocal": "voice",
        "voice": "voice",
        "drums": "beat",
        "drum": "beat",
        "beat": "beat",
        "bass": "bass",
        "other": "chord",
        "chords": "chord",
        "chord": "chord",
        "instrumental": "chord",
    }
    return mapping.get(stem.strip().lower(), stem.strip().lower())


def _ensure_library_publish_ok(result: dict[str, Any] | None) -> None:
    if result and result.get("enabled") and result.get("error"):
        raise RuntimeError(f"library publish failed: {result['error']}")


async def _upload_segment_manifest(
    *,
    job: JobRecord,
    context: JobContext,
    segment_dir: Path,
    segment: dict[str, Any],
    segment_id: str,
    stem_group: str,
    outputs: dict[str, str],
    library_publish: dict[str, Any] | None,
) -> str:
    manifest = {
        "schema_version": 1,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "job": {
            "id": job.id,
            "type": job.job_type.value,
            "stage": job.stage.value,
            "priority": job.priority,
            "payload": job.payload,
        },
        "source": {
            "source": job.artifacts.get("source"),
            "youtube_url": job.artifacts.get("youtube_url"),
            "spotify_metadata": job.artifacts.get("spotify_metadata"),
            "youtube_match": job.artifacts.get("youtube_match"),
        },
        "analysis": {
            "segment": segment,
            "segment_id": segment_id,
            "stem_group": stem_group,
            "process_mode": job.artifacts.get("process_mode"),
            "process_group": job.artifacts.get("process_group"),
            "root_job_id": job.artifacts.get("root_job_id"),
        },
        "outputs": outputs,
        "library_publish": library_publish,
    }
    manifest_path = segment_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return await context.gcs.upload(
        manifest_path,
        f"gpu-ingestion/{job.id}/segments/{segment_id}/manifest.json",
        content_type="application/json",
    )


def _ordered_upload_stems(stems: dict[str, str]) -> list[tuple[str, str]]:
    priority = {"chord": 0, "beat": 1, "bass": 2, "voice": 3}
    return sorted(stems.items(), key=lambda item: priority.get(_canonical_stem_type(item[0]), 99))


def _filter_stems(stems: dict[str, str], stem_group: str) -> list[tuple[str, str]]:
    if stem_group == "all":
        return list(stems.items())
    if stem_group == "chord":
        return [(stem, path) for stem, path in stems.items() if _canonical_stem_type(stem) == "chord"]
    if stem_group == "other":
        return [(stem, path) for stem, path in stems.items() if _canonical_stem_type(stem) != "chord"]
    raise ValueError(f"unknown stem group: {stem_group}")


def _safe_job_part(value: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_"} else "-" for character in value)


def _should_skip_segment_processing(segment: dict[str, Any]) -> bool:
    label = str(segment.get("label") or "").lower()
    duration = float(segment.get("end") or 0.0) - float(segment.get("start") or 0.0)
    return label in {"start", "end", "silence"} or duration < 5.0


def _truthy_payload_flag(job: JobRecord, key: str) -> bool:
    value = job.payload.get(key, job.artifacts.get(key))
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _analysis_cache_key(artifacts: dict[str, Any]) -> str | None:
    youtube_url = str(artifacts.get("youtube_url") or "")
    video_id = extract_youtube_video_id(youtube_url)
    if video_id:
        return f"youtube-{video_id}"
    match = artifacts.get("youtube_match") or {}
    if isinstance(match, dict):
        video_id = str(match.get("video_id") or "")
        if video_id:
            return f"youtube-{video_id}"
    return None


def _segment_stem_cache_key(artifacts: dict[str, Any], segment: dict[str, Any]) -> str | None:
    source_key = _analysis_cache_key(artifacts)
    if not source_key:
        return None
    payload = {
        "source": source_key,
        "segment_id": str(segment.get("id") or ""),
        "start": round(float(segment.get("start") or 0.0), 3),
        "end": round(float(segment.get("end") or 0.0), 3),
        "label": str(segment.get("label") or ""),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    return f"{_safe_job_part(source_key)}-{digest}"


def _analysis_cache_lock(cache_key: str) -> asyncio.Lock:
    return _bounded_cache_lock(_ANALYSIS_CACHE_LOCKS, cache_key)


def _analysis_cache_entry_complete(cache_dir: Path) -> bool:
    if not cache_dir.is_dir():
        return False
    if not (cache_dir / "metadata.json").is_file():
        return False
    if not (cache_dir / "analyzer_result.json").is_file():
        return False
    stems_dir = cache_dir / "stems"
    return all((stems_dir / f"{stem}.wav").is_file() for stem in _REQUIRED_DEMUCS_STEMS)


def _has_required_stems(stem_paths: dict[str, Any]) -> bool:
    return all(stem in stem_paths and Path(str(stem_paths[stem])).is_file() for stem in _REQUIRED_DEMUCS_STEMS)


def _restore_analysis_cache_sync(cache_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    analyzer_target = output_dir / "analyzer_result.json"
    _link_or_copy(cache_dir / "analyzer_result.json", analyzer_target)
    analysis = json.loads(analyzer_target.read_text(encoding="utf-8"))

    stems_target = output_dir / "cached_full_stems"
    stems_target.mkdir(parents=True, exist_ok=True)
    stem_paths = {}
    for stem in _REQUIRED_DEMUCS_STEMS:
        target = stems_target / f"{stem}.wav"
        _link_or_copy(cache_dir / "stems" / f"{stem}.wav", target)
        stem_paths[stem] = str(target)
    try:
        os.utime(cache_dir / "metadata.json", None)
    except OSError:
        pass
    return {"analysis": analysis, "analyzer_result_path": str(analyzer_target), "stem_paths": stem_paths}


def _store_analysis_cache_sync(
    cache_root: Path,
    cache_dir: Path,
    analyzer_result_path: str,
    stem_paths: dict[str, Any],
    analysis: dict[str, Any],
) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = cache_root / ".tmp" / f"{cache_dir.name}-{os.getpid()}-{time.time_ns()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    stems_dir = tmp_dir / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    try:
        analyzer_source = Path(analyzer_result_path)
        if analyzer_source.is_file():
            _link_or_copy(analyzer_source, tmp_dir / "analyzer_result.json")
        else:
            (tmp_dir / "analyzer_result.json").write_text(json.dumps(analysis), encoding="utf-8")
        for stem in _REQUIRED_DEMUCS_STEMS:
            _link_or_copy(Path(str(stem_paths[stem])), stems_dir / f"{stem}.wav")
        (tmp_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "written_at": datetime.now(timezone.utc).isoformat(),
                    "stems": list(_REQUIRED_DEMUCS_STEMS),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
        os.replace(tmp_dir, cache_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


async def _prune_analysis_cache(cache_root: Path, *, max_entries: int, max_bytes: int) -> None:
    if max_entries <= 0 and max_bytes <= 0:
        return

    def prune() -> None:
        entries = []
        paths = cache_root.iterdir() if cache_root.exists() else []
        for path in paths:
            if not path.is_dir() or path.name == ".tmp":
                continue
            metadata = path / "metadata.json"
            try:
                entries.append((metadata.stat().st_mtime, _directory_size(path), path))
            except OSError:
                entries.append((0.0, _directory_size(path), path))
        entries.sort(reverse=True, key=lambda item: item[0])
        total_bytes = sum(size for _, size, _ in entries)
        keep: list[tuple[float, int, Path]] = []
        for item in entries:
            if max_entries > 0 and len(keep) >= max_entries:
                total_bytes -= item[1]
                shutil.rmtree(item[2], ignore_errors=True)
                continue
            keep.append(item)
        for _, size, stale in reversed(keep):
            if max_bytes <= 0 or total_bytes <= max_bytes:
                break
            total_bytes -= size
            shutil.rmtree(stale, ignore_errors=True)

    await asyncio.to_thread(prune)


async def _restore_cached_segment_stems(
    *,
    cache_key: str | None,
    requested_stems: Any,
    output_dir: Path,
    context: JobContext,
) -> dict[str, str] | None:
    if not context.settings.segment_stem_cache_enabled or not cache_key:
        return None
    stem_names = [str(stem) for stem in requested_stems]
    if not stem_names:
        return {}
    cache_dir = context.settings.work_dir / "segment-stem-cache" / _safe_job_part(cache_key)
    lock = _segment_stem_cache_lock(cache_key)
    async with lock:
        if not all((cache_dir / f"{stem}.mp3").is_file() for stem in stem_names):
            return None
        output_dir.mkdir(parents=True, exist_ok=True)
        stems = {}
        for stem in stem_names:
            target = output_dir / f"{stem}.mp3"
            _link_or_copy(cache_dir / f"{stem}.mp3", target)
            stems[stem] = str(target)
        metadata = cache_dir / "metadata.json"
        if metadata.exists():
            try:
                os.utime(metadata, None)
            except OSError:
                pass
        return stems


async def _store_cached_segment_stems(
    *,
    cache_key: str | None,
    stems: dict[str, str],
    context: JobContext,
) -> None:
    if not context.settings.segment_stem_cache_enabled or not cache_key or not stems:
        return
    cache_root = context.settings.work_dir / "segment-stem-cache"
    cache_dir = cache_root / _safe_job_part(cache_key)
    lock = _segment_stem_cache_lock(cache_key)
    async with lock:
        await asyncio.to_thread(_store_cached_segment_stems_sync, cache_root, cache_dir, stems)
        await _prune_directory_cache(
            cache_root,
            max_entries=context.settings.segment_stem_cache_max_entries,
            max_bytes=context.settings.segment_stem_cache_max_bytes,
        )


def _store_cached_segment_stems_sync(cache_root: Path, cache_dir: Path, stems: dict[str, str]) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for stem, path in stems.items():
        source = Path(path)
        if source.is_file():
            _link_or_copy(source, cache_dir / f"{stem}.mp3")
    (cache_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "written_at": datetime.now(timezone.utc).isoformat(),
                "stems": sorted(stems),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


async def _prune_directory_cache(cache_root: Path, *, max_entries: int, max_bytes: int) -> None:
    if max_entries <= 0 and max_bytes <= 0:
        return

    def prune() -> None:
        entries = _directory_cache_entries(cache_root)
        total_bytes = sum(size for _, size, _ in entries)
        keep: list[tuple[float, int, Path]] = []
        for item in entries:
            if max_entries > 0 and len(keep) >= max_entries:
                total_bytes -= item[1]
                shutil.rmtree(item[2], ignore_errors=True)
                continue
            keep.append(item)
        for _, size, stale in reversed(keep):
            if max_bytes <= 0 or total_bytes <= max_bytes:
                break
            total_bytes -= size
            shutil.rmtree(stale, ignore_errors=True)

    await asyncio.to_thread(prune)


def _directory_cache_entries(cache_root: Path) -> list[tuple[float, int, Path]]:
    entries = []
    paths = cache_root.iterdir() if cache_root.exists() else []
    for path in paths:
        if not path.is_dir() or path.name == ".tmp":
            continue
        metadata = path / "metadata.json"
        try:
            mtime = metadata.stat().st_mtime
        except OSError:
            mtime = 0.0
        entries.append((mtime, _directory_size(path), path))
    entries.sort(reverse=True, key=lambda item: item[0])
    return entries


def _source_audio_cache_lock(video_id: str) -> asyncio.Lock:
    return _bounded_cache_lock(_SOURCE_AUDIO_CACHE_LOCKS, video_id)


def _segment_stem_cache_lock(cache_key: str) -> asyncio.Lock:
    return _bounded_cache_lock(_SEGMENT_STEM_CACHE_LOCKS, cache_key)


def _bounded_cache_lock(lock_map: dict[tuple[int, str], asyncio.Lock], cache_key: str) -> asyncio.Lock:
    key = (id(asyncio.get_running_loop()), cache_key)
    lock = lock_map.get(key)
    if lock is not None:
        return lock
    if len(lock_map) >= _MAX_CACHE_LOCKS:
        _prune_unlocked_cache_locks(lock_map)
    lock = asyncio.Lock()
    lock_map[key] = lock
    return lock


def _prune_unlocked_cache_locks(lock_map: dict[tuple[int, str], asyncio.Lock]) -> None:
    target_size = max(0, _MAX_CACHE_LOCKS // 2)
    for key, lock in list(lock_map.items()):
        if len(lock_map) <= target_size:
            break
        if not lock.locked():
            lock_map.pop(key, None)


def cache_lock_status() -> dict[str, Any]:
    return {
        "source_audio_locks": len(_SOURCE_AUDIO_CACHE_LOCKS),
        "analysis_locks": len(_ANALYSIS_CACHE_LOCKS),
        "segment_stem_locks": len(_SEGMENT_STEM_CACHE_LOCKS),
        "max_locks_per_map": _MAX_CACHE_LOCKS,
    }


async def _prune_source_audio_cache(cache_dir: Path, *, max_entries: int, max_bytes: int) -> None:
    if max_entries <= 0 and max_bytes <= 0:
        return

    def prune() -> None:
        files = sorted(
            (path for path in cache_dir.glob("*.wav") if path.is_file()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        sizes = {path: _safe_path_size(path) for path in files}
        total_bytes = sum(sizes.values())
        keep: list[Path] = []
        for path in files:
            if max_entries > 0 and len(keep) >= max_entries:
                total_bytes -= sizes[path]
                _safe_unlink(path)
                continue
            keep.append(path)
        for stale in reversed(keep):
            if max_bytes <= 0 or total_bytes <= max_bytes:
                break
            total_bytes -= sizes[stale]
            _safe_unlink(stale)

    await asyncio.to_thread(prune)


def _directory_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for child in path.rglob("*"):
        if child.is_file():
            total += _safe_path_size(child)
    return total


def _safe_path_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


_REQUIRED_DEMUCS_STEMS = ("bass", "drums", "other", "vocals")


def _minimal_youtube_resolved(source: str) -> dict[str, Any]:
    video_id = extract_youtube_video_id(source)
    youtube_url = f"https://www.youtube.com/watch?v={video_id}" if video_id else source
    return {
        "source": source,
        "youtube_url": youtube_url,
        "youtube_match": {
            "video_id": video_id,
            "title": None,
            "channel": "",
            "duration_seconds": 0,
            "url": youtube_url,
            "thumbnail": None,
            "metadata_source": "skipped_for_direct_youtube_download",
        },
        "spotify_metadata": {
            "spotify_id": None,
            "title": source,
            "artist": "",
            "artists": [],
            "album": "",
            "duration_ms": 0,
            "album_art_url": None,
            "album_art_highres": None,
            "album_art_medres": None,
            "album_art_lowres": None,
            "isrc": None,
            "popularity": 0,
            "query_only": True,
            "source_type": "youtube",
            "metadata_source": "skipped_for_direct_youtube_download",
        },
    }
