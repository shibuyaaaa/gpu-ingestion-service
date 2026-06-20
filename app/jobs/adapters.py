import asyncio
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from app.jobs.base import JobAdapter, StageResult
from app.jobs.context import JobContext
from app.legacy.audio import AudioOps
from app.legacy.utils.source import download_youtube_audio, resolve_source_metadata, resolve_youtube_match
from app.queue import JobRecord, JobStage, JobType


PROCESS_PRIORITY_QUICK_CHORD = 400
PROCESS_PRIORITY_BULK_CHORD = 300
PROCESS_PRIORITY_QUICK_OTHER = 200
PROCESS_PRIORITY_BULK_OTHER = 100

PROCESS_MODE_SEGMENT_CHORD = "segment_chord"
PROCESS_MODE_SEGMENT_OTHER = "segment_other"


class DissectAdapter(JobAdapter):
    job_types: set[str] = set()

    async def download(self, job: JobRecord, context: JobContext) -> StageResult:
        source = self._source_from_payload(job.payload)
        work_dir = context.job_work_dir(job)
        work_dir.mkdir(parents=True, exist_ok=True)

        if context.settings.dry_run_mode:
            source_path = work_dir / "source.wav"
            source_path.write_bytes(b"dry-run-audio")
            resolved = {
                "source": source,
                "youtube_url": "dry-run://youtube",
                "spotify_metadata": {"title": source, "artist": "", "query_only": True},
                "youtube_match": {"url": "dry-run://youtube"},
                "audio_path": str(source_path),
            }
        else:
            payload_metadata = job.payload.get("spotify_metadata")
            if isinstance(payload_metadata, dict) and payload_metadata.get("title"):
                resolved = {
                    "source": source,
                    "spotify_metadata": payload_metadata,
                }
            else:
                resolved = await resolve_source_metadata(source)
            library_result = await context.library.lookup(resolved.get("spotify_metadata"))
            if library_result.exists:
                return StageResult(
                    next_stage=None,
                    artifacts={
                        "source": source,
                        "youtube_url": resolved.get("youtube_url"),
                        "spotify_metadata": resolved.get("spotify_metadata"),
                        "youtube_match": resolved.get("youtube_match"),
                        "library_precheck": library_result.to_dict(),
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
            resolved = await resolve_youtube_match(resolved)
            audio_path = await download_youtube_audio(resolved["youtube_url"], work_dir)
            source_path = Path(audio_path)

        wav_path = work_dir / "input.wav"
        if source_path.suffix.lower() in {".wav", ".wave"}:
            if source_path != wav_path:
                shutil.copyfile(source_path, wav_path)
        else:
            await AudioOps.convert_to_wav(source_path, wav_path)

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
            },
        )

    async def analyze(self, job: JobRecord, context: JobContext) -> StageResult:
        audio_path = job.artifacts["audio_path"]
        output_dir = Path(job.artifacts["work_dir"]) / "analysis"
        async with context.models.gpu_lock:
            context.models.active_gpu_job_id = job.id
            context.models.active_gpu_model = "all-in-one"
            try:
                result = await context.models.all_in_one.analyze(audio_path, output_dir)
            finally:
                context.models.active_gpu_job_id = None
                context.models.active_gpu_model = None
        analysis = result.get("analysis", {})
        segments = self._normalize_segments(analysis)
        chorus_segment = self._find_chorus_segment(segments)
        full_stem_urls = result.get("stem_urls") or {}
        full_stem_paths = result.get("stem_paths") or result.get("full_stem_paths") or {}
        if not full_stem_paths and full_stem_urls and not context.settings.dry_run_mode:
            full_stem_paths = await self._download_full_stems(full_stem_urls, Path(job.artifacts["work_dir"]))
        artifacts = {
            "analysis": analysis,
            "analysis_timings": result.get("timings") or {},
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
            "analysis": artifacts.get("analysis", {}),
            "analyzer_result_path": artifacts.get("analyzer_result_path"),
            "analyzer_result_url": artifacts.get("analyzer_result_url"),
            "full_stem_urls": artifacts.get("full_stem_urls", {}),
            "full_stem_paths": artifacts.get("full_stem_paths", {}),
            "segments": artifacts.get("segments", []),
            "chorus_segment": artifacts.get("chorus_segment"),
            "skip_segment_ids": artifacts.get("skip_segment_ids", []),
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
                started = time.perf_counter()
                stems_to_extract = dict(_filter_stems(full_stem_paths, stem_group))
                stems = await self._extract_stem_segments(stems_to_extract, stems_dir, segment)
                timings["stem_segment_extract_seconds"] = round(time.perf_counter() - started, 6)
                stem_source = "all_in_one_full_song_stems"
            else:
                if segment_path is None:
                    raise RuntimeError("segment audio path was not prepared for HTDemucs processing")
                started = time.perf_counter()
                async with context.models.gpu_lock:
                    context.models.active_gpu_job_id = job.id
                    context.models.active_gpu_model = "htdemucs"
                    try:
                        stems = await context.models.htdemucs.separate(segment_path, stems_dir)
                    finally:
                        context.models.active_gpu_job_id = None
                        context.models.active_gpu_model = None
                timings["htdemucs_segment_seconds"] = round(time.perf_counter() - started, 6)
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
