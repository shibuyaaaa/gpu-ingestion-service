from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


_GCS_URI_RE = re.compile(r"^gs://(?P<bucket>[^/]+)/(?P<path>.+)$")


@dataclass(frozen=True)
class GpuIngestionObject:
    uri: str
    bucket: str
    job_id: str
    segment_id: str
    stem: str
    canonical_stem: str
    spotify_id: str | None
    crawler_session_id: str | None


def canonical_stem_type(stem: str) -> str:
    normalized = stem.strip().lower()
    return {
        "other": "chord",
        "chords": "chord",
        "chord": "chord",
        "instrumental": "chord",
        "vocals": "voice",
        "vocal": "voice",
        "voice": "voice",
        "drums": "beat",
        "drum": "beat",
        "beat": "beat",
        "bass": "bass",
    }.get(normalized, normalized)


def parse_gpu_ingestion_uri(uri: str) -> GpuIngestionObject | None:
    uri = uri.strip()
    if not uri or not uri.endswith(".mp3"):
        return None
    match = _GCS_URI_RE.match(uri)
    if not match:
        return None

    parts = match.group("path").split("/")
    try:
        prefix_index = parts.index("gpu-ingestion")
    except ValueError:
        return None
    tail = parts[prefix_index + 1 :]
    if len(tail) != 4 or tail[1] != "segments":
        return None

    job_id, _, segment_id, filename = tail
    stem = Path(filename).stem
    crawler_session_id = None
    spotify_id = None
    job_parts = job_id.split(":")
    if len(job_parts) >= 3 and job_parts[0] == "crawler":
        crawler_session_id = job_parts[1]
        spotify_id = job_parts[2]

    return GpuIngestionObject(
        uri=uri,
        bucket=match.group("bucket"),
        job_id=job_id,
        segment_id=segment_id,
        stem=stem,
        canonical_stem=canonical_stem_type(stem),
        spotify_id=spotify_id,
        crawler_session_id=crawler_session_id,
    )


def summarize(objects: Iterable[GpuIngestionObject]) -> dict[str, object]:
    rows = list(objects)
    by_song: dict[str, set[str]] = defaultdict(set)
    by_segment: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        song_key = row.spotify_id or row.job_id
        by_song[song_key].add(row.segment_id)
        by_segment[f"{song_key}:{row.segment_id}"].add(row.canonical_stem)

    complete_segments = sum(1 for stems in by_segment.values() if {"chord", "voice", "beat", "bass"} <= stems)
    chord_segments = sum(1 for stems in by_segment.values() if "chord" in stems)
    return {
        "objects": len(rows),
        "songs_or_jobs": len(by_song),
        "segments": len(by_segment),
        "segments_with_chord": chord_segments,
        "segments_with_all_four_stems": complete_segments,
        "stem_counts": dict(Counter(row.canonical_stem for row in rows)),
        "crawler_songs": len({row.spotify_id for row in rows if row.spotify_id}),
    }


def read_uris_from_gcloud(bucket: str, prefix: str) -> list[str]:
    target = f"gs://{bucket}/{prefix.rstrip('/')}/**"
    result = subprocess.run(
        ["gcloud", "storage", "ls", "--recursive", target],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout.splitlines()


def write_jsonl(path: Path, rows: Iterable[GpuIngestionObject]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")


def write_csv(path: Path, rows: Iterable[GpuIngestionObject]) -> None:
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()) if rows else ["uri"])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inventory gpu-ingestion audio artifacts in GCS.")
    parser.add_argument("--bucket", default="shibuya-assets")
    parser.add_argument("--prefix", default="gpu-ingestion")
    parser.add_argument("--from-file", type=Path, help="Read gcloud storage ls output from a file.")
    parser.add_argument("--jsonl", type=Path, help="Write parsed object rows as JSONL.")
    parser.add_argument("--csv", type=Path, help="Write parsed object rows as CSV.")
    parser.add_argument("--sample", type=int, default=10, help="Number of parsed rows to print.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.from_file:
        uris = args.from_file.read_text(encoding="utf-8").splitlines()
    else:
        uris = read_uris_from_gcloud(args.bucket, args.prefix)

    rows = [row for uri in uris if (row := parse_gpu_ingestion_uri(uri))]
    summary = summarize(rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.sample and rows:
        print("\nSample:")
        for row in rows[: args.sample]:
            print(json.dumps(asdict(row), sort_keys=True))
    if args.jsonl:
        write_jsonl(args.jsonl, rows)
        print(f"\nWrote {args.jsonl}")
    if args.csv:
        write_csv(args.csv, rows)
        print(f"Wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

