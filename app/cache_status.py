from pathlib import Path
from typing import Any


def local_cache_status(
    *,
    work_dir: Path,
    source_audio_enabled: bool,
    source_audio_max_entries: int,
    source_audio_max_bytes: int,
    analysis_enabled: bool,
    analysis_max_entries: int,
    analysis_max_bytes: int,
    lock_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = _source_audio_cache_status(
        work_dir / "source-cache",
        enabled=source_audio_enabled,
        max_entries=source_audio_max_entries,
        max_bytes=source_audio_max_bytes,
    )
    analysis = _analysis_cache_status(
        work_dir / "analysis-cache",
        enabled=analysis_enabled,
        max_entries=analysis_max_entries,
        max_bytes=analysis_max_bytes,
    )
    return {
        "source_audio": source,
        "analysis": analysis,
        "total_bytes": source["bytes"] + analysis["bytes"],
        "total_mib": round((source["bytes"] + analysis["bytes"]) / (1024 * 1024), 3),
        "locks": lock_status or {},
    }


def _source_audio_cache_status(path: Path, *, enabled: bool, max_entries: int, max_bytes: int) -> dict[str, Any]:
    files = [item for item in _safe_iterdir(path) if item.is_file() and item.suffix.lower() == ".wav"]
    total_bytes = sum(_safe_file_size(item) for item in files)
    return {
        "enabled": enabled,
        "path": str(path),
        "entries": len(files),
        "max_entries": max_entries,
        "max_bytes": max_bytes,
        "max_mib": round(max_bytes / (1024 * 1024), 3),
        "bytes": total_bytes,
        "mib": round(total_bytes / (1024 * 1024), 3),
    }


def _analysis_cache_status(path: Path, *, enabled: bool, max_entries: int, max_bytes: int) -> dict[str, Any]:
    entries = [item for item in _safe_iterdir(path) if item.is_dir() and item.name != ".tmp"]
    complete_entries = sum(1 for item in entries if _analysis_entry_complete(item))
    total_bytes = sum(_directory_size(item) for item in entries)
    return {
        "enabled": enabled,
        "path": str(path),
        "entries": len(entries),
        "complete_entries": complete_entries,
        "max_entries": max_entries,
        "max_bytes": max_bytes,
        "max_mib": round(max_bytes / (1024 * 1024), 3),
        "bytes": total_bytes,
        "mib": round(total_bytes / (1024 * 1024), 3),
    }


def _analysis_entry_complete(path: Path) -> bool:
    if not (path / "metadata.json").is_file() or not (path / "analyzer_result.json").is_file():
        return False
    stems_dir = path / "stems"
    return all((stems_dir / f"{stem}.wav").is_file() for stem in ("bass", "drums", "other", "vocals"))


def _directory_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for child in path.rglob("*"):
        if child.is_file():
            total += _safe_file_size(child)
    return total


def _safe_iterdir(path: Path) -> list[Path]:
    try:
        return list(path.iterdir())
    except OSError:
        return []


def _safe_file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0
