from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChartCandidate:
    spotify_id: str
    title: str
    artist: str
    artists: list[str]
    popularity: int
    playlist_source: str
    rank: int
    artist_ids: list[str] | None = None
    album_id: str | None = None
    album: str | None = None
    album_art_url: str | None = None
    album_art_highres: str | None = None
    album_art_medres: str | None = None
    album_art_lowres: str | None = None
    album_artists: list[dict[str, str]] | None = None
    album_type: str | None = None
    release_date: str | None = None
    total_tracks: int | None = None
    duration_ms: int | None = None
    isrc: str | None = None

    @property
    def source(self) -> str:
        return f"spotify:track:{self.spotify_id}"

    def to_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "spotify_id": self.spotify_id,
            "title": self.title,
            "artist": self.artist,
            "artists": self.artists,
            "popularity": self.popularity,
        }
        optional = {
            "artist_ids": self.artist_ids,
            "album_id": self.album_id,
            "album": self.album,
            "album_art_url": self.album_art_url,
            "album_art_highres": self.album_art_highres,
            "album_art_medres": self.album_art_medres,
            "album_art_lowres": self.album_art_lowres,
            "album_artists": self.album_artists,
            "album_type": self.album_type,
            "release_date": self.release_date,
            "total_tracks": self.total_tracks,
            "duration_ms": self.duration_ms,
            "isrc": self.isrc,
            "metadata_source": "spotify_chart_candidate",
        }
        for key, value in optional.items():
            if value not in (None, "", []):
                metadata[key] = value
        return metadata

    def to_dict(self) -> dict[str, Any]:
        data = {
            "spotify_id": self.spotify_id,
            "title": self.title,
            "artist": self.artist,
            "artists": self.artists,
            "popularity": self.popularity,
            "playlist_source": self.playlist_source,
            "rank": self.rank,
        }
        data.update({key: value for key, value in self.to_metadata().items() if key not in data})
        return data
