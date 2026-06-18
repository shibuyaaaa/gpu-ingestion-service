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

    @property
    def source(self) -> str:
        return f"spotify:track:{self.spotify_id}"

    def to_metadata(self) -> dict[str, Any]:
        return {
            "spotify_id": self.spotify_id,
            "title": self.title,
            "artist": self.artist,
            "artists": self.artists,
            "popularity": self.popularity,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "spotify_id": self.spotify_id,
            "title": self.title,
            "artist": self.artist,
            "artists": self.artists,
            "popularity": self.popularity,
            "playlist_source": self.playlist_source,
            "rank": self.rank,
        }
