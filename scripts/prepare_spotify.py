"""Create the app catalog from the Kaggle Spotify Tracks dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


FEATURES = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "tempo",
]


def prepare(source: Path, output: Path, max_tracks: int = 12_000) -> pd.DataFrame:
    raw = pd.read_csv(source)
    required = {
        "track_id",
        "track_name",
        "artists",
        "track_genre",
        "popularity",
        *FEATURES,
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Source is missing columns: {', '.join(missing)}")

    # The source repeats some Spotify IDs under multiple genre labels. Keep one
    # canonical row, favoring the most popular copy, then curate for recognition.
    catalog = raw.dropna(subset=list(required)).copy()
    catalog = catalog.sort_values("popularity", ascending=False)
    catalog = catalog.drop_duplicates("track_id").head(max_tracks)
    catalog = catalog.rename(
        columns={"track_name": "title", "artists": "artist", "track_genre": "genre"}
    )
    catalog = catalog[["track_id", "title", "artist", "genre", *FEATURES, "popularity"]]
    output.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(output, index=False)
    return catalog


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/music_catalog.csv"))
    parser.add_argument("--max-tracks", type=int, default=12_000)
    args = parser.parse_args()
    result = prepare(args.source, args.output, args.max_tracks)
    print(
        f"Wrote {len(result):,} unique tracks across "
        f"{result['genre'].nunique()} genres to {args.output}"
    )
