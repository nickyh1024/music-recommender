"""Create an app-ready catalog from the official FMA metadata archive."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


AUDIO_FEATURES = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "tempo",
]


def prepare(metadata_dir: Path, output: Path, max_tracks: int = 3000) -> pd.DataFrame:
    tracks = pd.read_csv(metadata_dir / "tracks.csv", index_col=0, header=[0, 1])
    echonest = pd.read_csv(metadata_dir / "echonest.csv", index_col=0, header=[0, 1, 2])

    catalog = pd.DataFrame(index=echonest.index)
    for feature in AUDIO_FEATURES:
        catalog[feature] = echonest[("echonest", "audio_features", feature)]
    catalog["title"] = tracks[("track", "title")]
    catalog["artist"] = tracks[("artist", "name")]
    catalog["genre"] = tracks[("track", "genre_top")]
    catalog["listens"] = tracks[("track", "listens")]
    catalog = catalog.dropna(subset=["title", "artist", "genre", *AUDIO_FEATURES])
    catalog = catalog[catalog["title"].str.strip().ne("") & catalog["artist"].str.strip().ne("")]

    # Keep genres represented and favor tracks with more listening evidence.
    per_genre = max(1, max_tracks // catalog["genre"].nunique())
    catalog = (
        catalog.sort_values("listens", ascending=False)
        .groupby("genre", group_keys=False)
        .head(per_genre)
        .head(max_tracks)
    )
    log_listens = np.log1p(catalog["listens"].clip(lower=0))
    span = log_listens.max() - log_listens.min()
    catalog["popularity"] = ((log_listens - log_listens.min()) / (span or 1) * 100).round(1)
    catalog.index = "fma_" + catalog.index.astype(str)
    catalog.index.name = "track_id"
    catalog = catalog[["title", "artist", "genre", *AUDIO_FEATURES, "popularity"]]
    output.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(output)
    return catalog


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/music_catalog.csv"))
    parser.add_argument("--max-tracks", type=int, default=3000)
    args = parser.parse_args()
    result = prepare(args.metadata_dir, args.output, args.max_tracks)
    print(f"Wrote {len(result):,} tracks across {result['genre'].nunique()} genres to {args.output}")
