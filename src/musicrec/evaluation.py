"""Offline diagnostics for a content-based recommender."""

from __future__ import annotations

import pandas as pd

from .model import MusicRecommender


def genre_hit_rate_at_k(catalog: pd.DataFrame, k: int = 5, sample_size: int = 200) -> float:
    """Share of seed tracks with a same-genre recommendation in the top K."""
    if k < 1:
        raise ValueError("k must be positive")
    sample = catalog.sample(min(sample_size, len(catalog)), random_state=42)
    model = MusicRecommender().fit(catalog)
    hits = 0
    indexed = catalog.set_index(catalog["track_id"].astype(str))
    for track_id in sample["track_id"].astype(str):
        expected_genre = str(indexed.loc[track_id, "genre"])
        recommendations = model.recommend([track_id], n=k)
        hits += recommendations["genre"].eq(expected_genre).any()
    return hits / len(sample)
