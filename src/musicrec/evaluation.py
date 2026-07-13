"""Offline diagnostics for a content-based recommender."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .model import MusicRecommender


@dataclass(frozen=True)
class EvaluationReport:
    genre_hit_rate_at_k: float
    artist_diversity_at_k: float
    catalog_coverage_at_k: float


def evaluate_recommender(
    catalog: pd.DataFrame, k: int = 5, sample_size: int = 200
) -> EvaluationReport:
    """Measure genre relevance, artist variety, and catalog exposure."""
    if k < 1:
        raise ValueError("k must be positive")
    sample = catalog.sample(min(sample_size, len(catalog)), random_state=42)
    model = MusicRecommender().fit(catalog)
    indexed = catalog.set_index(catalog["track_id"].astype(str))
    genre_hits = 0
    artist_diversity = 0.0
    exposed: set[str] = set()
    for track_id in sample["track_id"].astype(str):
        expected_genre = str(indexed.loc[track_id, "genre"])
        recommendations = model.recommend([track_id], n=k)
        genre_hits += recommendations["genre"].eq(expected_genre).any()
        primary_artists = recommendations["artist"].str.split(";").str[0].str.lower()
        artist_diversity += primary_artists.nunique() / max(1, len(recommendations))
        exposed.update(recommendations["track_id"].astype(str))
    size = len(sample)
    return EvaluationReport(
        genre_hit_rate_at_k=genre_hits / size,
        artist_diversity_at_k=artist_diversity / size,
        catalog_coverage_at_k=len(exposed) / len(catalog),
    )


def genre_hit_rate_at_k(catalog: pd.DataFrame, k: int = 5, sample_size: int = 200) -> float:
    """Share of seed tracks with a same-genre recommendation in the top K."""
    if k < 1:
        raise ValueError("k must be positive")
    return evaluate_recommender(catalog, k, sample_size).genre_hit_rate_at_k
