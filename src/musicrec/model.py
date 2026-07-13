"""Content-based music recommendations using audio features and genres."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURES = ("danceability", "energy", "valence", "acousticness", "instrumentalness", "tempo")
REQUIRED = ("track_id", "title", "artist", "genre", *FEATURES)


@dataclass
class MusicRecommender:
    """Rank unseen tracks against a profile built from seed tracks."""

    genre_weight: float = 0.20
    popularity_weight: float = 0.05

    def fit(self, catalog: pd.DataFrame) -> "MusicRecommender":
        missing = sorted(set(REQUIRED) - set(catalog.columns))
        if missing:
            raise ValueError(f"Catalog is missing columns: {', '.join(missing)}")
        if catalog["track_id"].duplicated().any():
            raise ValueError("track_id values must be unique")
        self.catalog_ = catalog.copy().reset_index(drop=True)
        raw = self.catalog_.loc[:, FEATURES].apply(pd.to_numeric, errors="raise").to_numpy(float)
        self.feature_min_ = raw.min(axis=0)
        span = raw.max(axis=0) - self.feature_min_
        self.feature_span_ = np.where(span == 0, 1.0, span)
        self.vectors_ = (raw - self.feature_min_) / self.feature_span_
        return self

    def recommend(self, seed_track_ids: list[str], n: int = 10) -> pd.DataFrame:
        if not hasattr(self, "catalog_"):
            raise RuntimeError("Call fit before recommend")
        if not seed_track_ids:
            raise ValueError("Choose at least one seed track")
        seed_track_ids = list(dict.fromkeys(seed_track_ids))
        seed_mask = self.catalog_["track_id"].astype(str).isin(map(str, seed_track_ids))
        if seed_mask.sum() != len(seed_track_ids):
            known = set(self.catalog_.loc[seed_mask, "track_id"].astype(str))
            unknown = [str(x) for x in seed_track_ids if str(x) not in known]
            raise ValueError(f"Unknown track_id values: {', '.join(unknown)}")
        profile = self.vectors_[seed_mask].mean(axis=0)
        distance = np.linalg.norm(self.vectors_ - profile, axis=1) / np.sqrt(len(FEATURES))
        audio_similarity = 1.0 - distance
        seed_genres = set(self.catalog_.loc[seed_mask, "genre"].str.lower())
        genre_match = self.catalog_["genre"].str.lower().isin(seed_genres).astype(float).to_numpy()
        popularity = pd.to_numeric(self.catalog_.get("popularity", 0), errors="coerce")
        popularity_score = (popularity.fillna(0).clip(0, 100).to_numpy() / 100 if not np.isscalar(popularity) else np.zeros(len(self.catalog_)))
        score = (
            audio_similarity
            + self.genre_weight * genre_match
            + self.popularity_weight * popularity_score
        ) / (1 + self.genre_weight + self.popularity_weight)
        candidates = self.catalog_.loc[~seed_mask].copy()
        candidates["score"] = score[~seed_mask]
        candidates["match_reason"] = np.where(genre_match[~seed_mask] > 0, "Similar sound and genre", "Similar audio profile")
        return candidates.sort_values(["score", "title"], ascending=[False, True]).head(n).reset_index(drop=True)
