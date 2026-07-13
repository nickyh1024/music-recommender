"""Small, explainable music recommendation toolkit."""

from .model import MusicRecommender
from .evaluation import EvaluationReport, evaluate_recommender, genre_hit_rate_at_k

__all__ = [
    "EvaluationReport",
    "MusicRecommender",
    "evaluate_recommender",
    "genre_hit_rate_at_k",
]
