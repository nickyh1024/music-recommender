import unittest

import pandas as pd

from musicrec import MusicRecommender, evaluate_recommender, genre_hit_rate_at_k


class MusicRecommenderTests(unittest.TestCase):
    def setUp(self):
        self.catalog = pd.DataFrame([
            ["a", "Bright Pop", "A", "pop", .8, .8, .9, .1, 0, 120, 80],
            ["b", "Sunny Pop", "B", "pop", .78, .75, .85, .12, 0, 118, 70],
            ["c", "Quiet Folk", "C", "folk", .2, .2, .3, .9, .1, 80, 60],
        ], columns=["track_id", "title", "artist", "genre", "danceability", "energy", "valence", "acousticness", "instrumentalness", "tempo", "popularity"])

    def test_similar_track_ranks_first_and_seed_is_excluded(self):
        result = MusicRecommender().fit(self.catalog).recommend(["a"])
        self.assertEqual(result.iloc[0].track_id, "b")
        self.assertNotIn("a", result.track_id.tolist())

    def test_rejects_unknown_seed(self):
        with self.assertRaisesRegex(ValueError, "Unknown track_id"):
            MusicRecommender().fit(self.catalog).recommend(["missing"])

    def test_genre_hit_rate_is_bounded(self):
        result = genre_hit_rate_at_k(self.catalog, k=1, sample_size=3)
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 1)

    def test_exclusions_and_artist_cap_are_applied(self):
        expanded = pd.concat(
            [
                self.catalog,
                pd.DataFrame(
                    [["d", "Another Sunny Pop", "B", "pop", .79, .76, .86, .11, 0, 119, 72]],
                    columns=self.catalog.columns,
                ),
            ],
            ignore_index=True,
        )
        result = MusicRecommender().fit(expanded).recommend(
            ["a"], n=3, exclude_track_ids={"c"}, max_per_artist=1
        )
        self.assertNotIn("c", result.track_id.tolist())
        self.assertEqual(result.artist.str.lower().nunique(), len(result))

    def test_mood_targets_change_scores(self):
        model = MusicRecommender().fit(self.catalog)
        baseline = model.recommend(["a"], n=2, max_per_artist=2)
        chilled = model.recommend(
            ["a"], n=2, feature_targets={"energy": 0.1}, max_per_artist=2
        )
        self.assertNotEqual(baseline.score.tolist(), chilled.score.tolist())

    def test_evaluation_report_is_bounded(self):
        report = evaluate_recommender(self.catalog, k=1, sample_size=3)
        for value in (
            report.genre_hit_rate_at_k,
            report.artist_diversity_at_k,
            report.catalog_coverage_at_k,
        ):
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 1)


if __name__ == "__main__":
    unittest.main()
