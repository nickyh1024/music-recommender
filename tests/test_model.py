import unittest

import pandas as pd

from musicrec import MusicRecommender


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


if __name__ == "__main__":
    unittest.main()
