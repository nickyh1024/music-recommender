# Music Recommender

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://byi8omgknvvsd36njg7qlz.streamlit.app/)

**Live app:** https://byi8omgknvvsd36njg7qlz.streamlit.app/

An explainable content-based music recommendation app built with a curated
catalog of 12,000 popular Spotify tracks. Choose songs you like
and the model recommends unseen tracks based on normalized audio similarity,
genre overlap, and a small popularity tie-breaker.

## Quick start

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
streamlit run app.py
```

Run tests with `python -m unittest discover -s tests -v`.

The bundled catalog contains the most popular unique tracks from a 114K-track
Kaggle dataset. You can upload another CSV
with `track_id`, `title`, `artist`, `genre`, `danceability`, `energy`, `valence`,
`acousticness`, `instrumentalness`, `tempo`, and optional `popularity` (0–100).

## Data

The default catalog is derived from the CC0-licensed **Spotify Tracks Dataset —
114K Songs Across 114 Genres** on Kaggle. The project stores metadata and audio
features only; it does not redistribute audio. Download the source CSV and run:

```bash
python scripts/prepare_spotify.py /path/to/spotify-tracks-dataset-detailed.csv
```

`genre_hit_rate_at_k` provides a deterministic offline diagnostic: whether at
least one of the top-K recommendations shares the seed track's genre. Genre is
only a proxy for relevance, not a substitute for evaluation with real users.

## Roadmap

- Add direct Spotify links and album artwork.
- Collect likes, skips, and listening history.
- Add collaborative filtering and offline ranking evaluation.
- Combine content and behavioral signals in a hybrid recommender.
