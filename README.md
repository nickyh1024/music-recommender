# Music Recommender

An explainable content-based music recommendation app. Choose songs you like
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

The bundled catalog contains fictional demo data. You can upload a real CSV
with `track_id`, `title`, `artist`, `genre`, `danceability`, `energy`, `valence`,
`acousticness`, `instrumentalness`, `tempo`, and optional `popularity` (0–100).

## Roadmap

- Replace the demo catalog with licensed or user-provided music metadata.
- Collect likes, skips, and listening history.
- Add collaborative filtering and offline ranking evaluation.
- Combine content and behavioral signals in a hybrid recommender.
