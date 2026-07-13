# Music Recommender

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://byi8omgknvvsd36njg7qlz.streamlit.app/)

**Live app:** https://byi8omgknvvsd36njg7qlz.streamlit.app/

An explainable content-based music recommendation app built with real metadata
from the Free Music Archive (FMA). Choose songs you like
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

The bundled catalog is a compact, genre-balanced sample of FMA tracks. You can upload another CSV
with `track_id`, `title`, `artist`, `genre`, `danceability`, `energy`, `valence`,
`acousticness`, `instrumentalness`, `tempo`, and optional `popularity` (0–100).

## Data

FMA contains 106,574 Creative Commons-licensed tracks. This project uses track
metadata and Echo Nest audio features from its CC BY 4.0 metadata release; it
does not redistribute audio. The committed catalog can be reproduced by
downloading `fma_metadata.zip` from the official FMA repository and running:

```bash
python scripts/prepare_fma.py /path/to/fma_metadata
```

`genre_hit_rate_at_k` provides a deterministic offline diagnostic: whether at
least one of the top-K recommendations shares the seed track's genre. Genre is
only a proxy for relevance, not a substitute for evaluation with real users.

## Roadmap

- Replace the demo catalog with licensed or user-provided music metadata.
- Collect likes, skips, and listening history.
- Add collaborative filtering and offline ranking evaluation.
- Combine content and behavioral signals in a hybrid recommender.
