"""Interactive music recommendation demo."""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from musicrec import MusicRecommender

st.set_page_config(page_title="Music Recommender", page_icon="🎧", layout="wide")
st.title("Music Recommender")
st.caption("Choose songs you like to create a taste profile.")

default_path = Path(__file__).parent / "data" / "music_catalog.csv"
uploaded = st.file_uploader("Optional: upload your own catalog CSV", type="csv")
catalog = pd.read_csv(uploaded if uploaded else default_path)
catalog["label"] = catalog["title"] + " — " + catalog["artist"]
source = "your uploaded catalog" if uploaded else "the Kaggle Spotify Tracks dataset"
st.caption(f"Searching {len(catalog):,} tracks across {catalog['genre'].nunique()} genres from {source}.")
choices = st.multiselect("Songs you like", catalog["label"], default=catalog["label"].head(1).tolist())
n = st.slider("Number of recommendations", 1, min(10, len(catalog) - 1), min(5, len(catalog) - 1))

if choices:
    seed_ids = catalog.loc[catalog["label"].isin(choices), "track_id"].astype(str).tolist()
    recommendations = MusicRecommender().fit(catalog).recommend(seed_ids, n=n)
    recommendations["match"] = recommendations["score"].map(lambda value: f"{value:.0%}")
    st.subheader("Recommended for you")
    st.dataframe(recommendations[["title", "artist", "genre", "match", "match_reason"]], hide_index=True, width="stretch")
else:
    st.info("Choose at least one song to get recommendations.")

st.caption(
    "The bundled catalog uses real Spotify track metadata and redistributes no audio. "
    "Upload another catalog with the same columns to replace it."
)
