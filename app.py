"""Interactive music recommendation demo."""

from html import escape
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from musicrec import MusicRecommender


st.set_page_config(
    page_title="SoundScout — Music Recommender",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@700;800&display=swap');
    :root { --ink: #f7f4ff; --muted: #aba6bd; --accent: #a78bfa; --pink: #f472b6; }
    .stApp {
        background:
          radial-gradient(circle at 76% 2%, rgba(124,58,237,.20), transparent 28rem),
          radial-gradient(circle at 15% 42%, rgba(236,72,153,.10), transparent 25rem),
          #0b0911;
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: rgba(11,9,17,.82); border-bottom: 1px solid rgba(255,255,255,.04); }
    [data-testid="stToolbar"] { right: 1rem; }
    .block-container { max-width: 1240px; padding-top: 1.5rem; padding-bottom: 2rem; }
    html, body, [class*="css"] { font-family: "DM Sans", sans-serif; }
    h1, h2, h3 { font-family: "Manrope", sans-serif !important; letter-spacing: -.035em; color:#f8f6ff !important; }
    p, label, [data-testid="stCaptionContainer"] { color:#b6b0c4; }
    [data-testid="stSidebar"] { background: rgba(17,14,27,.92); border-right: 1px solid #292337; }
    [data-testid="stSidebar"] > div { padding-top: 1.35rem; }
    [data-testid="stSidebar"] h3 { font-size:1.05rem; margin-top:.6rem; }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p { color:#bcb5ca !important; }
    [data-testid="stMetric"] {
        background: rgba(25,20,38,.72); border: 1px solid #302942;
        border-radius: 16px; padding: .9rem 1rem;
    }
    [data-testid="stMetricLabel"] p { color:#aaa3b8 !important; }
    [data-testid="stMetricValue"] { color:#f6f2ff !important; font:700 1.85rem "Manrope"; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(145deg, rgba(31,25,46,.88), rgba(18,15,28,.88));
        border: 1px solid #342d46 !important; border-radius: 18px !important;
        box-shadow: 0 18px 50px rgba(0,0,0,.16);
    }
    .hero { padding: 1.8rem 0 1.2rem; max-width: 760px; }
    .eyebrow { color: #c4b5fd; font-weight: 700; letter-spacing: .12em; font-size: .75rem; text-transform: uppercase; }
    .hero h1 { font-size: clamp(2.65rem, 5vw, 4.55rem); line-height: 1.01; margin: .55rem 0 1rem; }
    .gradient-text { background: linear-gradient(95deg,#c4b5fd,#f9a8d4); -webkit-background-clip:text; color:transparent; }
    .hero-copy { color: var(--muted); font-size: 1.08rem; max-width: 640px; line-height: 1.65; }
    .brand { font: 800 1.3rem "Manrope"; margin-bottom: .2rem; }
    .brand span { color: var(--accent); }
    .side-copy { color: #8f899f; font-size: .86rem; margin-bottom: 1.6rem; }
    .section-label { color:#928ca3; font-size:.74rem; font-weight:700; text-transform:uppercase; letter-spacing:.12em; margin-bottom:.15rem; }
    .track-title { font: 700 1.05rem "Manrope"; color:#fbfaff; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .track-artist { color:#aaa4b8; font-size:.87rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin:.15rem 0 .8rem; }
    .track-meta { display:flex; justify-content:space-between; align-items:center; gap:.5rem; }
    .genre-pill { color:#d8b4fe; background:rgba(168,85,247,.13); border:1px solid rgba(192,132,252,.24); border-radius:999px; padding:.24rem .55rem; font-size:.72rem; }
    .match { color:#86efac; font-size:.78rem; font-weight:700; }
    .profile { background:linear-gradient(100deg,rgba(124,58,237,.16),rgba(236,72,153,.09)); border:1px solid #3a2f51; border-radius:18px; padding:1rem 1.2rem; color:#c9c3d5; margin:.25rem 0 1.25rem; }
    .profile strong { color:#fff; }
    .footer { color:#716b80; font-size:.78rem; text-align:center; padding:2rem 0 1rem; }
    .stMultiSelect [data-baseweb="tag"] { background:#6d28d9; }
    [data-baseweb="select"] > div, [data-baseweb="input"] { background:#181323 !important; border-color:#3a314c !important; color:#f7f4ff !important; }
    [data-testid="stFileUploaderDropzone"] { background:#17131f !important; border:1px dashed #493e60 !important; border-radius:14px; }
    [data-testid="stFileUploaderDropzone"] button { background:#241c34; color:#eee9f8; border-color:#493e60; }
    [data-testid="stFileUploaderDropzoneInstructions"] span, [data-testid="stFileUploaderDropzoneInstructions"] small { color:#9e97ac !important; }
    [data-testid="stSlider"] [role="slider"] { background:#c4b5fd; }
    .stButton > button { border-radius:999px; border-color:#463b60; }
    @media (max-width: 700px) { .hero { padding-top:1rem; } .hero h1 { font-size:2.65rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


def taste_description(seeds: pd.DataFrame) -> str:
    energy = seeds["energy"].mean()
    valence = seeds["valence"].mean()
    acoustic = seeds["acousticness"].mean()
    pace = "high-energy" if energy > 0.68 else "laid-back" if energy < 0.38 else "balanced"
    mood = "bright" if valence > 0.62 else "moody" if valence < 0.38 else "bittersweet"
    texture = "acoustic" if acoustic > 0.58 else "polished"
    return f"A <strong>{mood}</strong>, <strong>{pace}</strong> taste with a {texture} edge."


default_path = Path(__file__).parent / "data" / "music_catalog.csv"

with st.sidebar:
    st.markdown('<div class="brand">Sound<span>Scout</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-copy">Your taste, translated into sound.</div>', unsafe_allow_html=True)
    with st.expander("Use your own catalog"):
        uploaded = st.file_uploader("Upload a CSV", type="csv", label_visibility="collapsed")

catalog = pd.read_csv(uploaded if uploaded else default_path)
catalog["label"] = catalog["title"] + " — " + catalog["artist"]

with st.sidebar:
    st.markdown("### Build your taste profile")
    st.caption("Pick one to five tracks you already love.")
    choices = st.multiselect(
        "Favorite tracks",
        catalog["label"],
        default=catalog["label"].head(1).tolist(),
        max_selections=5,
        placeholder="Search songs or artists",
    )
    n = st.slider("Playlist length", 3, 12, 6)
    st.divider()
    st.caption(f"{len(catalog):,} tracks  ·  {catalog['genre'].nunique()} genres")

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Personalized music discovery</div>
      <h1>Find your next<br><span class="gradient-text">favorite track.</span></h1>
      <div class="hero-copy">Choose a few songs that feel like you. SoundScout reads their mood, energy, and texture to build a playlist worth pressing play on.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric1, metric2, metric3 = st.columns(3)
metric1.metric("Music library", f"{len(catalog):,}")
metric2.metric("Genres to explore", f"{catalog['genre'].nunique()}")
metric3.metric("Audio signals", "6")

if choices:
    seed_rows = catalog[catalog["label"].isin(choices)]
    st.markdown("## Your discovery mix")
    st.markdown(
        f'<div class="profile"><span class="section-label">Taste profile</span><br>{taste_description(seed_rows)}</div>',
        unsafe_allow_html=True,
    )
    seed_ids = seed_rows["track_id"].astype(str).tolist()
    recommendations = MusicRecommender().fit(catalog).recommend(seed_ids, n=n)
    rows = [st.columns(2) for _ in range((len(recommendations) + 1) // 2)]
    for index, recommendation in recommendations.iterrows():
        column = rows[index // 2][index % 2]
        with column:
            with st.container(border=True):
                score = min(100, round(recommendation["score"] * 100))
                st.markdown(
                    f"""
                    <div class="section-label">#{index + 1:02d} · {escape(recommendation['match_reason'])}</div>
                    <div class="track-title">{escape(str(recommendation['title']))}</div>
                    <div class="track-artist">{escape(str(recommendation['artist']))}</div>
                    <div class="track-meta">
                      <span class="genre-pill">{escape(str(recommendation['genre']))}</span>
                      <span class="match">{score}% match</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
else:
    st.info("Choose at least one favorite track in the sidebar to build your mix.")

st.markdown(
    '<div class="footer">Built from real Spotify track metadata · No audio is redistributed · Independent educational project</div>',
    unsafe_allow_html=True,
)
