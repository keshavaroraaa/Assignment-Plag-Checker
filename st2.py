import streamlit as st
import os
from plag_predictor import read_file, tokenize, score_single_file

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Risk Predictor",
    page_icon="◈",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&family=Nunito+Sans:wght@300;400;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Nunito Sans', sans-serif;
}

.main .block-container {
    padding-top: 3rem;
    padding-bottom: 4rem;
    max-width: 720px;
}

/* ── Background ── */
.stApp {
    background: #0f0e0c;
}

/* ── Hero title ── */
.hero-title {
    font-family: 'Lora', serif;
    font-size: 2.4rem;
    font-weight: 600;
    color: #f0ede8;
    line-height: 1.15;
    margin-bottom: 0.3rem;
}
.hero-title em {
    font-style: italic;
    color: #d4a96a;
}
.hero-sub {
    font-family: 'Nunito Sans', sans-serif;
    font-weight: 300;
    font-size: 0.95rem;
    color: #8a8378;
    margin-bottom: 2rem;
    line-height: 1.6;
}
.eyebrow {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #5a5650;
    margin-bottom: 0.7rem;
}

/* ── Upload area ── */
.stFileUploader {
    background: #1a1916 !important;
    border: 1.5px dashed #2e2c28 !important;
    border-radius: 10px !important;
    padding: 1.2rem !important;
    transition: border-color 0.2s !important;
}
.stFileUploader:hover {
    border-color: #4a4640 !important;
}
.stFileUploader label {
    color: #8a8378 !important;
    font-size: 0.85rem !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #d4a96a !important;
    color: #0f0e0c !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Nunito Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.6rem 1.4rem !important;
    letter-spacing: 0.02em !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
}

/* ── Info / warning / error / success boxes ── */
.stAlert {
    border-radius: 8px !important;
    border-left-width: 3px !important;
    font-size: 0.87rem !important;
}

/* ── Divider ── */
hr {
    border-color: #2a2825 !important;
    margin: 1.5rem 0 !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #1a1916;
    border: 1px solid #252320;
    border-radius: 10px;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #5a5650 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.5rem !important;
    color: #f0ede8 !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.75rem !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #4a7c59, #d4a96a, #c0392b) !important;
    border-radius: 99px !important;
}
.stProgress > div > div > div {
    background: #2a2825 !important;
    border-radius: 99px !important;
    height: 6px !important;
}

/* ── Risk level pill ── */
.risk-pill {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.risk-low  { background: #1a2e22; color: #5db882; border: 1px solid #2d4f38; }
.risk-med  { background: #2e2414; color: #d4a96a; border: 1px solid #4a3820; }
.risk-high { background: #2e1414; color: #e05252; border: 1px solid #4a2020; }

/* ── Score number ── */
.score-big {
    font-family: 'Lora', serif;
    font-size: 3.2rem;
    font-weight: 600;
    color: #f0ede8;
    line-height: 1;
}
.score-big span {
    font-size: 1.2rem;
    color: #5a5650;
    font-family: 'Nunito Sans', sans-serif;
    font-weight: 300;
}

/* ── Section heading ── */
.section-heading {
    font-family: 'Lora', serif;
    font-style: italic;
    font-size: 1.15rem;
    color: #c8bfb0;
    margin-bottom: 1rem;
    margin-top: 0.5rem;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #1a1916 !important;
    border: 1px solid #252320 !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    color: #8a8378 !important;
}
.streamlit-expanderContent {
    background: #131210 !important;
    border: 1px solid #1e1d1a !important;
    border-top: none !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0c0b09 !important;
    border-right: 1px solid #1e1d1a !important;
}
section[data-testid="stSidebar"] * {
    color: #8a8378 !important;
}
section[data-testid="stSidebar"] h3 {
    font-family: 'Lora', serif !important;
    color: #c8bfb0 !important;
    font-size: 1rem !important;
}
.stSidebar .stInfo {
    background: #1a1916 !important;
    border-color: #2e2c28 !important;
    font-size: 0.82rem !important;
}

/* ── Caption text ── */
.stCaption {
    color: #4a4640 !important;
    font-size: 0.78rem !important;
}

/* ── Code / text blocks ── */
.stTextArea textarea, .stCode {
    background: #131210 !important;
    color: #8a8378 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
    border-color: #252320 !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #d4a96a !important;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### How it works")
    st.info("""
This tool uses **TF-IDF** and **heuristic AI scoring** to analyse the *DNA* of writing —
not just matching text online, but detecting whether it reads like a single human author
or a patchwork of sources.
    """)
    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.75rem; color:#3a3830;">Metrics include vocabulary richness, '
        'passive voice ratio, phrase density, structural uniformity, style shifts, and formality.</p>',
        unsafe_allow_html=True,
    )


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="eyebrow">Academic Integrity Tool</p>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Assignment <em>Risk</em> Predictor</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Upload a submission to analyse its stylistic fingerprint '
    'and surface indicators of plagiarism or AI-generated content.</p>',
    unsafe_allow_html=True,
)

# ── Upload ─────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Drop a file here — .txt, .py, .md, .html",
    type=["txt", "py", "md", "html"],
    label_visibility="visible",
)

# ── Analysis ───────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    text = bytes_data.decode("utf-8", errors="ignore")

    with st.spinner("Reading linguistic patterns…"):
        tokens = tokenize(text)

    if len(tokens) < 10:
        st.error("File is too short for a reliable analysis. Please upload a longer document.")
    else:
        results = score_single_file(text, tokens)
        risk_score = results["overall_risk_score"]
        pct = risk_score * 100

        # Determine risk tier
        if risk_score < 0.3:
            tier, pill_class, tier_label = "low", "risk-low", "Low Risk"
        elif risk_score < 0.6:
            tier, pill_class, tier_label = "med", "risk-med", "Moderate Risk"
        else:
            tier, pill_class, tier_label = "high", "risk-high", "High Risk"

        st.markdown("---")

        # ── Score block ───────────────────────────────────────────────────────
        col_score, col_bar = st.columns([1, 2], gap="large")

        with col_score:
            st.markdown(
                f'<div class="score-big">{pct:.1f}<span>%</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<span class="risk-pill {pill_class}">{tier_label}</span>',
                unsafe_allow_html=True,
            )

        with col_bar:
            st.markdown(
                '<p class="section-heading">Overall Risk Score</p>',
                unsafe_allow_html=True,
            )
            st.progress(risk_score)
            st.caption(
                f"Based on vocabulary richness, passive voice, phrase density, "
                f"structural uniformity, style shifts & formality. "
                f"File: **{uploaded_file.name}**"
            )

        st.markdown("---")

        # ── Feature breakdown ─────────────────────────────────────────────────
        st.markdown('<p class="section-heading">Feature Breakdown</p>', unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Vocab Richness",
            f"{results['vocabulary_richness']*100:.1f}%",
            help="Higher = more original vocabulary diversity",
        )
        m2.metric(
            "Phrase Density",
            f"{results['repeated_phrase_density']*100:.1f}%",
            delta_color="inverse",
            help="Higher = more repeated phrases",
        )
        m3.metric(
            "Style Shifts",
            f"{results['style_shift_score']}",
            help="Abrupt changes in writing complexity",
        )

        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

        m4, m5, m6 = st.columns(3)
        m4.metric(
            "Uniformity",
            f"{results['structural_uniformity']*100:.1f}%",
            help="High uniformity may indicate templated writing",
        )
        m5.metric(
            "Passive Voice",
            f"{results['passive_voice_ratio']*100:.1f}%",
            help="Elevated ratio common in AI-generated text",
        )
        m6.metric(
            "Formality",
            f"{results['formality_score']*100:.1f}%",
            help="Unusually high formality can indicate non-human authorship",
        )

        st.markdown("---")

        # ── Raw text preview ──────────────────────────────────────────────────
        with st.expander("View analysed text"):
            st.code(
                text[:2000] + ("…" if len(text) > 2000 else ""),
                language=None,
            )

else:
    st.info("Upload a file above to begin the analysis.")