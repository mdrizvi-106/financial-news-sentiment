import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from transformers import pipeline

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Market Pulse",
    page_icon="📡",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
}

.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0f0f1a 50%, #0a0f0a 100%);
}

/* Header */
.hero {
    padding: 2rem 0 2rem 0;
    border-bottom: 1px solid #1e1e2e;
    margin-bottom: 2rem;
    text-align: left;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    letter-spacing: -2px;
    color: #ffffff;
    line-height: 1;
    margin: 0;
}
.hero-title span {
    color: #00ff88;
}
.hero-subtitle {
    font-size: 0.95rem;
    color: #666680;
    margin-top: 0.5rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 300;
}

/* Ticker input area */
.stTextInput input {
    background: #12121e !important;
    border: 1px solid #2a2a3e !important;
    border-radius: 4px !important;
    color: #e8e8f0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 1.1rem !important;
    padding: 0.8rem 1rem !important;
    letter-spacing: 2px;
}
.stTextInput input:focus {
    border-color: #00ff88 !important;
    box-shadow: 0 0 0 1px #00ff88 !important;
}

/* Button */
.stButton button {
    background: #00ff88 !important;
    color: #0a0a0f !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.7rem 2rem !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    transition: all 0.2s;
}
.stButton button:hover {
    background: #00cc6a !important;
    transform: translateY(-1px);
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #12121e;
    border: 1px solid #1e1e2e;
    border-radius: 8px;
    padding: 1.2rem !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 2px;
    color: #666680 !important;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
}

/* Verdict banners */
.verdict-positive {
    background: linear-gradient(90deg, #001a0d, #002a15);
    border: 1px solid #00ff88;
    border-left: 4px solid #00ff88;
    border-radius: 4px;
    padding: 1rem 1.5rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #00ff88;
    letter-spacing: 1px;
    margin: 1rem 0;
}
.verdict-negative {
    background: linear-gradient(90deg, #1a0000, #2a0000);
    border: 1px solid #ff4444;
    border-left: 4px solid #ff4444;
    border-radius: 4px;
    padding: 1rem 1.5rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #ff4444;
    letter-spacing: 1px;
    margin: 1rem 0;
}

/* Headline cards */
.headline-card {
    background: #12121e;
    border: 1px solid #1e1e2e;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.headline-text {
    font-size: 0.9rem;
    color: #c8c8d8;
    flex: 1;
    padding-right: 1rem;
    line-height: 1.4;
}
.badge-positive {
    background: #002a15;
    color: #00ff88;
    border: 1px solid #00ff88;
    border-radius: 3px;
    padding: 0.2rem 0.6rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 1px;
    white-space: nowrap;
}
.badge-negative {
    background: #2a0000;
    color: #ff4444;
    border: 1px solid #ff4444;
    border-radius: 3px;
    padding: 0.2rem 0.6rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 1px;
    white-space: nowrap;
}
.badge-neutral {
    background: #1a1a2e;
    color: #888899;
    border: 1px solid #444455;
    border-radius: 3px;
    padding: 0.2rem 0.6rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 1px;
    white-space: nowrap;
}

/* Section headers */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 3px;
    color: #444455;
    text-transform: uppercase;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e1e2e;
}

/* Hide streamlit default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stAlert { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Hero Header ───────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-title">MARKET<span>PULSE</span></p>
    <p class="hero-subtitle">Real-time financial news sentiment · Powered by FinBERT</p>
</div>
""", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────
col_input, col_btn = st.columns([4, 1])
with col_input:
    ticker_input = st.text_input("", placeholder="Enter ticker — AAPL, TSLA, NVDA, GOOGL", label_visibility="collapsed")
with col_btn:
    st.write("")
    analyse = st.button("ANALYSE →")

# ── Analysis ──────────────────────────────────────────────────
if analyse and ticker_input:
    with st.spinner("Fetching news and running sentiment analysis..."):
        ticker = yf.Ticker(ticker_input.upper())
        news = ticker.news[:10]

    if not news:
        st.markdown('<div class="verdict-negative">⚠ NO NEWS FOUND — Try AAPL, TSLA, NVDA or GOOGL</div>', unsafe_allow_html=True)
    else:
        with st.spinner("Running FinBERT sentiment model..."):
            sentiment = pipeline("sentiment-analysis", model="ProsusAI/finbert")

            results = []
            for article in news:
                try:
                    title = article['content']['title']
                    result = sentiment(str(title))[0]
                    results.append({
                        'headline': title,
                        'sentiment': result['label'],
                        'score': round(result['score'], 3)
                    })
                except:
                    continue

        df = pd.DataFrame(results)
        df['short_headline'] = df['headline'].astype(str).str[:45] + '...'

        positive_count = len(df[df['sentiment'] == 'positive'])
        negative_count = len(df[df['sentiment'] == 'negative'])
        neutral_count  = len(df[df['sentiment'] == 'neutral'])
        total = len(df)

        # ── Metrics ───────────────────────────────────────────
        st.markdown('<div class="section-label">Sentiment Breakdown</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🟢 Bullish", positive_count)
        c2.metric("🔴 Bearish", negative_count)
        c3.metric("⚪ Neutral", neutral_count)
        c4.metric("📰 Total", total)

        # ── Verdict ───────────────────────────────────────────
        if negative_count > positive_count:
            st.markdown(f'<div class="verdict-negative">▼ BEARISH SIGNAL — {negative_count}/{total} headlines negative on {ticker_input.upper()}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="verdict-positive">▲ BULLISH SIGNAL — {positive_count}/{total} headlines positive on {ticker_input.upper()}</div>', unsafe_allow_html=True)

        # ── Chart ─────────────────────────────────────────────
        st.markdown('<div class="section-label">Confidence Scores</div>', unsafe_allow_html=True)

        fig = go.Figure()
        colors = {'positive': '#00ff88', 'negative': '#ff4444', 'neutral': '#444466'}
        for sentiment_type in ['positive', 'negative', 'neutral']:
            mask = df['sentiment'] == sentiment_type
            fig.add_trace(go.Bar(
                x=df[mask]['short_headline'],
                y=df[mask]['score'],
                name=sentiment_type.upper(),
                marker_color=colors[sentiment_type],
                marker_line_width=0,
            ))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Space Mono', color='#888899', size=11),
            barmode='group',
            xaxis=dict(tickangle=-35, gridcolor='#1e1e2e', title=''),
            yaxis=dict(gridcolor='#1e1e2e', title='Confidence', range=[0, 1]),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
            margin=dict(t=20, b=120),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Headlines list ────────────────────────────────────
        st.markdown('<div class="section-label">Headlines</div>', unsafe_allow_html=True)
        for _, row in df.iterrows():
            badge_class = f"badge-{row['sentiment']}"
            label = f"{row['sentiment'].upper()} {row['score']:.2f}"
            st.markdown(f"""
            <div class="headline-card">
                <span class="headline-text">{row['headline']}</span>
                <span class="{badge_class}">{label}</span>
            </div>
            """, unsafe_allow_html=True)