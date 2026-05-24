import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from transformers import pipeline
import io

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MarketPulse · FinBERT Sentiment",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --border: #1e1e2e;
    --accent: #00ff9d;
    --accent2: #ff4f4f;
    --accent3: #ffd166;
    --text: #e8e8f0;
    --muted: #6b6b8a;
    --positive: #00ff9d;
    --negative: #ff4f4f;
    --neutral: #6b6b8a;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2rem 4rem; max-width: 1400px; }

/* Hero header */
.hero {
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.5rem;
    margin-bottom: 2rem;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -1px;
    margin: 0;
    line-height: 1;
}
.hero-title span { color: var(--accent); }
.hero-sub {
    font-size: 0.78rem;
    letter-spacing: 0.18em;
    color: var(--muted);
    margin-top: 0.4rem;
    text-transform: uppercase;
}

/* Cards */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.card-label {
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.3rem;
}
.card-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
}
.positive { color: var(--positive); }
.negative { color: var(--negative); }
.neutral  { color: var(--neutral); }

/* Headline cards */
.headline-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--border);
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
    transition: border-left-color 0.2s;
}
.headline-card.positive { border-left-color: var(--positive); }
.headline-card.negative { border-left-color: var(--negative); }
.headline-card.neutral  { border-left-color: var(--neutral); }
.headline-text { font-size: 0.92rem; font-weight: 500; margin-bottom: 0.4rem; }
.headline-meta { font-size: 0.75rem; color: var(--muted); font-family: 'Space Mono', monospace; }
.badge {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.68rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-right: 0.5rem;
}
.badge.positive { background: rgba(0,255,157,0.12); color: var(--positive); }
.badge.negative { background: rgba(255,79,79,0.12);  color: var(--negative); }
.badge.neutral  { background: rgba(107,107,138,0.15); color: var(--neutral); }

/* Alert banner */
.alert-positive {
    background: rgba(0,255,157,0.07);
    border: 1px solid rgba(0,255,157,0.25);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    color: var(--positive);
    font-weight: 500;
    margin-bottom: 1.5rem;
}
.alert-negative {
    background: rgba(255,79,79,0.07);
    border: 1px solid rgba(255,79,79,0.25);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    color: var(--negative);
    font-weight: 500;
    margin-bottom: 1.5rem;
}

/* Section heading */
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 1.8rem 0 0.8rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
}

/* Ticker pill */
.ticker-pill {
    display: inline-block;
    background: rgba(0,255,157,0.1);
    border: 1px solid rgba(0,255,157,0.3);
    color: var(--accent);
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    font-weight: 700;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    margin-right: 0.4rem;
}

/* Slider label override */
.stSlider label { color: var(--muted) !important; font-size: 0.8rem !important; }

/* Input & button */
.stTextInput input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
    border-radius: 6px !important;
}
.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.55rem 1.4rem !important;
    letter-spacing: 0.05em !important;
    width: 100%;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    color: var(--muted);
    background: transparent;
    border: none;
    padding: 0.5rem 1rem;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: transparent !important;
}

/* Download button */
.stDownloadButton > button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    border-radius: 6px !important;
}
</style>
""", unsafe_allow_html=True)

# Plotly base colours

# ── Model cache ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

# ── Fetch + analyse helper ────────────────────────────────────────────────────
def fetch_and_analyse(ticker_sym: str, model, min_confidence: float) -> pd.DataFrame | None:
    ticker = yf.Ticker(ticker_sym)
    news = ticker.news[:10]
    if not news:
        return None
    results = []
    for article in news:
        title = article['content']['title']
        res = model(title)[0]
        results.append({
            'ticker': ticker_sym.upper(),
            'headline': title,
            'sentiment': res['label'],
            'score': round(res['score'], 4),
        })
    df = pd.DataFrame(results)
    df = df[df['score'] >= min_confidence]
    return df if len(df) else None

# ── Colour map ────────────────────────────────────────────────────────────────
COLOUR = {'positive': '#00ff9d', 'negative': '#ff4f4f', 'neutral': '#6b6b8a'}

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <p class="hero-title">MARKET<span>PULSE</span></p>
  <p class="hero-sub">Real-time financial news sentiment · Powered by FinBERT</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    min_conf = st.slider("Min. confidence threshold", 0.50, 0.99, 0.60, 0.01,
                         help="Hide predictions below this confidence score")
    st.markdown("---")
    st.markdown("Built with [FinBERT](https://huggingface.co/ProsusAI/finbert) · [yfinance](https://github.com/ranaroussi/yfinance)")

# ── Input row ─────────────────────────────────────────────────────────────────
col_inp, col_btn = st.columns([5, 1])
with col_inp:
    raw_input = st.text_input(
        label="tickers",
        label_visibility="collapsed",
        placeholder="Enter up to 3 tickers — AAPL, TSLA, NVDA",
        value="",
    )
with col_btn:
    analyse_clicked = st.button("ANALYSE →")

# ── Main logic ────────────────────────────────────────────────────────────────
if analyse_clicked and raw_input.strip():
    tickers = [t.strip().upper() for t in raw_input.replace(",", " ").split() if t.strip()][:3]

    with st.spinner("Loading FinBERT model…"):
        model = load_model()

    all_dfs = []
    progress = st.progress(0, text="Fetching news…")
    for i, sym in enumerate(tickers):
        progress.progress((i + 1) / len(tickers), text=f"Analysing {sym}…")
        df = fetch_and_analyse(sym, model, min_conf)
        if df is not None:
            all_dfs.append(df)
        else:
            st.warning(f"No news found for **{sym}** — skipping.")
    progress.empty()

    if not all_dfs:
        st.error("No results. Try tickers like AAPL, TSLA, NVDA, GOOGL.")
        st.stop()

    combined = pd.concat(all_dfs, ignore_index=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_overview, tab_headlines, tab_compare = st.tabs(
        ["OVERVIEW", "HEADLINES", "COMPARE"]
    )

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1 · OVERVIEW
    # ─────────────────────────────────────────────────────────────────────────
    with tab_overview:
        for sym in [d['ticker'].iloc[0] for d in all_dfs]:
            df_t = combined[combined['ticker'] == sym]
            pos = len(df_t[df_t['sentiment'] == 'positive'])
            neg = len(df_t[df_t['sentiment'] == 'negative'])
            neu = len(df_t[df_t['sentiment'] == 'neutral'])
            total = len(df_t)
            dominant = df_t['sentiment'].mode()[0]

            st.markdown(f'<p class="section-title"><span class="ticker-pill">{sym}</span> sentiment snapshot</p>', unsafe_allow_html=True)

            # Alert banner
            if dominant == 'positive':
                st.markdown(f'<div class="alert-positive">✅ Overall sentiment is <strong>BULLISH</strong> — {pos} of {total} headlines positive</div>', unsafe_allow_html=True)
            elif dominant == 'negative':
                st.markdown(f'<div class="alert-negative">⚠️ Overall sentiment is <strong>BEARISH</strong> — {neg} of {total} headlines negative</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="card">⚪ Overall sentiment is <strong>NEUTRAL</strong></div>', unsafe_allow_html=True)

            # Metric cards + donut side by side
            mc1, mc2, mc3, mc4 = st.columns([1, 1, 1, 2])
            with mc1:
                st.markdown(f'<div class="card"><div class="card-label">Positive</div><div class="card-value positive">{pos}</div></div>', unsafe_allow_html=True)
            with mc2:
                st.markdown(f'<div class="card"><div class="card-label">Negative</div><div class="card-value negative">{neg}</div></div>', unsafe_allow_html=True)
            with mc3:
                st.markdown(f'<div class="card"><div class="card-label">Neutral</div><div class="card-value neutral">{neu}</div></div>', unsafe_allow_html=True)
            with mc4:
                donut = go.Figure(go.Pie(
                    labels=['Positive', 'Negative', 'Neutral'],
                    values=[pos, neg, neu],
                    hole=0.68,
                    marker=dict(colors=['#00ff9d', '#ff4f4f', '#3a3a5c'],
                                line=dict(color='#0a0a0f', width=2)),
                    textinfo='none',
                    hovertemplate='%{label}: %{value}<extra></extra>',
                ))
                donut.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="DM Sans", color="#e8e8f0"),
                    showlegend=False,
                    height=130,
                    margin=dict(l=0, r=0, t=0, b=0),
                    annotations=[dict(
                        text=f"<b>{dominant.upper()}</b>",
                        x=0.5, y=0.5,
                        font=dict(size=13, color=COLOUR[dominant], family="Space Mono, monospace"),
                        showarrow=False,
                    )],
                )
                st.plotly_chart(donut, use_container_width=True, config={'displayModeBar': False})

            # Bar chart
            bar = px.bar(
                df_t.sort_values('score'),
                x='score', y=df_t['headline'].str[:45] + '…',
                color='sentiment',
                color_discrete_map=COLOUR,
                orientation='h',
                height=320,
            )
            bar.update_layout(
                              paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(family="DM Sans", color="#e8e8f0"),
                              xaxis=dict(showgrid=False, zeroline=False, color="#6b6b8a"),
                              yaxis=dict(showgrid=True, gridcolor="#1e1e2e", zeroline=False, color="#6b6b8a"),
                              xaxis_title="Confidence", yaxis_title="",
                              margin=dict(l=0, r=0, t=55, b=0),
                              legend=dict(orientation='h', y=1.18, x=0,
                                          font=dict(size=11)))
            bar.update_traces(marker_line_width=0)
            st.plotly_chart(bar, use_container_width=True, config={'displayModeBar': False})

            st.markdown("---")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2 · HEADLINES
    # ─────────────────────────────────────────────────────────────────────────
    with tab_headlines:
        st.markdown('<p class="section-title">All analysed headlines</p>', unsafe_allow_html=True)

        # Confidence filter
        conf_filter = st.slider("Show headlines with confidence ≥", min_conf, 1.0, min_conf, 0.01, key="headline_conf")
        filtered = combined[combined['score'] >= conf_filter]

        for _, row in filtered.iterrows():
            label_cls = row['sentiment']
            st.markdown(f"""
            <div class="headline-card {label_cls}">
                <div class="headline-text">{row['headline']}</div>
                <div class="headline-meta">
                    <span class="badge {label_cls}">{row['sentiment']}</span>
                    <span class="ticker-pill" style="font-size:0.65rem;padding:0.1rem 0.4rem;">{row['ticker']}</span>
                    Confidence: {row['score']:.2%}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # CSV export
        st.markdown('<p class="section-title">Export</p>', unsafe_allow_html=True)
        csv_buf = io.StringIO()
        filtered[['ticker', 'headline', 'sentiment', 'score']].to_csv(csv_buf, index=False)
        st.download_button(
            label="⬇ Download results as CSV",
            data=csv_buf.getvalue(),
            file_name="marketpulse_sentiment.csv",
            mime="text/csv",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3 · COMPARE (multi-ticker)
    # ─────────────────────────────────────────────────────────────────────────
    with tab_compare:
        if len(all_dfs) < 2:
            st.info("Enter 2 or 3 tickers separated by commas to compare them side by side.")
        else:
            st.markdown('<p class="section-title">Side-by-side sentiment comparison</p>', unsafe_allow_html=True)

            # Summary table
            summary_rows = []
            for df_t in all_dfs:
                sym = df_t['ticker'].iloc[0]
                pos = len(df_t[df_t['sentiment'] == 'positive'])
                neg = len(df_t[df_t['sentiment'] == 'negative'])
                neu = len(df_t[df_t['sentiment'] == 'neutral'])
                avg_conf = df_t['score'].mean()
                dominant = df_t['sentiment'].mode()[0]
                summary_rows.append({'Ticker': sym, 'Positive': pos, 'Negative': neg,
                                     'Neutral': neu, 'Avg Confidence': f"{avg_conf:.2%}",
                                     'Dominant': dominant.upper()})
            st.dataframe(pd.DataFrame(summary_rows).set_index('Ticker'),
                         use_container_width=True)

            # Grouped bar comparison
            st.markdown('<p class="section-title">Sentiment count by ticker</p>', unsafe_allow_html=True)
            comp_data = []
            for df_t in all_dfs:
                sym = df_t['ticker'].iloc[0]
                for sent in ['positive', 'negative', 'neutral']:
                    comp_data.append({'Ticker': sym, 'Sentiment': sent,
                                      'Count': len(df_t[df_t['sentiment'] == sent])})
            comp_df = pd.DataFrame(comp_data)
            comp_bar = px.bar(comp_df, x='Ticker', y='Count', color='Sentiment',
                              barmode='group',
                              color_discrete_map={'positive': '#00ff9d',
                                                  'negative': '#ff4f4f',
                                                  'neutral': '#3a3a5c'},
                              height=320)
            comp_bar.update_layout(
                                   paper_bgcolor="rgba(0,0,0,0)",
                                   plot_bgcolor="rgba(0,0,0,0)",
                                   font=dict(family="DM Sans", color="#e8e8f0"),
                                   xaxis=dict(showgrid=False, zeroline=False, color="#6b6b8a"),
                                   yaxis=dict(showgrid=True, gridcolor="#1e1e2e", zeroline=False, color="#6b6b8a"),
                                   margin=dict(l=0, r=0, t=50, b=0),
                                   legend=dict(orientation='h', y=1.12, x=0, font=dict(size=11)))
            comp_bar.update_traces(marker_line_width=0)
            st.plotly_chart(comp_bar, use_container_width=True, config={'displayModeBar': False})

            # Avg confidence radar / bar
            st.markdown('<p class="section-title">Average confidence per ticker</p>', unsafe_allow_html=True)
            conf_data = [{'Ticker': df_t['ticker'].iloc[0],
                          'Avg Confidence': df_t['score'].mean()} for df_t in all_dfs]
            conf_df = pd.DataFrame(conf_data)
            conf_bar = px.bar(conf_df, x='Ticker', y='Avg Confidence',
                              color='Ticker',
                              color_discrete_sequence=['#00ff9d', '#ffd166', '#00b4d8'],
                              height=250, text_auto='.2%')
            conf_bar.update_layout(
                                   paper_bgcolor="rgba(0,0,0,0)",
                                   plot_bgcolor="rgba(0,0,0,0)",
                                   font=dict(family="DM Sans", color="#e8e8f0"),
                                   xaxis=dict(showgrid=False, zeroline=False, color="#6b6b8a"),
                                   yaxis=dict(showgrid=True, gridcolor="#1e1e2e", zeroline=False, color="#6b6b8a"),
                                   margin=dict(l=0, r=0, t=30, b=0),
                                   showlegend=False,
                                   yaxis_tickformat='.0%')
            conf_bar.update_traces(marker_line_width=0)
            st.plotly_chart(conf_bar, use_container_width=True, config={'displayModeBar': False})

elif analyse_clicked and not raw_input.strip():
    st.warning("Please enter at least one ticker symbol.")
