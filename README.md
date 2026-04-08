# 📡 MarketPulse — Financial News Sentiment Analyser

A real-time NLP web app that analyses the sentiment of live financial news headlines for any stock ticker — built with FinBERT, Streamlit, and yfinance.

🔗 **Live Demo:** [your-app-url.streamlit.app](https://your-app-url.streamlit.app)

---

## What It Does

Enter any stock ticker (AAPL, TSLA, NVDA, GOOGL) and the app:

1. Fetches the latest 10 news headlines for that stock
2. Runs each headline through **FinBERT** — a transformer model pre-trained on financial text
3. Classifies each headline as **Positive**, **Negative**, or **Neutral**
4. Displays a confidence score, sentiment breakdown, and an overall bullish/bearish verdict

---

## Why FinBERT?

Most sentiment models are trained on general text like tweets or product reviews. Financial language is different — phrases like *"downward pressure"* or *"beats estimates"* require domain-specific understanding.

FinBERT is BERT fine-tuned specifically on financial news and earnings call transcripts, making it significantly more accurate for market-related text than general-purpose models.

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Sentiment Model | FinBERT (ProsusAI/finbert) via HuggingFace |
| Frontend | Streamlit |
| News Data | yfinance |
| Charts | Plotly |
| Language | Python 3 |

---

## Run Locally

```bash
# Clone the repo
git clone https://github.com/mdrizvi-106/financial-news-sentiment.git
cd financial-news-sentiment

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## Screenshots

> Add a screenshot of your app here after deployment

---

## How It Works

```
User enters ticker (e.g. TSLA)
        ↓
yfinance fetches latest 10 headlines
        ↓
Each headline passed to FinBERT
        ↓
Model returns label + confidence score
        ↓
Results displayed as chart + verdict
```

---

## Example Output

| Headline | Sentiment | Confidence |
|----------|-----------|------------|
| Tesla Faces $43.9B Free Cash Flow Challenge | Negative | 0.96 |
| Alphabet Climbs 4% on Waymo's Nashville Deal | Positive | 0.94 |
| Apple Announces Developer Conference Date | Neutral | 0.81 |

---

## Limitations

- News data depends on Yahoo Finance availability — some tickers may return no results
- FinBERT works best on clean headline text — very short or ambiguous headlines may score low confidence
- This is a demonstration tool, not financial advice

---

## Author

**Mohamed Rizvi Shaik Abdulla**
MSc Artificial Intelligence — University of Leicester
[LinkedIn](https://linkedin.com/in/sam1061) · [GitHub](https://github.com/mdrizvi-106)

---

> Built as part of a portfolio demonstrating end-to-end NLP deployment for financial intelligence applications.
