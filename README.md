<<<<<<< HEAD
# Mini Project: AI-Powered Product Feedback Analyzer

A one-day portfolio project for beginner AI Product Analysts. It ingests user reviews, runs **AI/ML** (sentiment analysis + topic modeling), and outputs **business-ready insights** with charts and an exec summary.

## What you get
- `data/reviews_sample.csv` — sample reviews (replace with real data later)
- `src/analyze_reviews.py` — end-to-end pipeline (clean → sentiment → topics → insights → charts → summary)
- `app.py` — optional Streamlit dashboard
- `outputs/` — where results are saved
- `requirements.txt` — install dependencies in a virtual env

## Quickstart (Terminal)
```bash
# 1) Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install requirements
pip install -r requirements.txt

# 3) Run the pipeline (uses data/reviews_sample.csv by default)
python src/analyze_reviews.py

# 4) (Optional) Launch the dashboard
streamlit run app.py
```

## Replace with your own data
Put a CSV at `data/reviews.csv` (or rename the sample). Required columns:
- `review_id` (int or str)
- `created_at` (YYYY-MM-DD or any parseable date)
- `rating` (1–5 optional)
- `review_text` (the feedback text)

## What the pipeline does
1. **Clean & prep** text (lowercase, strip punctuation, stopwords).
2. **Sentiment (AI)** using VADER (fast, no API keys).
3. **Topics (AI/ML)** with TF-IDF + **NMF** to discover themes.
4. **Business mapping**: auto-labels topics (e.g., Login, Performance, Pricing, Support, UI/UX, Features).
5. **KPIs & Insights** per theme (volume, % negative, example quotes).
6. **Charts** (Top Negative Themes) + **Word Cloud** of negatives.
7. **Executive summary** in plain business English.

## Outputs
- `outputs/clean_reviews.csv`
- `outputs/theme_summary.csv`
- `outputs/top_issues.png`
- `outputs/wordcloud_negative.png`
- `outputs/executive_summary.md`

## Notes
- This project uses **no paid APIs** and runs on a typical laptop.
- To impress recruiters, replace the sample CSV with **real reviews** from your target product (export App Store/Play Store or support tickets, where allowed).
=======
# ai-feedback-analyzer
AI-powered tool to extract insights from user feedback using NLP + ChatGPT
>>>>>>> aa0b4abdf4ceb938868a8b8fded8632cbf337449
