import os
import re
import math
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk

from src.gpt_summarizer import summarize_reviews  #  Added ChatGPT import

# Ensure NLTK stopwords are available
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

from nltk.corpus import stopwords

BASE = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

print("\n💡 STEP 2: EXPLORING THE DATA...\n")

# Load the dataset
df = pd.read_csv(os.path.join(DATA_DIR, "reviews_sample.csv"))

# Show column names
print("📊 Columns in dataset:", df.columns.tolist())

# Show first few rows
print("\n🔍 First 5 rows:")
print(df.head())

# Show total count
print(f"\n📦 Total reviews: {len(df)}")

# Date range
if "created_at" in df.columns:
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    print("\n📅 Review date range:")
    print(df["created_at"].min(), "→", df["created_at"].max())

# Rating distribution
if "rating" in df.columns:
    print("\n⭐ Rating distribution:")
    print(df["rating"].value_counts().sort_index())

    # Quick sentiment health
    positive = (df["rating"] >= 4).sum()
    negative = (df["rating"] <= 2).sum()
    print(f"\n👍 Positive reviews (≥4★): {positive}")
    print(f"👎 Negative reviews (≤2★): {negative}")

# Review text length
if "review_text" in df.columns:
    avg_len = df["review_text"].astype(str).str.len().mean()
    print(f"\n📝 Avg review length: {avg_len:.1f} characters")


def load_data():
    path_primary = os.path.join(DATA_DIR, "reviews.csv")
    path_sample = os.path.join(DATA_DIR, "reviews_sample.csv")
    path = path_primary if os.path.exists(path_primary) else path_sample
    df = pd.read_csv(path)
    if "review_text" not in df.columns:
        raise ValueError("CSV must contain a 'review_text' column.")
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    else:
        df["created_at"] = pd.Timestamp("today")
    if "rating" not in df.columns:
        df["rating"] = np.nan
    if "review_id" not in df.columns:
        df["review_id"] = np.arange(1, len(df) + 1)
    return df

def basic_clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_texts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text_clean"] = df["review_text"].astype(str).apply(basic_clean)
    return df

def sentiment_scores(texts):
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(t)["compound"] for t in texts]
    return np.array(scores)

def label_sentiment(compound, pos=0.2, neg=-0.2):
    if compound >= pos:
        return "positive"
    elif compound <= neg:
        return "negative"
    else:
        return "neutral"

def topic_model(texts, n_topics=6, max_features=5000):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1
    )
    X = vectorizer.fit_transform(texts)
    nmf = NMF(n_components=n_topics, random_state=42, init="nndsvda", max_iter=400)
    W = nmf.fit_transform(X)
    H = nmf.components_
    terms = np.array(vectorizer.get_feature_names_out())

    topic_terms = []
    for k in range(n_topics):
        top_idx = H[k].argsort()[::-1][:10]
        topic_terms.append(terms[top_idx].tolist())

    label_hints = {
        "Login/Auth": ["login", "password", "2fa", "code", "otp"],
        "Performance": ["slow", "lag", "crash", "battery"],
        "Pricing/Billing": ["price", "refund", "charged", "billing"],
        "Support": ["support", "help", "agent"],
        "UI/UX": ["ui", "design", "look", "search", "filter"],
        "Features": ["feature", "notification", "onboarding"]
    }

    topic_labels = []
    for k, terms_k in enumerate(topic_terms):
        label = "Other"
        joined = " ".join(terms_k)
        best_score = 0
        for cand, kws in label_hints.items():
            score = sum(kw in joined for kw in kws)
            if score > best_score:
                best_score = score
                label = cand
        topic_labels.append(label)

    doc_topics = W.argmax(axis=1)
    return W, H, topic_terms, topic_labels, doc_topics

def summarize_business(df, theme_summary):
    lines = []
    total = len(df)
    neg = (df["sentiment_label"]=="negative").mean()
    pos = (df["sentiment_label"]=="positive").mean()
    lines.append(f"Total reviews analyzed: {total}. Positive: {pos:.1%}, Negative: {neg:.1%}.")
    lines.append("Top problem themes (by negative share and volume):")
    top_issues = theme_summary.sort_values(["neg_share","count"], ascending=[False, False]).head(3)
    for _, r in top_issues.iterrows():
        lines.append(f"- {r['theme']}: {r['neg_share']:.1%} negative of {int(r['count'])} mentions; example: \"{r['example'][:120]}\"")
    quick_wins = theme_summary.query("neg_share >= 0.40 and count >= 2").head(3)
    if len(quick_wins):
        lines.append("Recommended quick wins:")
        for _, r in quick_wins.iterrows():
            rec = {
                "Login/Auth": "Fix OTP delivery/retry and improve error messages on failed login.",
                "Performance": "Optimize app startup and reduce battery usage on background tasks.",
                "Pricing/Billing": "Clarify pricing, prevent double-charge, and add self-serve refunds.",
                "Support": "Improve first response time and set clear SLAs.",
                "UI/UX": "Improve search relevance and add better filters.",
                "Features": "Prioritize onboarding simplification and export options.",
                "Other": "Review top example feedback to classify correctly."
            }.get(r["theme"], "Address top user complaints under this theme.")
            lines.append(f"- {r['theme']}: {rec}")
    else:
        lines.append("Overall sentiment looks balanced; maintain strengths and address minor friction points.")
    return "\n".join(lines)

def main():
    df = load_data()
    df = preprocess_texts(df)

    # Sentiment
    df["sent_compound"] = sentiment_scores(df["text_clean"].tolist())
    df["sentiment_label"] = df["sent_compound"].apply(label_sentiment)

    # Topics
    W, H, topic_terms, topic_labels, doc_topics = topic_model(df["text_clean"].tolist(), n_topics=6)
    df["topic_id"] = doc_topics
    df["theme"] = df["topic_id"].apply(lambda i: topic_labels[i])

    # Aggregates
    g = df.groupby("theme", as_index=False).agg(
        count=("review_id", "count"),
        avg_rating=("rating", "mean"),
        neg_share=("sentiment_label", lambda s: (s=="negative").mean()),
        pos_share=("sentiment_label", lambda s: (s=="positive").mean())
    )
    examples = (
        df.sort_values("sent_compound")
          .groupby("theme", as_index=False)
          .agg(example=("review_text","first"))
    )
    theme_summary = g.merge(examples, on="theme", how="left")

    df.to_csv(os.path.join(OUT_DIR, "clean_reviews.csv"), index=False)
    theme_summary.to_csv(os.path.join(OUT_DIR, "theme_summary.csv"), index=False)

    # Visualization
    theme_summary = theme_summary.sort_values(["neg_share","count"], ascending=[False, False])
    plt.figure(figsize=(8,4))
    plt.bar(theme_summary["theme"], theme_summary["neg_share"])
    plt.title("Top Negative Themes (by negative share)")
    plt.ylabel("Negative Share")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "top_issues.png"), dpi=160)
    plt.close()

    neg_text = " ".join(df.loc[df["sentiment_label"]=="negative","text_clean"].tolist())
    if neg_text.strip():
        wc = WordCloud(width=800, height=400, background_color="white").generate(neg_text)
        wc.to_file(os.path.join(OUT_DIR, "wordcloud_negative.png"))

    # Executive Summary
    summary = summarize_business(df, theme_summary)
    with open(os.path.join(OUT_DIR, "executive_summary.md"), "w") as f:
        f.write("# Executive Summary\n\n")
        f.write(summary + "\n")

    print("\n💬 GPT SUMMARY:")
    try:
        gpt_output = summarize_reviews(df["review_text"].tolist())
        print(gpt_output)
    except Exception as e:
        print("⚠️ GPT Summary Error:", e)

if __name__ == "__main__":
    main()
