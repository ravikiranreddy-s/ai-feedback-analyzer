import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import nltk, re
from nltk.corpus import stopwords
from src.gpt_summarizer import summarize_reviews


try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    import nltk as _n
    _n.download("stopwords")

BASE = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, "data")

st.title("AI-Powered Product Feedback Analyzer")
st.write("Upload a CSV of reviews or use the sample to discover themes and sentiment.")

uploaded = st.file_uploader("Upload CSV (columns: review_text, created_at, rating, review_id)", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    sample_path = os.path.join(DATA_DIR, "reviews_sample.csv")
    df = pd.read_csv(sample_path)

if "review_text" not in df.columns:
    st.error("CSV must have a 'review_text' column.")
    st.stop()

def basic_clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\\S+|www\\S+", " ", text)
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text

# Clean
df["text_clean"] = df["review_text"].astype(str).apply(basic_clean)

# Sentiment
analyzer = SentimentIntensityAnalyzer()
df["sent_compound"] = df["text_clean"].apply(lambda t: analyzer.polarity_scores(t)["compound"])
def lab(c):
    return "positive" if c >= 0.2 else ("negative" if c <= -0.2 else "neutral")
df["sentiment_label"] = df["sent_compound"].apply(lab)

# Topics
n_topics = st.slider("Number of themes", 4, 10, 6)
stop_words = set(stopwords.words("english"))
vec = TfidfVectorizer(max_features=5000, stop_words=stop_words, ngram_range=(1,2))
X = vec.fit_transform(df["text_clean"])
nmf = NMF(n_components=n_topics, random_state=42, init="nndsvda", max_iter=400)
W = nmf.fit_transform(X)
H = nmf.components_
terms = np.array(vec.get_feature_names_out())

topic_terms = []
for k in range(n_topics):
    top_idx = H[k].argsort()[::-1][:10]
    topic_terms.append(terms[top_idx].tolist())

# Simple labels
label_hints = {
    "Login/Auth": ["login","password","two factor","2fa","code","auth","otp"],
    "Performance": ["slow","lag","crash","battery","overheat","freeze","bug"],
    "Pricing/Billing": ["price","pricing","billed","refund","charged","billing"],
    "Support": ["support","help","respond","reply","customer","agent"],
    "UI/UX": ["ui","design","look","clean","navigation","filter","search","dark mode"],
    "Features": ["feature","export","notification","onboarding","tutorial"]
}
labels = []
for terms_k in topic_terms:
    label = "Other"
    best = 0
    joined = " ".join(terms_k)
    for cand, kws in label_hints.items():
        score = sum(kw in joined for kw in kws)
        if score > best:
            best = score
            label = cand
    labels.append(label)

df["topic_id"] = W.argmax(axis=1)
df["theme"] = df["topic_id"].apply(lambda i: labels[i] if i < len(labels) else "Other")

# KPIs
kpi = df.groupby("theme", as_index=False).agg(
    count=("review_text","count"),
    neg_share=("sentiment_label", lambda s: (s=="negative").mean()),
    pos_share=("sentiment_label", lambda s: (s=="positive").mean())
).sort_values(["neg_share","count"], ascending=[False, False])

st.subheader("Top Negative Themes")
fig = plt.figure(figsize=(8,4))
plt.bar(kpi["theme"], kpi["neg_share"])
plt.title("Top Negative Themes (by negative share)")
plt.ylabel("Negative Share")
plt.xticks(rotation=30, ha="right")
st.pyplot(fig)

st.subheader("Sample Insights")
st.dataframe(kpi)

st.subheader("Example Quotes (most negative per theme)")
ex = df.sort_values("sent_compound").groupby("theme").head(1)[["theme","review_text"]]
st.table(ex)

st.caption("Tip: Replace the sample CSV with your product's real reviews to make this portfolio-ready.")

if st.button("💡 Generate AI Summary with GPT"):
    with st.spinner("Generating AI insights..."):
        summary = summarize_reviews(df['review_text'].tolist())
        st.subheader("🧠 ChatGPT Summary of Customer Feedback")
        st.success(summary)
