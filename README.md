# 🧠 AI Feedback Analyzer

This is an AI-powered project that analyzes user feedback from product reviews to extract key insights using NLP and OpenAI's ChatGPT API. It performs sentiment classification, topic modeling, and auto-generates an executive summary to guide product and engineering teams.

---

## 🚀 Features

- 🔍 Text Cleaning & Preprocessing
- 🎯 Sentiment Analysis (using VADER)
- 🧵 Topic Modeling (using NMF + TF-IDF)
- 📊 Visualizations (bar chart + word cloud)
- 🤖 Executive Summary using **ChatGPT API** (GPT-3.5)
- 📁 Modular code ready for scaling


## 🧪 Sample Output

- ✅ Cleaned dataset with sentiment labels
- 📈 Top negative themes visualized
- ☁️ Word cloud for major complaint patterns
- 🤖 AI-generated business summary using ChatGPT API

---

## 📦 How to Run

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your OpenAI key
echo "OPENAI_API_KEY=your-api-key-here" > .env

# 4. Run the analyzer
python3 -m src.analyze_reviews
