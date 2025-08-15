# src/gpt_summarizer.py

import os
import openai
from dotenv import load_dotenv

# Load the .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def summarize_reviews(reviews: list[str]) -> str:
    prompt = f"""
    Summarize the key pain points, customer sentiment, and feature suggestions from the following product reviews:

    {reviews}

    Return a helpful summary for product and engineering teams.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful AI product analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.6,
        max_tokens=300
    )

    return response.choices[0].message.content.strip()
