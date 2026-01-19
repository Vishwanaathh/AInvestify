import pandas as pd
import re
import numpy as np
import joblib
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import feedparser

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def scrape_google_news_rss(stock_name, max_articles=10):
    query = stock_name.replace(" ", "+") + "+stock"
    rss_url = f"https://news.google.com/rss/search?q={query}"
    feed = feedparser.parse(rss_url)
    cleaned_news = []
    for entry in feed.entries[:max_articles]:
        combined_text = f"{entry.title} {entry.summary}"
        cleaned_news.append(clean_text(combined_text))
    return cleaned_news

nltk.download('punkt')
from nltk.tokenize import word_tokenize

print("Welcome to Stock Fundamentals and sentiment analysis")
print("This model uses NLP techniques and ML Random Forest to analyze stocks")
print("This script is designed for automatic news input but manual fundamentals")
print("Loading Fundamentals model...")
fund = joblib.load('../fundamentals_stock_model.joblib')
print("Finished loading")
print("Loading Sentiment Analyzer")
analyzer = SentimentIntensityAnalyzer()
print("Finished loading")

while True:
    n = input("Enter y to continue q to quit: ")
    if n.lower() == "y":
        name = input("Enter stock name: ")
        mkcap = int(input("Enter market cap: "))
        cval = float(input("Enter current value of stock: "))
        high = float(input("Enter 52 week high: "))
        low = float(input("Enter 52 week low: "))
        bval = float(input("Enter book value: "))
        pe = float(input("Enter P/E Ratio: "))
        div = float(input("Enter dividend percentage: "))
        roce = float(input("Enter ROCE: "))
        roe = float(input("Enter ROE: "))
        sal = float(input("Enter sales growth in last 3 yrs: "))

        news_articles = scrape_google_news_rss(name)
        if not news_articles:
            s = 0
        else:
            all_news = " ".join(news_articles)
            s = analyzer.polarity_scores(all_news)["compound"]

        f = fund.predict([[mkcap, cval, high, low, bval, pe, div, roce, roe, sal]])[0]

        if f == 0 and s < -0.2:
            print("Bad fundamentals and bearish sentiment.")
        elif f == 0 and -0.2 <= s <= 0.2:
            print("Bad fundamentals and neutral sentiment.")
        elif f == 0 and s > 0.2:
            print("Bad fundamentals but bullish sentiment.")
        elif f == 1 and s < -0.2:
            print("Good fundamentals but bearish sentiment.")
        elif f == 1 and -0.2 <= s <= 0.2:
            print("Good fundamentals and neutral sentiment.")
        else:
            print("Good fundamentals and bullish sentiment.")
    else:
        print("Bye bye!")
        break

