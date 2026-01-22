import pandas as pd
import re
import numpy as np
import joblib
import nltk
import time
import random
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import feedparser
import yfinance as yf
from yfinance.exceptions import YFRateLimitError


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def clean_news_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"&\w+;", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
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


def get_stock_fundamentals(ticker, retries=7):
    stock = yf.Ticker(ticker)

    for attempt in range(retries):
        try:
            info = stock.info

            if info is None or len(info) == 0:
                raise Exception("Empty fundamentals returned from Yahoo")

            fundamentals = {
                "Market Cap": info.get("marketCap"),
                "PE Ratio": info.get("trailingPE"),
                "Book Value": info.get("bookValue"),
                "Dividend Yield": info.get("dividendYield"),
                "Current Price": info.get("currentPrice"),
                "52 Week High": info.get("fiftyTwoWeekHigh"),
                "52 Week Low": info.get("fiftyTwoWeekLow"),
                "EBITDA": info.get("ebitda"),
                "Price/Sales": info.get("priceToSalesTrailing12Months"),
                "Price/Book": info.get("priceToBook")
            }

        
            if fundamentals["Dividend Yield"] is not None and fundamentals["Dividend Yield"] < 1:
                fundamentals["Dividend Yield"] *= 100

        
            missing = [k for k, v in fundamentals.items() if v is None]
            if missing:
                print("⚠️ Missing values from Yahoo Finance:", missing)

            return fundamentals

        except YFRateLimitError:
            wait = (attempt + 1) * 10 + random.randint(1, 3)
            print(f"⚠️ Rate limited. Waiting {wait}s... (Retry {attempt+1}/{retries})")
            time.sleep(wait)

        except Exception as e:
            wait = (attempt + 1) * 6
            print(f"⚠️ Yahoo returned incomplete data: {e}")
            print(f"Retrying after {wait}s... (Retry {attempt+1}/{retries})")
            time.sleep(wait)

    print("❌ Failed after retries.")
    return None



def safe_num(x):
    try:
        if x is None:
            return 0
        if isinstance(x, float) and np.isnan(x):
            return 0
        return float(x)
    except:
        return 0


nltk.download("punkt")
from nltk.tokenize import word_tokenize

print("Welcome to Stock Fundamentals and Sentiment Analysis")
print("This model uses NLP techniques and ML models to analyze stocks")
print("Automatic news input + automatic fundamentals\n")

print("Loading Fundamentals model...")
fund = joblib.load("../fundamentals_stock_model.joblib")
print("✅ Fundamentals model loaded")

print("Loading Sentiment Analyzer...")
analyzer = SentimentIntensityAnalyzer()
print("✅ Sentiment Analyzer loaded")

print("Loading Fundamentals scorer model...")
score = joblib.load("../stock_score_regression.pkl")
print("✅ Score model loaded\n")

while True:
    n = input("Enter y to continue, q to quit: ").strip().lower()

    if n == "y":
        name = input("Enter Stock Name: ").strip()
        tick = input("Enter stock ticker: ").strip().upper()

        ff = get_stock_fundamentals(tick)

        if ff is None:
            print("Could not fetch fundamentals. Try again later.\n")
            continue

        print("\nFundamentals are:\n")
        for k in ff:
            print(f"{k} - {ff[k]}")

    
        news_articles = scrape_google_news_rss(name)
        print("\nLatest news is:\n")

        cleaned_articles = [clean_news_text(a) for a in news_articles]

        for a in cleaned_articles:
            print(a)
            print("\n")

        if not news_articles:
            s = 0
        else:
            all_news = " ".join(cleaned_articles)
            s = analyzer.polarity_scores(all_news)["compound"]

        input_features = [[
            safe_num(ff["Market Cap"]),
            safe_num(ff["Current Price"]),
            safe_num(ff["52 Week High"]),
            safe_num(ff["52 Week Low"]),
            safe_num(ff["Book Value"]),
            safe_num(ff["PE Ratio"]),
            safe_num(ff["Dividend Yield"]),
            safe_num(ff["EBITDA"]),
            safe_num(ff["Price/Sales"]),
            safe_num(ff["Price/Book"])
        ]]

        f = fund.predict(input_features)[0]
        sc = score.predict(input_features)[0]

        print("\nStock Score is:")
        print(sc)

        
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

        print("\n" + "-" * 60 + "\n")

    else:
        print("Bye bye!")
        break
