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
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, jsonify
from flask_jwt_extended import jwt_required, JWTManager, create_access_token, get_jwt_identity
from flask_cors import CORS
from nltk.tokenize import word_tokenize


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

            return fundamentals

        except YFRateLimitError:
            wait = (attempt + 1) * 10 + random.randint(1, 3)
            time.sleep(wait)

        except Exception:
            wait = (attempt + 1) * 6
            time.sleep(wait)

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

fund = joblib.load("../AI_PART/fundamentals_stock_model.joblib")
analyzer = SentimentIntensityAnalyzer()
score = joblib.load("../AI_PART/stock_score_regression.pkl")
rfrscore = joblib.load("../AI_PART/rfr_stockfundamentalsscorer.pkl")

nn_model = load_model("../AI_PART/keras_stockfundamentalsscorer.h5", compile=False)
nn_X_scaler = joblib.load("../AI_PART/keras_X_scaler.pkl")
nn_y_scaler = joblib.load("../AI_PART/keras_Y_scaler.pkl")

senlogreg = joblib.load("../AI_PART/sentiment_logreg.pkl")
vectorizer = joblib.load("../AI_PART/tfidf_vectorizer.pkl")

app = Flask(__name__)


@app.route("/")
def home():
    return "Welcome to Ainvestify"


@app.route("/request/<stockname>/<stockticker>")
def reqq(stockname, stockticker):
    ff = get_stock_fundamentals(stockticker)
    if ff is None:
        return jsonify({"error": "Failed to fetch stock fundamentals"}), 500

    news_articles = scrape_google_news_rss(stockname)
    cleaned_articles = [clean_text(a) for a in news_articles]

    all_news = ""
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
    rfr = rfrscore.predict(input_features)[0]

    X_input_scaled = nn_X_scaler.transform(input_features)
    nn_scaled = nn_model.predict(X_input_scaled)
    nns = nn_y_scaler.inverse_transform(nn_scaled.reshape(-1, 1))[0][0]

    senlog = senlogreg.predict(vectorizer.transform([all_news]))

    return jsonify({
        "fundc": float(f),
        "XG": float(sc),
        "RFR": float(rfr),
        "nns": float(nns),
        "avg": float((sc + rfr + nns) / 3),
        "sent": float(s),
        "logsent": int(senlog[0])
    })


if __name__ == "__main__":
    app.run(debug=True)
