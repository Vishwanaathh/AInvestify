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
from flask import Flask, request, jsonify
from flask_jwt_extended import jwt_required, JWTManager, create_access_token, get_jwt_identity
from flask_cors import CORS
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import os
from google import genai
from dotenv import load_dotenv
from transformers import pipeline
from tabpfn_client import TabPFNRegressor

load_dotenv()


def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
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

            if info is None or len(info) == 0 or info.get("currentPrice") is None:
                raise Exception("Empty or invalid fundamentals returned from Yahoo")

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

analyzer = SentimentIntensityAnalyzer()

_news_cache = {}
_NEWS_CACHE_TTL = 300  # 5 minutes

print("Loading FinBERT sentiment model...")
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
print("FinBERT loaded.")

print("Training TabPFN fundamentals model via hosted API...")
_train_data = pd.read_csv("../AI_PART/datasets/financials_cleaned.csv", sep=",")
_train_data = _train_data.rename(columns={"52w_low": "52w_high_temp"})
_train_data = _train_data.rename(columns={"52w_high": "52w_low"})
_train_data = _train_data.rename(columns={"52w_high_temp": "52w_high"})
_numeric_cols = ["Price", "Price/Earnings", "Dividend_Yield", "52w_low", "52w_high",
                  "Market_Cap", "EBITDA", "Price/Sales", "Price/Book", "Book_Value"]
_train_data[_numeric_cols] = _train_data[_numeric_cols].apply(pd.to_numeric, errors="coerce")
_train_data = _train_data.dropna(subset=_numeric_cols)
for col in _numeric_cols:
    lower = _train_data[col].quantile(0.01)
    upper = _train_data[col].quantile(0.99)
    _train_data[col] = _train_data[col].clip(lower=lower, upper=upper)
_train_data["selling_zone"] = np.where(
    ((_train_data["52w_high"] - _train_data["Price"]) / _train_data["52w_high"]) <= 0.10, 1, 0
)
_train_data["ebitda_to_mcap"] = _train_data["EBITDA"] / _train_data["Market_Cap"]
_train_data["fundamental_score"] = (
    0.20 * (_train_data["Price/Earnings"].between(10, 25)).astype(int) +
    0.20 * (_train_data["Market_Cap"] >= 10000000000).astype(int) +
    0.20 * (_train_data["ebitda_to_mcap"].between(0.05, 0.15)).astype(int) +
    0.20 * ((_train_data["Price/Sales"] < 1) | (_train_data["Price/Sales"].between(1, 2))).astype(int) +
    0.20 * (
        (_train_data["Dividend_Yield"] > 3.5) |
        (_train_data["selling_zone"] == 1) |
        (_train_data["Price/Book"] < 3)
    ).astype(int)
)
_X_train = _train_data[["Market_Cap", "Price", "52w_high", "52w_low", "Book_Value",
                          "Price/Earnings", "Dividend_Yield", "EBITDA", "Price/Sales", "Price/Book"]]
_Y_train = _train_data["fundamental_score"]

tabpfn_model = TabPFNRegressor()
tabpfn_model.fit(_X_train, _Y_train)
print("TabPFN model trained and ready.")

app = Flask(__name__)
CORS(app)

api_keyy = os.getenv("GENAI_API_KEY")


@app.route("/")
def home():
    return "Welcome to Ainvestify"


@app.route("/chatbot/<path:query>")
def response(query):
    queryy = "Respond to this with the latest relevant news regarding the question mentioned and also the underlying fundamentals of the business " + query
    client = genai.Client(api_key=api_keyy)
    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=queryy
    )

    return jsonify({"response": response.text})


@app.route("/chart/<stockticker>")
def stock_chart(stockticker):
    try:
        stock = yf.Ticker(stockticker)
        hist = stock.history(period="max")

        if hist.empty:
            return jsonify({"error": "No historical data found"}), 404

        data = []
        for date, row in hist.iterrows():
            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "close": round(float(row["Close"]), 2)
            })

        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/fundamentals/<stockticker>")
def fundd(stockticker):
    ff = get_stock_fundamentals(stockticker)
    if ff is None:
        return jsonify({"error": "Failed to fetch stock fundamentals"}), 500
    input_features = {
        "Market Cap": safe_num(ff["Market Cap"]),
        "Current Price": safe_num(ff["Current Price"]),
        "52 Week High": safe_num(ff["52 Week High"]),
        "52 Week Low": safe_num(ff["52 Week Low"]),
        "Book Value": safe_num(ff["Book Value"]),
        "PE Ratio": safe_num(ff["PE Ratio"]),
        "Dividend Yield": safe_num(ff["Dividend Yield"]),
        "EBITDA": safe_num(ff["EBITDA"]),
        "Price/Sales": safe_num(ff["Price/Sales"]),
        "Price/Book": safe_num(ff["Price/Book"])
    }

    return jsonify(input_features)


@app.route("/news/<stockname>")
def news(stockname):
    news_articles = scrape_google_news_rss(stockname)
    cleaned_articles = [clean_text(a) for a in news_articles]
    return cleaned_articles


@app.route("/request/<stockname>/<stockticker>")
def reqq(stockname, stockticker):
    t_start = time.time()
    ff = get_stock_fundamentals(stockticker)
    t_fund_fetch = time.time()
    if ff is None:
        return jsonify({"error": "Failed to fetch stock fundamentals"}), 500

    cache_key = stockname.lower()
    now = time.time()
    if cache_key in _news_cache and (now - _news_cache[cache_key]["timestamp"]) < _NEWS_CACHE_TTL:
        cleaned_articles = _news_cache[cache_key]["articles"]
    else:
        news_articles = scrape_google_news_rss(stockname)
        cleaned_articles = [clean_text(a) for a in news_articles]
        _news_cache[cache_key] = {"articles": cleaned_articles, "timestamp": now}
    t_news_fetch = time.time()

    all_news = ""
    if not cleaned_articles:
        s = 0
    else:
        all_news = " ".join(cleaned_articles)
        s = analyzer.polarity_scores(all_news)["compound"]
    t_sentiment_score = time.time()

    input_features = pd.DataFrame([[
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
    ]], columns=["Market_Cap", "Price", "52w_high", "52w_low", "Book_Value", "Price/Earnings", "Dividend_Yield", "EBITDA", "Price/Sales", "Price/Book"])

    t0 = time.time()
    try:
        fundamental_score = float(tabpfn_model.predict(input_features)[0])
    except Exception as e:
        print(f"TabPFN prediction failed: {e}")
        fundamental_score = None
    t1 = time.time()

    try:
        if all_news.strip():
            finbert_result = finbert(all_news[:512])[0]
            finbert_label = finbert_result["label"]
            finbert_confidence = float(finbert_result["score"])
        else:
            finbert_label = "neutral"
            finbert_confidence = 0.0
    except Exception as e:
        print(f"FinBERT prediction failed: {e}")
        finbert_label = "unknown"
        finbert_confidence = None
    t2 = time.time()

    print(
        f"[TIMING] fundamentals_fetch: {t_fund_fetch - t_start:.4f}s | "
        f"news_fetch: {t_news_fetch - t_fund_fetch:.4f}s | "
        f"sentiment_score: {t_sentiment_score - t_news_fetch:.4f}s | "
        f"tabpfn_model: {t1 - t0:.4f}s | "
        f"finbert: {t2 - t1:.4f}s | "
        f"TOTAL: {t2 - t_start:.4f}s"
    )

    return jsonify({
        "fundamental_score": fundamental_score,
        "sent": float(s),
        "finbert_label": finbert_label,
        "finbert_confidence": finbert_confidence
    })

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
