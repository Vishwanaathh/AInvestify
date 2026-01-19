import pandas as pd
import re
import numpy as np
import joblib
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import feedparser
import yfinance as yf
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
def get_stock_fundamentals(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    
    fundamentals = {
        "Market Cap": info.get('marketCap'),
        "PE Ratio": info.get('trailingPE'),
        "Book Value": info.get('bookValue'),
        "Dividend Yield": info.get('dividendYield'),
        "ROE": info.get('returnOnEquity'),
        "ROCE": info.get('returnOnCapital'),
        "Revenue Growth": info.get('revenueGrowth'),
        "Current Price": info.get('currentPrice'),
        "52 Week High": info.get('fiftyTwoWeekHigh'),
        "52 Week Low": info.get('fiftyTwoWeekLow')
    }
    
    return fundamentals

nltk.download('punkt')
from nltk.tokenize import word_tokenize

print("Welcome to Stock Fundamentals and sentiment analysis")
print("This model uses NLP techniques and ML Random Forest to analyze stocks")
print("This script is designed for automatic news input but manual fundamentals")
print("Loading Fundamentals model...")
fund = joblib.load('./fundamentals_stock_model.joblib')
print("Finished loading")
print("Loading Sentiment Analyzer")
analyzer = SentimentIntensityAnalyzer()
print("Finished loading")

while True:
    n = input("Enter y to continue q to quit: ")
    if n.lower() == "y":
        name=input("Enter Stock Name")
        tick=input("Enter stock ticker")
        ff=get_stock_fundamentals(tick)
        print("Fundamentals are: ")
        
        for i in ff:
            print(i,end="")
            print("-",end="")
            print(ff[i])
            
            

        news_articles = scrape_google_news_rss(name)
        print("Latest news is")
        cleaned_articles = [clean_news_text(a) for a in news_articles]
        for i in cleaned_articles:
            print(i)
            print("/n")
        if not news_articles:
            s = 0
        else:
            
            all_news = " ".join(cleaned_articles)
            s = analyzer.polarity_scores(all_news)["compound"]

        f = fund.predict([[ff["Market Cap"],
    ff["Current Price"],
    ff["52 Week High"],
    ff["52 Week Low"],
    ff["Book Value"],
    ff["PE Ratio"],
    ff["Dividend Yield"],
    ff["ROCE"],
    ff["ROE"],
    ff["Revenue Growth"]]])[0]

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

