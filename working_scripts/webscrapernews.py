import feedparser
import re

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

if __name__ == "__main__":
    stock = input("Enter stock name: ")
    news_data = scrape_google_news_rss(stock)

    print("\nCleaned News Text:\n")
    for i, news in enumerate(news_data, 1):
        print(f"{i}. {news}")
