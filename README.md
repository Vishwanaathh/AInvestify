# Ainvestify – AI Powered Stock Analysis Platform

Ainvestify is an AI-powered stock analysis platform that combines **machine learning, sentiment analysis, and financial data visualization** to help users analyze stocks quickly and effectively.

The platform fetches **stock market data, financial fundamentals, news sentiment, and ML-based predictions** to provide a consolidated analysis of a stock.

---

# Features

* 📈 **Stock Price Visualization**

  * Interactive stock price charts using Chart.js.

* 🧠 **Machine Learning Stock Scoring**

  * Uses multiple ML models to analyze financial indicators and produce a stock score.

* 📰 **News Sentiment Analysis**

  * Collects recent stock-related news and evaluates sentiment using NLP models.

* 📊 **Fundamental Analysis**

  * Displays key financial metrics for the selected stock.

* 🤖 **AI Chat Assistant**

  * Users can ask questions related to stocks and markets.

* ⚡ **Real-time Data Fetching**

  * Uses APIs and web scraping to collect financial data and news.

---

# Project Architecture

The project is divided into **Frontend, Backend, and AI Modules**.

```
PROJECT_ROOT
│
├── backend
│   └── main.py
│
├── frontend
│   └── basic.html
│
└── AI_PART
    │
    ├── ML Models
    │
    ├── datasets
    │
    ├── Model_gens
    │
    └── working_scripts
```

---

# Tech Stack

### Frontend

* HTML
* CSS
* JavaScript
* Chart.js

### Backend

* Python
* Flask

### Machine Learning / AI

* TensorFlow / Keras
* Scikit-learn
* Pandas
* NumPy
* TF-IDF Vectorization
* Logistic Regression
* Random Forest

### Data Sources

* Yahoo Finance (yfinance)
* News RSS feeds
* Stock sentiment datasets

---

# Machine Learning Models Used

The system uses multiple models to generate stock insights.

| Model                   | Purpose                       |
| ----------------------- | ----------------------------- |
| Logistic Regression     | News sentiment analysis       |
| Random Forest Regressor | Fundamental stock scoring     |
| Neural Network (Keras)  | Advanced fundamental analysis |
| Stock Score Regression  | Combined stock scoring        |

---

# Datasets Used

The following datasets are used for training and evaluation:

* **financials_cleaned.csv**
  Financial indicators of companies

* **stock_tweets.csv**
  Tweets used for sentiment analysis

* **stock_yfinance_data.csv**
  Historical stock price data

---

# System Workflow

1. User enters **Stock Name and Ticker** in the UI.
2. Frontend sends request to the **Flask Backend API**.
3. Backend performs:

   * Stock data fetching from Yahoo Finance
   * News scraping
   * Sentiment analysis
   * Fundamental analysis using ML models
4. Backend returns:

   * Price data
   * ML predictions
   * Fundamentals
   * News sentiment
5. Frontend visualizes results using charts and metrics.

---

# API Endpoints

| Endpoint                         | Description                     |
| -------------------------------- | ------------------------------- |
| `/chart/<ticker>`                | Returns historical stock prices |
| `/fundamentals/<ticker>`         | Returns fundamental metrics     |
| `/request/<stock_name>/<ticker>` | Runs ML analysis                |
| `/news/<stock_name>`             | Fetches recent news             |
| `/chatbot/<message>`             | AI assistant response           |

---

# Installation

### 1 Clone the Repository

```bash
git clone https://github.com/yourusername/ainvestify.git
cd ainvestify
```

---

### 2 Install Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies:

* flask
* pandas
* numpy
* scikit-learn
* tensorflow
* yfinance
* feedparser
* vaderSentiment

---

### 3 Run the Backend Server

```bash
cd backend
python main.py
```

The Flask server will start at:

```
http://127.0.0.1:5000
```

---

### 4 Open the Frontend

Open the file:

```
frontend/basic.html
```

in your browser.

---

# Example Usage

1. Enter stock name (e.g., **Alphabet Inc**)
2. Enter ticker (e.g., **GOOG**)
3. Click **Analyze**
4. The system will display:

* Price chart
* ML score
* Fundamentals
* News sentiment
* Overall interpretation

---

# Future Improvements

* Add **technical indicators** (RSI, MACD, Moving Averages)
* Improve **news sentiment model using BERT**
* Add **portfolio recommendation system**
* Deploy as a **full web application**

---

# License

This project is for **educational and research purposes**.

---

# Author

Developed by **Sriram Kancherla** and **Viswanath Parashuram Yadavalli**

---
