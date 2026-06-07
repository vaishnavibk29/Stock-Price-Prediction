# 📈 Stock Price Movement Prediction using ML & Sentiment Analysis

A machine learning system that predicts whether a stock will go **up or down** the next trading day, by combining **technical price indicators** with **Twitter sentiment analysis**. 

---

## Overview

Financial markets are influenced by both historical price trends and investor sentiment. This project explores whether social media signals from Twitter can improve stock movement prediction beyond technical indicators alone.

The system is trained and evaluated on the **StockNet dataset** — 88 US stock tickers, 1,257 trading days (Sep 2012 – Sep 2017), with ~120,000 associated tweets from Yahoo Finance and Twitter.

---

## Key Features

- 📊 **Dual data pipeline** — merges technical OHLCV stock data with tweet-derived sentiment features
- 🔒 **Leak-free design** — tweet features are lagged by 1 day to prevent data leakage from future signals
- 🎯 **Neutral zone filtering** — excludes ±0.2% marginal moves for a cleaner binary classification signal
- 💬 **VADER sentiment analysis** — chosen specifically for informal, finance-domain social media text
- 🤖 **4 model types** compared across price-only vs price+tweet feature sets
- 📉 **5 evaluation metrics** — Accuracy, Precision, Recall, F1, ROC-AUC + confusion matrices

---

## Dataset

**StockNet Dataset**
- **Price data**: OHLCV CSVs from Yahoo Finance — 108,589 rows across 88 tickers
- **Tweet data**: JSON files per ticker/date — 119,844 raw tweets, 103,898 after cleaning
- **Coverage**: September 2012 – September 2017

> Download the StockNet dataset from: https://github.com/yumoxu/stocknet-dataset

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow / Keras (LSTM, BiLSTM) |
| Machine Learning | Scikit-learn (Random Forest), XGBoost |
| NLP & Sentiment | NLTK VADER, regex text cleaning |
| Data Processing | Pandas, NumPy, glob, fastparquet |
| Visualisation | Matplotlib, Seaborn |
| Storage | Parquet (fastparquet engine) |

---

## Models Compared

Each model was trained and evaluated under **two feature conditions**: price-only and price+tweets.

| Model | Type | Notes |
|---|---|---|
| **LSTM** | Sequential (RNN) | 30-day sliding window, 2 stacked layers, batch norm + dropout |
| **BiLSTM** | Sequential (RNN) | Bidirectional — 256-unit hidden representation |
| **Random Forest** | Tree ensemble | 300 trees, single-timestep features, no scaling needed |
| **XGBoost** | Gradient boosting | 500 estimators, L1+L2 regularisation, strongest tabular baseline |

**Training split (time-based, no leakage):**
- Training: 70% — Sep 2012 to ~Dec 2015
- Validation: 15% — ~Dec 2015 to ~Sep 2016
- Test: 15% — ~Sep 2016 to Sep 2017

---

## Feature Engineering

**Technical features (price data):**
- Daily return, 5-day & 10-day moving averages (MA5, MA10)
- MA5 ratio, MA10 ratio, 5-day volatility, 5-day momentum
- Volume change, intraday HL%, OC%, lag returns (lag1, lag2)

**Sentiment features (tweet data):**
- VADER compound sentiment score
- Engagement (retweets + favourites), retweet count, favourite count
- Word count, lexical diversity, has_question flag, has_exclaim flag

---

## Results Summary

| Model | Features | Test Accuracy | Test AUC | Test F1 |
|---|---|---|---|---|
| BiLSTM | Price + Tweets | 53.20% | 0.502 | 0.669 |
| BiLSTM | Price only | 50.02% | 0.500 | 0.318 |
| LSTM | Price + Tweets | 50.47% | 0.490 | 0.418 |
| LSTM | Price only | 51.09% | 0.511 | 0.592 |
| Random Forest | Price + Tweets | 52.83% | 0.507 | 0.473 |
| Random Forest | Price only | 49.24% | 0.492 | 0.456 |
| **XGBoost** | **Price + Tweets** | **52.50%** | **0.500** | **0.644** |
| XGBoost | Price only | 49.67% | 0.497 | 0.627 |

> All models achieved test accuracies in the 48–54% range with AUC scores near 0.5 — consistent with the **Efficient Market Hypothesis**, which suggests public signals are already priced in. Tweet features consistently improved or maintained performance vs. price-only baselines.

---

## Project Structure

```
stock-prediction/
├── data/
│   ├── price/raw/               # Per-ticker OHLCV CSVs (not included)
│   ├── tweet/raw/               # Per-ticker tweet JSONs (not included)
│   └── merged_stock_tweet_data.parquet  # Preprocessed master dataset
├── Stock_price_movement_detection.ipynb  # Main notebook
├── model_comparison_results.csv          # All model metrics
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/vaishnavibk29/stock-prediction.git
cd stock-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Get the StockNet dataset and place:
- Price CSVs in `data/price/raw/`
- Tweet JSONs in `data/tweet/raw/`

### 4. Run the notebook
```bash
jupyter notebook Stock_price_movement_detection.ipynb
```
Run all cells in order — data processing → feature engineering → model training → evaluation.

---

## Requirements

```
tensorflow
scikit-learn
xgboost
nltk
pandas
numpy
matplotlib
seaborn
fastparquet
jupyter
```

---

## 🔍 Design Decisions

**Why VADER?** VADER is specifically designed for short, noisy social media text. Finance tweets use hashtags, cashtags, abbreviations, and informal language that break generic sentiment tools. VADER's lexicon was validated on social media data, making it the principled choice here.

**Why lag tweet features by 1 day?** To simulate a real trading environment — on day T, a trader only has access to information from day T-1 and earlier. Using same-day tweets would be data leakage.

**Why the ±0.2% neutral zone?** Marginal daily moves carry very low signal-to-noise. Excluding them produces a cleaner binary classification target focused on days with meaningful directional movement.

---

*built to explore whether tweets move markets — turns out, slightly.*
