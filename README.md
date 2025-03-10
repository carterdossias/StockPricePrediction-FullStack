# Stock Price Prediction App

## Main Contributors
- @CarterDossias (Connections / Data Import / Data Prep) 
- @NoahMalewicki (Database Design / Mangament) 
- @RishiBokka (Creating ML Model For Predictions)

This repository contains a full-stack application for predicting stock prices based on:
1. **Historical stock data** stored in a MySQL database.
2. **News sentiment analysis** performed on daily articles related to each ticker.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Database Setup](#database-setup)
- [Sentiment Analysis](#sentiment-analysis)
- [ML Model](#ml-model)
- [Web App](#web-app)
- [Data Sources](#data-sources)
- [Future Plans](#future-plans)
- [Contributing](#contributing)
- [License](#license)

---

## Overview
The goal of this project is to leverage various data sources—such as historical stock data and news articles—to build a machine learning model capable of predicting a stock’s closing price for a specified day. The application consists of:

1. **A MySQL database** to store stock data and sentiment scores.
2. **Python scripts** for data ingestion, processing, and sentiment analysis.
3. **A Web App** (Flask) that allows users to query and visualize data, run predictions, and manage data import via an admin portal.

---

## Features
- **Automatic Data Import**: Python scripts fetch stock data (using Yahoo Finance) and import it into MySQL.
- **News Sentiment Analysis**: Daily news summaries are scored with a sentiment value \([-1, +1]\), which is stored alongside stock data.
- **Database Management**: A homelab MySQL server (managed by @NoahMalewicki) holds all data in structured tables (e.g., `<TICKER>_data`).
- **Admin Portal**: A password-protected interface that lets authorized users import new tickers and manage data.
- **Web Interface**: Users can view historical data, run predictions (once the ML model is in place), and see sentiment analysis.

---

## Project Structure
Here’s a simplified view of the repository layout:
---

## Database Setup
1. **MySQL Server**: A MySQL database is hosted on a homelab server (hosted by @NoahMalewicki).  
2. **Stock Data Tables**: Each ticker has its own table named `<TICKER>_data` with columns:
   - `date` (DATE, Primary Key)
   - `open`, `high`, `low`, `close` (FLOAT)
   - `volume` (BIGINT)
   - (Optional) `sentiment` (FLOAT) for daily sentiment scores.
3. **Data Import Scripts**: 
   - `ticker_import.py` fetches historical data from Yahoo Finance (`yfinance`) and imports it into `<TICKER>_data`.  
   - `app.py` includes an admin route `/admin/import_ticker` to trigger this script via the web interface.

---

## Sentiment Analysis
We use Python scripts to:
- Pull news articles from the Finnhub API (or other providers).
- Summarize or extract the relevant text.
- Apply a sentiment model (e.g., a pretrained Transformer or logistic regression) to generate a daily sentiment score for each article.  
- Aggregate sentiment scores into a single daily value (averaging or summation) and store it in the database.

---

## ML Model
**Not Currently Implemented**.  
Headed by @RishiBokka
Eventually, we plan to:
1. Train an LSTM or Transformer-based model that uses both **price history** and **sentiment** features.
2. Store the model artifacts (e.g., `.h5`, `.pkl`) in the `models/` directory.
3. Provide a route or a UI form for users to request a prediction for a specific date.

---

## Web App
The Flask web application serves as the front end:
- **Index Page** (`/`): Allows users to input a ticker and a target date to run (future) predictions.
- **Stock View** (`/stockview`): Plots historical closing prices (and eventually sentiment).
- **About Page** (`/about`): Provides details about the project.
- **Admin Portal** (`/admin`): Requires authentication (using Flask-BasicAuth or similar). 
  - **Import Ticker** (`/admin/import_ticker`): Admins can create a new table for a given ticker and import data automatically.

---

## Data Sources
1. **Yahoo Finance**: 
   - Used to retrieve daily stock data via the `yfinance` library.  
   - `hist = yf.Ticker(ticker_symbol).history(period="max")` is a common approach.
2. **Finnhub API**: 
   - Provides news articles for a given ticker.  
   - We fetch daily summaries and run sentiment analysis, storing the scores in MySQL.

---

## Future Plans
- **Implement the ML Model**:  
  Build and train an LSTM/Transformer on price + sentiment features, store the model in `models/`, and enable predictions in the web UI.
- **Enhanced Admin Features**:  
  - Scheduled tasks (e.g., daily data import).  
  - Detailed logs of data import and model training.
- **User Accounts**:  
  Replace BasicAuth with a more robust authentication/authorization system (Flask-Login or similar).
- **More Visualization**:  
  Graphs for sentiment over time, volume, or correlation between sentiment and price changes.

---

## License
*Unspecified currently*

---
