# Stock Price Prediction App

## Main Contributors
- @CarterDossias (FinBert Sentiment Analysis Fine Tuning / Connections / Data Import / Data Prep) 
- @NoahMalewicki (Database Design / Mangament) 
- @RishiBokka (Creating ML Model For Predictions)

This project is a FULL STACK application that is aimed at prediciting stock price based on the following:
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
The goal of the project is to leverage stock data sources and data sets along with financial news articles about each stock to build a maching learning model capable of predicting a stock's closing price for a specific day
The application consists of the following:

1. **A MySQL database** running in my homelab to store stock data and sentiment scores.
2. **Python scripts** for data ingestion, processing, and sentiment analysis along with machine learning.
3. **A Web App** (Flask) that allows users to query and visualize data, run predictions, and manage data import via an admin portal.

---

## Features
- **Automatic Data Import**: Python scripts can be run to fetch stock data (using Yahoo Finance) and import it into MySQL.
- **News Sentiment Analysis**: Daily news summaries are scored with my fine tuned FinnBert modelwith a sentiment value \([-1, +1]\), which is stored alongside stock data.
- **Database Management**: A homelab MySQL server running in my rack (managed by @NoahMalewicki) holds all data in structured tables (e.g., `<TICKER>_data`).
- **Admin Portal**: A password-protected interface that lets authorized users import new tickers and manage data.
- **Web Interface**: Users can view historical data, run predictions, and see sentiment analysis.
- **User Accounts**: Users can sign up / sign in to create an account where the passwords are stored in hash for security.

---

## Project Structure
Here’s a simplified view of the repository layout:
---

## Database Setup
1. **MySQL Server**: A MySQL database is hosted on a homelab linux server.  
2. **Stock Data Tables**: Each ticker has its own tables named `<TICKER>_data` `<TICKER>_news` with columns:
   - `date` (DATE, Primary Key)
   - `open`, `high`, `low`, `close` (FLOAT)
   - `volume` (BIGINT)
   - `sentiment` (FLOAT) for daily sentiment scores.
   - and many more
3. **Data Import Scripts**: 
   - `ticker_import.py` fetches historical data from Yahoo Finance (`yfinance`) and imports it into `<TICKER>_data`. It also grabs news from Finnhub with an API key defined privately and inserts news into `<TICKER>_news`
   - `app.py` includes an admin route `/admin/import_ticker` to trigger this script via the web interface.

---

## Sentiment Analysis
We use Python scripts to:
- Pull news articles from the Finnhub API.
- Summarize or extract the relevant text.
- Fine tune Finbert's model on thousands of labeled data
- Apply my fine tuned sentiment model to generate a daily sentiment score for each article.  
- Aggregate sentiment scores into a single daily value (summation) and store it in the database.

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
- And many more

---

## Data Sources
1. **Yahoo Finance**: 
   - Used to retrieve daily stock data via the `yfinance` library.  
   - `hist = yf.Ticker(ticker_symbol).history(period="max")` is our approach.
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
