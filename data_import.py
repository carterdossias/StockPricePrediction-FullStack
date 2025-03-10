import requests
import datetime
import time
import pandas as pd
import mysql.connector
import numpy as np

from credentials import (
    API_KEYcred,
    ipCred,
    usernameCred,
    passwordCred,
    databaseCred
)

# =================== Stock Data Import Function ===================

def safe_float(value):
    """Convert a value to float if not NaN, otherwise return None."""
    if pd.isnull(value):
        return None
    return float(value)

def create_new_ticker_table(ticker_symbol, server=ipCred, username=usernameCred, password=passwordCred, database=databaseCred):
    """
    Fetch historical stock data for a ticker via yfinance,
    create a table (if it doesn't exist) named <TICKER>_data,
    and insert the data using upsert functionality.
    """
    ticker_symbol = ticker_symbol.upper().strip()
    
    # Establish connection to MySQL
    conn = mysql.connector.connect(
        host=server,
        user=username,
        password=password,
        database=database
    )
    cursor = conn.cursor()
    
    # Fetch historical stock data using yfinance
    import yfinance as yf
    stock = yf.Ticker(ticker_symbol)
    hist = stock.history(period="max")
    
    # Localize the index to UTC
    if hist.index.tzinfo is None:
        hist.index = hist.index.tz_localize('UTC', nonexistent='shift_forward', ambiguous='NaT')
    else:
        hist.index = hist.index.tz_convert('UTC')
    
    # Define table name based on ticker symbol
    table_name = f"{ticker_symbol}_data"
    
    # Create table if it doesn't exist
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        date DATE PRIMARY KEY,
        open FLOAT,
        high FLOAT,
        low FLOAT,
        close FLOAT,
        volume BIGINT
    );
    """
    cursor.execute(create_table_query)
    conn.commit()
    
    # Insert data with upsert functionality
    insert_query = f"""
    INSERT INTO {table_name} (date, open, high, low, close, volume)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        open = VALUES(open),
        high = VALUES(high),
        low = VALUES(low),
        close = VALUES(close),
        volume = VALUES(volume)
    """
    
    for index, row in hist.iterrows():
        data_tuple = (
            index.date(),
            safe_float(row.get('Open')),
            safe_float(row.get('High')),
            safe_float(row.get('Low')),
            safe_float(row.get('Close')),
            int(row.get('Volume')) if not pd.isnull(row.get('Volume')) else None
        )
        cursor.execute(insert_query, data_tuple)
    
    conn.commit()
    cursor.close()
    conn.close()
    print(f"Stock data imported successfully for {ticker_symbol}.")

# =================== News Data Import Function ===================

def import_news_data(ticker, start_date=datetime.date(2024, 1, 1), end_date=datetime.date.today()):
    """
    Fetch news data from the FINNUB API for the given ticker within the given date range,
    then insert the news articles into a table named <TICKER>_news in the database.
    """
    ticker = ticker.upper().strip()
    API_KEY = API_KEYcred
    BASE_URL = "https://finnhub.io/api/v1/company-news"
    
    # Database Configuration
    db_config = {
        'host': ipCred,
        'user': usernameCred,
        'password': passwordCred,
        'database': databaseCred
    }
    
    all_news = []
    current_start = start_date
    MAX_RETRIES = 5

    while current_start <= end_date:
        current_end = current_start + datetime.timedelta(days=6)
        if current_end > end_date:
            current_end = end_date

        params = {
            "symbol": ticker,
            "from": current_start.strftime("%Y-%m-%d"),
            "to": current_end.strftime("%Y-%m-%d"),
            "token": API_KEY
        }
        
        print(f"Fetching news for {ticker} from {params['from']} to {params['to']}")
        retries = 0
        success = False
        while not success and retries < MAX_RETRIES:
            response = requests.get(BASE_URL, params=params)
            if response.status_code == 200:
                news_items = response.json()
                if news_items:
                    all_news.extend(news_items)
                success = True
            elif response.status_code == 429:
                retries += 1
                wait_time = 2 ** retries
                print(f"Rate limit reached. Retrying in {wait_time} seconds (attempt {retries}/{MAX_RETRIES})")
                time.sleep(wait_time)
            else:
                print(f"Error: {response.status_code} for range {params['from']} to {params['to']}")
                success = True  # exit loop on non-429 errors
        
        current_start = current_end + datetime.timedelta(days=1)
        time.sleep(1)

    if not all_news:
        print(f"No news data found for {ticker}.")
        return

    # Convert collected news to a DataFrame
    df = pd.DataFrame(all_news)
    
    # Helper to safely convert Unix timestamps
    def safe_convert(ts):
        try:
            ts_val = int(ts)
            if ts_val <= 0:
                return None
            return datetime.datetime.fromtimestamp(ts_val).date()
        except Exception:
            return None

    if not df.empty and 'datetime' in df.columns:
        df['datetime'] = df['datetime'].apply(safe_convert)
    
    # Connect to the database and create news table if it doesn't exist
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    
    table_name = f"{ticker}_news"
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        news_id BIGINT PRIMARY KEY,
        date_time DATE,
        headline TEXT,
        related VARCHAR(10),
        source_ VARCHAR(255),
        summary TEXT,
        sentiment DOUBLE
    );
    """
    cursor.execute(create_table_query)
    conn.commit()
    
    # Insert news data (ignoring sentiment for now)
    insert_query = f"""
    INSERT INTO {table_name} (news_id, date_time, headline, related, source_, summary)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE 
        date_time = VALUES(date_time),
        headline = VALUES(headline),
        related = VALUES(related),
        source_ = VALUES(source_),
        summary = VALUES(summary)
    """
    
    rows_inserted = 0
    for _, row in df.iterrows():
        if pd.isna(row.get('id')) or pd.isna(row.get('datetime')):
            continue
        data = (
            int(row['id']),
            row['datetime'],
            row.get('headline', ''),
            row.get('related', ''),
            row.get('source', ''),
            row.get('summary', '')
        )
        cursor.execute(insert_query, data)
        rows_inserted += 1
    
    conn.commit()
    cursor.close()
    conn.close()
    print(f"Inserted {rows_inserted} news articles into {ticker}_news table.")

# =================== Main Execution ===================

if __name__ == "__main__":
    # Example: Import stock data and news data for a given ticker.
    ticker_symbol = input("Enter ticker symbol: ").strip().upper()
    print(f"Importing stock data for {ticker_symbol}...")
    create_new_ticker_table(ticker_symbol)
    
    print(f"Importing news data for {ticker_symbol}...")
    # Optionally, you can adjust the start date for news import.
    import_news_start = datetime.date(2024, 1, 1)
    import_news_end = datetime.date.today()
    import_news_data(ticker_symbol, start_date=import_news_start, end_date=import_news_end)