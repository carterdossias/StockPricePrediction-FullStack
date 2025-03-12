### KEEP ME UP TO DATE WITH MASTER DATA PULL 
## THIS SCRIPT IS WHAT THE ADMIN PAGE USES TO IMPORT TICKER DATA INTO THE DATABASE
## THIS SCRIPT IS NOT USED IN THE MAIN APP
## KEEP UP TO DATE WITH MASTER DATA PULL!!!!!!!

from credentials import ipCred, usernameCred, passwordCred, databaseCred, API_KEYcred

import yfinance as yf
import mysql.connector
import pandas as pd
import numpy as np
import datetime
import requests
import time

# ----------------- Helper Function -----------------

def safe_float(value):
    """Convert a value to float if not NaN, otherwise return None."""
    if pd.isnull(value):
        return None
    return float(value)

# ----------------- Stock Data Import -----------------

def create_new_ticker_table(
    ticker_symbol,
    server=ipCred,
    username=usernameCred,
    password=passwordCred,
    database=databaseCred,
    log_queue=None
):
    """
    Fetch historical stock data for a ticker via yfinance,
    create a table (if it doesn't exist) named <TICKER>_data,
    and insert the data using upsert functionality.
    Also ensures the ticker is recorded in the `tickers` table with active=1.
    """
    ticker_symbol = ticker_symbol.upper().strip()
    if log_queue:
        log_queue.put(f"Importing stock data for {ticker_symbol}...")

    # Connect to MySQL
    conn = mysql.connector.connect(
        host=server,
        user=username,
        password=password,
        database=database
    )
    cursor = conn.cursor()

    # Fetch historical stock data using yfinance
    stock = yf.Ticker(ticker_symbol)
    hist = stock.history(period="max")

    # Localize the index to UTC to avoid DST/timezone issues
    if hist.index.tzinfo is None:
        hist.index = hist.index.tz_localize('UTC', nonexistent='shift_forward', ambiguous='NaT')
    else:
        hist.index = hist.index.tz_convert('UTC')

    # Create table name
    table_name = f"{ticker_symbol}_data"

    # Create table if not exists
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

    # Insert data with upsert
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

    # Now upsert into the `tickers` table
    # If the ticker doesn't exist, insert it with date_added = CURDATE(), last_update = CURDATE().
    # If it exists, set active=1 and update last_update=CURDATE().
    ticker_upsert_query = """
        INSERT INTO tickers (symbol, date_added, last_update)
        VALUES (%s, CURDATE(), CURDATE())
        ON DUPLICATE KEY UPDATE
            active = 1,
            last_update = CURDATE();
    """
    cursor.execute(ticker_upsert_query, (ticker_symbol,))
    conn.commit()

    cursor.close()
    conn.close()

    if log_queue:
        log_queue.put(f"Stock data imported successfully for {ticker_symbol}.")
    else:
        print(f"Stock data imported successfully for {ticker_symbol}.")


# ----------------- News Data Import -----------------

def import_news_data(
    ticker,
    start_date=datetime.date(2024, 1, 1),
    end_date=datetime.date.today(),
    log_queue=None
):
    """
    Fetch news data from the FINNUB API for the given ticker within the given date range,
    then insert the news articles into a table named <TICKER>_news in the database.
    """
    ticker = ticker.upper().strip()
    if log_queue:
        log_queue.put(f"Importing news data for {ticker} from {start_date} to {end_date}...")

    API_KEY = API_KEYcred
    BASE_URL = "https://finnhub.io/api/v1/company-news"

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

        msg = f"Fetching news for {ticker} from {params['from']} to {params['to']}"
        if log_queue:
            log_queue.put(msg)
        else:
            print(msg)

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
                msg = f"Rate limit reached. Retrying in {wait_time} seconds (attempt {retries}/{MAX_RETRIES})"
                if log_queue:
                    log_queue.put(msg)
                else:
                    print(msg)
                time.sleep(wait_time)
            else:
                err_msg = f"Error: {response.status_code} for range {params['from']} to {params['to']}"
                if log_queue:
                    log_queue.put(err_msg)
                else:
                    print(err_msg)
                success = True  # exit loop on non-429 errors

        current_start = current_end + datetime.timedelta(days=1)
        time.sleep(1)

    if not all_news:
        no_data_msg = f"No news data found for {ticker}."
        if log_queue:
            log_queue.put(no_data_msg)
        else:
            print(no_data_msg)
        return

    # Convert collected news to a DataFrame
    df = pd.DataFrame(all_news)

    def safe_ts(ts):
        try:
            ts_val = int(ts)
            if ts_val <= 0:
                return None
            return datetime.datetime.fromtimestamp(ts_val).date()
        except Exception:
            return None

    if not df.empty and 'datetime' in df.columns:
        df['datetime'] = df['datetime'].apply(safe_ts)

    # Connect to the database and create news table if not exists
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
        sentiment DOUBLE,
        sentiment_label VARCHAR(10) DEFAULT NULL
    );
    """
    cursor.execute(create_table_query)
    conn.commit()

    # Insert news data with upsert
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
    for _, row_data in df.iterrows():
        if pd.isna(row_data.get('id')) or pd.isna(row_data.get('datetime')):
            continue
        data = (
            int(row_data['id']),
            row_data['datetime'],
            row_data.get('headline', ''),
            row_data.get('related', ''),
            row_data.get('source', ''),
            row_data.get('summary', '')
        )
        cursor.execute(insert_query, data)
        rows_inserted += 1

    conn.commit()
    cursor.close()
    conn.close()

    msg_done = f"Inserted {rows_inserted} news articles into {ticker}_news table."
    if log_queue:
        log_queue.put(msg_done)
    else:
        print(msg_done)


# ----------------- CLI Testing -----------------

if __name__ == "__main__":
    # Example: Import stock data and news data for a given ticker
    ticker_symbol = input("Enter ticker symbol: ").strip().upper()
    print(f"Importing stock data for {ticker_symbol}...")
    create_new_ticker_table(ticker_symbol)

    print(f"Importing news data for {ticker_symbol}...")
    import_news_data(ticker_symbol)