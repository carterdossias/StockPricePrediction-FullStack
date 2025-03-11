## DAILY UPDATE SCRIPT
# This script will incrementally update stock and news data for all active tickers in the database.

import mysql.connector
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import requests
import time

from credentials import (
    ipCred,
    usernameCred,
    passwordCred,
    databaseCred,
    API_KEYcred
)

def safe_float(value):
    """Convert a value to float if not NaN, otherwise return None."""
    if pd.isnull(value):
        return None
    return float(value)

def update_tickers_incrementally():
    """
    1) Fetch all active tickers from the 'tickers' table.
    2) For each ticker, determine the date range (from last_update + 1 day to today).
    3) Fetch and upsert new stock data into <TICKER>_data.
    4) Fetch and upsert new news data into <TICKER>_news.
    5) Update tickers.last_update to today's date upon success.
    """
    # Connect to the database
    conn = mysql.connector.connect(
        host=ipCred,
        user=usernameCred,
        password=passwordCred,
        database=databaseCred
    )
    cursor = conn.cursor()

    # 1) Get all active tickers
    cursor.execute("SELECT symbol, COALESCE(last_update, date_added) AS last_update FROM tickers WHERE active=1;")
    rows = cursor.fetchall()
    if not rows:
        print("No active tickers found.")
        cursor.close()
        conn.close()
        return

    for (symbol, last_update) in rows:
        symbol = symbol.upper().strip()
        if not last_update:
            # If last_update is NULL, use the date_added or fallback to an older date
            last_update = datetime.date(2000, 1, 1)
        else:
            # last_update is a date object
            pass

        today = datetime.date.today()
        start_date = last_update + datetime.timedelta(days=1)
        if start_date > today:
            print(f"No new data needed for {symbol}. (Already up to date)")
            continue

        # Convert to string for yfinance and news calls
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = today.strftime('%Y-%m-%d')

        print(f"Updating {symbol} from {start_str} to {end_str}...")

        # 2) Update Stock Data incrementally
        try:
            update_stock_data(symbol, start_date, today)
        except Exception as e:
            print(f"Error updating stock data for {symbol}: {e}")
            continue

        # 3) Update News Data incrementally
        try:
            update_news_data(symbol, start_date, today)
        except Exception as e:
            print(f"Error updating news data for {symbol}: {e}")
            continue

        # 4) If we reach here, both stock & news updates succeeded. Update last_update in tickers table.
        try:
            cursor.execute(
                "UPDATE tickers SET last_update = %s WHERE symbol = %s",
                (today, symbol)
            )
            conn.commit()
            print(f"Successfully updated last_update for {symbol} to {today}")
        except Exception as e:
            print(f"Error updating last_update for {symbol}: {e}")

    cursor.close()
    conn.close()
    print("Incremental update process completed.")


def update_stock_data(symbol, start_date, end_date):
    """
    Fetch incremental stock data for <symbol> from <start_date> to <end_date>
    and upsert it into <SYMBOL>_data table.
    """
    conn = mysql.connector.connect(
        host=ipCred,
        user=usernameCred,
        password=passwordCred,
        database=databaseCred
    )
    cursor = conn.cursor()

    # 1) Fetch partial data from yfinance
    yf_symbol = yf.Ticker(symbol)
    hist = yf_symbol.history(
        start=start_date.strftime('%Y-%m-%d'),
        end=(end_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')  # inclusive end
    )

    # 2) Localize or convert to UTC
    if not hist.empty:
        if hist.index.tzinfo is None:
            hist.index = hist.index.tz_localize('UTC', nonexistent='shift_forward', ambiguous='NaT')
        else:
            hist.index = hist.index.tz_convert('UTC')

    # 3) Create table if not exists
    table_name = f"{symbol}_data"
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

    # 4) Upsert
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

    rows_inserted = 0
    for idx, row_data in hist.iterrows():
        date_val = idx.date()
        data_tuple = (
            date_val,
            safe_float(row_data.get('Open')),
            safe_float(row_data.get('High')),
            safe_float(row_data.get('Low')),
            safe_float(row_data.get('Close')),
            int(row_data['Volume']) if not pd.isnull(row_data['Volume']) else None
        )
        cursor.execute(insert_query, data_tuple)
        rows_inserted += 1

    conn.commit()
    cursor.close()
    conn.close()

    print(f"[Stock] Inserted/updated {rows_inserted} rows for {symbol} from {start_date} to {end_date}.")


def update_news_data(symbol, start_date, end_date):
    """
    Fetch incremental news data for <symbol> from <start_date> to <end_date>
    and upsert it into <SYMBOL>_news table.
    """
    db_config = {
        'host': ipCred,
        'user': usernameCred,
        'password': passwordCred,
        'database': databaseCred
    }

    API_KEY = API_KEYcred
    BASE_URL = "https://finnhub.io/api/v1/company-news"

    all_news = []
    current_start = start_date
    MAX_RETRIES = 5

    while current_start <= end_date:
        current_end = current_start + datetime.timedelta(days=6)
        if current_end > end_date:
            current_end = end_date

        params = {
            "symbol": symbol,
            "from": current_start.strftime("%Y-%m-%d"),
            "to": current_end.strftime("%Y-%m-%d"),
            "token": API_KEY
        }

        print(f"[News] Fetching for {symbol} from {params['from']} to {params['to']}...")
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
                success = True  # exit on non-429 errors

        current_start = current_end + datetime.timedelta(days=1)
        time.sleep(1)

    if not all_news:
        print(f"[News] No new articles found for {symbol}.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(all_news)

    def safe_ts(ts):
        try:
            val = int(ts)
            if val <= 0:
                return None
            return datetime.datetime.fromtimestamp(val).date()
        except:
            return None

    if not df.empty and 'datetime' in df.columns:
        df['datetime'] = df['datetime'].apply(safe_ts)

    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    table_name = f"{symbol}_news"
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
        data_tuple = (
            int(row_data['id']),
            row_data['datetime'],
            row_data.get('headline', ''),
            row_data.get('related', ''),
            row_data.get('source', ''),
            row_data.get('summary', '')
        )
        cursor.execute(insert_query, data_tuple)
        rows_inserted += 1

    conn.commit()
    cursor.close()
    conn.close()

    print(f"[News] Inserted/updated {rows_inserted} articles for {symbol} from {start_date} to {end_date}.")


if __name__ == "__main__":
    # Run the incremental update for all active tickers
    update_tickers_incrementally()