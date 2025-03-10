### KEEP ME UP TO DATE WITH MASTER DATA PULL 
## THIS SCRIPT IS WHAT THE ADMIN PAGE USES TO IMPORT TICKER DATA INTO THE DATABASE
## THIS SCRIPT IS NOT USED IN THE MAIN APP
## KEEP UP TO DATE WITH MASTER DATA PULL!!!!!!!

from credentials import ipCred, usernameCred, passwordCred, databaseCred

# ticker_import.py

import yfinance as yf
import mysql.connector
import pandas as pd
import numpy as np
from datetime import datetime

def safe_float(value):
    """Convert a value to float if not NaN, otherwise return None."""
    if pd.isnull(value):
        return None
    return float(value)

def create_new_ticker_table(ticker_symbol, server=ipCred, username=usernameCred, password=passwordCred, database=databaseCred):
    ticker_symbol = ticker_symbol.upper().strip()
    # Database connection parameters
    server = server
    username = username
    password = password
    database = database
    
    # Establish connection to MySQL
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
    
    # Localize the index to UTC to bypass DST-related issues
    if hist.index.tzinfo is None:
        hist.index = hist.index.tz_localize('UTC', nonexistent='shift_forward', ambiguous='NaT')
    else:
        hist.index = hist.index.tz_convert('UTC')
    
    # Create table name based on ticker symbol (e.g., DIS_data)
    table_name = f"{ticker_symbol}_data"
    
    # SQL to create table if it doesn't exist (trailing comma removed)
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
    
    # SQL to insert data into the table with upsert functionality
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
    
    # Iterate over DataFrame rows and insert into the table
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
    print(f"Data imported successfully into the MySQL database for {ticker_symbol}.")

if __name__ == "__main__":
    # For testing purposes
    ticker_symbol = input("Enter ticker symbol: ").strip().upper()
    create_new_ticker_table(ticker_symbol)