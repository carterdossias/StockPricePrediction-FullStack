import os
import mysql.connector
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for matplotlib
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from datetime import datetime, timedelta
import yfinance as yf
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for, flash
from tensorflow.keras.models import load_model
from flask_basicauth import BasicAuth

# ========== Configure Your Database ==========
db_config = {
    'host': '192.168.0.17',
    'user': 'admin',
    'password': 'spotify',
    'database': 'Stocks_DB'
}

app = Flask(__name__)
app.secret_key = "replace_with_a_secret_key"

# Configure Basic Auth for admin routes
app.config['BASIC_AUTH_USERNAME'] = 'admin'
app.config['BASIC_AUTH_PASSWORD'] = 'admin'  # change to a secure password
app.config['BASIC_AUTH_FORCE'] = False  # only force auth on specific routes
basic_auth = BasicAuth(app)

# Your existing functions here...
def load_model_and_objects(ticker):
    # ... (existing code)
    model_path = f"models/{ticker}_lstm_model.h5"
    scaler_path = f"models/{ticker}_scaler.pkl"
    look_back_path = f"models/{ticker}_look_back.pkl"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No file or directory found at {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"No file or directory found at {scaler_path}")
    if not os.path.exists(look_back_path):
        raise FileNotFoundError(f"No file or directory found at {look_back_path}")

    model = load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(look_back_path, 'rb') as f:
        look_back = pickle.load(f)
    
    return model, scaler, look_back

def fetch_historical_data(ticker):
    query = f"""
        SELECT Date_ AS date, Close_ AS close
        FROM {ticker}_stock_data
        ORDER BY Date_ ASC;
    """
    conn = mysql.connector.connect(**db_config)
    df = pd.read_sql(query, conn)
    conn.close()
    
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

def iterative_forecast(model, scaler, data, look_back, steps_ahead):
    forecast_sequence = data[-look_back:, 0].tolist()
    for _ in range(steps_ahead):
        X = np.array(forecast_sequence[-look_back:]).reshape(1, look_back, 1)
        scaled_pred = model.predict(X)
        forecast_sequence.append(scaled_pred[0, 0])
    predicted_scaled = forecast_sequence[-1]
    predicted_price = scaler.inverse_transform([[predicted_scaled]])[0, 0]
    return float(predicted_price)

def get_actual_price_yfinance(ticker, target_date):
    target_str = target_date.strftime('%Y-%m-%d')
    next_day_str = (target_date + timedelta(days=1)).strftime('%Y-%m-%d')
    ticker_obj = yf.Ticker(ticker)
    data = ticker_obj.history(start=target_str, end=next_day_str)
    if not data.empty and 'Close' in data.columns:
        price = float(data['Close'].iloc[0])
        print(f"DEBUG: Actual price for {ticker} on {target_str} is {price}")
        return price
    else:
        print(f"DEBUG: No actual price data found for {ticker} on {target_str}")
        return None

def create_plot(historical_df, target_date, predicted_price=None, actual_price=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(historical_df['date'], historical_df['close'], label="Historical Close", marker='o')
    last_known_date = historical_df['date'].iloc[-1]
    ax.axvline(last_known_date, color='gray', linestyle='--', label="Last Known Data")
    if predicted_price is not None:
        ax.plot(target_date, predicted_price, 'ro', label="Predicted Close")
        ax.text(target_date, predicted_price, f' {predicted_price:.2f}', color='red')
    if actual_price is not None:
        ax.plot(target_date, actual_price, 'go', label="Actual Close")
        ax.text(target_date, actual_price, f' {actual_price:.2f}', color='green')
    ax.set_title("Stock Price Prediction")
    ax.set_xlabel("Date")
    ax.set_ylabel("Closing Price")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded

# ---------- Admin Routes ----------
@app.route('/admin')
@basic_auth.required
def admin():
    # This is the main admin portal page.
    return render_template('admin.html')

@app.route('/admin/import_ticker', methods=['GET', 'POST'])
@basic_auth.required
def import_ticker():
    message = None
    if request.method == 'POST':
        ticker = request.form.get('ticker', '').upper().strip()
        if ticker:
            try:
                # Import the function from ticker_import.py
                from ticker_import import create_new_ticker_table
                create_new_ticker_table(ticker)
                message = f"Successfully imported data for {ticker}."
            except Exception as e:
                message = f"Error importing data for {ticker}: {e}"
        else:
            message = "Please enter a ticker symbol."
    return render_template('admin_import.html', message=message)

# ---------- Other Routes (index, stockview, about) ----------
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/stockview', methods=['GET', 'POST'])
def stock_view():
    if request.method == 'POST':
        ticker = request.form.get('ticker', '').upper().strip()
        if not ticker:
            return render_template('stockview.html', error="Please enter a ticker.")
        days_input = request.form.get('days', '').strip()
        try:
            days = int(days_input)
            if days <= 0:
                raise ValueError("Days must be positive.")
        except Exception:
            days = 30
        query = f"CALL SelectFromTimeFrame('{ticker}', {days});"
        try:
            conn = mysql.connector.connect(**db_config)
            df = pd.read_sql(query, conn)
            conn.close()
        except Exception as e:
            return render_template('stockview.html', error=f"Error fetching data for {ticker}: {e}")
        if df.empty:
            return render_template('stockview.html', error=f"No data available for {ticker} in the last {days} days.")
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df['date'], df['close'], marker='o', label="Close Price")
        ax.set_title(f"{ticker} Stock Data (Last {days} Days)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plot_png = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return render_template('stockview.html', ticker=ticker, plot_png=plot_png, days=days)
    return render_template('stockview.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form.get('ticker', '').strip().upper()
        date_str = request.form.get('date', '').strip()
        if not ticker or not date_str:
            return render_template('index.html', error="Please enter both ticker and date.")
        try:
            target_date = datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            return render_template('index.html', error="Invalid date format. Use YYYY-MM-DD.")
        try:
            model, scaler, look_back = load_model_and_objects(ticker)
        except Exception as e:
            return render_template('index.html', error=f"Could not load model for {ticker}: {e}")
        historical_df = fetch_historical_data(ticker)
        if historical_df.empty:
            return render_template('index.html', error=f"No historical data found for {ticker}.")
        today = datetime.today()
        if target_date > historical_df['date'].iloc[-1]:
            last_date = historical_df['date'].iloc[-1]
            steps_ahead = (target_date.date() - last_date.date()).days
            if steps_ahead < 1:
                steps_ahead = 1
            data_for_forecast = scaler.transform(historical_df[['close']].values)
        else:
            subset_df = historical_df[historical_df['date'] < target_date]
            if subset_df.empty:
                return render_template('index.html', error=f"No historical data available before {target_date.strftime('%Y-%m-%d')}.")
            last_date = subset_df['date'].iloc[-1]
            steps_ahead = (target_date.date() - last_date.date()).days
            if steps_ahead < 1:
                steps_ahead = 1
            data_for_forecast = scaler.transform(subset_df[['close']].values)
        predicted_price = iterative_forecast(model, scaler, data_for_forecast, look_back, steps_ahead)
        actual_price = None
        actual_msg = None
        if target_date.date() < today.date():
            actual_price = get_actual_price_yfinance(ticker, target_date)
            if actual_price is None:
                actual_msg = "There is no closing price data for the specified day (possible weekend or holiday)."
        plot_png = create_plot(historical_df, target_date, predicted_price, actual_price)
        return render_template(
            'index.html',
            ticker=ticker,
            date_str=date_str,
            predicted_price=predicted_price,
            actual_price=actual_price,
            actual_msg=actual_msg,
            plot_png=plot_png
        )
    return render_template('index.html')

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=7979)