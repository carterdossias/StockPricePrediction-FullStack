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
from flask import Flask, request, render_template, redirect, url_for, flash, Response, stream_with_context, session
from tensorflow.keras.models import load_model
from flask_basicauth import BasicAuth
import threading
import queue
import time
import bcrypt

from credentials import ipCred, usernameCred, passwordCred, databaseCred
from ticker_import import create_new_ticker_table, import_news_data

# ----------------- DB Configuration -----------------
db_config = {
    'host': ipCred,
    'user': usernameCred,
    'password': passwordCred,
    'database': databaseCred
}

app = Flask(__name__)
app.secret_key = "replace_with_a_secret_key"  # Replace with a secure key

# Configure Basic Auth for admin routes
app.config['BASIC_AUTH_USERNAME'] = 'admin'
app.config['BASIC_AUTH_PASSWORD'] = 'admin'  # Change to a secure password
app.config['BASIC_AUTH_FORCE'] = False
basic_auth = BasicAuth(app)

# ----------------- Global Log Queue and SSE Endpoint -----------------
log_queue = queue.Queue()  # Thread-safe queue for log messages

def sse_stream():
    """Generator function that yields log messages from log_queue."""
    while True:
        line = log_queue.get()
        if line is None:
            break
        yield f"data: {line}\n\n"

@app.route('/admin/import_logs')
@basic_auth.required
def import_logs():
    """SSE endpoint that streams log messages from log_queue."""
    return Response(stream_with_context(sse_stream()), mimetype='text/event-stream')

# ----------------- User Authentication Routes -----------------
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username').strip()
        email = request.form.get('email').strip()
        password = request.form.get('password').encode('utf-8')  # encode password as bytes

        # Hash the password with bcrypt
        hashed_password = bcrypt.hashpw(password, bcrypt.gensalt())

        try:
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor()
            insert_query = """
                INSERT INTO users (username, email, password_hash, account_balance, available_funds, portfolio_value, risk_profile, account_status)
                VALUES (%s, %s, %s, 0.00, 0.00, 0.00, 'moderate', 1);
            """
            cursor.execute(insert_query, (username, email, hashed_password.decode('utf-8')))
            conn.commit()
            flash("Sign-up successful. Please sign in.", "success")
            return redirect(url_for('signin'))
        except mysql.connector.Error as err:
            flash(f"Error: {err}", "danger")
        finally:
            cursor.close()
            conn.close()
    return render_template('signup.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        username_or_email = request.form.get('username_or_email').strip()
        password = request.form.get('password').encode('utf-8')
        try:
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor(dictionary=True)
            query = """
                SELECT user_id, username, email, password_hash
                FROM users
                WHERE username = %s OR email = %s
                LIMIT 1;
            """
            cursor.execute(query, (username_or_email, username_or_email))
            user = cursor.fetchone()
            if user and bcrypt.checkpw(password, user['password_hash'].encode('utf-8')):
                session['user_id'] = user['user_id']
                session['username'] = user['username']
                
                # Update the last_login field
                update_query = "UPDATE users SET last_login = NOW() WHERE user_id = %s"
                cursor.execute(update_query, (user['user_id'],))
                conn.commit()
                
                flash("Signed in successfully.", "success")
                return redirect(url_for('index'))
            else:
                flash("Invalid credentials.", "danger")
        except mysql.connector.Error as err:
            flash(f"Database error: {err}", "danger")
        finally:
            cursor.close()
            conn.close()
    return render_template('signin.html')

@app.route('/signout')
def signout():
    session.clear()
    flash("You have been signed out.", "info")
    return redirect(url_for('index'))

# ----------------- Existing Functions -----------------
def load_model_and_objects(ticker):
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

# ----------------- Admin Routes -----------------
@app.route('/admin')
@basic_auth.required
def admin():
    return render_template('admin.html')

@app.route('/admin/import_ticker', methods=['GET', 'POST'])
@basic_auth.required
def import_ticker():
    message = None
    if request.method == 'POST':
        ticker = request.form.get('ticker', '').upper().strip()
        if ticker:
            def import_job(ticker_symbol):
                try:
                    from ticker_import import create_new_ticker_table, import_news_data
                    log_queue.put(f"Starting import for {ticker_symbol}...")
                    create_new_ticker_table(ticker_symbol, log_queue=log_queue)
                    import_news_data(ticker_symbol, log_queue=log_queue)
                    log_queue.put(f"Finished import for {ticker_symbol}.")
                except Exception as e:
                    log_queue.put(f"Error importing data for {ticker_symbol}: {e}")
                finally:
                    log_queue.put(None)
            threading.Thread(target=import_job, args=(ticker,)).start()
            return render_template('admin_import_running.html', ticker=ticker)
        else:
            message = "Please enter a ticker symbol."
    return render_template('admin_import.html', message=message)

@app.route('/admin/manual_update', methods=['GET'])
@basic_auth.required
def manual_update():
    def daily_update_job():
        try:
            log_queue.put("Starting manual incremental update for all active tickers...")
            from daily_update import update_tickers_incrementally
            update_tickers_incrementally()
            log_queue.put("Finished manual incremental update.")
        except Exception as e:
            log_queue.put(f"Error during manual incremental update: {e}")
        finally:
            log_queue.put(None)
    threading.Thread(target=daily_update_job).start()
    return render_template('admin_import_running.html', ticker="All Active Tickers")

# ----------------- Other Routes -----------------
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