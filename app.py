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
from sklearn.preprocessing import MinMaxScaler
from credentials import ipCred, usernameCred, passwordCred, databaseCred
from ticker_import import create_new_ticker_table, import_news_data
from prediction_engine import predict_next_5_days
from flask import Flask, request, render_template, session


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
app.config['BASIC_AUTH_PASSWORD'] = 'awesomepassword'  # Change to a secure password
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
                # Update last login timestamp
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

@app.route('/admin/sentiment_analysis', methods=['GET', 'POST'])
@basic_auth.required
def admin_sentiment():
    message = None
    if request.method == 'POST':
        ticker = request.form.get('ticker', '').upper().strip()
        if ticker:
            def sentiment_job(ticker_symbol):
                try:
                    from transformers import pipeline
                    classifier = pipeline(
                        task="text-classification", 
                        model="./finbert-finetuned", 
                        tokenizer="./finbert-finetuned", 
                        device=-1
                    )
                    conn = mysql.connector.connect(**db_config)
                    cursor_fetch = conn.cursor()
                    fetch_query = f"""
                        SELECT news_id, summary
                        FROM {ticker_symbol}_news
                        WHERE sentiment IS NULL
                        LIMIT 10000;
                    """
                    cursor_fetch.execute(fetch_query)
                    rows = cursor_fetch.fetchall()
                    cursor_fetch.close()
                    if not rows:
                        log_queue.put("No rows to update.")
                        conn.close()
                        return
                    id_summary_pairs = [(news_id, summary) for news_id, summary in rows if summary and summary.strip()]
                    if not id_summary_pairs:
                        log_queue.put("No valid summaries found.")
                        conn.close()
                        return
                    news_ids, summaries = zip(*id_summary_pairs)
                    batch_size = 32
                    results = []
                    for i in range(0, len(summaries), batch_size):
                        batch = list(summaries[i:i+batch_size])
                        batch_results = classifier(batch, truncation=True)
                        results.extend(batch_results)
                    cursor_update = conn.cursor()
                    update_query = f"""
                        UPDATE {ticker_symbol}_news
                        SET sentiment = %s, sentiment_label = %s
                        WHERE news_id = %s
                    """
                    for news_id, result in zip(news_ids, results):
                        r = result[0] if isinstance(result, list) else result
                        label = r['label'].upper()  # "POSITIVE", "NEGATIVE", "NEUTRAL"
                        score = r['score']
                        if label == "POSITIVE":
                            sentiment_score = score
                        elif label == "NEGATIVE":
                            sentiment_score = -score
                        else:
                            sentiment_score = 0.0
                        cursor_update.execute(update_query, (sentiment_score, label, news_id))
                    conn.commit()
                    cursor_update.close()
                    conn.close()
                    log_queue.put("Sentiment scores and labels updated successfully.")
                except Exception as e:
                    log_queue.put(f"Error during sentiment analysis for {ticker_symbol}: {e}")
                finally:
                    log_queue.put(None)
            threading.Thread(target=sentiment_job, args=(ticker,)).start()
            return render_template('admin_sentiment_running.html', ticker=ticker)
        else:
            message = "Please enter a ticker symbol."
    return render_template('admin_sentiment.html', message=message)

@app.route('/admin/tickers', methods=['GET'])
@basic_auth.required
def admin_tickers():
    """
    Displays a table of all tickers in the database.
    """
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT symbol, active, date_added, last_update FROM tickers ORDER BY symbol;")
    tickers_data = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return render_template('admin_tickers.html', tickers=tickers_data)

@app.route('/admin/tickers/toggle', methods=['POST'])
@basic_auth.required
def toggle_ticker_active():
    """
    Toggles or sets the active state for a given ticker.
    """
    symbol = request.form.get('symbol', '').strip().upper()
    new_active = request.form.get('new_active', '1')
    if not symbol:
        flash("No ticker symbol provided.", "danger")
        return redirect(url_for('admin_tickers'))
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    update_query = """
        UPDATE tickers
        SET active = %s
        WHERE symbol = %s
    """
    cursor.execute(update_query, (new_active, symbol))
    conn.commit()
    cursor.close()
    conn.close()
    flash(f"Ticker '{symbol}' active state updated to {new_active}.", "success")
    return redirect(url_for('admin_tickers'))


@app.route('/watchlist', methods=['GET', 'POST'])
def watchlist():
    # Ensure user is logged in
    if not session.get('user_id'):
        flash("Please sign in to view your watchlist.", "danger")
        return redirect(url_for('signin'))

    user_id = session['user_id']

    # If POST, handle adding new stock to watchlist
    if request.method == 'POST':
        ticker = request.form.get('ticker', '').upper().strip()
        notes = request.form.get('notes', '').strip()
        alert_threshold = request.form.get('alert_threshold', '').strip()

        # Convert alert_threshold to float if provided
        if alert_threshold == '':
            alert_threshold = None
        else:
            try:
                alert_threshold = float(alert_threshold)
            except ValueError:
                alert_threshold = None

        try:
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor()
            insert_query = """
                INSERT INTO user_watchlist (user_id, symbol, notes, alert_threshold)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(insert_query, (user_id, ticker, notes, alert_threshold))
            conn.commit()
            flash(f"{ticker} added to your watchlist.", "success")
        except mysql.connector.Error as err:
            flash(f"Database error: {err}", "danger")
        finally:
            cursor.close()
            conn.close()

        return redirect(url_for('watchlist'))

    # For GET, fetch the user's watchlist and their recent news
    watchlist_items = []
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        # 1) Fetch the watchlist items for the current user
        select_query = """
            SELECT watchlist_id, symbol, notes, alert_threshold, date_added, stock_active
            FROM user_watchlist
            WHERE user_id = %s
            ORDER BY date_added DESC
        """
        cursor.execute(select_query, (user_id,))
        watchlist_items = cursor.fetchall()

        # 2) For each watchlist item, fetch up to 3 recent news articles
        for item in watchlist_items:
            ticker_symbol = item['symbol'].upper().strip()
            news_table = f"{ticker_symbol}_news"  # e.g., "AAPL_news"

            try:
                news_query = f"""
                    SELECT headline, sentiment_label, date_time
                    FROM {news_table}
                    ORDER BY date_time DESC
                    LIMIT 3;
                """
                cursor.execute(news_query)
                recent_news = cursor.fetchall()
                item['recent_news'] = recent_news
            except mysql.connector.Error:
                # If table doesn't exist or there's an error, just set an empty list
                item['recent_news'] = []

    except mysql.connector.Error as err:
        flash(f"Database error: {err}", "danger")
    finally:
        cursor.close()
        conn.close()

    return render_template('watchlist.html', watchlist=watchlist_items)
# ----------------- Stock View & Main Routes -----------------

@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/stockview', methods=['GET', 'POST'])
def stock_view():
    if request.method == 'POST':
        ticker = request.form.get('ticker', '').upper().strip()
        if not ticker:
            return render_template('stockview.html', error="Please enter a ticker.")
        
        # parse days
        days_input = request.form.get('days', '').strip()
        try:
            days = int(days_input)
            if days <= 0:
                raise ValueError("Days must be positive.")
        except Exception:
            days = 30

        # OPTION A: single‐statement SELECT over last `days` days
        query = f"""
            SELECT date, close
            FROM {ticker}_data
            WHERE date >= DATE_SUB(CURDATE(), INTERVAL {days} DAY)
            ORDER BY date ASC;
        """

        try:
            conn = mysql.connector.connect(**db_config)
            df = pd.read_sql(query, conn)
            conn.close()
        except Exception as e:
            return render_template('stockview.html',
                                   error=f"Error fetching data for {ticker}: {e}")

        if df.empty:
            return render_template('stockview.html',
                                   error=f"No data available for {ticker} in the last {days} days.")

        # ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])

        # build plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df['date'], df['close'], marker='o', label="Close Price")
        ax.set_title(f"{ticker} Stock Data (Last {days} Days)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        # encode as base64
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plot_png = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        return render_template('stockview.html',
                               ticker=ticker,
                               plot_png=plot_png,
                               days=days)

    return render_template('stockview.html')

@app.route('/watchlist/remove', methods=['POST'])
def remove_watchlist_item():
    # Ensure the user is logged in
    if not session.get('user_id'):
        flash("Please sign in to manage your watchlist.", "danger")
        return redirect(url_for('signin'))

    watchlist_id = request.form.get('watchlist_id', '').strip()
    if not watchlist_id:
        flash("No watchlist item specified.", "danger")
        return redirect(url_for('watchlist'))

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        # Only remove an item that belongs to the logged-in user.
        delete_query = """
            DELETE FROM user_watchlist
            WHERE watchlist_id = %s AND user_id = %s
        """
        cursor.execute(delete_query, (watchlist_id, session['user_id']))
        conn.commit()
        flash("Item removed from your watchlist.", "success")
    except mysql.connector.Error as err:
        flash(f"Error removing item: {err}", "danger")
    finally:
        cursor.close()
        conn.close()
    return redirect(url_for('watchlist'))

from datetime import datetime, timedelta
from io import BytesIO

from io import BytesIO
import base64

# app.py (only the index part shown)

from prediction_engine import predict_next_5_days

from prediction_engine import predict_next_5_days

@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    ticker = None
    plot_png = None
    forecasted_prices = None

    if request.method == "POST":
        ticker = request.form.get("ticker", "").strip().upper()
        if not ticker:
            error = "Please enter a ticker."
        else:
            try:
                plot_png, forecasted_prices = predict_next_5_days(ticker)
            except Exception as e:
                error = f"Prediction error: {e}"

    return render_template("index.html",
        error=error,
        ticker=ticker,
        plot_png=plot_png,
        forecasted_prices=forecasted_prices,
        watchlist=session.get("watchlist")
    )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=7979)