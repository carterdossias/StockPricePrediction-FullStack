# prediction_engine.py

import numpy as np
import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
from credentials import ipCred, usernameCred, passwordCred, databaseCred

# Database config
db_config = {
    'host': ipCred,
    'user': usernameCred,
    'password': passwordCred,
    'database': databaseCred
}

def predict_next_5_days(ticker: str):
    """
    Fetches the last 3 months of intraday 'view' data for `ticker`,
    trains an LSTM on everything up to the max date in those 3 months,
    then does a 5-day walk-forward forecast.
    Returns (plot_png_base64, forecasted_prices_list).
    """
    # 1) Load only last 3 months of data from <ticker>_view
    conn = mysql.connector.connect(**db_config)
    df = pd.read_sql(f"""
        SELECT date, open, high, low, close, volume, total_sentiment
        FROM {ticker}_view
        WHERE date >= DATE_SUB(CURDATE(), INTERVAL 3 MONTH)
        ORDER BY date ASC;
    """, conn, parse_dates=['date'], index_col='date')
    conn.close()

    # 2) Feature engineering
    df = df.sort_index()
    df['SMA_5']      = df['close'].rolling(5,  min_periods=1).mean()
    df['SMA_10']     = df['close'].rolling(10, min_periods=1).mean()
    df['pct_change'] = df['close'].pct_change().fillna(0)

    # 3) Prepare training data (no test split here, we forecast beyond last date)
    features = ['open','high','low','close','volume','total_sentiment','SMA_5','SMA_10','pct_change']
    lookback = 30

    # 4) Scale
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[features])

    # 5) Build sequences
    def create_sequences(data, seq_len):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len, features.index('close')])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(data_scaled, lookback)

    # 6) Build & train model
    model = Sequential([
        Bidirectional(LSTM(100, return_sequences=True), 
                      input_shape=(lookback, len(features))),
        Dropout(0.2),
        Bidirectional(LSTM(100)),
        Dropout(0.2),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(1e-3, clipnorm=1.0), loss='mae')
    es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    rl = ReduceLROnPlateau(monitor='loss', patience=5, factor=0.5, min_lr=1e-5)
    model.fit(X_train, y_train, epochs=100, batch_size=8, callbacks=[es, rl], verbose=0)

    # 7) 5-day walk-forward forecast
    current_seq = data_scaled[-lookback:].copy()
    preds_scaled = []
    for _ in range(5):
        nxt = model.predict(current_seq[np.newaxis, :, :])[0,0]
        preds_scaled.append(nxt)
        new_row = np.zeros(len(features))
        new_row[features.index('close')] = nxt
        current_seq = np.vstack([current_seq[1:], new_row])

    # 8) Denormalize
    forecasted_prices = []
    for val in preds_scaled:
        dummy = np.zeros((1, len(features)))
        dummy[0, features.index('close')] = val
        inv = scaler.inverse_transform(dummy)[0, features.index('close')]
        forecasted_prices.append(inv)

    # 9) Build dates for the 5 forecasts
    last_date = df.index[-1]
    forecast_dates = [last_date + timedelta(days=i+1) for i in range(5)]

    # 10) Plot: last 60 days + 5-day forecast
    hist_window = 60
    fig, ax = plt.subplots(figsize=(10, 5))
    hist = df['close'].iloc[-hist_window:]
    ax.plot(hist.index, hist.values, label='Historical Close', color='blue')
    ax.axvline(last_date, color='gray', linestyle='--', label='Forecast Start')
    ax.plot(forecast_dates, forecasted_prices, 'o-', color='red', label='5-Day Forecast')
    ax.set_title(f"{ticker} 5-Day Forecast from {last_date.date()}")
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plot_png = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return plot_png, forecasted_prices