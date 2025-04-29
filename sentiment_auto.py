import mysql.connector
from transformers import pipeline
import numpy as np
from credentials import ipCred, usernameCred, passwordCred, databaseCred

# Use your fine-tuned model (change paths as needed)
classifier = pipeline(
    task="text-classification", 
    model="./finbert-finetuned", 
    tokenizer="./finbert-finetuned", 
    device=-1
)

ticker = 'AAPL'
db_config = {
    'host': ipCred,
    'user': usernameCred,
    'password': passwordCred,
    'database': databaseCred
}

conn = mysql.connector.connect(**db_config)

# --- Cursor A: Fetch rows that need sentiment scores ---
cursor_fetch = conn.cursor()
fetch_query = f"""
    SELECT news_id, summary
    FROM {ticker}_news
    WHERE sentiment IS NULL
    LIMIT 15000;
"""
cursor_fetch.execute(fetch_query)
rows = cursor_fetch.fetchall()
cursor_fetch.close()  # Close fetch cursor

if not rows:
    print("No rows to update.")
    conn.close()
    exit()

# Filter out rows with empty summaries and unzip IDs and summaries
id_summary_pairs = [(news_id, summary) for news_id, summary in rows if summary]
if not id_summary_pairs:
    print("No valid summaries found.")
    conn.close()
    exit()

news_ids, summaries = zip(*id_summary_pairs)

# Process summaries in batches
batch_size = 32
results = []
for i in range(0, len(summaries), batch_size):
    batch = list(summaries[i:i+batch_size])
    batch_results = classifier(batch, truncation=True)
    results.extend(batch_results)

# --- Cursor B: Update rows with both numeric and label sentiment ---
cursor_update = conn.cursor()
update_query = f"""
    UPDATE {ticker}_news
    SET sentiment = %s, sentiment_label = %s
    WHERE news_id = %s
"""

# Process each result: store numeric score and its corresponding label.
for news_id, result in zip(news_ids, results):
    r = result[0] if isinstance(result, list) else result
    label = r['label'].upper()  # e.g., "POSITIVE", "NEGATIVE", "NEUTRAL"
    score = r['score']
    
    # Compute bipolar sentiment for the numeric column:
    if label == "POSITIVE":
        sentiment_score = score
    elif label == "NEGATIVE":
        sentiment_score = -score
    else:
        sentiment_score = 0.0

    # Update both columns: numeric sentiment and the label string.
    cursor_update.execute(update_query, (sentiment_score, label, news_id))

conn.commit()
cursor_update.close()
conn.close()

print("Sentiment scores and labels updated successfully.")