"""
train.py - Train an Isolation Forest model on collected system metrics.

Isolation Forest is a perfect first ML algorithm:
- Unsupervised: no labels needed — it learns what "normal" looks like
- Fast and lightweight
- Explainable: anomaly score tells you HOW anomalous something is

Run this after collecting at least ~30 minutes of data.
Re-run anytime you want to retrain on newer data.
"""

import sqlite3
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

DB_PATH  = os.path.join(os.path.dirname(__file__), "data", "metrics.db")
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "models", "isolation_forest.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "models", "scaler.pkl")

# Features used for training (all numeric columns)
FEATURES = [
    "cpu_percent", "cpu_freq_mhz", "load_avg_1m", "load_avg_5m",
    "mem_percent", "mem_used_mb", "swap_percent",
    "disk_read_mb", "disk_write_mb",
    "net_sent_mb", "net_recv_mb",
    "num_processes", "num_threads",
]


def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM metrics ORDER BY timestamp", conn)
    conn.close()
    return df


def train():
    print("[train] Loading data from database...")
    df = load_data()

    if len(df) < 20:
        print(f"[train] Only {len(df)} rows found. Collect more data first (aim for 100+).")
        return

    print(f"[train] Loaded {len(df)} rows spanning "
          f"{df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")

    X = df[FEATURES].fillna(0).values

    # Scale features so no single metric dominates
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Isolation Forest
    # contamination: expected fraction of anomalies in training data (~1-5% is reasonable)
    model = IsolationForest(
        n_estimators=200,
        contamination=0.03,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # Compute anomaly scores for training data (for threshold reference)
    scores = model.decision_function(X_scaled)  # higher = more normal
    predictions = model.predict(X_scaled)       # -1 = anomaly, 1 = normal

    n_anomalies = (predictions == -1).sum()
    print(f"[train] Flagged {n_anomalies} anomalies in training data "
          f"({100 * n_anomalies / len(df):.1f}%)")
    print(f"[train] Score range: {scores.min():.3f} → {scores.max():.3f}")

    # Save model and scaler
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"[train] Model saved to {MODEL_PATH}")
    print(f"[train] Scaler saved to {SCALER_PATH}")
    print("[train] Done! You can now run app.py to start the dashboard.")


if __name__ == "__main__":
    train()
