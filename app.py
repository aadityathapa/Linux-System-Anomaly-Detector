"""
app.py - Flask dashboard for the Linux Anomaly Detector.

Endpoints:
  GET  /              → main dashboard HTML
  GET  /api/metrics   → last N metric rows as JSON
  GET  /api/live      → current system snapshot + anomaly score
  GET  /api/anomalies → recent anomalies from DB
"""

import sqlite3
import pickle
import os
import numpy as np
from flask import Flask, jsonify, render_template
import psutil

DB_PATH     = os.path.join(os.path.dirname(__file__), "data", "metrics.db")
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "models", "isolation_forest.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "models", "scaler.pkl")

FEATURES = [
    "cpu_percent", "cpu_freq_mhz", "load_avg_1m", "load_avg_5m",
    "mem_percent", "mem_used_mb", "swap_percent",
    "disk_read_mb", "disk_write_mb",
    "net_sent_mb", "net_recv_mb",
    "num_processes", "num_threads",
]

app = Flask(__name__)

# Load model once at startup
model = scaler = None
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    print("[app] Model loaded successfully.")
else:
    print("[app] WARNING: No trained model found. Run train.py first.")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def score_row(row_dict):
    """Return anomaly score and label for a dict of metric values."""
    if model is None:
        return None, "no_model"
    X = np.array([[row_dict.get(f, 0) or 0 for f in FEATURES]])
    X_scaled = scaler.transform(X)
    score = float(model.decision_function(X_scaled)[0])
    label = "anomaly" if model.predict(X_scaled)[0] == -1 else "normal"
    # Normalise score to 0–100 (lower = more anomalous)
    # decision_function typically ranges roughly -0.5 to 0.5
    normalised = max(0, min(100, int((score + 0.5) * 100)))
    return normalised, label


@app.route("/")
def index():
    return render_template("index.html", model_loaded=model is not None)


@app.route("/api/metrics")
def api_metrics():
    """Return last 200 rows for charting."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM metrics ORDER BY timestamp DESC LIMIT 200"
    ).fetchall()
    conn.close()
    data = [dict(r) for r in reversed(rows)]

    # Attach anomaly scores
    for row in data:
        row["score"], row["label"] = score_row(row)

    return jsonify(data)


@app.route("/api/live")
def api_live():
    """Return a live system snapshot with anomaly score (no DB write)."""
    import os as _os
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    freq = psutil.cpu_freq()
    load1, load5, _ = _os.getloadavg()
    disk = psutil.disk_io_counters()
    net = psutil.net_io_counters()
    procs = list(psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]))

    snap = {
        "cpu_percent": cpu,
        "cpu_freq_mhz": freq.current if freq else 0,
        "load_avg_1m": round(load1, 2),
        "load_avg_5m": round(load5, 2),
        "mem_percent": mem.percent,
        "mem_used_mb": round(mem.used / 1e6, 1),
        "swap_percent": swap.percent,
        "disk_read_mb": 0,
        "disk_write_mb": 0,
        "net_sent_mb": 0,
        "net_recv_mb": 0,
        "num_processes": len(procs),
        "num_threads": sum(p.info.get("num_threads", 0) or 0
                           for p in psutil.process_iter(["num_threads"])),
    }
    snap["score"], snap["label"] = score_row(snap)
    return jsonify(snap)


@app.route("/api/anomalies")
def api_anomalies():
    """Return the most recent rows that were flagged as anomalies."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM metrics ORDER BY timestamp DESC LIMIT 500"
    ).fetchall()
    conn.close()

    anomalies = []
    for row in rows:
        d = dict(row)
        d["score"], d["label"] = score_row(d)
        if d["label"] == "anomaly":
            anomalies.append(d)
        if len(anomalies) >= 50:
            break

    return jsonify(anomalies)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
