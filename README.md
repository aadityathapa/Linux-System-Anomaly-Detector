# 🖥️ Linux Anomaly Detector

A machine learning-powered system monitor that learns what "normal" looks like on your Linux machine — then flags anomalies in real time via a slick terminal-aesthetic dashboard.

## How it works

1. **Collector** (`collector.py`) scrapes CPU, memory, disk, network, and process metrics every 5 seconds and stores them in a local SQLite database.
2. **Trainer** (`train.py`) reads that data and trains an **Isolation Forest** — an unsupervised ML model that learns the shape of normal behaviour without any labels.
3. **Dashboard** (`app.py`) serves a real-time web UI that scores live system snapshots and logs detected anomalies.

---

## Setup

### 1. Install dependencies

```bash
cd linux-anomaly-detector
pip install -r requirements.txt
```

### 2. Collect data (run for at least 30 minutes)

Open a terminal and run:

```bash
python collector.py
```

This saves metrics every 5 seconds to `data/metrics.db`. The more data you collect, the better the model learns what "normal" looks like. Try to collect data during typical usage — browsing, coding, etc.

> 💡 **Tip:** To generate interesting anomalies to detect later, try running `stress-ng --cpu 4 --timeout 60` in another terminal while the collector is running.

### 3. Train the model

Once you have at least ~100 rows (8 minutes), run:

```bash
python train.py
```

This trains an Isolation Forest and saves the model to `models/`. You can retrain anytime with newer data.

### 4. Launch the dashboard

```bash
python app.py
```

Open your browser at **http://localhost:5000**

---

## What you'll see

- **Anomaly Score ring** — 0 = extremely anomalous, 100 = completely normal
- **Live stats** — CPU, memory, disk I/O, network, top processes
- **Rolling charts** — last 100 samples of key metrics
- **Anomaly log** — timestamped table of flagged events
- **Alert banner** — lights up red when an anomaly is detected live

---

## Understanding the ML

### Isolation Forest

The core idea is elegant: **anomalies are easier to isolate than normal points**.

The algorithm builds many random decision trees. For each data point, it counts how many splits are needed to isolate it. Normal points (clustered together) need many splits. Anomalies (outliers) need very few.

The **anomaly score** is derived from the average path length across all trees.

### Features used

| Feature | Why it matters |
|---------|---------------|
| `cpu_percent` | Runaway processes |
| `load_avg_1m/5m` | Sustained high load |
| `mem_percent` | Memory leaks |
| `swap_percent` | Memory pressure |
| `disk_read/write_mb` | Unusual I/O (backup, cryptominer) |
| `net_sent/recv_mb` | Unusual network activity |
| `num_processes` | Fork bombs, unusual spawning |

---

## Project structure

```
linux-anomaly-detector/
├── collector.py      # Metric scraper (runs continuously)
├── train.py          # Model trainer (run once, re-run to retrain)
├── app.py            # Flask dashboard
├── requirements.txt
├── data/
│   └── metrics.db    # SQLite database (auto-created)
├── models/
│   ├── isolation_forest.pkl
│   └── scaler.pkl
└── templates/
    └── index.html    # Dashboard UI
```

---

## Ideas for extending the project

- **Email/desktop alerts** when anomalies are detected
- **Per-feature anomaly explanation** — which metric triggered it?
- **Multiple profiles** — learn separate "normal" for day vs night
- **LSTM model** — replace Isolation Forest with a time-series deep learning model
- **Systemd service** — run the collector as a background service on boot
- **Grafana integration** — export metrics to InfluxDB + Grafana
