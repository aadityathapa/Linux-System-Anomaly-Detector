"""
collector.py - System metrics collector
Scrapes CPU, memory, disk, network, and process data every N seconds
and stores it in a local SQLite database.
"""

import sqlite3
import time
import psutil
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "metrics.db")
INTERVAL = 5  # seconds between collections


def init_db():
    """Create the metrics table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,

            -- CPU
            cpu_percent         REAL,
            cpu_freq_mhz        REAL,
            load_avg_1m         REAL,
            load_avg_5m         REAL,

            -- Memory
            mem_percent         REAL,
            mem_used_mb         REAL,
            swap_percent        REAL,

            -- Disk I/O (delta per interval)
            disk_read_mb        REAL,
            disk_write_mb       REAL,

            -- Network (delta per interval)
            net_sent_mb         REAL,
            net_recv_mb         REAL,

            -- Processes
            num_processes       INTEGER,
            num_threads         INTEGER,
            top_cpu_proc        TEXT,
            top_mem_proc        TEXT
        )
    """)
    conn.commit()
    conn.close()


def collect_metrics(prev_disk=None, prev_net=None):
    """Collect a single snapshot of system metrics."""
    ts = datetime.now().isoformat()

    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_freq = psutil.cpu_freq()
    cpu_freq_mhz = cpu_freq.current if cpu_freq else 0.0
    load1, load5, _ = os.getloadavg()

    # Memory
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()

    # Disk I/O delta
    disk_io = psutil.disk_io_counters()
    if prev_disk:
        disk_read_mb = (disk_io.read_bytes - prev_disk.read_bytes) / 1e6
        disk_write_mb = (disk_io.write_bytes - prev_disk.write_bytes) / 1e6
    else:
        disk_read_mb = disk_write_mb = 0.0

    # Network delta
    net_io = psutil.net_io_counters()
    if prev_net:
        net_sent_mb = (net_io.bytes_sent - prev_net.bytes_sent) / 1e6
        net_recv_mb = (net_io.bytes_recv - prev_net.bytes_recv) / 1e6
    else:
        net_sent_mb = net_recv_mb = 0.0

    # Processes
    procs = list(psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]))
    num_processes = len(procs)
    num_threads = sum(p.info.get("num_threads", 0) or 0
                      for p in psutil.process_iter(["num_threads"]))

    try:
        top_cpu = max(procs, key=lambda p: p.info.get("cpu_percent") or 0)
        top_cpu_proc = f"{top_cpu.info['name']} ({top_cpu.info['cpu_percent']:.1f}%)"
    except Exception:
        top_cpu_proc = "unknown"

    try:
        top_mem = max(procs, key=lambda p: p.info.get("memory_percent") or 0)
        top_mem_proc = f"{top_mem.info['name']} ({top_mem.info['memory_percent']:.1f}%)"
    except Exception:
        top_mem_proc = "unknown"

    row = {
        "timestamp": ts,
        "cpu_percent": cpu_percent,
        "cpu_freq_mhz": cpu_freq_mhz,
        "load_avg_1m": round(load1, 2),
        "load_avg_5m": round(load5, 2),
        "mem_percent": mem.percent,
        "mem_used_mb": round(mem.used / 1e6, 1),
        "swap_percent": swap.percent,
        "disk_read_mb": round(disk_read_mb, 3),
        "disk_write_mb": round(disk_write_mb, 3),
        "net_sent_mb": round(net_sent_mb, 3),
        "net_recv_mb": round(net_recv_mb, 3),
        "num_processes": num_processes,
        "num_threads": num_threads,
        "top_cpu_proc": top_cpu_proc,
        "top_mem_proc": top_mem_proc,
    }

    return row, disk_io, net_io


def save_metric(row):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO metrics (
            timestamp, cpu_percent, cpu_freq_mhz, load_avg_1m, load_avg_5m,
            mem_percent, mem_used_mb, swap_percent,
            disk_read_mb, disk_write_mb,
            net_sent_mb, net_recv_mb,
            num_processes, num_threads,
            top_cpu_proc, top_mem_proc
        ) VALUES (
            :timestamp, :cpu_percent, :cpu_freq_mhz, :load_avg_1m, :load_avg_5m,
            :mem_percent, :mem_used_mb, :swap_percent,
            :disk_read_mb, :disk_write_mb,
            :net_sent_mb, :net_recv_mb,
            :num_processes, :num_threads,
            :top_cpu_proc, :top_mem_proc
        )
    """, row)
    conn.commit()
    conn.close()


def run():
    print(f"[collector] Initialising database at {DB_PATH}")
    init_db()
    prev_disk = prev_net = None
    print(f"[collector] Collecting metrics every {INTERVAL}s — press Ctrl+C to stop")
    while True:
        try:
            row, prev_disk, prev_net = collect_metrics(prev_disk, prev_net)
            save_metric(row)
            print(f"[{row['timestamp']}] CPU {row['cpu_percent']}% | "
                  f"MEM {row['mem_percent']}% | "
                  f"Procs {row['num_processes']}")
        except Exception as e:
            print(f"[collector] Error: {e}")
        time.sleep(INTERVAL)


if __name__ == "__main__":
    run()
