"""
Dashboard-Server
================
Einfacher Flask-Server, der die SQLite-Datenbank als JSON-API
für das HTML-Dashboard bereitstellt.

Installation:
    pip install flask

Starten:
    python dashboard_server.py
    → Öffne http://localhost:5000 im Browser

Hinweis: Zuerst pipeline.py ausführen, um die Datenbank zu befüllen!
"""

import json
import sqlite3
from pathlib import Path

from flask import Flask, jsonify, send_from_directory

DB_PATH = "finance.db"
app = Flask(__name__, static_folder=".")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.route("/")
def index():
    return send_from_directory(".", "dashboard.html")

@app.route("/api/tickers")
def api_tickers():
    conn = get_connection()
    rows = conn.execute("SELECT DISTINCT ticker FROM stock_features ORDER BY ticker").fetchall()
    conn.close()
    return jsonify([r["ticker"] for r in rows])

@app.route("/api/data/<ticker>")
def api_data(ticker):
    return api_data_days(ticker, 252)

@app.route("/api/data/<ticker>/<int:days>")
def api_data_days(ticker, days):
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM stock_features
        WHERE ticker = ?
        ORDER BY date DESC
        LIMIT ?
    """, (ticker.upper(), min(days, 500))).fetchall()
    conn.close()
    return jsonify([dict(r) for r in reversed(rows)])

@app.route("/api/summary/<ticker>")
def api_summary(ticker):
    conn = get_connection()
    latest = conn.execute("SELECT * FROM stock_features WHERE ticker=? ORDER BY date DESC LIMIT 1", (ticker.upper(),)).fetchone()
    prev   = conn.execute("SELECT close FROM stock_features WHERE ticker=? ORDER BY date DESC LIMIT 1 OFFSET 1", (ticker.upper(),)).fetchone()
    conn.close()
    if not latest:
        return jsonify({})
    d = dict(latest)
    d["prev_close"] = dict(prev)["close"] if prev else d["close"]
    d["change_pct"] = round((d["close"] - d["prev_close"]) / d["prev_close"] * 100, 2)
    return jsonify(d)

@app.route("/api/ml/<ticker>")
def api_ml(ticker):
    conn = get_connection()
    row = conn.execute("SELECT * FROM ml_results WHERE ticker=?", (ticker.upper(),)).fetchone()
    conn.close()
    if not row:
        return jsonify({"error": "no ML data"}), 404
    d = dict(row)
    d["feature_importances"] = json.loads(d["feature_importances"])
    d["backtest"]            = json.loads(d["backtest"])
    d["confusion_matrix"]    = json.loads(d["confusion_matrix"])
    return jsonify(d)

if __name__ == "__main__":
    if not Path(DB_PATH).exists():
        print(f"FEHLER: '{DB_PATH}' nicht gefunden — zuerst app.py ausführen!")
    else:
        print("Dashboard-Server läuft auf http://localhost:5000")
        app.run(debug=True, port=5000)