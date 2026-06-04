# 📈 Finanz-Dashboard – DAV Projekt

Ein datengetriebenes Analyse-Dashboard für Aktienmärkte mit Machine-Learning-Vorhersagen, technischen Indikatoren und interaktiver Visualisierung.

---

## 🗂️ Projektstruktur

```
Projekt-DAV/
├── app.py                  # Datenpipeline (Laden, Bereinigung, ML)
├── dashboard_server.py     # Flask-Server mit REST-API
├── dashboard.html          # Interaktives Web-Dashboard
├── finance.db              # SQLite-Datenbank (wird automatisch erstellt)
└── README.md
```

---

## ⚙️ Installation & Start

### Voraussetzungen
- Python 3.8+

### Schritt 1 – Abhängigkeiten installieren
```bash
python -m pip install pandas numpy flask yfinance scikit-learn
```

### Schritt 2 – Datenpipeline ausführen
```bash
python app.py
```
Dies lädt alle Aktiendaten, berechnet Features und trainiert das ML-Modell.  
Die Ergebnisse werden in `finance.db` gespeichert. *(Kann einige Minuten dauern)*

### Schritt 3 – Dashboard starten
```bash
python dashboard_server.py
```
Dann im Browser öffnen: **http://localhost:5000**

---

## 🔍 Wie funktioniert das Projekt?

### 1. Datenpipeline (`app.py`)

Die Pipeline läuft in 6 Schritten:

| Schritt | Beschreibung |
|---------|-------------|
| **1. Daten laden** | Historische Aktienkurse via Yahoo Finance (`yfinance`) für bis zu 1 Jahr |
| **2. Datenbereinigung** | Entfernung von Duplikaten, Ausreißern in OHLCV-Daten, Forward-Fill für fehlende Werte |
| **3. Feature Engineering** | Berechnung von 18+ technischen Indikatoren (siehe unten) |
| **4. Ausreißer-Erkennung** | Isolation Forest + Random Forest zur Erkennung anomaler Handelstage |
| **5. ML-Modell** | Random Forest Klassifikation zur Vorhersage der nächsten Kursbewegung |
| **6. Speicherung** | Alle Ergebnisse werden in einer SQLite-Datenbank gespeichert |

### 2. Technische Indikatoren (Feature Engineering)

| Kategorie | Indikator | Beschreibung |
|-----------|-----------|-------------|
| **Renditen** | Log Return, Return 5d/21d | Tages- und Mehrtagsrenditen |
| **Trend** | MA20, MA50, EMA12, EMA26 | Gleitende Durchschnitte |
| **Momentum** | MACD, MACD-Signal, RSI(14), Momentum(10) | Trendstärke und Umkehrsignale |
| **Volatilität** | Volatility 20d, ATR(14), Bollinger Bänder | Schwankungsbreite |
| **Volumen** | OBV, Volume Ratio, Vol Z-Score | Handelsvolumen-Analyse |

### 3. Machine Learning

- **Modell:** Random Forest Classifier (200 Estimatoren)
- **Ziel:** Vorhersage ob der Kurs am nächsten Tag steigt (1) oder fällt (0)
- **Validierung:** TimeSeriesSplit (5 Folds) – kein Data Leakage
- **Output:** Accuracy, Precision, Recall, Feature Importance, Backtest-Kurve

### 4. Dashboard-Server (`dashboard_server.py`)

Ein leichtgewichtiger Flask-Server stellt die Daten als REST-API bereit:

| Endpoint | Beschreibung |
|----------|-------------|
| `GET /` | Lädt das interaktive Dashboard |
| `GET /api/tickers` | Liste aller verfügbaren Aktien |
| `GET /api/data/<ticker>` | Historische Daten (Standard: 252 Tage) |
| `GET /api/data/<ticker>/<days>` | Historische Daten mit eigener Zeitspanne |
| `GET /api/summary/<ticker>` | Aktuelle Kennzahlen inkl. Tagesveränderung |
| `GET /api/ml/<ticker>` | ML-Ergebnisse, Feature Importance, Backtest |

---

## 📊 Enthaltene Aktien

```python
TICKERS = [
    "AAPL", "MSFT", "TSLA", "GOOG", "META", "AMZN", "NFLX",   # US Tech
    "COIN", "PLTR", "AMC", "GME",                                # Spezial
    "BMW.DE", "SAP.DE", "SIE.DE", "ALV.DE",                     # DAX
    "DTE.DE", "VOW3.DE", "MBG.DE", "BAYN.DE", "DBK.DE", "NVDA" # DAX + NVDA
]
```

Eigene Aktien können einfach in `app.py` in der `TICKERS`-Liste ergänzt werden.

---

## 🛠️ Technologien

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-REST--API-lightgrey?logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![SQLite](https://img.shields.io/badge/SQLite-Datenbank-blue?logo=sqlite)
![yfinance](https://img.shields.io/badge/yfinance-Marktdaten-green)

---

## 📝 Hinweise

- Die Datei `finance.db` wird automatisch erstellt und muss **nicht** manuell angelegt werden
- Ohne `yfinance` werden automatisch realistische Dummy-Daten generiert
- Das ML-Modell ist zu Analysezwecken erstellt und stellt **keine Anlageberatung** dar
