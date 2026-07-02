# Projekt-DAV — Feature-Engineering & Datenaufbereitung für Finanzzeitreihen

Modul **Datenverarbeitung und -bereinigung** · Thema 015 · SS 2026

Automatisierte Pipeline für historische Aktiendaten (Yahoo Finance API): Laden → Bereinigen → Feature-Engineering → Ausreißer-Erkennung → ML-Vorhersagemodell → interaktives Finanz-Dashboard.

---

## Inhaltsverzeichnis

- [Überblick](#überblick)
- [Architektur](#architektur)
- [Installation](#installation)
- [Nutzung](#nutzung)
- [Projektstruktur](#projektstruktur)
- [Methodik](#methodik)
  - [Datenbereinigung](#datenbereinigung)
  - [Feature Engineering](#feature-engineering)
  - [Ausreißer-Erkennung](#ausreißer-erkennung)
  - [ML-Vorhersagemodell](#ml-vorhersagemodell)
  - [Statistische Analyse](#statistische-analyse-analysepy)
- [Dashboard](#dashboard)
- [Datenbankschema](#datenbankschema)
- [Bekannte Einschränkungen](#bekannte-einschränkungen)
- [Autorinnen](#autorinnen)

---

## Überblick

Rohe Aktienkurse sind für Analysen und ML-Modelle in ihrer Rohform ungeeignet: fehlende Handelstage, Ausreißer und ein nicht-stationärer Trend verfälschen Ergebnisse. Dieses Projekt implementiert eine vollständige Datenpipeline, die:

1. echte historische OHLCV-Daten von Yahoo Finance lädt (`yfinance`),
2. sie realistisch bereinigt (keine Dummy-Daten, nachvollziehbare Interpolation),
3. daraus 18+ technische Indikatoren berechnet,
4. Ausreißer mit zwei unabhängigen Verfahren (Isolation Forest + Random Forest) erkennt,
5. ein Random-Forest-Klassifikationsmodell auf ausschließlich stationären Features trainiert,
6. alle Ergebnisse in SQLite persistiert,
7. und über ein interaktives Web-Dashboard zugänglich macht.

Ergänzend liefert `analyse.py` die statistische Tiefe (deskriptive Statistik, Stationaritätstests, Autokorrelation, ARIMA, empirischer Rohdaten-vs-stationär-Vergleich).

## Architektur

```
yfinance API
     │
     ▼
┌─────────────┐     ┌──────────────┐     ┌────────────────┐
│   app.py    │────▶│  finance.db  │◀────│   analyse.py    │
│  Pipeline   │     │   (SQLite)   │     │  Statistik/Plots │
└─────────────┘     └──────┬───────┘     └────────────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ dashboard_server.py│
                  │   Flask REST-API   │
                  └─────────┬──────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │  dashboard.html    │
                  │  (Browser-Frontend)│
                  └───────────────────┘
```

Vollständig lokal, keine Cloud-Abhängigkeiten, keine Kosten außer dem kostenlosen Yahoo-Finance-Zugriff.

**Tech-Stack:** Python · pandas · NumPy · scikit-learn · statsmodels · SQLite · Flask · Chart.js

## Installation

```bash
# Repository klonen
git clone https://github.com/emmasch937/Projekt-DAV.git
cd Projekt-DAV

# Abhängigkeiten installieren
pip install pandas numpy scikit-learn statsmodels matplotlib yfinance flask
```

## Nutzung

Die drei Skripte bauen aufeinander auf und müssen in dieser Reihenfolge ausgeführt werden:

```bash
# 1. Pipeline: Daten laden, bereinigen, Features berechnen, ML-Modell trainieren
python app.py

# 2. Optional: statistische Tiefenanalyse + Plots für einen einzelnen Ticker
python analyse.py

# 3. Dashboard starten
python dashboard_server.py
# → http://localhost:5000 im Browser öffnen
```

`app.py` befüllt `finance.db` neu (Standard-Tickerliste ist im Skript unter `TICKERS` definiert und kann angepasst werden). `analyse.py` liest ausschließlich aus der bereits befüllten Datenbank und muss daher **nach** `app.py` laufen.

## Projektstruktur

| Datei | Zweck |
|---|---|
| `app.py` | Hauptpipeline: Datenladen, Bereinigung, Feature-Engineering, Ausreißer-Erkennung, ML-Training, DB-Speicherung |
| `analyse.py` | Statistische Tiefenanalyse: deskriptive Statistik, Stationaritätstests, ACF/PACF, ARIMA, Rohdaten-vs-stationär-Vergleich, Plots |
| `dashboard_server.py` | Flask-Server, stellt `finance.db` als JSON-REST-API bereit und liefert das Frontend aus |
| `dashboard.html` | Interaktives Frontend (4 Tabs: Kursanalyse, ML-Vorhersage, Rendite-Vergleich, Portfolio-Simulator) |
| `finance.db` | SQLite-Datenbank (wird von `app.py` erzeugt) |

## Methodik

### Datenbereinigung

Umgesetzt in `clean_data()` (`app.py`). Da die Yahoo-Finance-Anbindung zeitweise unvollständige Daten liefert, wird bewusst **nicht** stillschweigend mit Platzhaltern aufgefüllt:

1. Duplikate und ungültige Zeilen (negatives Volumen) entfernen
2. Lückenlosen Werktags-Kalender herstellen, damit fehlende Handelstage überhaupt sichtbar werden
3. Unrealistische Kurssprünge (>50 %) werden als Fehler **markiert** (NaN gesetzt), nicht direkt verworfen
4. Zeitliche Interpolation (`limit=3`) füllt nur kurze Lücken; längere Ausfälle bleiben NaN und werden verworfen statt mit dem letzten bekannten Wert „eingefroren“
5. Fehlendes Volumen wird über den gleitenden 5-Tage-Median ersetzt (robust gegenüber Ausreißern)
6. OHLC-Konsistenz (High ≥ Low, High ≥ max(Open, Close) usw.) wird **erst nach** dem Füllen geprüft

### Feature Engineering

`compute_features()` berechnet u. a.:

- **Rendite:** `log_return`, `return_5d`, `return_21d`
- **Trend:** `ma_20`, `ma_50`, `ema_12`, `ema_26`, `macd`, `macd_signal`
- **Volatilität:** `volatility_20d` (annualisiert), Bollinger-Bänder, `atr_14`
- **Momentum/Volumen:** `rsi_14`, `momentum_10`, `volume_ratio`, `obv`

Insgesamt 18+ Indikatoren, gespeichert als 30 Spalten pro Ticker und Handelstag.

### Ausreißer-Erkennung

Zwei komplementäre Verfahren in `detect_outliers()`:

- **Isolation Forest** (unüberwacht, `contamination=0.05`) — isoliert anomale Punkte ohne Labels
- **Random Forest** (überwacht) — trainiert auf statistisch definierten Ausreißern (>2σ bei Return/Volumen, RSI <20 oder >80), liefert zusätzlich Feature Importance und eine Wahrscheinlichkeit pro Tag

Jeder erkannte Ausreißertag erhält eine lesbare Begründung (`outlier_reason`, z. B. „Kurseinbruch ↓ (-5.1 %)“).

### ML-Vorhersagemodell

`train_ml_model()` trainiert einen Random-Forest-Klassifikator, der vorhersagt, ob der Kurs am nächsten Handelstag steigt.

- **Nur stationäre Features** (Log-Returns, RSI, MACD, Bollinger-Position, z-normierte Werte …) — nie absolute Preise, da diese einen Trend enthalten und das Modell sonst nur das Preisniveau memoriert statt Muster zu lernen
- **`TimeSeriesSplit`** statt zufälligem Split, um Data Leakage aus der Zukunft zu vermeiden
- Ergebnisse (Accuracy, Precision, Recall, Feature Importance, Backtest gegen Buy-and-Hold) werden pro Ticker in der Tabelle `ml_results` gespeichert

### Statistische Analyse (`analyse.py`)

Ergänzt die Pipeline um:

- Deskriptive Statistik (Mittelwert, Median, Varianz, Schiefe, Kurtosis)
- Fehlende-Werte-Analyse mit Einordnung nach MCAR/MAR/MNAR
- Stationaritätstests: ADF, KPSS, Phillips-Perron (jeweils für rohen Kurs, Log-Returns, 1. Differenz)
- Autokorrelation (ACF/PACF) zur Einordnung als AR-/MA-Prozess
- ARIMA/ARMA-Modellierung mit automatischer Ordnungswahl nach AIC und Prognose mit Konfidenzintervall
- Empirischer Vergleich Random Forest auf Rohdaten vs. stationären Features (Train/Test-Accuracy-Lücke als Overfitting-Indikator)

Alle Plots werden automatisch im Ordner `plots/` gespeichert.

## Dashboard

`dashboard_server.py` stellt folgende Endpunkte bereit:

| Endpunkt | Beschreibung |
|---|---|
| `GET /api/tickers` | Liste aller verfügbaren Ticker |
| `GET /api/data/<ticker>/<days>` | Zeitreihe mit allen Features |
| `GET /api/summary/<ticker>` | Aktuellster Stand + Tagesveränderung |
| `GET /api/ml/<ticker>` | ML-Ergebnisse (Accuracy, Feature Importance, Backtest, Confusion Matrix) |

Das Frontend (`dashboard.html`) bietet vier Tabs: **Kursanalyse** (Chart mit markierten Ausreißern), **ML-Vorhersage** (Signal + Backtest gegen Buy-and-Hold), **Rendite-Vergleich** (mehrere Ticker) und **Portfolio-Simulator**.

## Datenbankschema

**`stock_features`** — ein Datensatz pro Ticker und Handelstag: OHLCV-Rohdaten, alle berechneten Features, Ausreißer-Kennzahlen (`outlier_iforest`, `outlier_score`, `outlier_rf`, `outlier_prob`, `outlier_reason`).

**`ml_results`** — ein Datensatz pro Ticker: Modellgüte (Accuracy, Precision, Recall), Feature Importance, Backtest-Zeitreihe, Konfusionsmatrix, Vorhersage für den nächsten Handelstag.

**`metadata`** — letzte Aktualisierung und Zeilenanzahl pro Ticker.

## Bekannte Einschränkungen

- Das ML-Modell wird pro Ticker unabhängig und auf einem vergleichsweise kleinen Testfenster trainiert (ein Jahr Daten, letzter `TimeSeriesSplit`-Fold); Accuracy schwankt daher deutlich zwischen Tickern und liegt teils nahe am Zufallsniveau.
- Random-Forest-Klassifikation auf Tagesbasis ist kein „echtes“ Zeitreihenmodell — ARIMA in `analyse.py` dient als methodischer Kontrast, nicht als Konkurrenzmodell für dieselbe Aufgabe.
- yfinance liefert nicht für jeden Ticker durchgehend saubere Daten (Delisting, Paywall-Fälle); solche Ticker werden übersprungen, nicht mit Dummy-Daten aufgefüllt.
- Aktuell nur Einzelaktien, kein ETF-Support (andere Datenstruktur bei yfinance).

## Autorinnen

Projekt im Modul *Datenverarbeitung und -bereinigung*, Thema 015, SS 2026.
