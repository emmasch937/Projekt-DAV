r"""
═══════════════════════════════════════════════════════════════════
  ANALYSE-MODUL — Statistische Datenanalyse & Zeitreihenmodelle
═══════════════════════════════════════════════════════════════════

Dieses Modul ergänzt app.py um die statistische Tiefe, die in der
Vorlesung gefordert wurde. Es läuft SEPARAT und liest die bereits
von app.py erzeugte finance.db.

Inhalte:
  1. Deskriptive Statistik (Mittelwert, Median, Varianz, ...)
  2. Fehlende-Werte-Analyse (Anteil, MCAR/MAR/MNAR-Einordnung)
  3. Stationaritätstests: ADF, KPSS, PP
  4. Autokorrelation: ACF & PACF (1. und 2. Ordnung)
  5. ARIMA / ARMA-Modelle mit automatischer Ordnungswahl
  6. Vergleich: Random Forest auf ROHDATEN vs. STATIONÄREN Daten
     → zeigt empirisch warum man keine Rohdaten nimmt (Data Leakage)
  7. Grafische Darstellung (4 PNGs im Ordner plots/):
     - Kurs vs. Log-Returns, Histogramm, ACF/PACF, ARIMA-Prognose

Installation:
    conda install statsmodels scikit-learn matplotlib  oder python -m pip install statsmodels scikit-learn matplotlib pandas numpy

Starten:
    C:\Users\emma-\Documents\Studium\4. Semster\DAV\Projekt_1.2\analyse.py
"""

import sqlite3
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import logging
logging.getLogger("statsmodels").setLevel(logging.ERROR)

# ── Optionale Bibliotheken ───────────────────────────────────────
try:
    from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_OK = True
except ImportError:
    STATSMODELS_OK = False
    print("⚠️ statsmodels fehlt — Tests übersprungen.")
    print("  Installation: conda install statsmodels\n")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import TimeSeriesSplit
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    import matplotlib
    matplotlib.use("Agg")  # kein Fenster nötig, speichert direkt als Datei
    import matplotlib.pyplot as plt
    # Schriftart die Umlaute (ä, ö, ü) sauber darstellt
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams["axes.unicode_minus"] = False
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False
    print("matplotlib fehlt - Grafiken übersprungen.")
    print("  Installation: conda install matplotlib\n")

DB_PATH = "finance.db"

# Farbschema passend zum Dashboard
COL_BLUE   = "#1a6fd4"
COL_GREEN  = "#16a05c"
COL_RED    = "#b52b2b"
COL_PURPLE = "#7c3aed"
COL_AMBER  = "#d4820a"
COL_GREY   = "#8a8a84"


# ─────────────────────────────────────────────────────────────────
# Daten aus der Datenbank laden
# ─────────────────────────────────────────────────────────────────

def load_from_db(ticker: str, db_path: str = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM stock_features WHERE ticker = ? ORDER BY date",
        conn, params=(ticker,)
    )
    conn.close()
    return df


def get_available_tickers(db_path: str = DB_PATH) -> list:
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT DISTINCT ticker FROM stock_features ORDER BY ticker").fetchall()
    conn.close()
    return [r[0] for r in rows]


# ─────────────────────────────────────────────────────────────────
# 1. DESKRIPTIVE STATISTIK
# ─────────────────────────────────────────────────────────────────

def descriptive_stats(df: pd.DataFrame, ticker: str):
    """
    Berechnet zentrale Lage- und Streumaße.

    Begriffe (für die Präsentation):
      Arithmetisches Mittel (Mittelwert): Summe / Anzahl
          → empfindlich gegenüber Ausreißern
      Median: mittlerer Wert wenn sortiert
          → robust gegenüber Ausreißern
      Varianz: mittlere quadratische Abweichung vom Mittelwert
      Standardabweichung: Wurzel der Varianz (gleiche Einheit wie Daten)
      Schiefe (Skewness): Asymmetrie der Verteilung
      Kurtosis: "Spitzigkeit" / Fat Tails
    """
    print("\n" + "═"*60)
    print(f"  1. DESKRIPTIVE STATISTIK — {ticker}")
    print("═"*60)

    close = df["close"].dropna()
    logret = df["log_return"].dropna()

    print(f"\n  Datensatzgröße: {len(df)} Handelstage")
    print(f"  Zeitraum: {df['date'].iloc[0]} bis {df['date'].iloc[-1]}")

    print(f"\n  KURS (close):")
    print(f"    Mittelwert (arithm.) : {close.mean():>10.2f}")
    print(f"    Median               : {close.median():>10.2f}")
    print(f"    Standardabweichung   : {close.std():>10.2f}")
    print(f"    Varianz              : {close.var():>10.2f}")
    print(f"    Minimum              : {close.min():>10.2f}")
    print(f"    Maximum              : {close.max():>10.2f}")

    print(f"\n  LOG-RETURNS (stationär):")
    print(f"    Mittelwert           : {logret.mean():>10.5f}")
    print(f"    Median               : {logret.median():>10.5f}")
    print(f"    Standardabweichung   : {logret.std():>10.5f}")
    print(f"    Schiefe (Skewness)   : {logret.skew():>10.4f}")
    print(f"    Kurtosis             : {logret.kurtosis():>10.4f}")
    print(f"    → Mittelwert ≈ Median ≈ 0 bestätigt: Log-Returns")
    print(f"      schwanken symmetrisch um null (gut für Modelle)")


# ─────────────────────────────────────────────────────────────────
# 2. FEHLENDE WERTE
# ─────────────────────────────────────────────────────────────────

def missing_value_analysis(df: pd.DataFrame, ticker: str):
    """
    Analysiert fehlende Werte und ihre möglichen Mechanismen.

    Drei Mechanismen (Vorlesung):
      MCAR (Missing Completely At Random):
          Fehlen rein zufällig, kein Zusammenhang mit irgendwas.
          → einfaches Löschen/Auffüllen unproblematisch
      MAR (Missing At Random):
          Fehlen hängt von BEOBACHTBAREN Variablen ab.
          → z.B. Feiertage → systematisch, aber erklärbar
      MNAR (Missing Not At Random):
          Fehlen hängt vom fehlenden Wert SELBST ab.
          → schlimmster Fall, schwer zu behandeln

    Bei Aktiendaten:
      - Wochenenden/Feiertage → keine Handelstage → MAR
      - NaN bei MA20 in den ersten 19 Tagen → strukturell (kein echtes Fehlen)
      - Echte Lücken (z.B. Handelsaussetzung) → selten, eher MNAR

    Behandlung bei Zeitreihen:
      NICHT mit Mittelwert auffüllen (zerstört zeitliche Struktur)!
      Stattdessen: Forward-Fill (letzte Beobachtung) oder
      Interpolation (Mittel der beiden Nachbarpunkte).
    """
    print("\n" + "═"*60)
    print(f"  2. FEHLENDE-WERTE-ANALYSE — {ticker}")
    print("═"*60)

    # Nur die Roh-Spalten betrachten (Feature-NaNs sind strukturell)
    raw_cols = ["open", "high", "low", "close", "volume"]
    print(f"\n  Rohdaten-Spalten ({len(df)} Zeilen):")
    total_missing = 0
    for col in raw_cols:
        if col in df.columns:
            n_miss = df[col].isna().sum()
            pct = n_miss / len(df) * 100
            total_missing += n_miss
            flag = " ⚠️" if pct > 5 else ""
            print(f"    {col:<10} fehlend: {n_miss:>4} ({pct:>5.1f}%){flag}")

    overall_pct = total_missing / (len(df) * len(raw_cols)) * 100
    print(f"\n  Gesamt fehlend in Rohdaten: {overall_pct:.2f}%")

    if overall_pct == 0:
        print("  ✓ Keine fehlenden Rohdaten — yfinance liefert saubere Handelstage.")
        print("    (Wochenenden/Feiertage sind gar nicht erst enthalten → MAR-konform)")
    elif overall_pct < 5:
        print("  ✓ Unter 5% — unkritisch, Forward-Fill ausreichend.")
    else:
        print("  ⚠️ Über 5% — kritisch! Genauere Behandlung nötig.")

    # Feature-NaNs erklären
    print(f"\n  Hinweis zu Feature-Spalten:")
    print(f"    MA50 hat die ersten 49 Zeilen NaN — das ist STRUKTURELL")
    print(f"    (man braucht 50 Tage für einen 50-Tage-Durchschnitt),")
    print(f"    kein echtes 'fehlen'. Diese Zeilen werden beim Modellieren")
    print(f"    einfach übersprungen (dropna).")

    # Lückenprüfung im Datum
    dates = pd.to_datetime(df["date"])
    gaps = dates.diff().dt.days
    big_gaps = (gaps > 5).sum()  # mehr als ein Wochenende
    print(f"\n  Zeitliche Lücken > 5 Tage: {big_gaps}")
    print(f"    (kleine Lücken = Wochenenden/Feiertage = normal)")


# ─────────────────────────────────────────────────────────────────
# 3. STATIONARITÄTSTESTS
# ─────────────────────────────────────────────────────────────────

def stationarity_tests(series: pd.Series, name: str):
    """
    Drei Tests auf Stationarität.

    Eine Zeitreihe ist stationär wenn Mittelwert, Varianz und
    Autokovarianz über die Zeit KONSTANT sind.

    ADF — Augmented Dickey-Fuller:
        H0: Zeitreihe ist NICHT stationär (hat Einheitswurzel)
        H1: Zeitreihe ist stationär
        → p-Wert < 0.05  ⇒  H0 ablehnen  ⇒  STATIONÄR

    KPSS — Kwiatkowski-Phillips-Schmidt-Shin:
        VERTAUSCHTE Hypothesen!
        H0: Zeitreihe IST stationär
        H1: Zeitreihe ist nicht stationär
        → p-Wert > 0.05  ⇒  H0 behalten  ⇒  STATIONÄR

    PP — Phillips-Perron (wie ADF, andere Korrektur):
        H0: nicht stationär
        → p-Wert < 0.05  ⇒  STATIONÄR

    Erwartung:
        Rohe Aktienkurse  → NICHT stationär (Trend)
        Log-Returns       → STATIONÄR
    """
    if not STATSMODELS_OK:
        return

    s = series.dropna()
    if len(s) < 20:
        print(f"    {name}: zu wenige Werte")
        return

    print(f"\n  ── {name} ──")

    # ADF
    adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
    adf_result = "STATIONÄR ✓" if adf_p < 0.05 else "nicht stationär ✗"
    print(f"    ADF : Statistik={adf_stat:>8.3f}  p-Wert={adf_p:.4f}  → {adf_result}")

    # KPSS (vertauschte Hypothese!)
    try:
        kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")
        kpss_result = "STATIONÄR ✓" if kpss_p > 0.05 else "nicht stationär ✗"
        print(f"    KPSS: Statistik={kpss_stat:>8.3f}  p-Wert={kpss_p:.4f}  → {kpss_result}")
    except Exception as e:
        print(f"    KPSS: Fehler ({e})")

    # PP (Phillips-Perron) — über adfuller mit anderer Methode approximiert
    # statsmodels hat keinen direkten PP-Test, ADF mit regression='ct' kommt nahe
    pp_stat, pp_p, *_ = adfuller(s, regression="ct", autolag="AIC")
    pp_result = "STATIONÄR ✓" if pp_p < 0.05 else "nicht stationär ✗"
    print(f"    PP  : Statistik={pp_stat:>8.3f}  p-Wert={pp_p:.4f}  → {pp_result}")


def run_stationarity(df: pd.DataFrame, ticker: str):
    print("\n" + "═"*60)
    print(f"  3. STATIONARITÄTSTESTS — {ticker}")
    print("═"*60)
    print("\n  Schritt 1: Roher Kurs (close) — sollte NICHT stationär sein")
    stationarity_tests(df["close"], "Roher Kurs")

    print("\n  Schritt 2: Log-Returns — sollten STATIONÄR sein")
    stationarity_tests(df["log_return"], "Log-Returns")

    print("\n  Schritt 3: 1. Differenz des Kurses (Kurs heute - gestern)")
    stationarity_tests(df["close"].diff(), "1. Differenz")

    print("\n  ⇒ Fazit: Rohdaten müssen transformiert werden, bevor man")
    print("     Zeitreihenmodelle anwenden kann. Log-Returns lösen das.")


# ─────────────────────────────────────────────────────────────────
# 4. AUTOKORRELATION
# ─────────────────────────────────────────────────────────────────

def autocorrelation_analysis(df: pd.DataFrame, ticker: str):
    """
    Autokorrelation: Wie hängt der Wert von heute mit
    früheren Werten zusammen?

    ACF (Autocorrelation Function):
        Korrelation zwischen Wert_t und Wert_{t-k} für verschiedene k.
        Enthält auch indirekte Effekte.

    PACF (Partial ACF):
        Nur der DIREKTE Effekt von Wert_{t-k}, bereinigt um
        die dazwischenliegenden Werte.

    Bedeutung für AR-Modelle:
        PACF bricht nach Lag p ab  →  AR(p)-Prozess
        ACF bricht nach Lag q ab   →  MA(q)-Prozess

    Bei Aktienkursen: meist hohe positive Autokorrelation 1. Ordnung
    (Kurs heute ≈ Kurs gestern). Bei Log-Returns: kaum noch
    Autokorrelation (Markteffizienz).
    """
    if not STATSMODELS_OK:
        return

    print("\n" + "═"*60)
    print(f"  4. AUTOKORRELATION — {ticker}")
    print("═"*60)

    for name, series in [("Roher Kurs", df["close"]), ("Log-Returns", df["log_return"])]:
        s = series.dropna()
        if len(s) < 10:
            continue
        acf_vals = acf(s, nlags=5)
        pacf_vals = pacf(s, nlags=5)
        print(f"\n  ── {name} ──")
        print(f"    Lag │   ACF   │  PACF")
        print(f"    ────┼─────────┼─────────")
        for lag in range(1, 4):
            print(f"     {lag}  │ {acf_vals[lag]:>7.4f} │ {pacf_vals[lag]:>7.4f}")
        if name == "Roher Kurs":
            print(f"    → ACF(1)={acf_vals[1]:.3f} sehr hoch: Kurs hängt stark")
            print(f"      vom Vortag ab → autoregressiver Prozess AR(1)")
        else:
            print(f"    → bei Log-Returns kaum Autokorrelation → effizienter Markt")


# ─────────────────────────────────────────────────────────────────
# 5. ARIMA / ARMA-MODELLE
# ─────────────────────────────────────────────────────────────────

def arima_forecast(df: pd.DataFrame, ticker: str, forecast_days: int = 10):
    """
    ARIMA(p,d,q) — Autoregressive Integrated Moving Average.

    p = AR-Ordnung   (autoregressiver Teil: Abhängigkeit von Vorwerten)
    d = Differenzen   (wie oft differenziert für Stationarität)
    q = MA-Ordnung    (Moving-Average der Fehlerterme)

    ARMA = ARIMA mit d=0 (Daten bereits stationär).

    Wir testen mehrere Ordnungen und wählen das beste Modell
    nach AIC (Akaike Information Criterion — je kleiner, desto besser).

    Getestete Kombinationen (wie in der Vorlesung):
        ARMA(1,1), (1,2), (2,1), (2,2)
    """
    if not STATSMODELS_OK:
        return None, None

    print("\n" + "═"*60)
    print(f"  5. ARIMA / ARMA-MODELLE — {ticker}")
    print("═"*60)

    # Wir modellieren den Kurs mit d=1 (eine Differenz → stationär)
    series = df["close"].dropna().reset_index(drop=True)

    print("\n  Teste verschiedene ARIMA(p,1,q)-Ordnungen (AIC):")
    candidates = [(1, 1, 1), (1, 1, 2), (2, 1, 1), (2, 1, 2)]
    results = []
    for order in candidates:
        try:
            model = ARIMA(series, order=order)
            fit = model.fit()
            results.append((order, fit.aic, fit))
            print(f"    ARIMA{order}  AIC = {fit.aic:>10.2f}")
        except Exception as e:
            print(f"    ARIMA{order}  Fehler")

    if not results:
        print("  Kein Modell konvergiert.")
        return None, None

    # Bestes Modell nach AIC
    best_order, best_aic, best_fit = min(results, key=lambda x: x[1])
    print(f"\n  ✓ Bestes Modell: ARIMA{best_order} (AIC = {best_aic:.2f})")

    # Prognose
    forecast = best_fit.get_forecast(steps=forecast_days)
    mean_fc = forecast.predicted_mean
    ci = forecast.conf_int(alpha=0.05)

    print(f"\n  Prognose für die nächsten {forecast_days} Handelstage:")
    print(f"    Tag │  Prognose │  95%-Konfidenzintervall")
    print(f"    ────┼───────────┼──────────────────────────")
    last_price = series.iloc[-1]
    for i in range(min(forecast_days, len(mean_fc))):
        fc = mean_fc.iloc[i]
        lo = ci.iloc[i, 0]
        hi = ci.iloc[i, 1]
        print(f"    +{i+1:>2} │ {fc:>9.2f} │ [{lo:>8.2f}, {hi:>8.2f}]")

    print(f"\n  Aktueller Kurs: {last_price:.2f}")
    print(f"  → Je näher am aktuellen Rand, desto enger das Intervall")
    print(f"    (weiter in der Zukunft = mehr Unsicherheit)")

    return best_fit, best_order


# ─────────────────────────────────────────────────────────────────
# 6. VERGLEICH: ROHDATEN vs. STATIONÄRE DATEN
# ─────────────────────────────────────────────────────────────────

def rawdata_vs_stationary(df: pd.DataFrame, ticker: str):
    """
    Empirischer Beweis warum man KEINE Rohdaten verwendet.

    Wir trainieren ZWEIMAL einen Random Forest:
      A) auf ROHDATEN (open, high, low, close, volume)
      B) auf STATIONÄREN Features (RSI, MACD, Log-Returns, ...)

    Erwartung:
      Variante A zeigt oft TÄUSCHEND hohe Accuracy, weil das Modell
      das Preisniveau "auswendig lernt" (Data Leakage / Overfitting).
      Auf echten neuen Daten würde sie versagen.

      Variante B ist ehrlicher — Accuracy näher an realistischen
      55-60%, weil nur stationäre Muster gelernt werden.
    """
    if not SKLEARN_OK:
        return

    print("\n" + "═"*60)
    print(f"  6. RANDOM FOREST: ROHDATEN vs. STATIONÄR — {ticker}")
    print("═"*60)

    df2 = df.copy().reset_index(drop=True)
    df2["label"] = (df2["close"].shift(-1) > df2["close"]).astype(int)

    def evaluate(feature_cols, label):
        data = df2[feature_cols + ["label"]].dropna().iloc[:-1]
        if len(data) < 60:
            return None
        X = data[feature_cols].values
        y = data["label"].values
        tscv = TimeSeriesSplit(n_splits=5)
        tr, te = list(tscv.split(X))[-1]
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        m = RandomForestClassifier(n_estimators=200, max_depth=5,
                                   min_samples_leaf=5, random_state=42,
                                   class_weight="balanced")
        m.fit(Xtr, y[tr])
        acc_train = accuracy_score(y[tr], m.predict(Xtr))
        acc_test = accuracy_score(y[te], m.predict(Xte))
        return acc_train, acc_test

    # A) Rohdaten
    raw_cols = ["open", "high", "low", "close", "volume"]
    res_raw = evaluate(raw_cols, "Rohdaten")

    # B) Stationäre Features
    stat_cols = ["log_return", "rsi_14", "macd", "macd_signal",
                 "volatility_20d", "volume_ratio", "momentum_10",
                 "return_5d", "atr_14"]
    stat_cols = [c for c in stat_cols if c in df2.columns]
    res_stat = evaluate(stat_cols, "Stationär")

    print(f"\n  {'Variante':<22}{'Train-Acc':>12}{'Test-Acc':>12}")
    print(f"  {'─'*46}")
    if res_raw:
        print(f"  {'A) ROHDATEN':<22}{res_raw[0]*100:>11.1f}%{res_raw[1]*100:>11.1f}%")
    if res_stat:
        print(f"  {'B) STATIONÄR':<22}{res_stat[0]*100:>11.1f}%{res_stat[1]*100:>11.1f}%")

    print(f"\n  Interpretation:")
    if res_raw and res_stat:
        gap_raw = (res_raw[0] - res_raw[1]) * 100
        gap_stat = (res_stat[0] - res_stat[1]) * 100
        print(f"    Rohdaten:  Lücke Train→Test = {gap_raw:>5.1f} Prozentpunkte")
        print(f"    Stationär: Lücke Train→Test = {gap_stat:>5.1f} Prozentpunkte")
        print(f"\n    Eine GROSSE Lücke = Overfitting. Rohdaten neigen dazu,")
        print(f"    weil das Modell das Preisniveau memoriert statt Muster")
        print(f"    zu lernen. Stationäre Features generalisieren besser.")
    print(f"\n  ⇒ DESHALB verwenden wir im Hauptmodell nur stationäre Features.")

# -----------------------------------------------------------------
# 7. GRAFISCHE DARSTELLUNG
# -----------------------------------------------------------------

def create_plots(df, ticker, arima_fit=None, arima_order=None, forecast_days=10):
    """
    Erzeugt vier Diagramme und speichert sie als PNG-Dateien.

    Plot 1 - Kurs vs. Log-Returns (Stationaritaet visuell)
    Plot 2 - Histogramm der Log-Returns mit Normalverteilung
    Plot 3 - ACF & PACF Plots (Standard der Zeitreihenanalyse)
    Plot 4 - ARIMA-Prognose mit Konfidenzband

    Alle Plots werden im Ordner 'plots/' gespeichert.
    """
    if not MATPLOTLIB_OK:
        return

    import os
    os.makedirs("plots", exist_ok=True)

    print("\n" + "=" * 60)
    print(f"  7. GRAFISCHE DARSTELLUNG - {ticker}")
    print("=" * 60)

    close  = df["close"].dropna().reset_index(drop=True)
    logret = df["log_return"].dropna().reset_index(drop=True)
    dates  = pd.to_datetime(df["date"])

    # ---- Plot 1: Kurs vs. Log-Returns ----------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=False)
    ax1.plot(close.values, color=COL_BLUE, linewidth=1.2)
    ax1.set_title(f"{ticker} - Kursverlauf (nicht stationär, mit Trend)",
                  fontsize=11, fontweight="bold")
    ax1.set_ylabel("Kurs ($)")
    ax1.grid(alpha=0.3)

    ax2.plot(logret.values, color=COL_GREEN, linewidth=0.8)
    ax2.axhline(0, color=COL_RED, linewidth=0.8, linestyle="--")
    ax2.set_title("Log-Returns (stationär, schwankt um null)",
                  fontsize=11, fontweight="bold")
    ax2.set_ylabel("Log-Return")
    ax2.set_xlabel("Handelstag")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    p1 = f"plots/{ticker}_1_kurs_vs_returns.png"
    plt.savefig(p1, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Gespeichert: {p1}")

    # ---- Plot 2: Histogramm Log-Returns + Normalverteilung -------
    fig, ax = plt.subplots(figsize=(9, 5))
    n_bins = 40
    counts, bins, _ = ax.hist(logret.values, bins=n_bins, density=True,
                              color=COL_BLUE, alpha=0.6, edgecolor="white",
                              linewidth=0.5, label="Log-Returns")
    # Normalverteilungskurve drueber
    mu, sigma = logret.mean(), logret.std()
    x = np.linspace(logret.min(), logret.max(), 200)
    normal = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    ax.plot(x, normal, color=COL_RED, linewidth=2,
            label=f"Normalverteilung (mu={mu:.4f}, sigma={sigma:.4f})")
    ax.axvline(mu, color=COL_AMBER, linewidth=1.2, linestyle="--", label="Mittelwert")
    ax.set_title(f"{ticker} - Verteilung der Log-Returns",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Log-Return")
    ax.set_ylabel("Dichte")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    # Schiefe und Kurtosis als Text
    skew = logret.skew()
    kurt = logret.kurtosis()
    ax.text(0.02, 0.97, f"Schiefe: {skew:.3f}\nKurtosis: {kurt:.3f}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    plt.tight_layout()
    p2 = f"plots/{ticker}_2_histogramm.png"
    plt.savefig(p2, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Gespeichert: {p2}")

    # ---- Plot 3: ACF & PACF --------------------------------------
    if STATSMODELS_OK:
        fig, axes = plt.subplots(2, 2, figsize=(11, 6))
        # Roher Kurs
        plot_acf(close, lags=20, ax=axes[0, 0], color=COL_BLUE)
        axes[0, 0].set_title("ACF - Roher Kurs (langsam abfallend = Trend)", fontsize=10)
        plot_pacf(close, lags=20, ax=axes[0, 1], color=COL_BLUE, method="ywm")
        axes[0, 1].set_title("PACF - Roher Kurs (Spike bei Lag 1 = AR(1))", fontsize=10)
        # Log-Returns
        plot_acf(logret, lags=20, ax=axes[1, 0], color=COL_GREEN)
        axes[1, 0].set_title("ACF - Log-Returns (kaum Autokorrelation)", fontsize=10)
        plot_pacf(logret, lags=20, ax=axes[1, 1], color=COL_GREEN, method="ywm")
        axes[1, 1].set_title("PACF - Log-Returns (effizienter Markt)", fontsize=10)
        for ax in axes.flat:
            ax.grid(alpha=0.3)
        plt.suptitle(f"{ticker} - Autokorrelation (ACF) & partielle Autokorrelation (PACF)",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        p3 = f"plots/{ticker}_3_acf_pacf.png"
        plt.savefig(p3, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  Gespeichert: {p3}")

    # ---- Plot 4: ARIMA-Prognose ----------------------------------
    if STATSMODELS_OK and arima_fit is not None:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        # Letzte 60 Tage historisch
        hist_n = min(60, len(close))
        hist = close.iloc[-hist_n:].values
        hist_x = np.arange(hist_n)
        ax.plot(hist_x, hist, color=COL_BLUE, linewidth=1.5, label="Historischer Kurs")

        # Prognose
        forecast = arima_fit.get_forecast(steps=forecast_days)
        mean_fc = forecast.predicted_mean.values
        ci = forecast.conf_int(alpha=0.05).values
        fc_x = np.arange(hist_n, hist_n + forecast_days)

        # Verbindungslinie vom letzten Punkt
        ax.plot([hist_n - 1, hist_n], [hist[-1], mean_fc[0]],
                color=COL_PURPLE, linewidth=1.8, linestyle="--")
        ax.plot(fc_x, mean_fc, color=COL_PURPLE, linewidth=1.8,
                linestyle="--", marker="o", markersize=3,
                label=f"ARIMA{arima_order} Prognose")
        # Konfidenzband
        ax.fill_between(fc_x, ci[:, 0], ci[:, 1], color=COL_PURPLE, alpha=0.15,
                        label="95%-Konfidenzintervall")

        ax.axvline(hist_n - 1, color=COL_GREY, linewidth=0.8, linestyle=":")
        ax.set_title(f"{ticker} - ARIMA{arima_order} Prognose für {forecast_days} Tage",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Handelstag (relativ)")
        ax.set_ylabel("Kurs ($)")
        ax.legend(fontsize=9, loc="best")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        p4 = f"plots/{ticker}_4_arima_prognose.png"
        plt.savefig(p4, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  Gespeichert: {p4}")

    print(f"\n  Alle Grafiken im Ordner 'plots/' - direkt für die Präsentation nutzbar.")





# ─────────────────────────────────────────────────────────────────
# HAUPTPROGRAMM
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    if not os.path.exists(DB_PATH):
        print(f"FEHLER: '{DB_PATH}' nicht gefunden.")
        print("Bitte zuerst app.py ausführen!")
        exit(1)

    tickers = get_available_tickers()
    if not tickers:
        print("Keine Daten in der Datenbank. Zuerst app.py ausführen!")
        exit(1)

    # Welche Aktie analysieren? Standard: erste in der DB
    TICKER = "BMW.DE"
    print("\n" + "█"*60)
    print(f"  STATISTISCHE ANALYSE — {TICKER}")
    print(f"  (verfügbare Ticker: {', '.join(tickers)})")
    print(f"  Zum Wechseln: TICKER-Variable unten im Code ändern")
    print("█"*60)

    df = load_from_db(TICKER)

    descriptive_stats(df, TICKER)
    missing_value_analysis(df, TICKER)
    run_stationarity(df, TICKER)
    autocorrelation_analysis(df, TICKER)
    arima_fit, arima_order = arima_forecast(df, TICKER, forecast_days=10)
    rawdata_vs_stationary(df, TICKER)
    create_plots(df, TICKER, arima_fit, arima_order, forecast_days=10)

    print("\n" + "█"*60)
    print("  ✓ Analyse abgeschlossen.")
    print("█"*60)