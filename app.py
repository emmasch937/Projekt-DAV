"""
Finanz-Dashboard Pipeline
=========================
Schritt 1: Aktiendaten von Yahoo Finance laden
Schritt 2: Datenbereinigung (realistische Interpolation, KEINE Dummy-Daten)
Schritt 3: Feature Engineering
Schritt 4: Ausreißer-Erkennung (Isolation Forest + Random Forest)
Schritt 5: ML-Vorhersagemodell (Random Forest Klassifikation)
Schritt 6: Ergebnisse in SQLite speichern
 
Installation:
    conda install pandas numpy scikit-learn
    pip install yfinance
 
Starten:
      python app.py
"""
 
import sqlite3
import numpy as np
import pandas as pd
 
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("yfinance nicht installiert – bitte zuerst 'pip install yfinance' ausführen.")
 
try:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from sklearn.model_selection import TimeSeriesSplit
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("scikit-learn nicht installiert.")
 
 
# ─────────────────────────────────────────────
# 1. DATEN LADEN
# ─────────────────────────────────────────────
 
def load_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Lädt echte Kursdaten von Yahoo Finance.
    Kein stiller Fallback auf Dummy-Daten – bei Problemen wird ein
    aussagekräftiger Fehler geworfen (siehe bekannte yfinance-Probleme
    seit dem Yahoo-Redesign 2025: Paywall, Delisting-Meldungen, Lücken).
    """
    if not YFINANCE_AVAILABLE:
        raise RuntimeError(
            "yfinance ist nicht installiert. Bitte 'pip install yfinance' ausführen."
        )
 
    stock = yf.Ticker(ticker)
    try:
        df = stock.history(period=period)
    except Exception as e:
        raise RuntimeError(f"{ticker}: Daten konnten nicht geladen werden ({e})")
 
    # Leerer DataFrame → Paywall / Delisting (Yahoo-Redesign ab Feb/März 2025)
    if df is None or df.empty:
        raise RuntimeError(
            f"{ticker}: Yahoo Finance lieferte keine Daten "
            f"(möglich: Paywall, Delisting oder ungültiges Symbol)."
        )
 
    df = df.reset_index()
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df.columns = [c.lower() for c in df.columns]
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  ✅  {ticker}: {len(df)} Handelstage geladen.")
    return df
 
 
# ─────────────────────────────────────────────
# 2. DATENBEREINIGUNG
# ─────────────────────────────────────────────
 
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realistische Bereinigung fehlender / falscher Werte.
 
    Vorgehen:
      b) Lückenlosen Werktags-Kalender herstellen → fehlende Tage werden
         überhaupt erst sichtbar (z.B. der "fehlender-Vortag"-Bug Nov 2025).
      c) Unrealistische Kurssprünge (>50 %) als Fehler MARKIEREN statt löschen
         (falsche Historie ab Jan 2025) → werden anschließend interpoliert.
      d) Echte Lückenfüllung per ZEITLICHER Interpolation (limit=3):
         gewichtet nach Datumsabstand statt den letzten Wert "einzufrieren".
         Längere Ausfälle bleiben NaN und werden verworfen.
      e) Volumen: fehlende Tage über gleitenden 5-Tage-MEDIAN (robust).
      g) Logische OHLC-Konsistenz wird ERST NACH dem Füllen geprüft.
    """
    price_cols = ["open", "high", "low", "close"]
 
    # a) Doppelte & grob ungültige Zeilen entfernen
    df = df.drop_duplicates(subset="date")
    df = df[df["volume"] >= 0]
    df = df.sort_values("date").reset_index(drop=True)
 
    # b) Lückenlosen Handelskalender (Werktage) herstellen
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    full_range = pd.bdate_range(start=df.index.min(), end=df.index.max())
    df = df.reindex(full_range)
    df.index.name = "date"
 
    # c) Unrealistische Kurssprünge (>50 %) zur Korrektur markieren
    ret = df["close"].pct_change().abs()
    outliers = ret > 0.50
    if outliers.any():
        print(f"  ⚠️  {int(outliers.sum())} unrealistische Kurssprünge zur Korrektur markiert.")
        df.loc[outliers, price_cols] = np.nan
 
    # d) Echte Lückenfüllung per zeitlicher Interpolation (nur kurze Lücken)
    n_missing = int(df[price_cols].isna().any(axis=1).sum())
    df[price_cols] = df[price_cols].interpolate(
        method="time", limit=3, limit_direction="both"
    )
    if n_missing:
        print(f"  🔧  {n_missing} fehlende Kurstage per Interpolation gefüllt.")
 
    # e) Volumen: fehlende Tage über gleitenden 5-Tage-Median
    df["volume"] = df["volume"].fillna(
        df["volume"].rolling(5, min_periods=1).median()
    )
 
    # f) Zu lange Lücken (immer noch NaN) verwerfen
    df = df.dropna(subset=price_cols)
 
    # g) Logische Konsistenz NACH dem Füllen prüfen
    df = df[df["close"] > 0]
    df = df[df["high"] >= df["low"]]
    df = df[df["high"] >= df[["open", "close"]].max(axis=1)]
    df = df[df["low"]  <= df[["open", "close"]].min(axis=1)]
 
    df = df.reset_index()
    df["date"] = df["date"].dt.date
    df = df.sort_values("date").reset_index(drop=True)
    return df
 
 
# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
 
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    v = df["volume"].values
    n = len(df)
 
    log_ret = np.zeros(n)
    log_ret[1:] = np.log(c[1:] / c[:-1])
 
    def rolling_return(prices, window):
        r = np.full(n, np.nan)
        r[window:] = prices[window:] / prices[:-window] - 1
        return r
 
    def sma(arr, w):
        result = np.full(n, np.nan)
        for i in range(w - 1, n):
            result[i] = arr[i - w + 1: i + 1].mean()
        return result
 
    def ema(arr, span):
        return pd.Series(arr).ewm(span=span, adjust=False).mean().values
 
    ma20 = sma(c, 20)
    ma50 = sma(c, 50)
    ema12 = ema(c, 12)
    ema26 = ema(c, 26)
    macd_line = ema12 - ema26
    macd_signal_line = ema(macd_line, 9)
 
    vol_20 = np.full(n, np.nan)
    bb_std = np.full(n, np.nan)
    for i in range(19, n):
        window_ret = log_ret[i - 19: i + 1]
        vol_20[i] = window_ret.std() * np.sqrt(252)
        bb_std[i] = c[i - 19: i + 1].std()
 
    bb_upper = ma20 + 2 * bb_std
    bb_lower = ma20 - 2 * bb_std
 
    tr = np.maximum(h - l,
         np.maximum(np.abs(h - np.roll(c, 1)),
                    np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr14 = pd.Series(tr).ewm(span=14, adjust=False).mean().values
 
    delta = np.diff(c, prepend=c[0])
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gains).ewm(com=13, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(com=13, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100.0)
    rsi = 100 - (100 / (1 + rs))
 
    momentum10 = np.full(n, np.nan)
    momentum10[10:] = (c[10:] / c[:-10] - 1) * 100
 
    vol_ma20 = sma(v, 20)
    vol_ratio = np.where(vol_ma20 > 0, v / vol_ma20, np.nan)
 
    obv = np.zeros(n)
    for i in range(1, n):
        if c[i] > c[i - 1]:   obv[i] = obv[i - 1] + v[i]
        elif c[i] < c[i - 1]: obv[i] = obv[i - 1] - v[i]
        else:                  obv[i] = obv[i - 1]
 
    features = df.copy()
    features["log_return"]      = np.round(log_ret, 6)
    features["return_5d"]       = np.round(rolling_return(c, 5), 4)
    features["return_21d"]      = np.round(rolling_return(c, 21), 4)
    features["ma_20"]           = np.round(ma20, 2)
    features["ma_50"]           = np.round(ma50, 2)
    features["ema_12"]          = np.round(ema12, 2)
    features["ema_26"]          = np.round(ema26, 2)
    features["macd"]            = np.round(macd_line, 4)
    features["macd_signal"]     = np.round(macd_signal_line, 4)
    features["volatility_20d"]  = np.round(vol_20, 4)
    features["bollinger_upper"] = np.round(bb_upper, 2)
    features["bollinger_lower"] = np.round(bb_lower, 2)
    features["atr_14"]          = np.round(atr14, 2)
    features["rsi_14"]          = np.round(rsi, 2)
    features["momentum_10"]     = np.round(momentum10, 2)
    features["volume_ma_20"]    = np.round(vol_ma20, 0)
    features["volume_ratio"]    = np.round(vol_ratio, 3)
    features["obv"]             = obv.astype(int)
    return features
 
 
# ─────────────────────────────────────────────
# 4. AUSREISSER-ERKENNUNG
# ─────────────────────────────────────────────
 
def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    feature_cols = ["log_return", "volume_ratio", "volatility_20d",
                    "rsi_14", "momentum_10", "atr_14"]
    df_ml = df[feature_cols].copy()
    valid_mask = df_ml.notna().all(axis=1)
    df_ml = df_ml[valid_mask]
 
    if SKLEARN_AVAILABLE:
        scaler = StandardScaler()
        X = scaler.fit_transform(df_ml)
 
        iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
        iso_labels = iso.fit_predict(X)
        iso_scores = iso.score_samples(X)
        df["outlier_iforest"] = 0
        df["outlier_score"]   = 0.0
        df.loc[valid_mask, "outlier_iforest"] = (iso_labels == -1).astype(int)
        df.loc[valid_mask, "outlier_score"]   = np.round(iso_scores, 4)
 
        log_ret_vals   = df_ml["log_return"].values
        vol_ratio_vals = df_ml["volume_ratio"].values
        rsi_vals       = df_ml["rsi_14"].values
        ret_mean, ret_std = log_ret_vals.mean(), log_ret_vals.std()
        vol_mean, vol_std = vol_ratio_vals.mean(), vol_ratio_vals.std()
        statistical_outlier = (
            (np.abs(log_ret_vals - ret_mean) > 2 * ret_std) |
            (vol_ratio_vals > vol_mean + 2 * vol_std) |
            (rsi_vals < 20) | (rsi_vals > 80)
        ).astype(int)
 
        rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42, class_weight="balanced")
        rf.fit(X, statistical_outlier)
        rf_pred  = rf.predict(X)
        rf_proba = rf.predict_proba(X)[:, 1]
        df["outlier_rf"]   = 0
        df["outlier_prob"] = 0.0
        df.loc[valid_mask, "outlier_rf"]   = rf_pred
        df.loc[valid_mask, "outlier_prob"] = np.round(rf_proba, 3)
 
        importances = rf.feature_importances_
        print("\n  Ausreißer — Feature Importance:")
        for col, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
            print(f"    {col:<20} {'█'*int(imp*40)} {imp:.3f}")
    else:
        log_ret_vals = df["log_return"].fillna(0).values
        ret_mean, ret_std = log_ret_vals.mean(), log_ret_vals.std()
        z = np.abs((log_ret_vals - ret_mean) / (ret_std + 1e-9))
        df["outlier_iforest"] = (z > 2).astype(int)
        df["outlier_score"]   = np.round(-z, 4)
        df["outlier_rf"]      = df["outlier_iforest"]
        df["outlier_prob"]    = np.round(np.minimum(z / 5, 1.0), 3)
 
    def get_reason(row):
        reasons = []
        if abs(row.get("log_return", 0)) > 0.04:
            direction = "Kurssprung ↑" if row["log_return"] > 0 else "Kurseinbruch ↓"
            reasons.append(f"{direction} ({row['log_return']*100:.1f}%)")
        if row.get("volume_ratio", 1) > 2.5:
            reasons.append(f"Volumen-Spike ({row['volume_ratio']:.1f}x)")
        if row.get("rsi_14", 50) > 80: reasons.append("RSI überkauft")
        if row.get("rsi_14", 50) < 20: reasons.append("RSI überverkauft")
        if row.get("volatility_20d", 0) > 0.5: reasons.append(f"Hohe Volatilität")
        return "; ".join(reasons) if reasons else "—"
 
    df["outlier_reason"] = df.apply(get_reason, axis=1)
    print(f"\n  Ausreißer — IsoForest: {df['outlier_iforest'].sum()} | RF: {df['outlier_rf'].sum()}")
    return df
 
 
# ─────────────────────────────────────────────
# 5. ML-VORHERSAGEMODELL
# ─────────────────────────────────────────────
 
def train_ml_model(df: pd.DataFrame, ticker: str, db_path: str = "finance.db"):
    """
    Random Forest Klassifikation: Steigt der Kurs morgen?
 
    ══════════════════════════════════════════════════════════════
    WARUM KEINE ROHDATEN ALS FEATURES?
    ══════════════════════════════════════════════════════════════
    Rohe Aktienkurse (close, open, high, low) sind NICHT-STATIONÄR:
    Sie steigen langfristig immer, haben also einen Trend.
    Ein Modell das auf Rohdaten trainiert wird, lernt nur diesen
    Trend — nicht echte Muster. Es würde auf neuen Daten versagen.
 
    Stationäre Alternativen die wir verwenden:
      1. Log-Returns        → relative Veränderung, kein Trend
      2. Technische Ind.    → RSI, MACD, BB-Position (normiert 0–1)
      3. Z-Score (rolling)  → jeder Wert relativ zu seinem eigenen
                              20-Tage-Mittelwert und -Standardabw.
      4. Verhältnisse       → volume_ratio, ma_distance (dimensionslos)
 
    Das Modell sieht NIEMALS einen absoluten Preis —
    nur abgeleitete, stationäre Kennzahlen.
 
    ══════════════════════════════════════════════════════════════
    FEATURE-AUSWAHL
    ══════════════════════════════════════════════════════════════
    Alle Features sind stationär und preisunabhängig:
 
      rsi_14          – Momentum 0–100, kein Preis
      macd            – Differenz zweier EMAs, kein Preis
      macd_signal     – EMA des MACD
      volatility_20d  – annualisierte Vola als Dezimalzahl
      volume_ratio    – heute / MA20-Volumen (dimensionslos)
      momentum_10     – 10-Tage Rendite in %
      log_return      – tägliche Log-Rendite
      atr_14          – Average True Range (wird z-normiert)
      return_5d       – 5-Tage kumulierte Rendite
      return_21d      – 21-Tage kumulierte Rendite
      bb_position     – (close - BB_lower) / BB_range → 0 bis 1
      ma_distance     – (MA20 - MA50) / MA50 → relativ, kein Preis
      rsi_zscore      – RSI z-normiert über 20-Tage-Fenster
      vol_zscore      – Volumen z-normiert über 20-Tage-Fenster
 
    ══════════════════════════════════════════════════════════════
    TRAIN/TEST SPLIT
    ══════════════════════════════════════════════════════════════
    TimeSeriesSplit — kein zufälliges Aufteilen!
    Bei Zeitreihen würde zufälliges Splitten bedeuten, dass
    zukünftige Daten ins Training einfließen (Data Leakage).
    Wir nehmen immer das letzte Fold als Testset.
 
    Label:
        ml_label = 1  wenn morgen Close > heute Close
        ml_label = 0  wenn morgen Close ≤ heute Close
    """
    import json
 
    if not SKLEARN_AVAILABLE:
        print("  scikit-learn nicht verfügbar — ML-Schritt übersprungen.")
        return
 
    print(f"\n  ML-Modell wird trainiert für {ticker}...")
    print("  → Nur stationäre Features (keine Rohdaten)")
 
    df2 = df.copy().reset_index(drop=True)
 
    # ── Stationäre Features berechnen ────────────────────────────
    # Bollinger-Position: wo liegt der Kurs relativ zum BB?
    # Ergebnis: 0 = am unteren Band, 1 = am oberen Band
    # → dimensionslos, kein absoluter Preis
    bb_range = df2["bollinger_upper"] - df2["bollinger_lower"]
    df2["bb_position"] = np.where(
        bb_range > 0,
        (df2["close"] - df2["bollinger_lower"]) / bb_range,
        0.5
    )
 
    # MA-Abstand: relative Differenz MA20 zu MA50
    # → dimensionslos, kein absoluter Preis
    df2["ma_distance"] = np.where(
        df2["ma_50"] > 0,
        (df2["ma_20"] - df2["ma_50"]) / df2["ma_50"],
        0.0
    )
 
    # Rolling Z-Score für RSI und Volumen über 20 Tage
    # → normiert jeden Wert relativ zu seinem eigenen Kurzzeitverhalten
    # → entfernt eventuelle Niveauunterschiede zwischen Aktien
    df2["rsi_zscore"] = (
        (df2["rsi_14"] - df2["rsi_14"].rolling(20).mean()) /
        (df2["rsi_14"].rolling(20).std() + 1e-9)
    )
    df2["vol_zscore"] = (
        (df2["volume_ratio"] - df2["volume_ratio"].rolling(20).mean()) /
        (df2["volume_ratio"].rolling(20).std() + 1e-9)
    )
 
    # ── Feature-Liste — AUSSCHLIESSLICH stationäre Werte ─────────
    feature_cols = [
        # Rendite-basiert (stationär per Definition)
        "log_return",        # ln(P_t / P_{t-1})
        "return_5d",         # kumulierte 5-Tage Rendite
        "return_21d",        # kumulierte 21-Tage Rendite
        "momentum_10",       # (P_t - P_{t-10}) / P_{t-10}
        # Momentum-Indikatoren (normiert 0–100 bzw. dimensionslos)
        "rsi_14",            # Relative Strength Index
        "rsi_zscore",        # RSI z-normiert über 20 Tage
        "macd",              # EMA12 - EMA26
        "macd_signal",       # 9-Tage EMA des MACD
        # Volatilität (als Dezimalzahl, kein Preis)
        "volatility_20d",    # annualisierte Vola
        "atr_14",            # Average True Range (wird StandardScaler-normiert)
        # Verhältnis-Features (dimensionslos)
        "bb_position",       # Position im Bollinger-Band (0–1)
        "ma_distance",       # (MA20 - MA50) / MA50
        "volume_ratio",      # Tagesvolumen / MA20-Volumen
        "vol_zscore",        # Volumen z-normiert über 20 Tage
    ]
 
    # Label: steigt Kurs am nächsten Tag?
    # Wichtig: close wird NUR für das Label verwendet, nie als Feature!
    df2["ml_label"] = (df2["close"].shift(-1) > df2["close"]).astype(int)
 
    df_clean = df2[feature_cols + ["ml_label", "date", "close"]].dropna()
    df_clean = df_clean.iloc[:-1]  # letzten Tag: kein Label bekannt
 
    if len(df_clean) < 60:
        print("  Zu wenige Daten für ML-Training.")
        return
 
    X = df_clean[feature_cols].values
    y = df_clean["ml_label"].values
 
    # ── TimeSeriesSplit — kein Data Leakage ───────────────────────
    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X))
    train_idx, test_idx = splits[-1]
 
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
 
    # StandardScaler normiert jeden Feature auf Mittelwert=0, Std=1
    # Fit NUR auf Trainingsdaten — nie auf Testdaten!
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
 
    # ── Random Forest ─────────────────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=5,    # verhindert Overfitting
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train_s, y_train)
 
    y_pred  = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]
 
    accuracy  = accuracy_score(y_test, y_pred)
    cm        = confusion_matrix(y_test, y_pred).tolist()
    report    = classification_report(y_test, y_pred, output_dict=True)
    precision = report["1"]["precision"]
    recall    = report["1"]["recall"]
 
    importances = dict(zip(feature_cols, model.feature_importances_.tolist()))
    sorted_imp  = sorted(importances.items(), key=lambda x: -x[1])
 
    print(f"\n  ML-Ergebnis für {ticker}:")
    print(f"    Accuracy  : {accuracy*100:.1f}%")
    print(f"    Precision : {precision*100:.1f}%")
    print(f"    Recall    : {recall*100:.1f}%")
    print(f"    Features  : {len(feature_cols)} (alle stationär)")
    print(f"    Testgröße : {len(y_test)} Tage")
    print("\n  Feature Importance (Top 6):")
    for feat, imp in sorted_imp[:6]:
        print(f"    {feat:<22} {'█'*int(imp*40)} {imp:.3f}")
 
    # ── Backtest ──────────────────────────────────────────────────
    test_dates         = df_clean["date"].iloc[test_idx].tolist()
    test_closes        = df_clean["close"].iloc[test_idx].values
    daily_returns_all  = np.diff(test_closes) / test_closes[:-1]
    daily_returns_ml   = np.where(y_pred[:-1] == 1, daily_returns_all, 0.0)
    cumret_model       = np.cumprod(1 + daily_returns_ml) * 100
    cumret_bah         = np.cumprod(1 + daily_returns_all) * 100
 
    backtest = {
        "dates":        [str(d) for d in test_dates[1:]],
        "model":        [round(v, 2) for v in cumret_model.tolist()],
        "buy_and_hold": [round(v, 2) for v in cumret_bah.tolist()],
        "final_model":  round(float(cumret_model[-1]), 2),
        "final_bah":    round(float(cumret_bah[-1]), 2),
    }
 
    # ── Vorhersage für morgen ─────────────────────────────────────
    last_row    = df2[feature_cols].dropna().iloc[-1].values.reshape(1, -1)
    last_scaled = scaler.transform(last_row)
    tomorrow_prob = float(model.predict_proba(last_scaled)[0][1])
    tomorrow_pred = int(model.predict(last_scaled)[0])
 
    print(f"\n  Vorhersage für morgen:")
    print(f"    Wahrscheinlichkeit Kursanstieg: {tomorrow_prob*100:.1f}%")
    print(f"    Signal: {'↑ Steigt' if tomorrow_pred==1 else '↓ Fällt'}")
 
    # ── In Datenbank speichern ────────────────────────────────────
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ml_results (
            ticker              TEXT PRIMARY KEY,
            accuracy            REAL,
            precision_score     REAL,
            recall_score        REAL,
            test_size           INTEGER,
            train_size          INTEGER,
            tomorrow_prob       REAL,
            tomorrow_signal     INTEGER,
            feature_importances TEXT,
            backtest            TEXT,
            confusion_matrix    TEXT,
            last_updated        TEXT
        )
    """)
    conn.execute("""
        INSERT OR REPLACE INTO ml_results VALUES (?,?,?,?,?,?,?,?,?,?,?,datetime('now'))
    """, (
        ticker,
        round(accuracy, 4),
        round(precision, 4),
        round(recall, 4),
        len(y_test),
        len(y_train),
        round(tomorrow_prob, 4),
        tomorrow_pred,
        json.dumps(dict(sorted_imp)),
        json.dumps(backtest),
        json.dumps(cm),
    ))
    conn.commit()
    conn.close()
    print(f"  ML-Ergebnisse gespeichert → {db_path}")
 
 
# ─────────────────────────────────────────────
# 6. DATENBANK
# ─────────────────────────────────────────────
 
def init_db(db_path: str = "finance.db"):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stock_features (
            ticker           TEXT,
            date             TEXT,
            open             REAL, high REAL, low REAL, close REAL, volume REAL,
            log_return       REAL, return_5d REAL, return_21d REAL,
            ma_20            REAL, ma_50 REAL, ema_12 REAL, ema_26 REAL,
            macd             REAL, macd_signal REAL,
            volatility_20d   REAL, bollinger_upper REAL, bollinger_lower REAL,
            atr_14           REAL, rsi_14 REAL, momentum_10 REAL,
            volume_ma_20     REAL, volume_ratio REAL, obv INTEGER,
            outlier_iforest  INTEGER DEFAULT 0, outlier_score REAL DEFAULT 0,
            outlier_rf       INTEGER DEFAULT 0, outlier_prob REAL DEFAULT 0,
            outlier_reason   TEXT DEFAULT '',
            PRIMARY KEY (ticker, date)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            ticker TEXT PRIMARY KEY, last_updated TEXT, row_count INTEGER
        )
    """)
    conn.commit()
    conn.close()
 
 
def save_to_db(df: pd.DataFrame, ticker: str, db_path: str = "finance.db"):
    conn = sqlite3.connect(db_path)
    df_db = df.copy()
    df_db["ticker"] = ticker
    df_db["date"]   = df_db["date"].astype(str)
    conn.execute("DELETE FROM stock_features WHERE ticker = ?", (ticker,))
    df_db.to_sql("stock_features", conn, if_exists="append", index=False)
    conn.execute("INSERT OR REPLACE INTO metadata VALUES (?,datetime('now'),?)", (ticker, len(df)))
    conn.commit()
    conn.close()
    print(f"  {ticker}: {len(df)} Zeilen gespeichert → {db_path}")
 
 
# ─────────────────────────────────────────────
# 7. HAUPTPROGRAMM
# ─────────────────────────────────────────────
 
TICKERS = ["AAPL", "MSFT", "TSLA","GOOG", "META", "AMZN", "NFLX", "COIN", "PLTR", "AMC", "BMW.DE", "SAP.DE", "NVDA", "GME", "SIE.DE", "ALV.DE", "DTE.DE", "VOW3.DE","MBG.DE", "BAYN.DE", "DBK.DE"]  # ← eigene Aktien eintragen
DB_PATH  = "finance.db"
PERIOD   = "1y"
 
if __name__ == "__main__":
    print("=" * 50)
    print("  Finanz-Dashboard Pipeline")
    print("=" * 50)
 
    init_db(DB_PATH)
 
    erfolgreich, fehlgeschlagen = [], []
 
    for ticker in TICKERS:
        print(f"\n{'─'*40}\n[{ticker}]")
        try:
            raw      = load_stock_data(ticker, period=PERIOD)
            cleaned  = clean_data(raw)
            if len(cleaned) < 60:
                print(f"  ⚠️  {ticker}: zu wenige gültige Tage ({len(cleaned)}) – übersprungen.")
                fehlgeschlagen.append(ticker)
                continue
            features = compute_features(cleaned)
            features = detect_outliers(features)
            save_to_db(features, ticker, DB_PATH)
            train_ml_model(features, ticker, DB_PATH)
            erfolgreich.append(ticker)
        except Exception as e:
            # Ein fehlerhafter Ticker (Paywall/Delisting/Netzfehler)
            # bricht NICHT die ganze Pipeline ab.
            print(f"  ❌  {ticker}: übersprungen – {e}")
            fehlgeschlagen.append(ticker)
 
    print("\n" + "="*50)
    print("✓ Pipeline abgeschlossen.")
    print(f"  Erfolgreich   : {len(erfolgreich)} → {', '.join(erfolgreich) if erfolgreich else '—'}")
    print(f"  Fehlgeschlagen: {len(fehlgeschlagen)} → {', '.join(fehlgeschlagen) if fehlgeschlagen else '—'}")
    print(f"  Datenbank: {DB_PATH}")
    print("  Starte jetzt: python dashboard_server.py")
 
