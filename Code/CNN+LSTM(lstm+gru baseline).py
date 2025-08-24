#!/usr/bin/env python3
"""
ADR (Price) Forecasting via LSTM / GRU / CNN+LSTM

- Robust CSV loader for hotel_booking-like data (single date column OR year+month+day),
  tolerant to month names/numbers and varied column cases.
- Pipeline: feature engineering (safe), ADF stationarity check, time-aware split,
  sequence windowing (LOOKBACK), multi-step horizon (HORIZON), model training,
  evaluation (MAE/RMSE/R2/Directional Accuracy), and plots/artifacts to ./outputs.

Run:
  source "/Users/luoyi/Desktop/ML Research/.venv/bin/activate"
  python adr_forecasting.py
"""

import os, sys, math, warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------ Display shim --------------------------------------
try:
    from IPython.display import display  # type: ignore
except Exception:
    def display(x):
        try:
            print(x.to_string())
        except Exception:
            print(x)
# ----------------------------------------------------------------------------------

# ------------------------------ Config --------------------------------------------
# Robust discovery of hotel_booking.csv:
# 1) common desktop path
# 2) /mnt/data (when running in hosted envs)
# 3) same folder as this script
script_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
CANDIDATE_PATHS = [
    Path("/Users/luoyi/Desktop/ML Research/hotel_booking.csv"),
    Path("/mnt/data/hotel_booking.csv"),
    script_dir / "hotel_booking.csv",
]
DATA_PATH = next((p for p in CANDIDATE_PATHS if p.exists()), None)
if DATA_PATH is None:
    raise FileNotFoundError(
        "Could not find hotel_booking.csv. Put it next to this script or update CANDIDATE_PATHS."
    )
print(f"[INFO] Using data: {DATA_PATH}")

# Forecasting controls (imitate paper)
LOOKBACK = 60
HORIZON  = 30      # use 1 for next-day; 30 to mimic 30-day-ahead point forecast
TARGET   = "ADR"

EPOCHS = 50
BATCH  = 64
RANDOM_SEED = 1337
# ----------------------------------------------------------------------------------

warnings.filterwarnings("ignore", category=FutureWarning)
np.random.seed(RANDOM_SEED)

# Optional ADF test
try:
    from statsmodels.tsa.stattools import adfuller
    HAS_ADF = True
except Exception:
    HAS_ADF = False

# Deep learning
try:
    import tensorflow as tf
    tf.random.set_seed(RANDOM_SEED)
    from tensorflow.keras import layers, models, callbacks
    HAS_TF = True
except Exception:
    HAS_TF = False
    print("\n[INFO] TensorFlow not available. On Apple Silicon:")
    print("       pip install tensorflow-macos tensorflow-metal\n")

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------ Helpers -------------------------------------------
def _get_col(df: pd.DataFrame, *candidates):
    """Return the first existing column name (case-insensitive)."""
    names = list(df.columns)
    lower_map = {c.lower(): c for c in names}
    for cand in candidates:
        if cand in names:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None

def coerce_month_series(s: pd.Series) -> pd.Series:
    """Map month names/numbers (any case/whitespace) → 1..12."""
    month_map = {
        'january':1, 'february':2, 'march':3, 'april':4, 'may':5, 'june':6,
        'july':7, 'august':8, 'september':9, 'october':10, 'november':11, 'december':12
    }
    s_str = s.astype(str).str.strip().str.lower()
    mapped = s_str.map(month_map)
    numeric = pd.to_numeric(s_str, errors='coerce')
    return mapped.fillna(numeric)

def load_and_build_daily_adr(csv_path: Path) -> pd.DataFrame:
    df_raw = pd.read_csv(csv_path)

    # ADR/price column
    adr_col = _get_col(df_raw, 'adr', 'ADR', 'price', 'Price', 'amount', 'rate')
    if adr_col is None:
        raise ValueError("Could not find ADR/price column (looked for: adr/price/amount/rate).")

    # Case 1: single date column
    date_col = _get_col(df_raw, 'reservation_status_date', 'reservation_date',
                        'date', 'Date', 'booking_date', 'arrival_date')
    if date_col is not None:
        tmp = pd.DataFrame({
            'Date': pd.to_datetime(df_raw[date_col], errors='coerce'),
            'ADR' : pd.to_numeric(df_raw[adr_col], errors='coerce'),
        }).dropna(subset=['Date', 'ADR'])
        daily = (tmp.groupby('Date')['ADR'].mean()
                   .sort_index()
                   .asfreq('D')
                   .to_frame())
        daily['ADR'] = daily['ADR'].interpolate(limit_direction='both')
        if len(daily) == 0:
            raise ValueError("Parsed a date column but got zero rows after cleaning; inspect CSV.")
        return daily

    # Case 2: year + month + day (Kaggle hotel_bookings style)
    year_col  = _get_col(df_raw, 'arrival_date_year', 'year', 'Year')
    month_col = _get_col(df_raw, 'arrival_date_month', 'month', 'Month')
    day_col   = _get_col(df_raw, 'arrival_date_day_of_month', 'day', 'Day')

    if all([year_col, month_col, day_col]):
        y = pd.to_numeric(df_raw[year_col], errors='coerce')
        m = coerce_month_series(df_raw[month_col])
        d = pd.to_numeric(df_raw[day_col], errors='coerce')

        tmp = pd.DataFrame({
            'Date': pd.to_datetime(dict(year=y, month=m, day=d), errors='coerce'),
            'ADR' : pd.to_numeric(df_raw[adr_col], errors='coerce'),
        }).dropna(subset=['Date', 'ADR'])

        if tmp.empty:
            print("[DEBUG] Sample of raw month values:", df_raw[month_col].astype(str).head(10).tolist())
            raise ValueError("Could not construct valid dates from year/month/day.")

        daily = (tmp.groupby('Date')['ADR'].mean()
                   .sort_index()
                   .asfreq('D')
                   .to_frame())
        daily['ADR'] = daily['ADR'].interpolate(limit_direction='both')
        return daily

    raise ValueError(
        "No usable date columns found. Provide a single date column (e.g., reservation_status_date) "
        "or the triple arrival_date_year + arrival_date_month + arrival_date_day_of_month."
    )

def feature_engineering(daily: pd.DataFrame) -> pd.DataFrame:
    """Safe feature set: prevents ±inf from pct_change/log and drops NaNs."""
    feat = daily.copy()

    # avoid inf in returns/log
    adr_safe = feat["ADR"].clip(lower=1e-6)

    feat["ADR_return"]   = adr_safe.pct_change()
    feat["ADR_log"]      = np.log(adr_safe)
    feat["ADR_log_diff"] = feat["ADR_log"].diff()

    for w in [7, 14, 30]:
        feat[f"ADR_roll_mean_{w}"] = feat["ADR"].rolling(w).mean()
        feat[f"ADR_roll_std_{w}"]  = feat["ADR"].rolling(w).std()

    # Replace ±inf with NaN, then drop
    feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat.dropna(inplace=True)
    return feat

def run_adf(series: pd.Series):
    if not HAS_ADF:
        print("[INFO] statsmodels not installed; skipping ADF test.")
        return
    try:
        s = series.dropna().values
        stat, pval, usedlag, nobs, crit, icbest = adfuller(s, autolag="AIC")
        print(f"[ADF] Statistic={stat:.4f}, p-value={pval:.4f}, usedlag={usedlag}, nobs={nobs}")
        print(f"      Critical values: {crit}")
        print("      (p < 0.05 suggests stationarity)")
    except Exception as e:
        print("[WARN] ADF test failed:", e)

def make_sequences(X, y, lookback=60, horizon=1):
    Xs, ys = [], []
    for i in range(lookback, len(X) - horizon + 1):
        Xs.append(X[i - lookback : i])
        ys.append(y[i + horizon - 1])
    return np.array(Xs), np.array(ys)

def time_aware_split(feat: pd.DataFrame, feature_cols, target_col):
    X_all = feat[feature_cols].values
    y_all = feat[target_col].values.reshape(-1, 1)

    n = len(feat)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    X_train_raw = X_all[:train_end]
    y_train_raw = y_all[:train_end]

    X_val_raw   = X_all[train_end:val_end]
    y_val_raw   = y_all[train_end:val_end]

    X_test_raw  = X_all[val_end:]
    y_test_raw  = y_all[val_end:]

    scaler_X = StandardScaler().fit(X_train_raw)
    scaler_y = MinMaxScaler().fit(y_train_raw)

    X_train = scaler_X.transform(X_train_raw)
    y_train = scaler_y.transform(y_train_raw)

    X_val   = scaler_X.transform(X_val_raw)
    y_val   = scaler_y.transform(y_val_raw)

    X_test  = scaler_X.transform(X_test_raw)
    y_test  = scaler_y.transform(y_test_raw)

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            scaler_X, scaler_y, train_end, val_end, n)

# ------------------------------ Models --------------------------------------------
def build_lstm(input_shape):
    inp = layers.Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.LSTM(32)(x)
    out = layers.Dense(1)(x)
    m = models.Model(inp, out)
    m.compile(optimizer="adam", loss="mse")
    return m

def build_gru(input_shape):
    inp = layers.Input(shape=input_shape)
    x = layers.GRU(64, return_sequences=True)(inp)
    x = layers.GRU(32)(x)
    out = layers.Dense(1)(x)
    m = models.Model(inp, out)
    m.compile(optimizer="adam", loss="mse")
    return m

def build_cnn_lstm(input_shape):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv1D(filters=32, kernel_size=3, padding="causal", activation="relu")(inp)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1)(x)
    m = models.Model(inp, out)
    m.compile(optimizer="adam", loss="mse")
    return m

# ------------------------------ Evaluation ----------------------------------------
def evaluate_model(model, name, X_te, y_te, scaler_y, feat_df, lookback, horizon, target='ADR'):
    yhat_scaled = model.predict(X_te, verbose=0)
    yhat  = scaler_y.inverse_transform(yhat_scaled)
    ytrue = scaler_y.inverse_transform(y_te)

    mae  = mean_absolute_error(ytrue, yhat)
    rmse = math.sqrt(mean_squared_error(ytrue, yhat))
    r2   = r2_score(ytrue, yhat)

    # Align test index back to original feat index
    n_total   = len(feat_df)
    test_start = int(n_total * 0.85)
    seq_start  = test_start + lookback + horizon - 1
    idx = feat_df.index[seq_start : seq_start + len(ytrue)]

    # Directional accuracy vs previous actual
    prev_vals  = feat_df[target].shift(1).loc[idx].values.reshape(-1, 1)
    sign_true  = np.sign(ytrue - prev_vals)
    sign_pred  = np.sign(yhat  - prev_vals)
    directional_acc = float(np.mean((sign_true == sign_pred)[~np.isnan(sign_true)]))

    return {
        "name": name,
        "y_true": ytrue.flatten(),
        "y_pred": yhat.flatten(),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "directional_acc": directional_acc,
        "index": idx.astype(str),  # for CSV
    }

def plot_series(idx, y_true, y_pred, title, outpath_prefix):
    # Actual vs Predicted
    plt.figure(figsize=(10, 4))
    plt.plot(idx, y_true, label="Actual")
    plt.plot(idx, y_pred, label="Predicted")
    plt.title(title + " — Actual vs Predicted")
    plt.xlabel("Date"); plt.ylabel("ADR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outpath_prefix}_actual_vs_pred.png", dpi=120)
    try: plt.show()
    except Exception: pass

    # Residuals
    residuals = np.array(y_true) - np.array(y_pred)
    plt.figure(figsize=(10, 4))
    plt.plot(idx, residuals)
    plt.title(title + " — Residuals (Test)")
    plt.xlabel("Date"); plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(f"{outpath_prefix}_residuals.png", dpi=120)
    try: plt.show()
    except Exception: pass

# ------------------------------ Main ----------------------------------------------
def main():
    print(f"[INFO] Using data: {DATA_PATH}")

    daily = load_and_build_daily_adr(DATA_PATH)
    print("\nDaily ADR head:")
    display(daily.head(10))
    print("Rows:", len(daily))

    feat = feature_engineering(daily)
    print("\nEngineered features tail:")
    display(feat.tail(5))

    # ADF on log-diff
    run_adf(feat["ADR_log_diff"])

    # Feature set (multivariate)
    feature_cols = [
        "ADR", "ADR_return", "ADR_log_diff",
        "ADR_roll_mean_7", "ADR_roll_std_7",
        "ADR_roll_mean_14", "ADR_roll_std_14",
        "ADR_roll_mean_30", "ADR_roll_std_30",
    ]

    # Finite-guard before split
    cols = feature_cols + [TARGET]
    arr  = feat[cols].to_numpy()
    mask = np.isfinite(arr).all(axis=1)
    if mask.sum() < len(mask):
        print(f"[CLEAN] Dropping {len(mask) - int(mask.sum())} rows with NaN/Inf in features/target.")
    feat = feat.loc[mask].copy()

    if len(feat) < (LOOKBACK + HORIZON + 50):
        print("[WARN] Few rows after cleaning; consider lowering LOOKBACK/HORIZON.")

    (X_train, y_train, X_val, y_val, X_test, y_test,
     scaler_X, scaler_y, train_end, val_end, n_total) = time_aware_split(feat, feature_cols, TARGET)

    # Windowing
    X_tr, y_tr = make_sequences(X_train, y_train, LOOKBACK, HORIZON)
    X_va, y_va = make_sequences(X_val,   y_val,   LOOKBACK, HORIZON)
    X_te, y_te = make_sequences(X_test,  y_test,  LOOKBACK, HORIZON)

    print("\nShapes:")
    print("  X_tr, y_tr:", X_tr.shape, y_tr.shape)
    print("  X_va, y_va:", X_va.shape, y_va.shape)
    print("  X_te, y_te:", X_te.shape, y_te.shape)

    OUT = Path("./outputs")
    OUT.mkdir(parents=True, exist_ok=True)

    if not HAS_TF:
        print("\n[EXIT] TensorFlow not found — install and re-run.")
        sys.exit(0)

    input_shape = (X_tr.shape[1], X_tr.shape[2])
    mdl_lstm    = build_lstm(input_shape)
    mdl_gru     = build_gru(input_shape)
    mdl_cnnlstm = build_cnn_lstm(input_shape)

    es = callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss")

    print("\n[TRAIN] LSTM ...")
    mdl_lstm.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                 epochs=EPOCHS, batch_size=BATCH, callbacks=[es], verbose=1)

    print("\n[TRAIN] GRU ...")
    mdl_gru.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                epochs=EPOCHS, batch_size=BATCH, callbacks=[es], verbose=1)

    print("\n[TRAIN] CNN+LSTM ...")
    mdl_cnnlstm.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                    epochs=EPOCHS, batch_size=BATCH, callbacks=[es], verbose=1)

    # Evaluate
    results = []
    for model, name in [(mdl_lstm, "LSTM"), (mdl_gru, "GRU"), (mdl_cnnlstm, "CNN+LSTM")]:
        r = evaluate_model(model, name, X_te, y_te, scaler_y, feat, LOOKBACK, HORIZON, TARGET)
        results.append(r)
        pred_df = pd.DataFrame({"date": r["index"], "y_true": r["y_true"], "y_pred": r["y_pred"]})
        pred_df.to_csv(OUT / f"predictions_{name}.csv", index=False)
        plot_series(pred_df["date"], pred_df["y_true"], pred_df["y_pred"], name, str(OUT / f"{name}"))

    # Summary table
    summary = pd.DataFrame([
        {"Model": r["name"], "MAE": r["mae"], "RMSE": r["rmse"], "R2": r["r2"], "Directional_Acc": r["directional_acc"]}
        for r in results
    ]).sort_values("RMSE")
    print("\nMetric Summary (lower MAE/RMSE better, higher R²/Directional_Acc better):")
    display(summary)
    summary.to_csv(OUT / "metrics_summary.csv", index=False)

    # Save models
    try:
        mdl_lstm.save(OUT / "lstm_model.keras")
        mdl_gru.save(OUT / "gru_model.keras")
        mdl_cnnlstm.save(OUT / "cnn_lstm_model.keras")
    except Exception as e:
        print("[WARN] Model saving failed:", e)

    print(f"\n[DONE] Artifacts written to: {OUT.resolve()}")
    print("      - metrics_summary.csv")
    print("      - predictions_*.csv")
    print("      - *_actual_vs_pred.png, *_residuals.png")
    print("      - *.keras (saved models)")

if __name__ == "__main__":
    main()
