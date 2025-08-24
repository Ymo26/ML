#!/usr/bin/env python3
"""
CNN+LSTM Price Forecasting — Quantile (Pinball) Loss Edition
- Duplicate-date safe, same features/splits/windows as before
- Uses Quantile (Pinball) loss by default (τ=0.5 → median forecast)
"""

import os, sys, math, warnings
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------- CONFIG ---------------------------
DATA_PATH   = Path("/Users/luoyi/Desktop/ML Research/hotel_booking.csv")  # your CSV
DATE_COL    = None   # e.g., "Date" or "reservation_status_date"; leave None to auto-detect
PRICE_COL   = None   # e.g., "close" or "adr"; leave None to auto-detect

LOOKBACK    = 60     # history per sample
HORIZON     = 30     # predict t+30
TEST_SPLIT  = 0.15   # last 15% as test
VAL_SPLIT   = 0.15   # last 15% (of pre-test) for validation

EPOCHS      = 60
BATCH       = 64
USE_BASELINES = True  # also train LSTM & GRU
RANDOM_SEED = 1337

# 🔁 Loss: Quantile (Pinball) – set tau in (0,1); e.g., 0.1/0.5/0.9
PINBALL_TAU = 0.5
# -------------------------------------------------------------

warnings.filterwarnings("ignore", category=FutureWarning)
np.random.seed(RANDOM_SEED)

# ML/DL
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import tensorflow as tf
    tf.random.set_seed(RANDOM_SEED)
    from tensorflow.keras import layers, models, callbacks
except Exception as e:
    print("[ERROR] TensorFlow not available. Try: pip install tensorflow-macos tensorflow-metal")
    raise

# --------------------------- Loss ---------------------------
def pinball_loss(tau: float):
    tau = float(tau)
    def _loss(y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(tau*e, (tau-1)*e))
    return _loss

# --------------------------- Helpers ---------------------------
def infer_date_price(df: pd.DataFrame) -> Tuple[str, str]:
    if DATE_COL and DATE_COL in df.columns:
        date_col = DATE_COL
    else:
        cand_dates = ["Date","date","timestamp","Timestamp","reservation_status_date","arrival_date","booking_date"]
        date_col = next((c for c in cand_dates if c in df.columns), None)

    if PRICE_COL and PRICE_COL in df.columns:
        price_col = PRICE_COL
    else:
        cand_prices = ["adr","ADR","close","Close","price","Price","Adj Close","adj_close","rate","amount"]
        price_col = next((c for c in cand_prices if c in df.columns), None)

    if not date_col or not price_col:
        raise ValueError(f"Could not find DATE or PRICE column. Saw: {list(df.columns)[:20]}")
    return date_col, price_col

def add_tech_indicators(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    s = df[price_col].clip(lower=1e-6)
    out = pd.DataFrame(index=df.index)
    out["price"]      = s
    out["ret_1"]      = s.pct_change()
    out["log_ret_1"]  = np.log(s).diff()

    for w in (5, 10, 20, 30, 60):
        out[f"ma_{w}"]  = s.rolling(w).mean()
        out[f"std_{w}"] = s.rolling(w).std()

    # RSI(14)
    delta = s.diff()
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=df.index).rolling(14).mean()
    roll_dn = pd.Series(loss, index=df.index).rolling(14).mean()
    rs = roll_up / (roll_dn + 1e-12)
    out["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    # MACD (12,26,9)
    ema12 = s.ewm(span=12, adjust=False).mean()
    ema26 = s.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    out["macd"]  = macd
    out["macd9"] = macd.ewm(span=9, adjust=False).mean()

    # Bollinger(20,2)
    ma20  = s.rolling(20).mean()
    std20 = s.rolling(20).std()
    out["bb_up"] = ma20 + 2*std20
    out["bb_dn"] = ma20 - 2*std20

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out.dropna(inplace=True)
    return out

def make_sequences(X: np.ndarray, y: np.ndarray, lookback=60, horizon=1):
    Xs, ys = [], []
    for i in range(lookback, len(X) - horizon + 1):
        Xs.append(X[i - lookback:i])
        ys.append(y[i + horizon - 1])
    return np.array(Xs), np.array(ys)

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, prev: np.ndarray) -> float:
    st = np.sign(y_true - prev)
    sp = np.sign(y_pred - prev)
    mask = ~np.isnan(st)
    return float(np.mean((st[mask] == sp[mask])))

# --------------------------- Load & Fix Duplicates ---------------------------
df_raw = pd.read_csv(DATA_PATH)
date_col, price_col = infer_date_price(df_raw)

df_raw[date_col] = pd.to_datetime(df_raw[date_col], errors="coerce")
df_raw = df_raw[[date_col, price_col]].dropna().sort_values(date_col)
df_raw = df_raw.rename(columns={date_col: "Date", price_col: "Price"})

# Collapse duplicate timestamps to one row per day
df_raw["Date"] = df_raw["Date"].dt.floor("D")
daily = (df_raw.groupby("Date", as_index=True)["Price"]
               .mean()
               .to_frame()
               .sort_index())

# Daily calendar & gap fill
daily = daily.asfreq("D")
daily["Price"] = daily["Price"].interpolate(limit_direction="both")

print("[INFO] Built daily series with unique dates.")
print(daily.head().to_string())

# --------------------------- Features ---------------------------
feat = add_tech_indicators(daily, "Price")
feat["target"] = feat["price"]  # predict raw price at t+HORIZON

# --------------------------- Splits & scaling ---------------------------
n = len(feat)
test_start = int(n * (1 - TEST_SPLIT))
val_start  = int(test_start * (1 - VAL_SPLIT))  # chronological

X_cols = [c for c in feat.columns if c != "target"]
y_col  = "target"

X_all = feat[X_cols].values
y_all = feat[y_col].values.reshape(-1, 1)

# Scale on train only (no leakage)
scaler_X = StandardScaler().fit(X_all[:val_start])
scaler_y = MinMaxScaler().fit(y_all[:val_start])

X_sc = scaler_X.transform(X_all)
y_sc = scaler_y.transform(y_all)

# Window AFTER scaling
X_tr, y_tr = make_sequences(X_sc[:val_start],           y_sc[:val_start],           LOOKBACK, HORIZON)
X_va, y_va = make_sequences(X_sc[val_start:test_start], y_sc[val_start:test_start], LOOKBACK, HORIZON)
X_te, y_te = make_sequences(X_sc[test_start:],          y_sc[test_start:],          LOOKBACK, HORIZON)

print("Shapes:")
print(" X_tr, y_tr:", X_tr.shape, y_tr.shape)
print(" X_va, y_va:", X_va.shape, y_va.shape)
print(" X_te, y_te:", X_te.shape, y_te.shape)

# --------------------------- Models ---------------------------
def build_cnn_lstm(input_shape, loss_obj):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv1D(32, kernel_size=3, padding="causal", activation="relu")(inp)
    x = layers.Conv1D(32, kernel_size=3, padding="causal", activation="relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)
    m = models.Model(inp, out)
    m.compile(optimizer="adam", loss=loss_obj)
    return m

def build_lstm(input_shape, loss_obj):
    inp = layers.Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.LSTM(32)(x)
    out = layers.Dense(1)(x)
    m = models.Model(inp, out)
    m.compile(optimizer="adam", loss=loss_obj)
    return m

def build_gru(input_shape, loss_obj):
    inp = layers.Input(shape=input_shape)
    x = layers.GRU(64, return_sequences=True)(inp)
    x = layers.GRU(32)(x)
    out = layers.Dense(1)(x)
    m = models.Model(inp, out)
    m.compile(optimizer="adam", loss=loss_obj)
    return m

input_shape = (X_tr.shape[1], X_tr.shape[2])
loss_obj = pinball_loss(PINBALL_TAU)
cb = [callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss")]

models_to_train = [("CNN+LSTM", lambda: build_cnn_lstm(input_shape, loss_obj))]
if USE_BASELINES:
    models_to_train += [
        ("LSTM", lambda: build_lstm(input_shape, loss_obj)),
        ("GRU",  lambda: build_gru(input_shape, loss_obj)),
    ]

# --------------------------- Train/Eval ---------------------------
OUT = Path(f"./outputs_cnnlstm_pinball_tau{str(PINBALL_TAU).replace('.','_')}")
OUT.mkdir(parents=True, exist_ok=True)

def evaluate(trained, name: str):
    # Predict & invert scaling
    yhat_te = trained.predict(X_te, verbose=0)
    yhat  = scaler_y.inverse_transform(yhat_te).reshape(-1)
    ytrue = scaler_y.inverse_transform(y_te).reshape(-1)

    # Align dates for test sequences
    seq_start = test_start + LOOKBACK + HORIZON - 1
    idx = feat.index[seq_start: seq_start + len(ytrue)]
    prev = feat["target"].shift(1).loc[idx].values

    # Metrics
    mae  = mean_absolute_error(ytrue, yhat)
    rmse = math.sqrt(mean_squared_error(ytrue, yhat))
    mape = float(np.mean(np.abs((ytrue - yhat) / np.maximum(1e-6, ytrue)))) * 100.0
    r2   = r2_score(ytrue, yhat)
    dira = directional_accuracy(ytrue.reshape(-1,1), yhat.reshape(-1,1), prev.reshape(-1,1))

    # Save CSV & plots
    pd.DataFrame({"date": idx.astype(str), "y_true": ytrue, "y_pred": yhat}).to_csv(OUT / f"pred_{name}.csv", index=False)

    plt.figure(figsize=(10,4))
    plt.plot(idx, ytrue, label="Actual")
    plt.plot(idx, yhat,  label="Predicted")
    plt.title(f"{name} — Actual vs Predicted (H={HORIZON}, pinball τ={PINBALL_TAU})")
    plt.xlabel("Date"); plt.ylabel("Price")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT / f"{name}_actual_vs_pred.png", dpi=120)
    try: plt.show()
    except: pass

    resid = ytrue - yhat
    plt.figure(figsize=(10,4))
    plt.plot(idx, resid)
    plt.title(f"{name} — Residuals (Test)")
    plt.xlabel("Date"); plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(OUT / f"{name}_residuals.png", dpi=120)
    try: plt.show()
    except: pass

    return {"Model": name, "Loss": f"pinball_tau={PINBALL_TAU}", "MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2, "Directional_Acc": dira}

results = []
for name, builder in models_to_train:
    print(f"\n[TRAIN] {name} (pinball τ={PINBALL_TAU}) ...")
    m = builder()
    m.fit(X_tr, y_tr, validation_data=(X_va, y_va),
          epochs=EPOCHS, batch_size=BATCH, callbacks=cb, verbose=1)
    res = evaluate(m, name)
    results.append(res)
    try: m.save(OUT / f"{name.replace('+','_')}.keras")
    except Exception as e: print("[WARN] save failed:", e)

summary = pd.DataFrame(results).sort_values("RMSE")
print("\n=== Metric Summary (lower MAE/RMSE/MAPE better; higher R²/Directional_Acc better) ===")
print(summary.to_string(index=False))
summary.to_csv(OUT / "metrics_summary.csv", index=False)

print(f"\n[DONE] Outputs saved in: {OUT.resolve()}")
