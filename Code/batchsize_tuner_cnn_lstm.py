#!/usr/bin/env python3
"""
Batch-size tuner for CNN+LSTM (ReLU) with Huber loss.
- Duplicate-date safe preprocessing for your hotel_booking.csv
- LR √-scaling with batch size (Adam-friendly)
- EarlyStopping for the sweep, ReduceLROnPlateau on full train
- Skips OOM runs automatically

Outputs:
  ./out_batch_tune/  (per-BS results, plots, predictions, metrics)
  ./out_batch_tune/bs_sweep_summary.csv
"""

import math, warnings
from pathlib import Path
from typing import Tuple, List
import numpy as np, pandas as pd, matplotlib.pyplot as plt

# ======================= CONFIG =======================
DATA_PATH   = Path("/Users/luoyi/Desktop/ML Research/hotel_booking.csv")
DATE_COL    = None   # e.g., "reservation_status_date" or "Date"; None -> auto
PRICE_COL   = None   # e.g., "adr" or "Price"; None -> auto

LOOKBACK    = 60     # history window
HORIZON     = 30     # predict t+30
TEST_SPLIT  = 0.15
VAL_SPLIT   = 0.15

# Sweep these batch sizes (kept modest for 8GB M2). Adjust if you want.
BATCH_GRID: List[int] = [16, 24, 32, 48, 64, 96, 128, 192]

# Training budget
EPOCHS_TUNE = 12     # short sweep
EPOCHS_FULL = 60     # full training for the selected BS

# Optimizer / loss
BASE_LR     = 5e-4   # reference LR for REF_BS with Huber + ReLU (stable)
REF_BS      = 64     # reference batch size for LR scaling
HUBER_DELTA = 1.0
CLIPNORM    = 1.0
MIN_LR      = 1e-6
PLATEAU_FACTOR   = 0.5
PLATEAU_PATIENCE = 4

SEED = 1337
# ======================================================

warnings.filterwarnings("ignore", category=FutureWarning)
np.random.seed(SEED)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
tf.random.set_seed(SEED)
from tensorflow.keras import layers, models, callbacks, losses, optimizers

# ------------------ helpers ------------------
def infer_date_price(df: pd.DataFrame) -> Tuple[str, str]:
    date_cands = [DATE_COL, "Date","date","timestamp","Timestamp",
                  "reservation_status_date","arrival_date","booking_date"]
    price_cands= [PRICE_COL,"adr","ADR","close","Close","price","Price",
                  "Adj Close","adj_close","rate","amount"]
    date_cands  = [c for c in date_cands  if c]
    price_cands = [c for c in price_cands if c]
    d = next((c for c in date_cands  if c in df.columns), None)
    p = next((c for c in price_cands if c in df.columns), None)
    if not d or not p:
        raise ValueError(f"Could not find DATE/PRICE columns. Saw: {list(df.columns)[:20]}")
    return d, p

def add_indicators(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    s = df[price_col].clip(lower=1e-6)
    out = pd.DataFrame(index=df.index)
    out["price"]      = s
    out["ret_1"]      = s.pct_change()
    out["log_ret_1"]  = np.log(s).diff()
    for w in (5, 10, 20, 30, 60):
        out[f"ma_{w}"]  = s.rolling(w).mean()
        out[f"std_{w}"] = s.rolling(w).std()
    # RSI(14)
    d = s.diff(); up = np.where(d>0, d, 0.0); dn = np.where(d<0, -d, 0.0)
    rsi = 100.0 - 100.0/(1.0 + (pd.Series(up,index=df.index).rolling(14).mean()
                                /(pd.Series(dn,index=df.index).rolling(14).mean()+1e-12)))
    out["rsi_14"] = rsi
    # MACD (12,26,9)
    ema12 = s.ewm(span=12,adjust=False).mean()
    ema26 = s.ewm(span=26,adjust=False).mean()
    macd  = ema12 - ema26
    out["macd"]  = macd
    out["macd9"] = macd.ewm(span=9,adjust=False).mean()
    # Bollinger(20,2)
    ma20, sd20 = s.rolling(20).mean(), s.rolling(20).std()
    out["bb_up"] = ma20 + 2*sd20
    out["bb_dn"] = ma20 - 2*sd20
    out.replace([np.inf,-np.inf], np.nan, inplace=True)
    out.dropna(inplace=True)
    return out

def make_sequences(X: np.ndarray, y: np.ndarray, lookback=60, horizon=1):
    Xs, ys = [], []
    for i in range(lookback, len(X) - horizon + 1):
        Xs.append(X[i - lookback:i])
        ys.append(y[i + horizon - 1])
    return np.array(Xs), np.array(ys)

def build_cnn_lstm_relu(input_shape, lr: float):
    opt  = optimizers.Adam(learning_rate=lr, clipnorm=CLIPNORM)
    loss = losses.Huber(delta=HUBER_DELTA)
    inp = layers.Input(shape=input_shape)
    x = layers.Conv1D(32, 3, padding="causal", activation="relu")(inp)
    x = layers.Conv1D(32, 3, padding="causal", activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)
    m = models.Model(inp, out)
    m.compile(optimizer=opt, loss=loss)
    return m

def sqrt_lr_scale(base_lr: float, batch_size: int, ref_bs: int = 64) -> float:
    """Adam works well with √-scaling of LR."""
    return float(base_lr * np.sqrt(batch_size / ref_bs))

# ------------------ load & prepare ------------------
raw = pd.read_csv(DATA_PATH)
dcol, pcol = infer_date_price(raw)
raw[dcol] = pd.to_datetime(raw[dcol], errors="coerce")
raw = raw[[dcol, pcol]].dropna().sort_values(dcol).rename(columns={dcol: "Date", pcol: "Price"})

# collapse duplicates to daily, resample, interpolate gaps
raw["Date"] = raw["Date"].dt.floor("D")
daily = raw.groupby("Date", as_index=True)["Price"].mean().to_frame().sort_index()
daily = daily.asfreq("D")
daily["Price"] = daily["Price"].interpolate(limit_direction="both")

feat = add_indicators(daily, "Price")
feat["target"] = feat["price"]

n = len(feat)
test_start = int(n * (1 - TEST_SPLIT))
val_start  = int(test_start * (1 - VAL_SPLIT))

X_cols = [c for c in feat.columns if c != "target"]
X_all  = feat[X_cols].values
y_all  = feat["target"].values.reshape(-1, 1)

# scale on train only (no leakage)
scX = StandardScaler().fit(X_all[:val_start])
scy = MinMaxScaler().fit(y_all[:val_start])

X_sc = scX.transform(X_all)
y_sc = scy.transform(y_all)

X_tr, y_tr = make_sequences(X_sc[:val_start],           y_sc[:val_start],           LOOKBACK, HORIZON)
X_va, y_va = make_sequences(X_sc[val_start:test_start], y_sc[val_start:test_start], LOOKBACK, HORIZON)
X_te, y_te = make_sequences(X_sc[test_start:],          y_sc[test_start:],          LOOKBACK, HORIZON)

input_shape = (X_tr.shape[1], X_tr.shape[2])

# ------------------ sweep ------------------
out_root = Path("./out_batch_tune"); out_root.mkdir(parents=True, exist_ok=True)
sweep_rows = []

print(f"[INFO] Starting batch-size sweep over: {BATCH_GRID}")
for bs in BATCH_GRID:
    lr = sqrt_lr_scale(BASE_LR, bs, REF_BS)
    print(f"\n[TUNE] BS={bs}, scaled LR={lr:.6g}")
    try:
        m = build_cnn_lstm_relu(input_shape, lr)
        es = callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        hist = m.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                     epochs=EPOCHS_TUNE, batch_size=bs, verbose=0, callbacks=[es])
        best_val = float(np.min(hist.history["val_loss"]))
        sweep_rows.append({"batch_size": bs, "scaled_lr": lr, "best_val_loss": best_val})
        print(f"  -> best val_loss={best_val:.6f}")
    except tf.errors.ResourceExhaustedError:
        print("  -> OOM at this batch size. Skipping.")
        sweep_rows.append({"batch_size": bs, "scaled_lr": lr, "best_val_loss": np.inf})
    except Exception as e:
        print("  -> Error:", e)
        sweep_rows.append({"batch_size": bs, "scaled_lr": lr, "best_val_loss": np.inf})

sweep_df = pd.DataFrame(sweep_rows).sort_values("best_val_loss")
sweep_df.to_csv(out_root / "bs_sweep_summary.csv", index=False)
print("\n=== Sweep summary (lower val_loss is better) ===")
print(sweep_df.to_string(index=False))

# pick best batch size
best_row = sweep_df.iloc[0]
BEST_BS  = int(best_row["batch_size"])
BEST_LR  = float(best_row["scaled_lr"])
print(f"\n[SELECT] Best batch size = {BEST_BS} (scaled LR={BEST_LR:.6g})")

# ------------------ full train with best BS ------------------
full_model = build_cnn_lstm_relu(input_shape, BEST_LR)
cbs = [
    callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=PLATEAU_FACTOR,
                                patience=PLATEAU_PATIENCE, min_lr=MIN_LR, verbose=1),
]
hist = full_model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                      epochs=EPOCHS_FULL, batch_size=BEST_BS, verbose=1, callbacks=cbs)

# ------------------ evaluate on test ------------------
yhat_te = full_model.predict(X_te, verbose=0)
yhat  = scy.inverse_transform(yhat_te).ravel()
ytrue = scy.inverse_transform(y_te).ravel()

seq_start = test_start + LOOKBACK + HORIZON - 1
idx = feat.index[seq_start: seq_start + len(ytrue)]

mae  = mean_absolute_error(ytrue, yhat)
rmse = math.sqrt(mean_squared_error(ytrue, yhat))
mape = float(np.mean(np.abs((ytrue - yhat) / np.maximum(1e-6, ytrue)))) * 100.0
r2   = r2_score(ytrue, yhat)

pred_df = pd.DataFrame({"date": idx.astype(str), "y_true": ytrue, "y_pred": yhat})
pred_df.to_csv(out_root / f"pred_CNN+LSTM_ReLU_Huber_bs{BEST_BS}.csv", index=False)

plt.figure(figsize=(10,4))
plt.plot(idx, ytrue, label="Actual")
plt.plot(idx, yhat,  label="Pred")
plt.title(f"CNN+LSTM(ReLU)+Huber — Best BS={BEST_BS}, LR={BEST_LR:.6g}")
plt.xlabel("Date"); plt.ylabel("Price"); plt.legend(); plt.tight_layout()
plt.savefig(out_root / f"CNN+LSTM_best_bs{BEST_BS}_actual_vs_pred.png", dpi=120)
try: plt.show()
except: pass

metrics = {"Best_BatchSize": BEST_BS, "Scaled_LR": BEST_LR,
           "MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}
print("\n=== Final metrics (test) ===")
print(pd.Series(metrics))

pd.DataFrame([metrics]).to_csv(out_root / "best_bs_metrics.csv", index=False)
try: full_model.save(out_root / f"CNN_LSTM_best_bs{BEST_BS}.keras")
except Exception as e: print("[WARN] save failed:", e)

print(f"\n[DONE] All artifacts in: {out_root.resolve()}")
