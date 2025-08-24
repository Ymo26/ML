#!/usr/bin/env python3
"""
LR tuner for CNN+LSTM with three activations (ReLU, Sigmoid, KAN) using Huber loss.
- Duplicate-date safe preprocessing
- Short LR sweep -> pick best -> full train with ReduceLROnPlateau
"""

import math, warnings
from pathlib import Path
from typing import Tuple
import numpy as np, pandas as pd, matplotlib.pyplot as plt

# ------------ Config ------------
DATA_PATH = Path("/Users/luoyi/Desktop/ML Research/hotel_booking.csv")
DATE_COL  = None
PRICE_COL = None

LOOKBACK  = 60
HORIZON   = 30
TEST_SPLIT= 0.15
VAL_SPLIT = 0.15

EPOCHS_TUNE = 12
EPOCHS_FULL = 60
BATCH       = 64
SEED        = 1337

# LR grids per activation (based on your plots)
GRID_RELU    = [2e-4, 3e-4, 5e-4, 7.5e-4, 1e-3]
GRID_SIGMOID = [3e-4, 5e-4, 7.5e-4, 8e-4, 1e-3, 1.25e-3]
GRID_KAN     = [5e-4, 7.5e-4, 1e-3, 1.25e-3, 1.5e-3]

HUBER_DELTA = 1.0
CLIPNORM    = 1.0
MIN_LR      = 1e-6
REDUCE_FACTOR   = 0.5
REDUCE_PATIENCE = 4
KAN_NUM_KER = 8
# --------------------------------

warnings.filterwarnings("ignore", category=FutureWarning)
np.random.seed(SEED)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
tf.random.set_seed(SEED)
from tensorflow.keras import layers, models, callbacks, losses, optimizers, constraints, initializers

# ---------- KAN activation ----------
class KANActivation1D(layers.Layer):
    """y = x + sum_i a_i * exp(-b_i * (x - c_i)^2), shared across time & channels."""
    def __init__(self, num_kernels=8, **kw): super().__init__(**kw); self.num_kernels=int(num_kernels)
    def build(self, input_shape):
        self.amp  = self.add_weight(name="amp",  shape=(self.num_kernels,), initializer=initializers.Zeros(), trainable=True)
        self.cent = self.add_weight(name="cent", shape=(self.num_kernels,), initializer=initializers.RandomUniform(-1.0,1.0), trainable=True)
        self.beta = self.add_weight(name="beta", shape=(self.num_kernels,), initializer=initializers.Constant(1.0),
                                    constraint=constraints.NonNeg(), trainable=True)
        super().build(input_shape)
    def call(self, x):
        x = tf.cast(x, tf.float32)
        amp = tf.reshape(tf.cast(self.amp,  tf.float32),(1,1,1,self.num_kernels))
        cen = tf.reshape(tf.cast(self.cent, tf.float32),(1,1,1,self.num_kernels))
        bet = tf.reshape(tf.cast(self.beta, tf.float32),(1,1,1,self.num_kernels))
        e = tf.exp(- bet * tf.square(tf.expand_dims(x,-1) - cen))
        return x + tf.reduce_sum(amp*e, axis=-1)

# ---------- data utils ----------
def infer_date_price(df: pd.DataFrame) -> Tuple[str, str]:
    d = next((c for c in [DATE_COL,"Date","date","timestamp","Timestamp","reservation_status_date","arrival_date","booking_date"] if c and c in df.columns), None)
    p = next((c for c in [PRICE_COL,"adr","ADR","close","Close","price","Price","Adj Close","adj_close","rate","amount"] if c and c in df.columns), None)
    if not d or not p: raise ValueError(f"Need date/price columns. Saw: {list(df.columns)[:20]}")
    return d,p

def add_indicators(df, price_col):
    s=df[price_col].clip(lower=1e-6)
    out=pd.DataFrame(index=df.index)
    out["price"]=s; out["ret_1"]=s.pct_change(); out["log_ret_1"]=np.log(s).diff()
    for w in (5,10,20,30,60):
        out[f"ma_{w}"]=s.rolling(w).mean(); out[f"std_{w}"]=s.rolling(w).std()
    d=s.diff(); up=np.where(d>0,d,0.0); dn=np.where(d<0,-d,0.0)
    out["rsi_14"]=100-100/(1+(pd.Series(up,index=df.index).rolling(14).mean()/(pd.Series(dn,index=df.index).rolling(14).mean()+1e-12)))
    ema12=s.ewm(span=12,adjust=False).mean(); ema26=s.ewm(span=26,adjust=False).mean(); macd=ema12-ema26
    out["macd"]=macd; out["macd9"]=macd.ewm(span=9,adjust=False).mean()
    ma20=s.rolling(20).mean(); sd20=s.rolling(20).std(); out["bb_up"]=ma20+2*sd20; out["bb_dn"]=ma20-2*sd20
    out.replace([np.inf,-np.inf],np.nan, inplace=True); out.dropna(inplace=True)
    return out

def make_sequences(X,y,lookback=60,horizon=1):
    Xs,ys=[],[]
    for i in range(lookback,len(X)-horizon+1):
        Xs.append(X[i-lookback:i]); ys.append(y[i+horizon-1])
    return np.array(Xs), np.array(ys)

# ---------- load/prepare ----------
raw=pd.read_csv(DATA_PATH)
dcol,pcol=infer_date_price(raw)
raw[dcol]=pd.to_datetime(raw[dcol],errors="coerce")
raw=raw[[dcol,pcol]].dropna().sort_values(dcol).rename(columns={dcol:"Date",pcol:"Price"})
raw["Date"]=raw["Date"].dt.floor("D")
daily=(raw.groupby("Date",as_index=True)["Price"].mean().to_frame().sort_index()).asfreq("D")
daily["Price"]=daily["Price"].interpolate(limit_direction="both")
feat=add_indicators(daily,"Price"); feat["target"]=feat["price"]

n=len(feat); test_start=int(n*(1-TEST_SPLIT)); val_start=int(test_start*(1-VAL_SPLIT))
X_cols=[c for c in feat.columns if c!="target"]; y_col="target"
X_all, y_all = feat[X_cols].values, feat[y_col].values.reshape(-1,1)
scX=StandardScaler().fit(X_all[:val_start]); scy=MinMaxScaler().fit(y_all[:val_start])
X_sc, y_sc = scX.transform(X_all), scy.transform(y_all)

X_tr,y_tr = make_sequences(X_sc[:val_start],           y_sc[:val_start],           LOOKBACK,HORIZON)
X_va,y_va = make_sequences(X_sc[val_start:test_start], y_sc[val_start:test_start], LOOKBACK,HORIZON)
X_te,y_te = make_sequences(X_sc[test_start:],          y_sc[test_start:],          LOOKBACK,HORIZON)
input_shape=(X_tr.shape[1], X_tr.shape[2])

# ---------- model builders ----------
def build_relu(lr):
    opt=optimizers.Adam(learning_rate=lr, clipnorm=CLIPNORM); loss=losses.Huber(delta=HUBER_DELTA)
    inp=layers.Input(shape=input_shape)
    x=layers.Conv1D(32,3,padding="causal",activation="relu")(inp)
    x=layers.Conv1D(32,3,padding="causal",activation="relu")(x)
    x=layers.MaxPooling1D(2)(x)
    x=layers.LSTM(64, return_sequences=True)(x); x=layers.LSTM(32)(x)
    x=layers.Dense(32,activation="relu")(x); out=layers.Dense(1)(x)
    m=models.Model(inp,out); m.compile(optimizer=opt, loss=loss); return m

def build_sigmoid(lr):
    opt=optimizers.Adam(learning_rate=lr, clipnorm=CLIPNORM); loss=losses.Huber(delta=HUBER_DELTA)
    inp=layers.Input(shape=input_shape)
    x=layers.Conv1D(32,3,padding="causal",activation="sigmoid")(inp)
    x=layers.Conv1D(32,3,padding="causal",activation="sigmoid")(x)
    x=layers.MaxPooling1D(2)(x)
    x=layers.LSTM(64, return_sequences=True)(x); x=layers.LSTM(32)(x)
    x=layers.Dense(32,activation="sigmoid")(x); out=layers.Dense(1)(x)
    m=models.Model(inp,out); m.compile(optimizer=opt, loss=loss); return m

def build_kan(lr):
    opt=optimizers.Adam(learning_rate=lr, clipnorm=CLIPNORM); loss=losses.Huber(delta=HUBER_DELTA)
    inp=layers.Input(shape=input_shape)
    x=layers.Conv1D(32,3,padding="causal",activation=None)(inp); x=KANActivation1D(KAN_NUM_KER)(x)
    x=layers.Conv1D(32,3,padding="causal",activation=None)(x);  x=KANActivation1D(KAN_NUM_KER)(x)
    x=layers.MaxPooling1D(2)(x)
    x=layers.LSTM(64, return_sequences=True)(x); x=layers.LSTM(32)(x)
    x=layers.Dense(32,activation="linear")(x); out=layers.Dense(1)(x)
    m=models.Model(inp,out); m.compile(optimizer=opt, loss=loss); return m

# ---------- train helpers ----------
def run_one(model_fn, lr, tag, epochs=EPOCHS_TUNE):
    m=model_fn(lr)
    es=callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    hist=m.fit(X_tr,y_tr,validation_data=(X_va,y_va),epochs=epochs,batch_size=BATCH,verbose=0,callbacks=[es])
    best=float(np.min(hist.history["val_loss"]))
    return best, m

def full_train(model_fn, lr, tag):
    m=model_fn(lr)
    cbs=[
        callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=REDUCE_FACTOR, patience=REDUCE_PATIENCE,
                                    min_lr=MIN_LR, verbose=1),
    ]
    m.fit(X_tr,y_tr,validation_data=(X_va,y_va),epochs=EPOCHS_FULL,batch_size=BATCH,verbose=1,callbacks=cbs)
    return m

def evaluate(m, name, outdir: Path):
    yhat = scy.inverse_transform(m.predict(X_te, verbose=0)).ravel()
    ytrue= scy.inverse_transform(y_te).ravel()
    seq_start = int(len(feat)*(1-TEST_SPLIT)) + LOOKBACK + HORIZON - 1
    idx = feat.index[seq_start: seq_start+len(ytrue)]
    mae = mean_absolute_error(ytrue,yhat); rmse = math.sqrt(mean_squared_error(ytrue,yhat))
    mape = float(np.mean(np.abs((ytrue-yhat)/np.maximum(1e-6,ytrue))))*100.0
    r2   = r2_score(ytrue,yhat)
    pd.DataFrame({"date":idx.astype(str),"y_true":ytrue,"y_pred":yhat}).to_csv(outdir/f"pred_{name}.csv", index=False)
    plt.figure(figsize=(10,4)); plt.plot(idx,ytrue,label="Actual"); plt.plot(idx,yhat,label="Pred"); plt.legend(); plt.tight_layout()
    plt.title(f"{name} (H={HORIZON})"); plt.savefig(outdir/f"{name}_actual_vs_pred.png",dpi=120)
    return {"Model":name,"MAE":mae,"RMSE":rmse,"MAPE":mape,"R2":r2}

# ---------- sweeps ----------
experiments = [
    ("ReLU",    build_relu,    GRID_RELU),
    ("Sigmoid", build_sigmoid, GRID_SIGMOID),
    ("KAN",     build_kan,     GRID_KAN),
]

results = []
for tag, fn, grid in experiments:
    print(f"\n[TUNE] {tag} sweep over {grid}")
    scores=[]
    for lr in grid:
        best,_ = run_one(fn, lr, tag)
        scores.append((best, lr))
        print(f"  lr={lr:.6f} -> best val_loss={best:.6f}")
    best_val, best_lr = sorted(scores, key=lambda x: x[0])[0]
    print(f"[SELECT] {tag}: best_lr={best_lr} (val_loss={best_val:.6f})")

    out = Path(f"./out_best_{tag.lower()}"); out.mkdir(parents=True, exist_ok=True)
    m = full_train(fn, best_lr, tag)
    metrics = evaluate(m, f"{tag}+Huber_lr{best_lr}", out)
    metrics["best_lr"]=best_lr
    results.append(metrics)
    try: m.save(out/f"{tag}.keras")
    except: pass

print("\n=== Best LR per activation (picked by validation loss) ===")
print(pd.DataFrame(results).to_string(index=False))
pd.DataFrame(results).to_csv("./out_best_lr_summary.csv", index=False)
print("\n[DONE] Outputs in ./out_best_* folders and out_best_lr_summary.csv")
