#!/usr/bin/env python3
# CNN+LSTM Price Forecasting — Sigmoid activations in Conv/Dense
import os, sys, math, warnings
from pathlib import Path
from typing import Tuple
import numpy as np, pandas as pd, matplotlib.pyplot as plt

# ================ CONFIG (same knobs) ================
DATA_PATH   = Path("/Users/luoyi/Desktop/ML Research/hotel_booking.csv")
DATE_COL    = None
PRICE_COL   = None
LOOKBACK    = 60
HORIZON     = 30
TEST_SPLIT  = 0.15
VAL_SPLIT   = 0.15
EPOCHS      = 60
BATCH       = 64
RANDOM_SEED = 1337
USE_BASELINES = True
LOSS_NAME   = "logcosh"
# =====================================================

warnings.filterwarnings("ignore", category=FutureWarning)
np.random.seed(RANDOM_SEED)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import tensorflow as tf
    tf.random.set_seed(RANDOM_SEED)
    from tensorflow.keras import layers, models, callbacks, losses
except Exception:
    print("[ERROR] Install TF: pip install tensorflow-macos tensorflow-metal")
    raise

def infer_date_price(df: pd.DataFrame) -> Tuple[str, str]:
    dcands=["Date","date","timestamp","Timestamp","reservation_status_date","arrival_date","booking_date"]
    pcands=["adr","ADR","close","Close","price","Price","Adj Close","adj_close","rate","amount"]
    d=next((c for c in dcands if c in df.columns),None)
    p=next((c for c in pcands if c in df.columns),None)
    if not d or not p: raise ValueError("Missing date/price columns.")
    return d,p

def add_indicators(df, price_col):
    s=df[price_col].clip(lower=1e-6); out=pd.DataFrame(index=df.index)
    out["price"]=s; out["ret_1"]=s.pct_change(); out["log_ret_1"]=np.log(s).diff()
    for w in (5,10,20,30,60): out[f"ma_{w}"]=s.rolling(w).mean(); out[f"std_{w}"]=s.rolling(w).std()
    d=s.diff(); up=np.where(d>0,d,0.0); dn=np.where(d<0,-d,0.0)
    out["rsi_14"]=100-100/(1+(pd.Series(up,index=df.index).rolling(14).mean()/(pd.Series(dn,index=df.index).rolling(14).mean()+1e-12)))
    ema12=s.ewm(span=12,adjust=False).mean(); ema26=s.ewm(span=26,adjust=False).mean(); macd=ema12-ema26
    out["macd"]=macd; out["macd9"]=macd.ewm(span=9,adjust=False).mean()
    ma20=s.rolling(20).mean(); sd20=s.rolling(20).std(); out["bb_up"]=ma20+2*sd20; out["bb_dn"]=ma20-2*sd20
    out.replace([np.inf,-np.inf],np.nan,inplace=True); out.dropna(inplace=True); return out

def make_sequences(X,y,lookback=60,horizon=1):
    Xs,ys=[],[]
    for i in range(lookback,len(X)-horizon+1): Xs.append(X[i-lookback:i]); ys.append(y[i+horizon-1])
    return np.array(Xs),np.array(ys)

def dir_acc(y_true,y_pred,prev):
    st,sp=np.sign(y_true-prev),np.sign(y_pred-prev); m=~np.isnan(st)
    return float(np.mean((st[m]==sp[m])))

raw=pd.read_csv(DATA_PATH); dcol,pcol=infer_date_price(raw)
raw[dcol]=pd.to_datetime(raw[dcol],errors="coerce")
raw=raw[[dcol,pcol]].dropna().sort_values(dcol).rename(columns={dcol:"Date",pcol:"Price"})
raw["Date"]=raw["Date"].dt.floor("D")
daily=raw.groupby("Date",as_index=True)["Price"].mean().to_frame().sort_index()
daily=daily.asfreq("D"); daily["Price"]=daily["Price"].interpolate(limit_direction="both")

feat=add_indicators(daily,"Price"); feat["target"]=feat["price"]
n=len(feat); test_start=int(n*(1-TEST_SPLIT)); val_start=int(test_start*(1-VAL_SPLIT))
X_cols=[c for c in feat.columns if c!="target"]; X_all=feat[X_cols].values; y_all=feat["target"].values.reshape(-1,1)
scX=StandardScaler().fit(X_all[:val_start]); scy=MinMaxScaler().fit(y_all[:val_start])
X_sc=scX.transform(X_all); y_sc=scy.transform(y_all)
X_tr,y_tr=make_sequences(X_sc[:val_start],y_sc[:val_start],LOOKBACK,HORIZON)
X_va,y_va=make_sequences(X_sc[val_start:test_start],y_sc[val_start:test_start],LOOKBACK,HORIZON)
X_te,y_te=make_sequences(X_sc[test_start:],y_sc[test_start:],LOOKBACK,HORIZON)

loss={"logcosh":losses.LogCosh(),"huber":losses.Huber(1.0),"mse":losses.MeanSquaredError(),"mae":losses.MeanAbsoluteError()}[LOSS_NAME]

def build_cnn_lstm_sigmoid(input_shape):
    inp=layers.Input(shape=input_shape)
    x=layers.Conv1D(32,3,padding="causal",activation="sigmoid")(inp)
    x=layers.Conv1D(32,3,padding="causal",activation="sigmoid")(x)
    x=layers.MaxPooling1D(2)(x)
    x=layers.LSTM(64, return_sequences=True)(x)
    x=layers.LSTM(32)(x)
    x=layers.Dense(32, activation="sigmoid")(x)
    out=layers.Dense(1)(x)
    m=models.Model(inp,out); m.compile(optimizer="adam", loss=loss); return m

def build_lstm(input_shape):
    inp=layers.Input(shape=input_shape); x=layers.LSTM(64,return_sequences=True)(inp); x=layers.LSTM(32)(x); out=layers.Dense(1)(x)
    m=models.Model(inp,out); m.compile(optimizer="adam", loss=loss); return m

def build_gru(input_shape):
    inp=layers.Input(shape=input_shape); x=layers.GRU(64,return_sequences=True)(inp); x=layers.GRU(32)(x); out=layers.Dense(1)(x)
    m=models.Model(inp,out); m.compile(optimizer="adam", loss=loss); return m

inp_shape=(X_tr.shape[1],X_tr.shape[2])
cb=[callbacks.EarlyStopping(patience=8,restore_best_weights=True,monitor="val_loss")]
runs=[("CNN+LSTM(Sigmoid)", lambda: build_cnn_lstm_sigmoid(inp_shape))]
if USE_BASELINES: runs += [("LSTM", lambda: build_lstm(inp_shape)), ("GRU", lambda: build_gru(inp_shape))]
OUT=Path("./out_sigmoid"); OUT.mkdir(parents=True, exist_ok=True)

def evaluate(m,name):
    yhat=scy.inverse_transform(m.predict(X_te,verbose=0)).ravel()
    ytrue=scy.inverse_transform(y_te).ravel()
    seq_start=test_start+LOOKBACK+HORIZON-1; idx=feat.index[seq_start: seq_start+len(ytrue)]
    prev=feat["target"].shift(1).loc[idx].values
    mae=mean_absolute_error(ytrue,yhat); rmse=math.sqrt(mean_squared_error(ytrue,yhat))
    mape=float(np.mean(np.abs((ytrue-yhat)/np.maximum(1e-6,ytrue))))*100.0
    r2=r2_score(ytrue,yhat); dira=dir_acc(ytrue.reshape(-1,1),yhat.reshape(-1,1),prev.reshape(-1,1))
    pd.DataFrame({"date":idx.astype(str),"y_true":ytrue,"y_pred":yhat}).to_csv(OUT/f"pred_{name}.csv",index=False)
    plt.figure(figsize=(10,4)); plt.plot(idx,ytrue,label="Actual"); plt.plot(idx,yhat,label="Pred"); plt.legend(); plt.tight_layout()
    plt.savefig(OUT/f"{name}_actual_vs_pred.png",dpi=120)
    plt.figure(figsize=(10,4)); plt.plot(idx,ytrue-yhat); plt.tight_layout(); plt.savefig(OUT/f"{name}_residuals.png",dpi=120)
    return {"Model":name,"Loss":LOSS_NAME,"MAE":mae,"RMSE":rmse,"MAPE":mape,"R2":r2,"Directional_Acc":dira}

res=[]
for name,build in runs:
    print(f"\n[TRAIN] {name} ..."); m=build()
    m.fit(X_tr,y_tr,validation_data=(X_va,y_va),epochs=EPOCHS,batch_size=BATCH,callbacks=cb,verbose=1)
    res.append(evaluate(m,name)); 
    try: m.save(OUT/f"{name.replace('+','_')}.keras")
    except: pass
summ=pd.DataFrame(res).sort_values("RMSE"); print(summ.to_string(index=False))
summ.to_csv(OUT/"metrics_summary.csv", index=False); print(f"[DONE] {OUT.resolve()}")
