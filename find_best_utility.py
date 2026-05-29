"""Zoek de drempel die de PhysioNet utility maximaliseert,
zodat we beide drempelkeuzes (fp-budget vs. utility-optimaal) kunnen rapporteren.
"""
import json, sys, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from evaluation import compute_prediction_utility

# Herbouw val_meta + val_proba zoals in de pipeline
DATA_ZIP = Path("data/Prediction-of-sepsis.zip")
ARCHIVE_PREFIX = "Prediction-of-sepsis/"
RANDOM_STATE = 42

print("Inladen…", flush=True)
with zipfile.ZipFile(DATA_ZIP) as z:
    with z.open(ARCHIVE_PREFIX + "train_data.csv") as f:
        train = pd.read_csv(f)
if "Unnamed: 0" in train.columns:
    train = train.drop(columns=["Unnamed: 0"])

LIMITS = {"HR": (20, 250), "O2Sat": (50, 100), "Temp": (30, 43),
          "SBP": (40, 250), "MAP": (30, 200), "DBP": (20, 180),
          "Resp": (4, 60)}
for col, (lo, hi) in LIMITS.items():
    if col in train.columns:
        train[col] = train[col].where(train[col].between(lo, hi))

ROLL_VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "Resp"]
ROLL_LABS = ["Lactate", "WBC", "Creatinine", "BUN", "Glucose", "Platelets"]

def add_features(df):
    df = df.sort_values(["Patient_ID", "ICULOS"]).copy()
    g = df.groupby("Patient_ID", group_keys=False)
    for col in ROLL_LABS:
        if col in df.columns:
            df[f"{col}_isna"] = df[col].isna().astype(np.int8)
    for col in ROLL_LABS:
        if col in df.columns:
            df[col] = g[col].ffill()
    for col in ROLL_VITALS:
        if col in df.columns:
            roll = g[col].rolling(window=6, min_periods=1)
            df[f"{col}_mean6"] = roll.mean().reset_index(level=0, drop=True)
            df[f"{col}_std6"] = roll.std().reset_index(level=0, drop=True)
            df[f"{col}_min6"] = roll.min().reset_index(level=0, drop=True)
            df[f"{col}_max6"] = roll.max().reset_index(level=0, drop=True)
    if {"HR", "SBP"}.issubset(df.columns):
        df["ShockIndex"] = df["HR"] / df["SBP"]
    if {"SBP", "DBP"}.issubset(df.columns):
        df["PulsePressure"] = df["SBP"] - df["DBP"]
    df["qSOFA_resp"] = (df["Resp"] >= 22).astype(np.int8)
    df["qSOFA_sbp"] = (df["SBP"] <= 100).astype(np.int8)
    df["qSOFA"] = df["qSOFA_resp"] + df["qSOFA_sbp"]
    return df

print("Feature engineering…", flush=True)
train_fe = add_features(train)
all_patients = train_fe["Patient_ID"].unique()
rng = np.random.default_rng(RANDOM_STATE)
val_pat = rng.choice(all_patients, size=int(0.2 * len(all_patients)),
                     replace=False)
val_mask = train_fe["Patient_ID"].isin(val_pat)
feat_cols = [c for c in train_fe.columns if c not in ("SepsisLabel", "Patient_ID")]
X_val = train_fe.loc[val_mask, feat_cols]
y_val = train_fe.loc[val_mask, "SepsisLabel"]
val_meta = train_fe.loc[val_mask, ["Patient_ID", "ICULOS", "SepsisLabel"]].reset_index(drop=True)

print("Model laden…", flush=True)
model = lgb.Booster(model_file="lgbm_model.txt")
val_proba = model.predict(X_val)

def normalized_utility(meta, proba, threshold):
    df = meta.copy()
    df["pred"] = (proba >= threshold).astype(int)
    df = df.sort_values(["Patient_ID", "ICULOS"])
    obs, best, inact = 0.0, 0.0, 0.0
    for _, g in df.groupby("Patient_ID", sort=False):
        labels = g["SepsisLabel"].to_numpy()
        preds = g["pred"].to_numpy()
        n = len(labels)
        if labels.any():
            t_sepsis = int(np.argmax(labels)) - (-6)
            best_pred = np.zeros(n)
            lo = max(0, t_sepsis + (-12))
            hi = min(n, t_sepsis + 3 + 1)
            best_pred[lo:hi] = 1
        else:
            best_pred = np.zeros(n)
        obs += compute_prediction_utility(labels, preds, check_errors=False)
        best += compute_prediction_utility(labels, best_pred, check_errors=False)
        inact += compute_prediction_utility(labels, np.zeros(n), check_errors=False)
    return (obs - inact) / (best - inact + 1e-9)

print("Utility over thresholds…", flush=True)
results = []
for t in np.linspace(0.30, 0.95, 14):
    t0 = time.time()
    u = normalized_utility(val_meta, val_proba, t)
    results.append({"threshold": float(t), "utility": float(u)})
    print(f"  thr={t:.3f}  utility={u:+.4f}  ({time.time()-t0:.1f}s)", flush=True)

best = max(results, key=lambda r: r["utility"])
print("\nBest threshold by utility:")
print(json.dumps(best, indent=2))

# Recompute recall/precision/fp24 at best threshold
pred = (val_proba >= best["threshold"]).astype(int)
y_v = y_val.to_numpy()
tp = int(((y_v == 1) & (pred == 1)).sum())
fp = int(((y_v == 0) & (pred == 1)).sum())
fn = int(((y_v == 1) & (pred == 0)).sum())
neg_hours = int((y_v == 0).sum())
rec = tp / max(tp + fn, 1)
pr = tp / max(tp + fp, 1)
fp24 = fp / max(neg_hours, 1) * 24
print(f"At best threshold {best['threshold']:.3f}: recall={rec:.3f}, "
      f"precision={pr:.3f}, fp_per_24h={fp24:.2f}")

# update metrics.json
m = json.loads(Path("metrics.json").read_text(encoding="utf-8"))
m["lightgbm"]["utility_curve"] = results
m["lightgbm"]["utility_best"] = {
    "threshold": best["threshold"],
    "utility": best["utility"],
    "recall": rec,
    "precision": pr,
    "fp_per_24h": fp24,
}
Path("metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False),
                                encoding="utf-8")
print("metrics.json bijgewerkt.")
