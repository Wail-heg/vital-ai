"""Runt dezelfde pipeline als het notebook op de volledige dataset
en exporteert alle key-metrics naar metrics.json. Geen plotting — alles
quiet en zo snel mogelijk.
"""
from __future__ import annotations
import json, time, warnings, zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
)
import lightgbm as lgb

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_ZIP = Path("data/Prediction-of-sepsis.zip")
ARCHIVE_PREFIX = "Prediction-of-sepsis/"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- LOAD ------
log("Inladen train_data.csv…")
t0 = time.time()
with zipfile.ZipFile(DATA_ZIP) as z:
    with z.open(ARCHIVE_PREFIX + "train_data.csv") as f:
        train = pd.read_csv(f)
log(f"  klaar in {time.time()-t0:.1f}s — shape {train.shape}")

if "Unnamed: 0" in train.columns:
    train = train.drop(columns=["Unnamed: 0"])

log(f"Patiënten: {train['Patient_ID'].nunique()}")
log(f"Klassebalans rij-niveau   : {train['SepsisLabel'].mean():.4f}")
log(f"Klassebalans patient-level: "
    f"{train.groupby('Patient_ID')['SepsisLabel'].max().mean():.4f}")


# ---------------------------------------------------------------- PREP -----
LIMITS = {
    "HR": (20, 250), "O2Sat": (50, 100), "Temp": (30, 43),
    "SBP": (40, 250), "MAP": (30, 200), "DBP": (20, 180),
    "Resp": (4, 60),
}
for col, (lo, hi) in LIMITS.items():
    if col in train.columns:
        train[col] = train[col].where(train[col].between(lo, hi))
log("Outliers geclipt.")

ROLL_VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "Resp"]
ROLL_LABS = ["Lactate", "WBC", "Creatinine", "BUN", "Glucose", "Platelets"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
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


log("Feature engineering…")
t0 = time.time()
train_fe = add_features(train)
log(f"  klaar in {time.time()-t0:.1f}s — shape {train_fe.shape}")


# ---------------------------------------------------------------- SPLIT ----
all_patients = train_fe["Patient_ID"].unique()
rng = np.random.default_rng(RANDOM_STATE)
val_pat = rng.choice(all_patients, size=int(0.2 * len(all_patients)),
                     replace=False)
tr_mask = ~train_fe["Patient_ID"].isin(val_pat)
val_mask = train_fe["Patient_ID"].isin(val_pat)

DROP_COLS = ["SepsisLabel", "Patient_ID"]
feat_cols = [c for c in train_fe.columns if c not in DROP_COLS]

X_tr = train_fe.loc[tr_mask, feat_cols]
y_tr = train_fe.loc[tr_mask, "SepsisLabel"]
X_val = train_fe.loc[val_mask, feat_cols]
y_val = train_fe.loc[val_mask, "SepsisLabel"]
val_meta = train_fe.loc[val_mask, ["Patient_ID", "ICULOS",
                                   "SepsisLabel", "Gender", "Age"]
                       ].reset_index(drop=True)
log(f"Train: {len(X_tr)} rijen | Val: {len(X_val)} rijen | "
    f"pos {y_tr.mean():.4f}/{y_val.mean():.4f}")


# ---------------------------------------------------------------- BASELINES
log("qSOFA-baseline…")
q_proba = (train_fe.loc[val_mask, "qSOFA"].to_numpy() / 2.0)
q_pred = (q_proba >= 0.5).astype(int)
qsofa_auroc = roc_auc_score(y_val, q_proba)
qsofa_auprc = average_precision_score(y_val, q_proba)
qsofa_recall = q_pred[y_val.to_numpy() == 1].sum() / max(int((y_val == 1).sum()), 1)
qsofa_precision = q_pred[y_val.to_numpy() == 1].sum() / max(int(q_pred.sum()), 1)
log(f"  AUROC={qsofa_auroc:.3f}  AUPRC={qsofa_auprc:.3f}  "
    f"rec={qsofa_recall:.3f}  prec={qsofa_precision:.3f}")


log("Logistische regressie…")
base_cols = [c for c in ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp",
                         "Age", "Gender"] if c in feat_cols]
pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc", StandardScaler()),
    ("lr", LogisticRegression(max_iter=1000, class_weight="balanced",
                              random_state=RANDOM_STATE)),
])
pipe.fit(X_tr[base_cols], y_tr)
lr_proba = pipe.predict_proba(X_val[base_cols])[:, 1]
lr_auroc = roc_auc_score(y_val, lr_proba)
lr_auprc = average_precision_score(y_val, lr_proba)
log(f"  LR AUROC={lr_auroc:.3f}  AUPRC={lr_auprc:.3f}")


# ---------------------------------------------------------------- LGBM ----
pos = int(y_tr.sum())
neg = len(y_tr) - pos
scale = neg / max(pos, 1)
log(f"LightGBM training… scale_pos_weight={scale:.1f}")

lgb_train = lgb.Dataset(X_tr, label=y_tr)
lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

params = {
    "objective": "binary",
    "metric": ["auc", "average_precision"],
    "learning_rate": 0.05,
    "num_leaves": 64,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "scale_pos_weight": scale,
    "verbose": -1,
    "seed": RANDOM_STATE,
}

t0 = time.time()
model = lgb.train(
    params, lgb_train,
    num_boost_round=800,
    valid_sets=[lgb_train, lgb_val],
    valid_names=["train", "val"],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
)
log(f"  trained in {time.time()-t0:.1f}s | best_iter={model.best_iteration}")

val_proba = model.predict(X_val, num_iteration=model.best_iteration)
lgb_auroc = roc_auc_score(y_val, val_proba)
lgb_auprc = average_precision_score(y_val, val_proba)
brier = brier_score_loss(y_val, val_proba)
log(f"  LightGBM AUROC={lgb_auroc:.3f}  AUPRC={lgb_auprc:.3f}  "
    f"Brier={brier:.3f}")


# ---------------------------------------------------------------- THRESHOLD
def fp_per_24h(y_true, pred):
    fps = int(((y_true == 0) & (pred == 1)).sum())
    neg_hours = int((y_true == 0).sum())
    return fps / max(neg_hours, 1) * 24

thresholds = np.linspace(0.05, 0.80, 50)
rows = []
y_v = y_val.to_numpy()
for t in thresholds:
    pred = (val_proba >= t).astype(int)
    tp = int(((y_v == 1) & (pred == 1)).sum())
    fp = int(((y_v == 0) & (pred == 1)).sum())
    fn = int(((y_v == 1) & (pred == 0)).sum())
    rec = tp / max(tp + fn, 1)
    pr = tp / max(tp + fp, 1)
    rows.append({"threshold": float(t), "recall": rec, "precision": pr,
                 "fp_per_24h": fp_per_24h(y_v, pred)})
metrics_df = pd.DataFrame(rows)
chosen = metrics_df.iloc[(metrics_df["fp_per_24h"] - 3).abs().idxmin()]
THRESHOLD = float(chosen["threshold"])
log(f"Drempel (~3 FP/24u): {THRESHOLD:.3f}  "
    f"rec={chosen['recall']:.3f}  prec={chosen['precision']:.3f}  "
    f"fp24={chosen['fp_per_24h']:.2f}")


# ---------------------------------------------------------------- UTILITY -
import sys; sys.path.insert(0, ".")
from evaluation import compute_prediction_utility

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
        obs += compute_prediction_utility(labels, preds, check_errors=False,
                                          dt_early=-12, dt_optimal=-6, dt_late=3)
        best += compute_prediction_utility(labels, best_pred, check_errors=False,
                                           dt_early=-12, dt_optimal=-6, dt_late=3)
        inact += compute_prediction_utility(labels, np.zeros(n), check_errors=False,
                                            dt_early=-12, dt_optimal=-6, dt_late=3)
    return (obs - inact) / (best - inact + 1e-9)

log("PhysioNet utility score…")
t0 = time.time()
util = normalized_utility(val_meta, val_proba, THRESHOLD)
log(f"  utility = {util:.3f} (in {time.time()-t0:.1f}s)")


# ---------------------------------------------------------------- LEAD TIME
log("Lead time…")
pred_bin = (val_proba >= THRESHOLD).astype(int)
lead_df = val_meta.assign(pred=pred_bin)
lt_list = []
for _, g in lead_df.groupby("Patient_ID"):
    g = g.sort_values("ICULOS")
    if not g["SepsisLabel"].any():
        continue
    t_sepsis = g.loc[g["SepsisLabel"] == 1, "ICULOS"].min()
    alarms = g.loc[g["pred"] == 1, "ICULOS"]
    if len(alarms) > 0:
        lt_list.append(t_sepsis - alarms.min())
lt = pd.Series(lt_list, dtype=float)
n_septic_val = lead_df.groupby("Patient_ID")["SepsisLabel"].max().sum()
lead_median = float(lt.median()) if len(lt) else None
detect_rate = len(lt) / max(int(n_septic_val), 1)
log(f"  lead median = {lead_median} h  | detect_rate = {detect_rate:.2%}")


# ---------------------------------------------------------------- FAIRNESS
def subgroup(meta, proba, y_true, col, bins=None):
    df = meta.copy()
    df["proba"] = proba
    df["y"] = y_true.values
    if bins is not None:
        df["_group"] = pd.cut(df[col], bins=bins, include_lowest=True)
    else:
        df["_group"] = df[col]
    out = []
    for g, sub in df.groupby("_group", observed=True):
        if sub["y"].nunique() < 2:
            continue
        pred = (sub["proba"] >= THRESHOLD).astype(int)
        rec = int(((sub["y"] == 1) & (pred == 1)).sum()) / max(int((sub["y"] == 1).sum()), 1)
        out.append({
            "group": str(g),
            "n_rows": int(len(sub)),
            "n_pos": int(sub["y"].sum()),
            "AUROC": float(roc_auc_score(sub["y"], sub["proba"])),
            "AUPRC": float(average_precision_score(sub["y"], sub["proba"])),
            "recall_thr": float(rec),
        })
    return out


gender_tbl = subgroup(val_meta, val_proba, y_val, "Gender")
age_tbl = subgroup(val_meta, val_proba, y_val, "Age",
                   bins=[0, 40, 55, 70, 80, 120])

gap_gender = (max(r["AUROC"] for r in gender_tbl)
              - min(r["AUROC"] for r in gender_tbl)) if gender_tbl else None
gap_age = (max(r["AUROC"] for r in age_tbl)
           - min(r["AUROC"] for r in age_tbl)) if age_tbl else None
log(f"Fairness gap AUROC — gender={gap_gender}, age={gap_age}")


# ---------------------------------------------------------------- EXPORT --
metrics = {
    "n_patients_train": int(train["Patient_ID"].nunique()),
    "n_rows_train": int(len(train)),
    "class_balance_row": float(train["SepsisLabel"].mean()),
    "class_balance_patient": float(train.groupby("Patient_ID")["SepsisLabel"].max().mean()),
    "qsofa": {"AUROC": qsofa_auroc, "AUPRC": qsofa_auprc,
              "recall": qsofa_recall, "precision": qsofa_precision},
    "logreg": {"AUROC": lr_auroc, "AUPRC": lr_auprc},
    "lightgbm": {
        "AUROC": lgb_auroc, "AUPRC": lgb_auprc, "Brier": brier,
        "best_iteration": int(model.best_iteration),
        "threshold_3fp24": THRESHOLD,
        "recall_at_thr": float(chosen["recall"]),
        "precision_at_thr": float(chosen["precision"]),
        "fp_per_24h_at_thr": float(chosen["fp_per_24h"]),
        "utility": float(util),
        "lead_time_median_h": lead_median,
        "detection_rate": float(detect_rate),
    },
    "fairness": {
        "gender": gender_tbl,
        "age": age_tbl,
        "gap_gender_AUROC": gap_gender,
        "gap_age_AUROC": gap_age,
    },
}
Path("metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
log("metrics.json geschreven.")

# Voor evt. SHAP later — sla model + sample op
import pickle
model.save_model("lgbm_model.txt")
X_val.sample(min(5000, len(X_val)), random_state=RANDOM_STATE).to_parquet(
    "val_sample.parquet")
log("Model + val_sample opgeslagen.")
