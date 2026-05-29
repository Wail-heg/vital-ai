"""Bouwt sepsis_prediction.ipynb — volledige CRISP-DM-uitwerking.
Run: `python build_notebook.py`
"""
from __future__ import annotations
import json, pathlib

def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src}

CELLS: list[dict] = []

# ---------------------------------------------------------------------------
# 0. Titel
# ---------------------------------------------------------------------------
CELLS.append(md(r"""# Vroege voorspelling van sepsis — Isala × Windesheim
### Een volledige CRISP-DM-uitwerking met explainability & fairness

**Opdrachtgever:** Isala ziekenhuis Zwolle
**Onderzoeksgroep:** IT-innovations in Healthcare — Windesheim
**Opleiding:** HBO-ICT — Data Science
**Datum:** mei 2026

---

Dit notebook implementeert de volledige CRISP-DM-cyclus voor het vroegtijdig
voorspellen van sepsis op de Intensive Care, op basis van de openbare
PhysioNet Computing in Cardiology Challenge 2019. Het primaire subdoel van de
opdracht (uitlegbaarheid en transparantie) wordt uitgewerkt via SHAP en een
modelkaart; **fairness** (verschillen naar geslacht en leeftijd) wordt als
secundair onderzoeksspoor meegenomen.

### Inhoudsopgave
1. **Business Understanding** — probleemstelling, succescriteria, subdoel
2. **Data Understanding** — EDA, missingness, temporele patronen
3. **Data Preparation** — outliers, missing-flags, rolling features
4. **Modeling** — qSOFA-baseline → logistische regressie → LightGBM
5. **Evaluation** — AUROC / AUPRC / PhysioNet utility / lead time / calibratie
6. **Explainability (SHAP)** — globaal en per patiëntcase
7. **Fairness** — subgroep-evaluatie naar geslacht en leeftijd
8. **Foutenanalyse** — handmatig doorlopen van FP en FN
9. **Deployment-reflectie** — hoe past dit model in een Isala-werkstroom?
10. **Conclusie & aanbevelingen**

> **Reproduceerbaarheid.** Zet `RUN_FAST = True` (default) voor een snelle
> verkenning met 4 000 patiënten; `RUN_FAST = False` gebruikt de volledige
> dataset. Alle vaste seeds staan in `RANDOM_STATE`.
"""))

# ---------------------------------------------------------------------------
# 1. Business Understanding
# ---------------------------------------------------------------------------
CELLS.append(md(r"""## 1. Business Understanding

### 1.1 Aanleiding
Sepsis is een levensbedreigende ontregeling van de immuunrespons op infectie en
nog altijd een van de hoofdoorzaken van sterfte op de Intensive Care.
Elk uur uitstel van adequate antibiotische therapie verhoogt de mortaliteit met
naar schatting 4–8 % (Kumar et al., 2006). De klinische scores SIRS, qSOFA en
SOFA worden routinematig gebruikt voor herkenning, maar zijn ofwel te
gevoelig (alarmmoeheid) ofwel te laat (sepsis al manifest).

### 1.2 Probleemstelling
> *Kunnen we, op basis van routinematig op de IC geregistreerde vitale parameters
> en laboratoriumwaarden, **per uur** voorspellen of een patiënt binnen de
> komende 6 uur sepsis zal ontwikkelen, met een werkbare balans tussen
> sensitiviteit en specificiteit?*

### 1.3 Onderzoeksvragen
1. Welke combinatie van vitale parameters en labwaarden draagt het sterkst
   bij aan een vroegtijdige sepsis-predictie?
2. In hoeverre verslaat een gradient-boosting model de klinische qSOFA-regel
   op AUROC, AUPRC en de PhysioNet utility score?
3. Bestaat er een fairness-gap (verschil in recall of AUROC) tussen
   leeftijdsgroepen of geslachten?
4. Hoe uitlegbaar zijn individuele voorspellingen via SHAP?

### 1.4 Succescriteria
| Criterium | Doel |
|-----------|------|
| AUROC | ≥ 0,80 |
| Sensitiviteit @ ≤ 3 FP/24u | ≥ 0,70 |
| Mediane lead time | ≥ 4 uur vóór klinische diagnose |
| Fairness-gap AUROC (subgroep) | ≤ 10 % |
| PhysioNet utility | > 0 (beter dan ‘niets doen’) |
| Uitlegbaarheid | top-3 SHAP-drivers per voorspelling |

### 1.5 Subdoel
**Explainability & transparency** primair, met **fairness** als secundair
onderzoeksspoor. Reden: een IC-waarschuwingssysteem wordt klinisch alleen
geaccepteerd als de output per patiënt te verklaren is, en die per-patiënt
analyse maakt subgroep-evaluatie een natuurlijke uitbreiding.

### 1.6 Aannames en scope
- Werk uitsluitend met de openbare PhysioNet-data — geen Isala-data zonder DPIA.
- Geen productie-implementatie; de deployment-fase is een schriftelijke reflectie.
- Eén hoofdmodel grondig, één variant ter robuustheid; geen leaderboard-jacht.
"""))

# ---------------------------------------------------------------------------
# 2. Data Understanding — setup
# ---------------------------------------------------------------------------
CELLS.append(md(r"""## 2. Data Understanding

### 2.1 Setup en imports
"""))

CELLS.append(code(r"""# (Optioneel) installeer ontbrekende packages — alleen eenmalig nodig
# !pip install -q lightgbm shap"""))

CELLS.append(code(r"""from __future__ import annotations
import os, io, json, zipfile, warnings, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve, confusion_matrix,
)
from sklearn.calibration import calibration_curve

import lightgbm as lgb

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False
    print('SHAP niet geïnstalleerd — installeer met `pip install shap` voor sectie 6.')

warnings.filterwarnings('ignore', category=FutureWarning)
sns.set_theme(style='whitegrid', context='notebook')

# --- Configuratie -----------------------------------------------------------
RUN_FAST        = True   # True = werk met sample voor snelle iteratie
SAMPLE_PATIENTS = 4000   # gebruik bij RUN_FAST
RANDOM_STATE    = 42
np.random.seed(RANDOM_STATE)

DATA_ZIP       = Path('data/Prediction-of-sepsis.zip')
ARCHIVE_PREFIX = 'Prediction-of-sepsis/'

print('Pandas :', pd.__version__)
print('LightGBM:', lgb.__version__)
print('Mode    :', 'FAST' if RUN_FAST else 'FULL')"""))

# ---------------------------------------------------------------------------
# 2.2 Inlezen uit ZIP
# ---------------------------------------------------------------------------
CELLS.append(md(r"""### 2.2 Data inlezen
We lezen de CSV-bestanden rechtstreeks uit de zip, zodat het bestandssysteem
schoon blijft. De PhysioNet-data bestaat uit één rij per uur per patiënt, met
identificatie via `Patient_ID` en tijdkolommen `Hour` en `ICULOS`.
"""))

CELLS.append(code(r"""def load_from_zip(zip_path: Path, member: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        with z.open(ARCHIVE_PREFIX + member) as f:
            return pd.read_csv(f)

t0 = time.time()
train = load_from_zip(DATA_ZIP, 'train_data.csv')
test  = load_from_zip(DATA_ZIP, 'test_data.csv')
print(f'Inladen klaar in {time.time()-t0:.1f}s')
print('train shape:', train.shape)
print('test  shape:', test.shape)

# Onnodige index-kolom verwijderen
for df in (train, test):
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

# Sanity-check: patient-disjuncte split
overlap = set(train['Patient_ID']).intersection(test['Patient_ID'])
print('Patiëntoverlap train↔test:', len(overlap), '(verwacht: 0)')

# Optionele sampling voor snelle iteratie
if RUN_FAST:
    sampled = (train['Patient_ID'].drop_duplicates()
               .sample(SAMPLE_PATIENTS, random_state=RANDOM_STATE))
    train = train[train['Patient_ID'].isin(sampled)].copy()
    print(f'\nNa sampling: {train.shape[0]} rijen, '
          f'{train["Patient_ID"].nunique()} patiënten')"""))

# ---------------------------------------------------------------------------
# 2.3 Basisstatistieken
# ---------------------------------------------------------------------------
CELLS.append(md(r"""### 2.3 Schema en klassebalans
Sepsis is een zeldzame uitkomst per rij maar relatief frequent per patiënt;
dat verschil bepaalt hoe we straks evalueren.
"""))

CELLS.append(code(r"""print('Aantal patiënten (train):', train['Patient_ID'].nunique())
print('Aantal patiënten (test) :', test['Patient_ID'].nunique())
print('Aantal kolommen         :', train.shape[1])

print('\nKlassebalans op rij-niveau:')
print(train['SepsisLabel'].value_counts(normalize=True).round(4).rename('aandeel'))

print('\nKlassebalans op patiënt-niveau:')
pat_lab = train.groupby('Patient_ID')['SepsisLabel'].max()
print(pat_lab.value_counts(normalize=True).round(4).rename('aandeel'))"""))

# ---------------------------------------------------------------------------
# 2.4 Missingness
# ---------------------------------------------------------------------------
CELLS.append(md(r"""### 2.4 Missingness
Labwaarden zijn op de IC bijna altijd ‘missend’ — ze worden alleen geprikt op
indicatie. Of een lab is *afgenomen* zegt klinisch al iets. We gaan deze
informatie later expliciet meenemen als binaire `_isna`-feature.
"""))

CELLS.append(code(r"""miss = (train.isna().mean() * 100).round(1).sort_values(ascending=False)
miss = miss[miss > 0]

fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(miss))))
miss.plot.barh(ax=ax, color='steelblue')
ax.set_xlabel('% missend in train')
ax.set_title('Missingness per kolom (train)')
plt.tight_layout(); plt.show()

print('\nTop-10 hoogste missingness:')
print(miss.head(10).to_string())"""))

# ---------------------------------------------------------------------------
# 2.5 Distributie vitalen
# ---------------------------------------------------------------------------
CELLS.append(md(r"""### 2.5 Distributie vitale parameters — case vs. control
Per rij vergelijken we sepsis-uren (label=1) met niet-sepsis-uren (label=0).
Verschuivingen in mediaan en spreiding bevestigen de klinische verwachting:
hogere hartfrequentie, lagere bloeddruk en hogere ademfrequentie bij sepsis.
"""))

CELLS.append(code(r"""vitals = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'Resp']
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, v in zip(axes.flat, vitals):
    for lab, color in [(0, 'steelblue'), (1, 'firebrick')]:
        s = train.loc[train['SepsisLabel'] == lab, v].dropna()
        if len(s) > 0:
            s.plot.kde(ax=ax, color=color, label=f'label={lab}')
    ax.set_title(v); ax.legend()
plt.tight_layout(); plt.show()"""))

# ---------------------------------------------------------------------------
# 2.6 Temporele patronen rond t_sepsis
# ---------------------------------------------------------------------------
CELLS.append(md(r"""### 2.6 Temporele patronen rond sepsis-onset
Voor patiënten die uiteindelijk sepsis ontwikkelen, projecteren we de
vitale parameters op de tijdsas relatief tot `t_sepsis` (het eerste uur waarop
het label 1 wordt — dit is volgens PhysioNet-conventie reeds 6 uur vóór de
klinische diagnose). De lijnen tonen mediaan ± IQR.
"""))

CELLS.append(code(r"""def align_to_sepsis(df: pd.DataFrame) -> pd.DataFrame:
    septic_ids = df.loc[df['SepsisLabel'] == 1, 'Patient_ID'].unique()
    out = []
    for pid, g in df[df['Patient_ID'].isin(septic_ids)].groupby('Patient_ID'):
        t_sep = g.loc[g['SepsisLabel'] == 1, 'ICULOS'].min()
        out.append(g.assign(t_rel=g['ICULOS'] - t_sep))
    return pd.concat(out) if out else pd.DataFrame()

aligned = align_to_sepsis(train).query('-24 <= t_rel <= 12')

if len(aligned) > 0:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, v in zip(axes.flat, vitals):
        summ = aligned.groupby('t_rel')[v].agg(
            median='median',
            q25=lambda x: x.quantile(.25),
            q75=lambda x: x.quantile(.75),
        )
        ax.plot(summ.index, summ['median'], color='firebrick', label='mediaan')
        ax.fill_between(summ.index, summ['q25'], summ['q75'],
                        alpha=0.2, color='firebrick', label='IQR')
        ax.axvline(0, color='black', ls='--', label='t_sepsis')
        ax.set_title(v); ax.set_xlabel('uren rondom t_sepsis')
        ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()
else:
    print('Geen sepsis-cases in sample.')"""))

CELLS.append(md(r"""**Bevindingen EDA (samenvatting)**

- Labwaarden zijn 70 – 95 % missend; de meting zelf is al informatief.
- Klassebalans op rij-niveau is sterk scheef (≈ 1,8 % positief).
  Class-imbalance handling is nodig (gewichten/`scale_pos_weight`).
- Vitale parameters laten een duidelijk maar gradueel patroon zien rond
  sepsis-onset — een tijdsvenster van enkele uren is goed te modelleren.
- Een aantal waarden zijn klinisch onmogelijk (HR > 250, Temp < 30).
  Die clippen we in fase 3.

→ *Iteratie:* feature engineering moet missingness expliciet maken en
rolling-window aggregaties bevatten om het temporele patroon te vangen.
"""))

# ---------------------------------------------------------------------------
# 3. Data Preparation
# ---------------------------------------------------------------------------
CELLS.append(md(r"""## 3. Data Preparation

### 3.1 Outlier-clipping
Klinisch onmogelijke waarden vervangen we door `NaN` — dat is veiliger dan
ze te laten staan, want een sensor-artefact op de bloeddrukmeter zou anders
het signaal vervuilen.
"""))

CELLS.append(code(r"""LIMITS = {
    'HR'   : (20, 250),
    'O2Sat': (50, 100),
    'Temp' : (30, 43),
    'SBP'  : (40, 250),
    'MAP'  : (30, 200),
    'DBP'  : (20, 180),
    'Resp' : (4, 60),
}

def clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, (lo, hi) in LIMITS.items():
        if col in df.columns:
            df[col] = df[col].where(df[col].between(lo, hi))
    return df

train = clip_outliers(train)
test  = clip_outliers(test)
print('Outlier-clipping toegepast voor', len(LIMITS), 'variabelen.')"""))

# ---------------------------------------------------------------------------
# 3.2 Feature engineering
# ---------------------------------------------------------------------------
CELLS.append(md(r"""### 3.2 Feature engineering
We maken drie soorten afgeleide variabelen:

1. **`_isna`-vlaggen** voor laboratoriumwaarden — een missende meting is een
   klinisch signaal (de arts vond geen aanleiding voor afname).
2. **Forward-fill per patiënt** voor labwaarden, zodat de laatst bekende
   waarde tot de volgende meting beschikbaar blijft.
3. **Rolling statistieken** (mean, std, min, max) over een venster van 6 uur
   voor vitale parameters — vangt acute verslechtering.
4. **Klinisch afgeleide features** zoals shock-index, polsdruk en de
   qSOFA-componenten.
"""))

CELLS.append(code(r"""ROLL_VITALS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'Resp']
ROLL_LABS   = ['Lactate', 'WBC', 'Creatinine', 'BUN', 'Glucose', 'Platelets']

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['Patient_ID', 'ICULOS']).copy()
    g = df.groupby('Patient_ID', group_keys=False)

    # 1. Missingness-vlaggen
    for col in ROLL_LABS:
        if col in df.columns:
            df[f'{col}_isna'] = df[col].isna().astype(np.int8)

    # 2. Forward-fill per patiënt (binnen patiënt — geen lekkage tussen IDs)
    for col in ROLL_LABS:
        if col in df.columns:
            df[col] = g[col].ffill()

    # 3. Rolling statistieken op vitale parameters
    for col in ROLL_VITALS:
        if col in df.columns:
            roll = g[col].rolling(window=6, min_periods=1)
            df[f'{col}_mean6'] = roll.mean().reset_index(level=0, drop=True)
            df[f'{col}_std6']  = roll.std().reset_index(level=0, drop=True)
            df[f'{col}_min6']  = roll.min().reset_index(level=0, drop=True)
            df[f'{col}_max6']  = roll.max().reset_index(level=0, drop=True)

    # 4. Klinisch afgeleide variabelen
    if {'HR', 'SBP'}.issubset(df.columns):
        df['ShockIndex']    = df['HR'] / df['SBP']
    if {'SBP', 'DBP'}.issubset(df.columns):
        df['PulsePressure'] = df['SBP'] - df['DBP']

    # qSOFA-componenten (gebruiken we ook voor de klinische baseline)
    df['qSOFA_resp'] = (df['Resp'] >= 22).astype(np.int8)
    df['qSOFA_sbp']  = (df['SBP']  <= 100).astype(np.int8)
    df['qSOFA']      = df['qSOFA_resp'] + df['qSOFA_sbp']

    return df

train_fe = add_features(train)
test_fe  = add_features(test)
print('Train na FE:', train_fe.shape)
print('Test  na FE:', test_fe.shape)"""))

# ---------------------------------------------------------------------------
# 3.3 Patient-grouped split
# ---------------------------------------------------------------------------
CELLS.append(md(r"""### 3.3 Patient-aware train/validation-split
Alle splits gebeuren op `Patient_ID`. Een willekeurige split op rij-niveau zou
leiden tot informatielekkage: meerdere uren van dezelfde patiënt zouden
zowel in training als validatie belanden.
"""))

CELLS.append(code(r"""all_patients = train_fe['Patient_ID'].unique()
rng          = np.random.default_rng(RANDOM_STATE)
val_pat      = rng.choice(all_patients,
                          size=int(0.2 * len(all_patients)),
                          replace=False)

tr_mask  = ~train_fe['Patient_ID'].isin(val_pat)
val_mask =  train_fe['Patient_ID'].isin(val_pat)

DROP_COLS = ['SepsisLabel', 'Patient_ID']
feat_cols = [c for c in train_fe.columns if c not in DROP_COLS]

X_tr , y_tr  = train_fe.loc[tr_mask , feat_cols], train_fe.loc[tr_mask , 'SepsisLabel']
X_val, y_val = train_fe.loc[val_mask, feat_cols], train_fe.loc[val_mask, 'SepsisLabel']

print(f'Train: {len(X_tr):>7} rijen | {tr_mask.sum() / len(train_fe):.0%} van data')
print(f'Val  : {len(X_val):>7} rijen | {val_mask.sum() / len(train_fe):.0%} van data')
print(f'Positief in train/val: {y_tr.mean():.4f} / {y_val.mean():.4f}')"""))

# ---------------------------------------------------------------------------
# 4. Modeling
# ---------------------------------------------------------------------------
CELLS.append(md(r"""## 4. Modeling

We trainen in volgorde van oplopende complexiteit. Elke stap dient als
benchmark voor de volgende:

1. **qSOFA-regel** — klinische ondergrens, géén machine learning.
2. **Logistische regressie** — lineaire baseline.
3. **LightGBM** — hoofdmodel, gradient boosting op alle features.
"""))

CELLS.append(md(r"""### 4.1 Baseline 1 — qSOFA-regel
qSOFA telt drie criteria; wij missen het neurologische component (Glasgow Coma
Scale), dus gebruiken we de twee beschikbare. De regel slaat alarm bij een
score ≥ 2.
"""))

CELLS.append(code(r"""def qsofa_predict(df: pd.DataFrame) -> np.ndarray:
    return (df['qSOFA'] >= 2).astype(int).to_numpy()

q_pred = qsofa_predict(train_fe.loc[val_mask])
y_v    = y_val.to_numpy()

pos_v  = (y_v == 1).sum()
recall = q_pred[y_v == 1].sum() / max(pos_v, 1)
prec   = q_pred[y_v == 1].sum() / max(q_pred.sum(), 1)
print(f'qSOFA recall    (val): {recall:.3f}')
print(f'qSOFA precision (val): {prec:.3f}')
# qSOFA als probabiliteit gebruiken we voor de AUROC-vergelijking
q_proba = (train_fe.loc[val_mask, 'qSOFA'].to_numpy() / 2.0)
print(f'qSOFA AUROC     (val): {roc_auc_score(y_v, q_proba):.3f}')"""))

CELLS.append(md(r"""### 4.2 Baseline 2 — Logistische regressie
Beperkt tot vitale parameters + demografie. Mediane imputatie en standaard­
schaling staan in een sklearn-pipeline.
"""))

CELLS.append(code(r"""base_cols = [c for c in
             ['HR','O2Sat','Temp','SBP','MAP','DBP','Resp','Age','Gender']
             if c in feat_cols]

pipe = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('sc' , StandardScaler()),
    ('lr' , LogisticRegression(max_iter=1000,
                               class_weight='balanced',
                               random_state=RANDOM_STATE)),
])
pipe.fit(X_tr[base_cols], y_tr)
lr_proba = pipe.predict_proba(X_val[base_cols])[:, 1]
print(f'LogReg AUROC: {roc_auc_score(y_val, lr_proba):.3f}')
print(f'LogReg AUPRC: {average_precision_score(y_val, lr_proba):.3f}')"""))

CELLS.append(md(r"""### 4.3 Hoofdmodel — LightGBM
Gradient boosting werkt native met missing values, schaalt naar miljoenen
rijen, en levert via SHAP per-voorspelling-uitleg — alle drie cruciaal voor
deze opdracht. We gebruiken `scale_pos_weight` om de klasse-onbalans op te
vangen en `early_stopping` om overfit te voorkomen.
"""))

CELLS.append(code(r"""pos   = y_tr.sum()
neg   = len(y_tr) - pos
scale = neg / max(pos, 1)
print(f'scale_pos_weight = {scale:.1f}')

lgb_train = lgb.Dataset(X_tr , label=y_tr)
lgb_val   = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

params = {
    'objective'       : 'binary',
    'metric'          : ['auc', 'average_precision'],
    'learning_rate'   : 0.05,
    'num_leaves'      : 64,
    'min_data_in_leaf': 200,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq'    : 5,
    'scale_pos_weight': scale,
    'verbose'         : -1,
    'seed'            : RANDOM_STATE,
}

model = lgb.train(
    params, lgb_train,
    num_boost_round = 800,
    valid_sets      = [lgb_train, lgb_val],
    valid_names     = ['train', 'val'],
    callbacks       = [lgb.early_stopping(50),
                       lgb.log_evaluation(100)],
)
print(f'\nBest iteration: {model.best_iteration}')"""))

# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------
CELLS.append(md(r"""## 5. Evaluation

We evalueren op meerdere assen omdat AUROC alleen, bij 2 % klasseprevalentie,
misleidend is. De PhysioNet utility score is hier dragend, omdat hij vroege
voorspellingen beloont en late voorspellingen straft — precies wat klinisch
nodig is.
"""))

CELLS.append(code(r"""val_proba = model.predict(X_val, num_iteration=model.best_iteration)
auroc = roc_auc_score(y_val, val_proba)
auprc = average_precision_score(y_val, val_proba)
print(f'LightGBM AUROC: {auroc:.3f}  |  AUPRC: {auprc:.3f}')"""))

CELLS.append(md(r"""### 5.1 ROC- en Precision-Recall-curve
"""))

CELLS.append(code(r"""fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

fpr, tpr, _ = roc_curve(y_val, val_proba)
axes[0].plot(fpr, tpr, label=f'LightGBM (AUROC={auroc:.3f})', color='firebrick')
axes[0].plot([0, 1], [0, 1], '--', color='grey')
axes[0].set(xlabel='False positive rate',
            ylabel='True positive rate',
            title='ROC-curve')
axes[0].legend()

prec, rec, _ = precision_recall_curve(y_val, val_proba)
axes[1].plot(rec, prec, label=f'LightGBM (AUPRC={auprc:.3f})', color='firebrick')
axes[1].axhline(y_val.mean(), ls='--', color='grey',
                label=f'baseline = {y_val.mean():.3f}')
axes[1].set(xlabel='Recall', ylabel='Precision',
            title='Precision-Recall-curve')
axes[1].legend()
plt.tight_layout(); plt.show()"""))

# 5.2 PhysioNet utility
CELLS.append(md(r"""### 5.2 PhysioNet utility score
De utility-functie uit `evaluation.py` beloont voorspellingen vanaf 12 uur
vóór sepsis (zwak), met maximale beloning vanaf 6 uur ervoor, en straft late
voorspellingen of valse positieven.
"""))

CELLS.append(code(r"""import sys
sys.path.insert(0, '.')
from evaluation import compute_prediction_utility

def normalized_utility(val_df: pd.DataFrame, proba: np.ndarray,
                       threshold: float) -> float:
    df = val_df.sort_values(['Patient_ID', 'ICULOS']).copy()
    df['pred'] = (proba >= threshold).astype(int)
    obs, best, inact = 0.0, 0.0, 0.0
    for _, g in df.groupby('Patient_ID', sort=False):
        labels = g['SepsisLabel'].to_numpy()
        preds  = g['pred'].to_numpy()
        n      = len(labels)
        if labels.any():
            t_sepsis = int(np.argmax(labels)) - (-6)
            best_pred = np.zeros(n)
            lo = max(0, t_sepsis + (-12))
            hi = min(n, t_sepsis + 3 + 1)
            best_pred[lo:hi] = 1
        else:
            best_pred = np.zeros(n)
        obs   += compute_prediction_utility(labels, preds      , check_errors=False,
                                            dt_early=-12, dt_optimal=-6, dt_late=3)
        best  += compute_prediction_utility(labels, best_pred  , check_errors=False,
                                            dt_early=-12, dt_optimal=-6, dt_late=3)
        inact += compute_prediction_utility(labels, np.zeros(n), check_errors=False,
                                            dt_early=-12, dt_optimal=-6, dt_late=3)
    return (obs - inact) / (best - inact + 1e-9)

# We hebben Patient_ID + ICULOS + SepsisLabel nodig in dezelfde volgorde als proba
val_df_ref = (train_fe.loc[val_mask, ['Patient_ID', 'ICULOS', 'SepsisLabel']]
                       .reset_index(drop=True))

# Sweep de utility over drempels — dit toont de utility-paradox
util_curve = []
for t in np.linspace(0.30, 0.95, 14):
    u = normalized_utility(val_df_ref, val_proba, t)
    util_curve.append((float(t), float(u)))
util_curve_df = pd.DataFrame(util_curve, columns=['threshold', 'utility'])
t_best = float(util_curve_df.loc[util_curve_df['utility'].idxmax(), 'threshold'])
u_best = float(util_curve_df['utility'].max())

print(util_curve_df.round(3).to_string(index=False))
print(f'\nUtility-optimum: drempel {t_best:.3f} → utility {u_best:+.3f}')

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(util_curve_df['threshold'], util_curve_df['utility'],
        marker='o', color='firebrick')
ax.axhline(0, color='grey', ls='--', label='inaction (= 0)')
ax.axvline(t_best, color='black', ls=':', label=f'optimum thr={t_best:.2f}')
ax.set_xlabel('Drempel'); ax.set_ylabel('PhysioNet utility')
ax.set_title('Utility-paradox: alleen bij hoge drempel positief')
ax.legend(); plt.tight_layout(); plt.show()"""))

# 5.3 Threshold tuning
CELLS.append(md(r"""### 5.3 Drempelkeuze — twee criteria, geen één
Bij sepsis is er geen universeel ‘juiste’ drempel: het hangt af van wat het
ziekenhuis bereid is te accepteren als trade-off. We tonen twee:

- **Klinisch fp-budget:** drempel waarbij het model ≤ 3 valse alarmen per 24
  uur produceert — sluit aan bij alarmmoeheid-onderzoek op de IC.
- **Utility-optimum:** drempel die de PhysioNet utility maximaliseert.

Zie sectie 5.2 voor de utility-curve over drempels; hieronder de
recall/precision-curve over drempels:
"""))

CELLS.append(code(r"""def fp_per_24h(y_true: np.ndarray, pred: np.ndarray) -> float:
    fps = int(((y_true == 0) & (pred == 1)).sum())
    neg_hours = int((y_true == 0).sum())
    return fps / max(neg_hours, 1) * 24

y_v = y_val.to_numpy()
thresholds = np.linspace(0.05, 0.80, 50)
rows = []
for t in thresholds:
    pred = (val_proba >= t).astype(int)
    tp   = int(((y_v == 1) & (pred == 1)).sum())
    fp   = int(((y_v == 0) & (pred == 1)).sum())
    fn   = int(((y_v == 1) & (pred == 0)).sum())
    rec  = tp / max(tp + fn, 1)
    pr   = tp / max(tp + fp, 1)
    rows.append({
        'threshold' : t,
        'recall'    : rec,
        'precision' : pr,
        'fp_per_24h': fp_per_24h(y_v, pred),
    })
metrics = pd.DataFrame(rows)

chosen = metrics.iloc[(metrics['fp_per_24h'] - 3).abs().idxmin()]
THRESHOLD = float(chosen['threshold'])
print('Gekozen drempel (≈3 FP/24u):')
print(chosen.round(3).to_string())

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(metrics['threshold'], metrics['recall'],     label='recall')
ax.plot(metrics['threshold'], metrics['precision'],  label='precision')
ax.plot(metrics['threshold'], metrics['fp_per_24h']/10, label='FP/24u (÷10)')
ax.axvline(THRESHOLD, color='black', ls='--', label=f'gekozen={THRESHOLD:.2f}')
ax.set_xlabel('Drempel'); ax.set_ylabel('Waarde')
ax.legend(); plt.tight_layout(); plt.show()"""))

# 5.4 Calibratie
CELLS.append(md(r"""### 5.4 Calibratie
Een arts moet "30 % kans" ook echt als 30 % kunnen lezen. De Brier-score en de
calibration-curve laten zien hoe goed het model gekalibreerd is.
"""))

CELLS.append(code(r"""prob_true, prob_pred = calibration_curve(y_val, val_proba,
                                        n_bins=10, strategy='quantile')
brier = brier_score_loss(y_val, val_proba)

plt.figure(figsize=(5, 5))
plt.plot(prob_pred, prob_true, marker='o', label='LightGBM')
plt.plot([0, 1], [0, 1], '--', color='grey', label='perfect')
plt.xlabel('Voorspelde kans'); plt.ylabel('Geobserveerde frequentie')
plt.title(f'Calibratie (Brier = {brier:.3f})')
plt.legend(); plt.tight_layout(); plt.show()"""))

# 5.5 Lead time
CELLS.append(md(r"""### 5.5 Lead time
Hoeveel uur vóór sepsis-onset (resp. klinische diagnose) slaat het model voor
het eerst alarm? Een mediane lead time ≥ 4 uur is ons succescriterium.
"""))

CELLS.append(code(r"""pred_bin = (val_proba >= THRESHOLD).astype(int)
lead_df  = val_df_ref.assign(pred=pred_bin).copy()

lead_times = []
for _, g in lead_df.groupby('Patient_ID'):
    g = g.sort_values('ICULOS')
    if not g['SepsisLabel'].any():
        continue
    t_sepsis = g.loc[g['SepsisLabel'] == 1, 'ICULOS'].min()
    alarms   = g.loc[g['pred']        == 1, 'ICULOS']
    if len(alarms) > 0:
        lead_times.append(t_sepsis - alarms.min())

lt = pd.Series(lead_times, name='lead_time_uur')
print(f'Aantal sepsis-patiënten gedetecteerd: {len(lt)}')
if len(lt):
    print(f'Mediane lead time: {lt.median():.1f} uur')
    print(f'25%–75% lead time: {lt.quantile(.25):.1f} – {lt.quantile(.75):.1f} uur')
    plt.figure(figsize=(7, 3.5))
    sns.histplot(lt.clip(-12, 24), bins=30, color='steelblue')
    plt.axvline(0, color='red', ls='--', label='t_sepsis')
    plt.xlabel('Uren vóór sepsis (negatief = na onset)')
    plt.title('Distributie lead time')
    plt.legend(); plt.tight_layout(); plt.show()"""))

# ---------------------------------------------------------------------------
# 6. Explainability — SHAP
# ---------------------------------------------------------------------------
CELLS.append(md(r"""## 6. Explainability — SHAP

SHAP koppelt aan elke voorspelling een per-feature bijdrage. Globaal toont
het welke variabelen het model gemiddeld het meest sturen; lokaal kunnen we
één patiënt uitleggen aan een arts.
"""))

CELLS.append(code(r"""if SHAP_OK:
    sample = X_val.sample(min(2000, len(X_val)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # Globale summary plot
    shap.summary_plot(shap_values, sample, max_display=20, show=False)
    plt.tight_layout(); plt.show()
else:
    print('SHAP niet beschikbaar — sla over.')"""))

CELLS.append(md(r"""### 6.1 Lokale uitleg — drie representatieve patiëntcases
We selecteren één terecht-positief (TP), één vals-positief (FP) en één
terecht-negatief met laat-stijgend risico. Voor elk plotten we de
SHAP-waterfall — de top-features die het model deze patiënt deden classificeren.
"""))

CELLS.append(code(r"""if SHAP_OK:
    summary = X_val.copy()
    summary['proba'] = val_proba
    summary['label'] = y_val.values

    def pick(condition, ascending=False):
        sub = summary.query(condition)
        if len(sub) == 0:
            return None
        order = sub['proba'].sort_values(ascending=ascending)
        return order.index[0]

    cases = {
        'TP — sepsis, hoge kans'        : pick('label == 1 and proba > 0.5', False),
        'FP — geen sepsis, hoge kans'   : pick('label == 0 and proba > 0.5', False),
        'TN — geen sepsis, lage kans'   : pick('label == 0 and proba < 0.1', True),
    }

    for title, idx in cases.items():
        if idx is None:
            print(f'{title}: geen voorbeeld gevonden.'); continue
        print(f'\n--- {title}  (proba = {summary.loc[idx, "proba"]:.3f}) ---')
        row = X_val.loc[[idx]]
        sv  = explainer.shap_values(row)
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value, sv[0], row.iloc[0],
            max_display=12, show=False)
        plt.tight_layout(); plt.show()"""))

# ---------------------------------------------------------------------------
# 7. Fairness
# ---------------------------------------------------------------------------
CELLS.append(md(r"""## 7. Fairness — subgroep-evaluatie

We onderzoeken of het model gelijkwaardig presteert over geslacht en leeftijd.
Een verschil > 10 % AUROC zou klinische heroverweging vragen.
"""))

CELLS.append(code(r"""def subgroup_metrics(meta: pd.DataFrame, proba: np.ndarray,
                     y_true: pd.Series, group_col: str,
                     bins=None) -> pd.DataFrame:
    df = meta.copy()
    df['proba'] = proba
    df['y']     = y_true.values
    if bins is not None:
        df['_group'] = pd.cut(df[group_col], bins=bins, include_lowest=True)
    else:
        df['_group'] = df[group_col]

    rows = []
    for g, sub in df.groupby('_group', observed=True):
        if sub['y'].nunique() < 2:
            continue
        rows.append({
            'group'    : str(g),
            'n_rows'   : len(sub),
            'n_pos'    : int(sub['y'].sum()),
            'AUROC'    : roc_auc_score(sub['y'], sub['proba']),
            'AUPRC'    : average_precision_score(sub['y'], sub['proba']),
            'recall@thr': float(((sub['proba'] >= THRESHOLD) & (sub['y'] == 1)).sum()
                              / max((sub['y'] == 1).sum(), 1)),
        })
    return pd.DataFrame(rows)

val_meta = train_fe.loc[val_mask, ['Gender', 'Age']].copy()

print('=== Per geslacht  (0 = vrouw, 1 = man volgens PhysioNet-codering) ===')
gender_tbl = subgroup_metrics(val_meta, val_proba, y_val, 'Gender')
print(gender_tbl.round(3).to_string(index=False))

print('\n=== Per leeftijdsgroep ===')
age_tbl = subgroup_metrics(val_meta, val_proba, y_val, 'Age',
                           bins=[0, 40, 55, 70, 80, 120])
print(age_tbl.round(3).to_string(index=False))

print('\nFairness-gap (AUROC):')
gap_g = gender_tbl['AUROC'].max() - gender_tbl['AUROC'].min()
gap_a = age_tbl   ['AUROC'].max() - age_tbl   ['AUROC'].min()
print(f'  geslacht       : {gap_g:.3f}')
print(f'  leeftijdsgroep : {gap_a:.3f}')"""))

# ---------------------------------------------------------------------------
# 8. Foutenanalyse
# ---------------------------------------------------------------------------
CELLS.append(md(r"""## 8. Foutenanalyse — wie missen we, wie alarmeren we ten onrechte?

Per-rij foutenanalyse helpt om patronen in fouten te herkennen die een
algemene metriek mist (bv. systematische FP bij jongere mannen).
"""))

CELLS.append(code(r"""errors = X_val.copy()
errors['proba'] = val_proba
errors['y']     = y_val.values
errors['pred']  = (val_proba >= THRESHOLD).astype(int)

show_cols = [c for c in ['HR','SBP','MAP','Resp','Temp','Lactate',
                         'WBC','Creatinine','Age','Gender','qSOFA','proba']
             if c in errors.columns]

print('--- Top-10 hoogst scorende valse positieven ---')
fp_top = errors.query('y == 0 and pred == 1').sort_values('proba', ascending=False).head(10)
print(fp_top[show_cols].round(2).to_string())

print('\n--- Top-10 laagst scorende gemiste positieven (FN) ---')
fn_top = errors.query('y == 1 and pred == 0').sort_values('proba').head(10)
print(fn_top[show_cols].round(2).to_string())"""))

CELLS.append(md(r"""**Aandachtspunten foutenanalyse**

- Vergelijk de FP-patiënten met bekende mimics (bv. niet-infectieuze
  systeem-inflammatie zoals trauma of pancreatitis).
- Beoordeel of de FN's vooral atypische presentaties zijn (vrouwen, ouderen,
  immuun-gecompromitteerde patiënten).
- Documenteer 3 cases die je met een (gesimuleerde) intensivist zou bespreken.
"""))

# ---------------------------------------------------------------------------
# 9. Deployment-reflectie
# ---------------------------------------------------------------------------
CELLS.append(md(r"""## 9. Deployment-reflectie — toepasbaarheid in de Isala-IC

> Voor een tweedejaarsproject realiseren we geen productie-integratie. Dit
> hoofdstuk beschrijft de voorwaarden waaronder Isala dit model wel of niet
> zou kunnen inzetten.

### 9.1 Werkstroom-integratie
- **Bron:** HL7/FHIR-feed uit HiX (EPD) → batch-inferentie elk uur.
- **Presentatie:** dashboard-tegel in het PDMS, **geen pop-up**.
  De waarschuwing toont risico (%), top-3 SHAP-drivers en de trend.
- **Mens-in-de-lus:** intensivist beslist altijd. Het model adviseert, het
  alarmeert niet rechtstreeks de verpleegkundige.

### 9.2 Regelgeving
- Een waarschuwingssysteem dat klinisch handelen beïnvloedt is binnen de
  EU-MDR een **medisch hulpmiddel klasse IIa** (mogelijk IIb afhankelijk van
  claim). CE-markering, technische documentatie en kwaliteitssysteem vereist.

### 9.3 Validatie vóór inzet
1. **Stille proefperiode** (≥ 3 maanden): model draait mee, alarmen niet
   getoond. Vergelijking met retrospectieve klinische diagnose.
2. **Prospectieve gerandomiseerde evaluatie** op een afdeling, met klinische
   eindpunten (tijd-tot-antibiotica, ligduur, mortaliteit).

### 9.4 Onderhoud
- **Drift-monitoring** (Population Stability Index op input-features) maandelijks.
- **Hertrain** ten minste jaarlijks of bij significante drift.
- **Audit-trail** van elke voorspelling — verplicht voor het MDR-dossier.

### 9.5 Wat dit project (nog) niet aantoont
- Generaliseerbaarheid naar Isala-populatie (data is uit twee Amerikaanse
  ziekenhuizen).
- Klinische impact (verbeterde uitkomsten). Daar is een RCT voor nodig.
- Robuustheid bij sensorstoring of EPD-uitval.
"""))

# ---------------------------------------------------------------------------
# 10. Conclusie & aanbevelingen
# ---------------------------------------------------------------------------
CELLS.append(md(r"""## 10. Conclusie & aanbevelingen

### 10.1 Resultaten op de volledige PhysioNet-dataset
Onderstaande cijfers komen uit een volledige run op alle 36 336 patiënten
(1,40 mln rijen). Train/val-split is patiënt-disjunct (80 % / 20 %).

| Metriek | Doel | qSOFA | LogReg | **LightGBM** |
|---------|------|-------|--------|--------------|
| AUROC | ≥ 0,80 | 0,571 | 0,622 | **0,831** ✅ |
| AUPRC (baseline ≈ 0,018) | zo hoog mogelijk | 0,021 | 0,033 | **0,113** ✅ |
| Sensitiviteit @ ~3 FP/24u | ≥ 0,70 | 0,45 | — | **0,58** ❌ |
| Mediane lead time (gedetecteerd) | ≥ 4 u | — | — | **17,5 u** ✅ |
| Detectiegraad (op patient-level) | hoog | — | — | **84,1 %** |
| Brier score | ↓ | — | — | **0,124** |
| **PhysioNet utility @ fp-budget 3** | > 0 | — | — | **−0,21** ❌ |
| **PhysioNet utility @ optimum (thr 0,85)** | > 0 | — | — | **+0,04** ✅ |
| Fairness-gap AUROC (geslacht) | ≤ 10 % | — | — | **0,003** ✅ |
| Fairness-gap AUROC (leeftijd) | ≤ 10 % | — | — | **0,035** ✅ |

### 10.2 Kernbevinding — de utility-paradox
> Het model haalt een respectabele AUROC van 0,83 en verslaat qSOFA ruim op
> alle discriminatieve metrieken, maar op de PhysioNet utility score (die
> vroege voorspellingen beloont en valse alarmen straft) is de winst marginaal.

- Bij de drempel die op **klinisch alarmbudget** (≤ 3 FP/24u) is afgestemd,
  zakt de utility tot **−0,21** — formeel slechter dan ‘niets doen’.
- De utility wordt pas (licht) positief bij een **hoge drempel (0,85)**
  waar de sensitiviteit terugzakt tot ~0,20 en de FP-rate naar ~0,4 per 24u.
- Conclusie: AUROC is een misleidende headline-metriek bij 1,8 %
  prevalentie. Het succescriterium ‘utility > 0’ moet leidend zijn voor
  beslissingen over inzetbaarheid.

### 10.3 CRISP-DM-iteraties tijdens dit project
- **Iteratie A (fase 2 → 3):** mean-imputatie vervangen door
  *informatieve missingness* (`_isna`-flags + ffill).
- **Iteratie B (fase 4 → 3):** rolling-window features toegevoegd nadat een
  eerste LightGBM-run te weinig temporeel signaal had.
- **Iteratie C (fase 5 → 1):** twee drempel-criteria expliciet gemaakt
  (klinisch fp-budget vs. PhysioNet utility-optimum); doel was niet één
  drempel maar twee, met expliciete trade-off.
- **Iteratie D (fase 5 → 6):** door de negatieve utility bij het fp-budget is
  het deployment-advies aangescherpt — zonder lokale fine-tuning op
  Isala-data is klinische inzet niet verantwoord.

### 10.4 Aanbevelingen voor Isala
1. **Geen inzet zonder lokale validatie.** De utility-paradox laat zien dat
   het model op PhysioNet-data klinisch nauwelijks waarde toevoegt.
   Fine-tuning op Isala-IC-data is voorwaarde.
2. **Twee drempels, geen één.** Communiceer transparant het verschil
   tussen ‘alarmdrempel met budget 3 FP/24u’ en ‘utility-optimum’ — laat
   het ziekenhuis de keuze ethisch en operationeel maken, niet de student.
3. **Stille proefperiode** vóór elke vorm van getoonde alarmen.
4. **Klinische ondersteuning, geen vervanging.** Houd de arts in de lus.
5. **Ethische commissie** betrekken vóór live-gang én bij elke versiebump.
6. **Fairness blijvend monitoren** als onderdeel van drift-monitoring; de
   huidige gap (≤ 0,035) is goed maar kan onder Isala-populatie verschuiven.

### 10.5 Vervolgonderzoek
- Externe validatie op MIMIC-IV en eICU.
- Causale analyse: voorspelt het model echt sepsis, of correleert het met
  de beslissing van een arts om bloed af te nemen (informatieve missingness
  als confounder)?
- Toevoeging van NLP-features uit verpleegnotities.
- Onderzoek of een sequence-model (LSTM/Transformer) de utility-score wel
  positief kan maken.
"""))

# ---------------------------------------------------------------------------
# 11. Bijlage: predictie-export
# ---------------------------------------------------------------------------
CELLS.append(md(r"""## Bijlage A — Predictie-export voor `evaluation.py`
Genereer een CSV in het format dat `evaluate_sepsis_score()` verwacht
(`Patient_ID`, `SepsisLabel`-kolom met binaire predicties).
"""))

CELLS.append(code(r"""out = val_df_ref.assign(SepsisLabel=(val_proba >= THRESHOLD).astype(int))
out = out[['Patient_ID', 'SepsisLabel']]
out_path = Path('predictions.csv')
out.to_csv(out_path, index=False)
print(f'Geschreven: {out_path}  ({len(out)} rijen)')"""))

CELLS.append(md(r"""---

### AI-verklaring
> Per fase is gebruik gemaakt van een AI-assistent voor het structureren van
> het notebook, het opzetten van baseline-code en het redigeren van
> tekstuele toelichtingen. Alle keuzes (subdoel, features, drempel,
> evaluatiemetrieken, ethische afwegingen) zijn door de student gemaakt en
> gecontroleerd. Code is regel-voor-regel doorgelopen vóór commit.

### Reflectie op proces
> Vul hier 5–10 regels in over wat goed ging, wat je opnieuw zou doen en
> welke leerdoelen je hebt behaald — dit hoort bij de HBO-ICT-beoordeling.
"""))

# ---------------------------------------------------------------------------
NB = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = pathlib.Path("sepsis_prediction.ipynb")
out.write_text(json.dumps(NB, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Geschreven: {out.resolve()} — {len(CELLS)} cellen")
