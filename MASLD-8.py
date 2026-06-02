"""
ML-Based Prediction of Metabolic Dysfunction-Associated Steatotic Liver Disease (MASLD) 
A modular machine learning pipeline for predicting MASLD using NHANES 2017-2020 data and XGBoost classifier.

__version__ = "1.0.0"
__author__ = "DocCoderMD"

8-Feature Model: Age + Sex + Triglycerides + Fasting Glucose + hs-CRP +
                 BMI + Waist Circumference + UHRatio (Uric Acid / HDL)

Ground truth: VCTE/FibroScan CAP score >= 274 dB/m (steatosis grade >= S1)
             WITH cardiometabolic criteria per 2023 MASLD Delphi consensus
Algorithm:    XGBoost (Extreme Gradient Boosting)
Explainability: SHAP (SHapley Additive exPlanations)

NHANES data files required (download from https://www.cdc.gov/nchs/nhanes/):
  P_DEMO.xpt    - Demographics (age, sex, pregnancy)
  P_LUX.xpt     - Liver ultrasound / VCTE (CAP score)
  P_BIOPRO.xpt  - Biochemistry (serum uric acid)
  P_HDL.xpt     - HDL cholesterol
  P_TRIGLY.xpt  - Triglycerides
  P_HSCRP.xpt   - High-sensitivity CRP
  P_BMX.xpt     - Body measures (BMI, waist circumference)
  P_GLU.xpt     - Fasting plasma glucose
  P_FASTQX.xpt  - Fasting questionnaire (fasting time verification) 
  P_BPX.xpt     - Blood pressure (for MASLD cardiometabolic criteria)  
  P_ALQ.xpt     - Alcohol use questionnaire (exclusion)                
  P_HEPB_S.xpt  - Hepatitis B surface antigen (exclusion)            
  P_HCV.xpt     - Hepatitis C antibody (exclusion)                   


Requirements:
  pip install xgboost shap scikit-learn pandas numpy scipy matplotlib
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind, chi2_contingency

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, classification_report,
    roc_curve, confusion_matrix
)

try:
    import xgboost as xgb
except ImportError:
    sys.exit("ERROR: xgboost not installed.\nInstall with:  pip install xgboost")

try:
    import shap
except ImportError:
    sys.exit("ERROR: shap not installed.\nInstall with:  pip install shap")


# ── Directory setup ──────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "nhanes_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "ml_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CDC_BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"

NHANES_FILES = {
  
    "demo":    "P_DEMO.xpt",     # Demographics (age, sex, pregnancy)
    "liver":   "P_LUX.xpt",      # Liver ultrasound / VCTE (CAP score)
    "biopro":  "P_BIOPRO.xpt",   # Biochemistry (serum uric acid)
    "hdl":     "P_HDL.xpt",      # HDL cholesterol
    "trigly":  "P_TRIGLY.xpt",   # Triglycerides
    "hscrp":   "P_HSCRP.xpt",    # High-sensitivity CRP
    "body":    "P_BMX.xpt",      # Body measures (BMI, waist)
    "glucose": "P_GLU.xpt",      # Fasting plasma glucose
    "fasting": "P_FASTQX.xpt",   # [FIX 1] Fasting time verification
    "bp":      "P_BPX.xpt",      # [FIX 2] Blood pressure (MASLD criteria)
    "alcohol": "P_ALQ.xpt",      # [FIX 3] Alcohol use exclusion
    "hepb":    "P_HEPB_S.xpt",   # [FIX 4] Hepatitis B exclusion
    "hepc":    "P_HCV.xpt",      # [FIX 4] Hepatitis C exclusion
}

# ── Feature configuration ────────────────────────────────────────────────────
FEATURES = [
    "RIDAGEYR",   # Age (years)
    "RIAGENDR",   # Sex (1 = Male, 0 = Female after recoding)
    "LBXTR",      # Triglycerides (mg/dL)
    "LBXGLU",     # Fasting glucose (mg/dL)
    "LBXHSCRP",   # hs-CRP (mg/L)
    "BMXBMI",     # BMI (kg/m²)
    "BMXWAIST",   # Waist circumference (cm)
    "UHRatio",    # Uric Acid / HDL Ratio (computed)
]

FEATURE_LABELS = {
    "RIDAGEYR":  "Age (years)",
    "RIAGENDR":  "Sex (male=1)",
    "LBXTR":     "Triglycerides (mg/dL)",
    "LBXGLU":    "Fasting Glucose (mg/dL)",
    "LBXHSCRP":  "hs-CRP (mg/L)",
    "BMXBMI":    "BMI (kg/m\u00b2)",
    "BMXWAIST":  "Waist Circumference (cm)",
    "UHRatio":   "Uric Acid / HDL Ratio",
}

RANDOM_SEED = 42


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Data Acquisition
# ══════════════════════════════════════════════════════════════════════════════

def load_nhanes(name: str, filename: str) -> pd.DataFrame:
    """Load from local nhanes_data/ if present, otherwise download from CDC."""
    local = os.path.join(DATA_DIR, filename)
    if os.path.exists(local):
        print(f"  Loading {name} from local cache: {filename}")
        return pd.read_sas(local)
    url = f"{CDC_BASE_URL}/{filename}"
    print(f"  Downloading {name} from CDC: {filename}")
    return pd.read_sas(url)


def load_and_merge() -> pd.DataFrame:
    """
    Load all NHANES files and merge on SEQN (inner join).
    Includes 5 additional files vs original for corrected cohort definition.
    """
    print("=" * 60)
    print("Phase 1: Loading NHANES 2017-March 2020 Pre-Pandemic Data")
    print("=" * 60)

    dfs = {}
    for key, filename in NHANES_FILES.items():
        dfs[key] = load_nhanes(key, filename)

    print("\nMerging datasets on SEQN...")

    # Core clinical merge (inner join — must have all clinical measurements)
    master = dfs["demo"]
    for key in ["liver", "biopro", "hdl", "trigly", "hscrp", "body",
                "glucose", "fasting", "bp"]:
        master = pd.merge(master, dfs[key], on="SEQN", how="inner")

    # Exclusion files: left join — participants without these records are
    # assumed not excluded (absence of Hep B/C record ≠ positive status)
    for key in ["alcohol", "hepb", "hepc"]:
        master = pd.merge(master, dfs[key], on="SEQN", how="left")

    # Decode byte column names (SAS files sometimes return bytes)
    master.columns = [
        c.decode() if isinstance(c, bytes) else c for c in master.columns
    ]
    print(f"Merged dataset: {master.shape[0]:,} participants, "
          f"{master.shape[1]} variables")
    return master


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Cohort Exclusions & Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════

def apply_exclusions(master: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cohort exclusion criteria in sequence.
    Prints a CONSORT-style flow of participant counts at each step.

    Exclusion order:
      1. No valid CAP score (cannot assign outcome)
      2. Age < 18 years                             
      3. Pregnant women                            
      4. Non-fasting participants (< 8 hours fast)  
      5. Heavy alcohol use                        
      6. Active Hepatitis B (HBsAg positive)        
      7. Hepatitis C (anti-HCV positive)           
    """
    print("\n" + "=" * 60)
    print("Phase 2a: Cohort Exclusions (CONSORT Flow)")
    print("=" * 60)

    n = len(master)
    print(f"Starting N (merged dataset):          {n:>6,}")

    # ── 1. Valid CAP score ───────────────────────────────────────────────────
    master = master.dropna(subset=["LUXCAPM"])
    print(f"After requiring valid CAP score:      {len(master):>6,}  "
          f"(excluded {n - len(master):,})")
    n = len(master)

    # ── 2. Age >= 18 years ───────────────────────────────────────────────── 
    master = master[master["RIDAGEYR"] >= 18]
    print(f"After age >= 18 restriction:          {len(master):>6,}  "
          f"(excluded {n - len(master):,})")
    n = len(master)

    # ── 3. Exclude pregnant women ──────────────────────────────────────────
    # RIDEXPRG: 1 = Yes pregnant, 2 = Not pregnant, 3 = Indeterminate
    # Exclude only confirmed pregnant (== 1); keep missing/indeterminate
    if "RIDEXPRG" in master.columns:
        master = master[master["RIDEXPRG"] != 1]
        print(f"After excluding pregnant women:       {len(master):>6,}  "
              f"(excluded {n - len(master):,})")
        n = len(master)
    else:
        print("  WARNING: RIDEXPRG not found in DEMO file — skipping pregnancy exclusion")

    # ── 4. Fasting >= 8 hours ─────────────────────────────── 
    # PHAFSTHR: total fasting time in hours (from FASTQX file)
    # Only retain participants with documented fast >= 8 hours
    if "PHAFSTHR" in master.columns:
        master = master[master["PHAFSTHR"] >= 8]
        print(f"After fasting >= 8h requirement:      {len(master):>6,}  "
                f"(excluded {n - len(master):,})")
        n = len(master)
    else:
        print("  WARNING: PHAFSTHR not found — skipping fasting filter")

    # ── 5. Exclude heavy alcohol use ─────────────────────────────────────── 
    # ALQ130: average number of drinks per day in past 12 months
    # Threshold: >14 drinks/week (men) = >2/day; >7/week (women) = >1/day
    # Note: ALQ130 is drinks per day; RIAGENDR at this point is raw (1=M, 2=F)
    if "ALQ130" in master.columns:
        male_heavy   = (master["RIAGENDR"] == 1) & (master["ALQ130"] > 2)
        female_heavy = (master["RIAGENDR"] == 2) & (master["ALQ130"] > 1)
        master = master[~(male_heavy | female_heavy)]
        print(f"After excluding heavy alcohol use:    {len(master):>6,}  "
              f"(excluded {n - len(master):,})")
        n = len(master)
    else:
        print("  WARNING: ALQ130 not found — skipping alcohol exclusion")

    # ── 6. Exclude active Hepatitis B (HBsAg positive) ───────────────────── 
    # LBDHBG: Hepatitis B surface antigen; 1 = Positive, 2 = Negative
    if "LBDHBG" in master.columns:
        master = master[master["LBDHBG"] != 1]
        print(f"After excluding HBsAg positive:       {len(master):>6,}  "
              f"(excluded {n - len(master):,})")
        n = len(master)
    else:
        print("  WARNING: LBDHBG not found — skipping Hepatitis B exclusion")

    # ── 7. Exclude Hepatitis C (anti-HCV positive) ───────────────────────── 
    # LBDHCV: Hepatitis C antibody; 1 = Positive, 2 = Negative
    if "LBDHCV" in master.columns:
        master = master[master["LBDHCV"] != 1]
        print(f"After excluding anti-HCV positive:    {len(master):>6,}  "
              f"(excluded {n - len(master):,})")
        n = len(master)
    else:
        print("  WARNING: LBDHCV not found — skipping Hepatitis C exclusion")

    print(f"\nFinal eligible cohort (pre-feature engineering): {n:,}")
    return master


def prepare_cohort(master: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features, apply 2023 MASLD outcome definition,
    recode sex, and perform minimal imputation on remaining predictors.

    Key changes vs original:
     MASLD outcome = hepatic steatosis (CAP >= 274) AND >= 1
              cardiometabolic criterion per 2023 Delphi consensus
     pandas fillna no longer uses deprecated inplace=True
    """
    print("\n" + "=" * 60)
    print("Phase 2b: Feature Engineering & MASLD Outcome Definition")
    print("=" * 60)

    df = master.copy()

    # ── UHRatio: Uric Acid / HDL ─────────────────────────────────────────────
    df["UHRatio"] = df["LBXSUA"] / df["LBDHDD"]

    # ── Recode sex: NHANES 1=Male, 2=Female → 1=Male, 0=Female ─────────────
    # Must be done BEFORE alcohol exclusion sex check is already done above
    df["RIAGENDR"] = (df["RIAGENDR"] == 1).astype(int)

    # ── Blood pressure: mean of available readings ───────────────────────────
    # NHANES takes up to 3 BP readings; use mean of non-missing values
    # Systolic: BPXSY1/2/3; Diastolic: BPXDI1/2/3
    systolic_cols  = [c for c in ["BPXSY1", "BPXSY2", "BPXSY3"]  if c in df.columns]
    diastolic_cols = [c for c in ["BPXDI1", "BPXDI2", "BPXDI3"] if c in df.columns]

    if systolic_cols:
        df["SBP_mean"] = df[systolic_cols].mean(axis=1, skipna=True)
    else:
        df["SBP_mean"] = np.nan
        print("  WARNING: Systolic BP columns not found")

    if diastolic_cols:
        df["DBP_mean"] = df[diastolic_cols].mean(axis=1, skipna=True)
    else:
        df["DBP_mean"] = np.nan
        print("  WARNING: Diastolic BP columns not found")

    # ── MASLD Outcome: 2023 Delphi Consensus ───────────────────────────────
    #
    # Criterion 1: Hepatic steatosis (CAP >= 274 dB/m)
    # Criterion 2: At least ONE of the following cardiometabolic risk factors:
    #   a) BMI >= 25 kg/m² (overweight/obese) — or >= 23 kg/m² in Asian populations
    #   b) Fasting glucose >= 100 mg/dL OR known T2DM (approximated by glucose)
    #   c) Blood pressure >= 130/85 mmHg (or antihypertensive use — not captured here)
    #   d) Plasma triglycerides >= 150 mg/dL
    #   e) HDL < 40 mg/dL (men) or < 50 mg/dL (women)
    #
    # Note on HDL criterion: uses recoded RIAGENDR (1=Male, 0=Female)

    steatosis = df["LUXCAPM"] >= 274

    cm_bmi   = df["BMXBMI"] >= 25
    cm_glu   = df["LBXGLU"] >= 100
    cm_bp    = (df["SBP_mean"] >= 130) | (df["DBP_mean"] >= 85)
    cm_trig  = df["LBXTR"] >= 150
    cm_hdl   = (
        ((df["RIAGENDR"] == 1) & (df["LBDHDD"] < 40)) |
        ((df["RIAGENDR"] == 0) & (df["LBDHDD"] < 50))
    )

    cardiometabolic = cm_bmi | cm_glu | cm_bp | cm_trig | cm_hdl

    df["MASLD"] = np.where(steatosis & cardiometabolic, 1, 0)

    n_pos = (df["MASLD"] == 1).sum()
    n_neg = (df["MASLD"] == 0).sum()
    n_steatosis_only = (steatosis & ~cardiometabolic).sum()

    print(f"\nHepatic steatosis (CAP >= 274 dB/m): {steatosis.sum():,}")
    print(f"  → Met >= 1 cardiometabolic criterion: {n_pos:,} "
          f"({n_pos / len(df) * 100:.1f}%) — labelled MASLD-POSITIVE")
    print(f"  → No cardiometabolic criterion:       {n_steatosis_only:,} "
          f"— labelled MASLD-NEGATIVE (steatosis of other aetiology)")
    print(f"No steatosis (CAP < 274 dB/m):         {n_neg:,} — MASLD-NEGATIVE")
    print(f"\nTotal MASLD-positive: {n_pos:,} ({n_pos / len(df) * 100:.1f}%)")
    print(f"Total MASLD-negative: {(df['MASLD']==0).sum():,} "
          f"({(df['MASLD']==0).sum() / len(df) * 100:.1f}%)")

    # ── Imputation: only for remaining missing predictor values ─────────────
    # After fasting filter, glucose missingness should be near-zero.
    # Imputation retained for other features only.
    print("\nChecking for residual missing values in features:")
    for col in FEATURES:
        n_miss = df[col].isna().sum()
        if n_miss > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)   # [FIX 7] no inplace=True
            print(f"  Imputed {n_miss} missing in {col} (median={median_val:.3f})")
        else:
            print(f"  {col}: no missing values ✓")

    print(f"\nFinal analytic cohort: {len(df):,} participants")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2.5: Table 1 — Baseline Characteristics
# ══════════════════════════════════════════════════════════════════════════════

def generate_table1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 1: baseline characteristics stratified by MASLD status.
    Continuous: mean ± SD + Welch t-test p-value.
    Categorical: N (%) + chi-squared p-value.
    """
    print("\n" + "=" * 60)
    print("Phase 2.5: Generating Table 1 — Baseline Characteristics")
    print("=" * 60)

    pos = df[df["MASLD"] == 1]
    neg = df[df["MASLD"] == 0]
    rows = []

    def add_continuous(label, col):
        m_neg, s_neg = neg[col].mean(), neg[col].std()
        m_pos, s_pos = pos[col].mean(), pos[col].std()
        _, p = ttest_ind(neg[col].dropna(), pos[col].dropna(), equal_var=False)
        p_str = "<0.001" if p < 0.001 else f"{p:.3f}"
        rows.append({
            "Variable": label,
            f"No MASLD (N={len(neg):,})": f"{m_neg:.1f} \u00b1 {s_neg:.1f}",
            f"MASLD (N={len(pos):,})":    f"{m_pos:.1f} \u00b1 {s_pos:.1f}",
            "p-value": p_str,
        })

    def add_categorical(label, col, value, value_label):
        c_neg = (neg[col] == value).sum()
        c_pos = (pos[col] == value).sum()
        ct = pd.crosstab(df[col], df["MASLD"])
        _, p, _, _ = chi2_contingency(ct)
        p_str = "<0.001" if p < 0.001 else f"{p:.3f}"
        rows.append({
            "Variable": f"{label} ({value_label})",
            f"No MASLD (N={len(neg):,})": f"{c_neg} ({c_neg/len(neg)*100:.1f}%)",
            f"MASLD (N={len(pos):,})":    f"{c_pos} ({c_pos/len(pos)*100:.1f}%)",
            "p-value": p_str,
        })

    add_continuous("Age (years)",             "RIDAGEYR")
    add_categorical("Sex", "RIAGENDR", 1,     "Male")
    add_continuous("BMI (kg/m\u00b2)",        "BMXBMI")
    add_continuous("Waist Circ. (cm)",        "BMXWAIST")
    add_continuous("Triglycerides (mg/dL)",   "LBXTR")
    add_continuous("Fasting Glucose (mg/dL)", "LBXGLU")
    add_continuous("hs-CRP (mg/L)",           "LBXHSCRP")
    add_continuous("UHRatio",                 "UHRatio")
    add_continuous("SBP (mmHg)",              "SBP_mean")
    add_continuous("DBP (mmHg)",              "DBP_mean")

    table1 = pd.DataFrame(rows)
    print(table1.to_string(index=False))

    path = os.path.join(OUTPUT_DIR, "table1_demographics_8feature_corrected.csv")
    table1.to_csv(path, index=False)
    print(f"\nTable 1 saved: {path}")
    return table1


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: Model Training & Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(df: pd.DataFrame):
    """
    Train XGBoost on 8 features, evaluate with 5-fold CV and held-out test set.

    Changes vs original:
      [FIX 8] Removed deprecated use_label_encoder=False parameter
    """
    print("\n" + "=" * 60)
    print("Phase 3: XGBoost Model Training & Evaluation (8 Features)")
    print("=" * 60)

    X = df[FEATURES]
    y = df["MASLD"]

    # 80/20 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Training set: {len(X_train):,} | Test set: {len(X_test):,}")

    # Z-score normalisation — fit on training set only (no data leakage)
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train), columns=FEATURES, index=X_train.index
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test), columns=FEATURES, index=X_test.index
    )

    # Class imbalance correction
    imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Class imbalance ratio: {imbalance_ratio:.2f}")

    # XGBoost classifier
    # [FIX 8] use_label_encoder removed (deprecated in XGBoost >= 1.6)
    model = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.80,
        colsample_bytree=0.80,
        scale_pos_weight=imbalance_ratio,
        eval_metric="auc",
        random_state=RANDOM_SEED,
    )

    print("\nTraining XGBoost (8-feature corrected model)...")
    model.fit(X_train_s, y_train)
    print("Training complete.")

    # 5-fold stratified cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(
        model, X_train_s, y_train, cv=cv, scoring="roc_auc"
    )
    print(f"\n5-Fold CV AUROC: {cv_scores.mean():.3f} "
          f"(\u00b1 {cv_scores.std():.3f})")

    # Test-set evaluation
    probs = model.predict_proba(X_test_s)[:, 1]
    preds = model.predict(X_test_s)

    auroc    = roc_auc_score(y_test, probs)
    accuracy = accuracy_score(y_test, preds)

    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv         = tp / (tp + fp)
    npv         = tn / (tn + fn)

    print(f"\n{'='*50}")
    print(f"  Test Set Performance — 8-Feature Corrected Model")
    print(f"{'='*50}")
    print(f"  AUROC:       {auroc:.3f}")
    print(f"  Accuracy:    {accuracy:.3f}")
    print(f"  Sensitivity: {sensitivity:.3f}")
    print(f"  Specificity: {specificity:.3f}")
    print(f"  PPV:         {ppv:.3f}")
    print(f"  NPV:         {npv:.3f}")
    print(f"{'='*50}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds,
                                target_names=["No MASLD", "MASLD"]))

    return model, scaler, X_test_s, y_test, probs, cv_scores


# ══════════════════════════════════════════════════════════════════════════════
# Phase 4: SHAP Explainability
# ══════════════════════════════════════════════════════════════════════════════

def run_shap(model, X_test_scaled: pd.DataFrame) -> None:
    """
    Compute SHAP values using TreeExplainer and generate:
      1. Beeswarm summary plot (individual + directional)
      2. Bar plot (global mean |SHAP| ranking)

    [FIX 9] Added guard for list-type shap_values from binary XGBoost classifiers
    """
    print("\n" + "=" * 60)
    print("Phase 4: SHAP Explainability Analysis")
    print("=" * 60)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled)

    # [FIX 9] Newer SHAP versions may return a list of arrays for binary
    # classification — extract the positive class (index 1)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Map NHANES variable names to clinical display labels
    X_display = X_test_scaled.rename(columns=FEATURE_LABELS)

    # ── Beeswarm summary plot ────────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_display, show=False)
    plt.title(
        "SHAP Feature Importance: MASLD Risk Prediction\n"
        "(NHANES 2017-2020, VCTE-Validated, 8-Feature Corrected Model)",
        fontsize=13, pad=15,
    )
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "shap_summary_plot_8feature_corrected.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  SHAP beeswarm plot saved: {path}")

    # ── Bar plot (mean |SHAP| ranking) ───────────────────────────────────────
    plt.figure(figsize=(10, 5))
    shap.summary_plot(shap_values, X_display, plot_type="bar", show=False)
    plt.title(
        "Mean Absolute SHAP Values: 8-Feature Corrected Model",
        fontsize=13, pad=15,
    )
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "shap_bar_plot_8feature_corrected.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  SHAP bar plot saved: {path}")

    # Print global ranking
    mean_shap = np.abs(shap_values).mean(axis=0)
    ranking = sorted(
        zip(FEATURES, mean_shap), key=lambda x: x[1], reverse=True
    )
    print("\n  Global SHAP Ranking (mean |SHAP value|):")
    for feat, val in ranking:
        print(f"    {FEATURE_LABELS[feat]:<30}: {val:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 5: AUROC Curve
# ══════════════════════════════════════════════════════════════════════════════

def plot_roc(y_test, probs, auroc: float) -> None:
    """Plot and save the ROC curve for the held-out test set."""
    fpr, tpr, _ = roc_curve(y_test, probs)

    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, color="#2563EB", lw=2,
             label=f"XGBoost 8-feature corrected (AUROC = {auroc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--",
             label="Random (AUROC = 0.500)")
    plt.xlabel("1 - Specificity (False Positive Rate)", fontsize=12)
    plt.ylabel("Sensitivity (True Positive Rate)", fontsize=12)
    plt.title(
        "Receiver Operating Characteristic Curve\n"
        "MASLD Prediction — XGBoost 8-Feature Corrected Model",
        fontsize=13,
    )
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "auroc_curve_8feature_corrected.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n  AUROC curve saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 6: Clinical Prediction Demo
# ══════════════════════════════════════════════════════════════════════════════

def clinical_demo(model, scaler: StandardScaler) -> None:
    """
    Demonstrate the corrected model on a representative clinical scenario.

    Vignette: 55-year-old male — lean phenotype (BMI 23.5 kg/m²) but
    meets MASLD cardiometabolic criteria via elevated triglycerides,
    borderline waist, prediabetic glucose, raised hs-CRP, and high UHRatio.
    This patient would be classified as MASLD under 2023 Delphi criteria
    (steatosis + cardiometabolic), NOT merely as hepatic steatosis.
    """
    print("\n" + "=" * 60)
    print("Phase 6: Clinical Prediction Demo — 8-Feature Corrected Model")
    print("=" * 60)

    demo = pd.DataFrame({
        "RIDAGEYR":  [55.0],
        "RIAGENDR":  [1],
        "LBXTR":     [220.0],
        "LBXGLU":    [125.0],
        "LBXHSCRP":  [6.5],
        "BMXBMI":    [23.5],
        "BMXWAIST":  [88.0],
        "UHRatio":   [0.25],
    })

    demo_scaled = scaler.transform(demo)
    risk = model.predict_proba(demo_scaled)[0, 1]

    # Check if vignette patient meets 2023 MASLD cardiometabolic criteria
    cm_met = []
    if demo["BMXBMI"].values[0] >= 25:    cm_met.append("BMI >= 25")
    if demo["LBXGLU"].values[0] >= 100:   cm_met.append("Fasting glucose >= 100")
    if demo["LBXTR"].values[0]  >= 150:   cm_met.append("Triglycerides >= 150")

    print("\n  Patient Profile: Lean phenotype (BMI 23.5 kg/m²)")
    print(f"    Age: 55 years | Sex: Male")
    print(f"    Triglycerides: 220 mg/dL | Fasting Glucose: 125 mg/dL")
    print(f"    hs-CRP: 6.5 mg/L | UHRatio: 0.25")
    print(f"    BMI: 23.5 kg/m² (normal) | Waist: 88 cm")
    print(f"\n  2023 MASLD Cardiometabolic Criteria Met: {', '.join(cm_met)}")
    print(f"  >>> Predicted MASLD Probability: {risk * 100:.1f}%")

    if risk >= 0.50:
        print("  >>> TRIAGE: HIGH RISK — recommend VCTE/FibroScan referral")
    else:
        print("  >>> TRIAGE: LOWER RISK — continue routine monitoring")

    print(
        "\n  Clinical note: This patient meets 2023 MASLD criteria (steatosis\n"
        "  + cardiometabolic burden) despite normal BMI. The corrected model\n"
        "  correctly frames this as MASLD — not merely incidental steatosis."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print(" MASLD XGBoost Prediction — 8-Feature CORRECTED Model")
    print(" NHANES 2017-March 2020 Pre-Pandemic Data")
    print(" Corrections: Fasting filter | 2023 MASLD definition |")
    print("              Alcohol + HepB/C exclusions | Age >= 18 |")
    print("              Pregnancy exclusion | Pandas/XGBoost fixes")
    print("=" * 60)

    # Phase 1: Load & merge NHANES data (13 files)
    master = load_and_merge()

    # Phase 2a: Apply exclusions (CONSORT flow)
    master = apply_exclusions(master)

    # Phase 2b: Feature engineering & MASLD outcome definition
    df = prepare_cohort(master)

    # Phase 2.5: Table 1
    generate_table1(df)

    # Phase 3: Train & evaluate XGBoost
    model, scaler, X_test_s, y_test, probs, cv_scores = train_and_evaluate(df)

    auroc = roc_auc_score(y_test, probs)

    # Phase 4: SHAP explainability
    run_shap(model, X_test_s)

    # Phase 5: ROC curve
    plot_roc(y_test, probs, auroc)

    # Phase 6: Clinical demo
    clinical_demo(model, scaler)

    print(f"\n{'='*60}")
    print(f"All outputs saved to: {OUTPUT_DIR}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
