# MASLD XGBoost Risk Prediction Model (NHANES 2017-2020)

## Overview
This repository contains the machine learning pipeline for predicting Metabolic Dysfunction-Associated Steatotic Liver Disease (MASLD) using routine clinical biomarkers, demographic variables, and novel digital indices. The model utilizes Extreme Gradient Boosting (XGBoost) and SHAP (SHapley Additive exPlanations) for interpretability, adhering to the 2023 MASLD Delphi consensus criteria.

## Abstract (ACG 2026 Submission)
**TITLE** 
An Enzyme-Free, Eight-Feature XGBoost Model for Metabolic Dysfunction-Associated Steatotic Liver Disease Prediction Incorporating a Novel Biomarker: A Nationally Representative Cross-Sectional Analysis

**Introduction**

Metabolic dysfunction-associated steatotic liver disease (MASLD) affects about 30% of United States adults, yet is substantially under-diagnosed. Existing non-invasive prediction models typically require liver enzymes—alanine aminotransferase (ALT), aspartate aminotransferase (AST), or gamma-glutamyl transferase (GGT)—unavailable at many primary care encounters. The serum uric acid-to-HDL cholesterol ratio (UHRatio) captures synergistic metabolic risk signals but has never been evaluated as a feature in a machine learning (ML) MASLD prediction model. We developed an enzyme-free XGBoost classifier incorporating UHRatio alongside seven routine clinical parameters, validated against vibration-controlled transient elastography (VCTE).

**Methods**

Cross-sectional secondary analysis of NHANES 2017–March 2020 was performed. MASLD was defined as VCTE-derived controlled attenuation parameter (CAP) ≥274 dB/m. Eight features—age, sex, triglycerides, fasting glucose, high-sensitivity C-reactive protein (hs-CRP), BMI, waist circumference, and UHRatio (uric acid [mg/dL] ÷ HDL-C [mg/dL])—were used with no liver enzymes. XGBoost was trained on 80% of participants (N=3,806) and evaluated on a 20% held-out test set (N=952) with five-fold stratified cross-validation. SHapley Additive exPlanations (SHAP) assessed feature-level interpretability.

**Results**

Analytic cohort: 4,758 adults (MASLD+: 1,853 [38.9%]). All features differed significantly between groups (p< 0.001), including UHRatio (0.13±0.05 vs. 0.10±0.04). The enzyme-free model achieved AUROC 0.818 (CV AUROC: 0.838±0.004; sensitivity: 0.844; specificity: 0.898; PPV: 0.841; NPV: 0.900). SHAP ranked waist circumference highest, followed by triglycerides, BMI, and fasting glucose; and UHRatio in fifth—ahead of age, hs-CRP, and sex. A lean male (age 55, BMI 23.5 kg/m²) with elevated triglycerides, glucose, hs-CRP, and UHRatio received a 92% predicted MASLD probability.

**Discussion**

This is the first study embedding UHRatio as a predictive feature in an ML MASLD model. The enzyme-free design achieves AUROC 0.818, exceeding published NHANES-based models requiring enzymes like ALT and GGT (AUROC 0.809–0.874). All features derive from a standard metabolic panel and anthropometrics, enabling first-encounter primary care deployment without liver function testing. SHAP-confirmed UHRatio directionality aligns with its established biology. This interpretable model has potential as a scalable first-contact triage tool for MASLD.

## Technical Requirements
This script requires Python 3.8+ and the following libraries:
* `pandas`
* `numpy`
* `scikit-learn`
* `xgboost`
* `shap`
* `matplotlib`
* `scipy`

Install dependencies via pip:
```bash 
pip install pandas numpy scikit-learn xgboost shap matplotlib scipy
```

## Data Acquisition & Setup (Important)
Because the CDC periodically updates its server structures, this script is configured to read NHANES data locally to ensure stability. **You must download the datasets manually before running the code.**

1. Create a folder named `nhanes_data` in the same directory as the Python script.
2. Download the following 13 `.xpt` files from the [CDC NHANES 2017-March 2020 Pre-Pandemic cycle](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2017-2020):
   * **Demographics & Questionnaires:** `P_DEMO.xpt`, `P_FASTQX.xpt`, `P_ALQ.xpt`
   * **Examination:** `P_LUX.xpt`, `P_BMX.xpt`, `P_BPXO.xpt` *(Note: You must rename P_BPXO.xpt to P_BPX.xpt after downloading)*
   * **Laboratory:** `P_BIOPRO.xpt`, `P_HDL.xpt`, `P_TRIGLY.xpt`, `P_GLU.xpt`, `P_HSCRP.xpt`, `P_HEPB_S.xpt`, `P_HEPC.xpt` *(Note: You must rename P_HEPC.xpt to P_HCV.xpt after downloading)*
3. Place all 13 files into the `nhanes_data` folder.

## Usage
Once the data is staged, run the script from your terminal or IDE:
```bash
python "MASLD-8.py"
```

## Pipeline Outputs
The script automatically generates a new folder called `ml_output/` containing:

### 1. SHAP Feature Importance (Beeswarm)
![SHAP Summary Plot](MASLD_Output.md/shap_summary_plot_8feature_corrected.png)

### 2. SHAP Global Importance (Bar)
![SHAP Bar Plot](MASLD_Output.md/shap_bar_plot_8feature_corrected.png)

### 3. AUROC Curve
![AUROC Curve](MASLD_Output.md/auroc_curve_8feature_corrected.png)

### 4. Baseline Characteristics
[Click here to view Table 1: Baseline Demographics & Clinical Characteristics](table1_demographics_8feature.csv)


## Clinical Demo
The end of the script outputs a simulated clinical vignette to the console, demonstrating how the model evaluates a "lean phenotype" patient (normal BMI) who possesses underlying cardiometabolic risk factors, illustrating the model's alignment with the 2023 Delphi consensus.
