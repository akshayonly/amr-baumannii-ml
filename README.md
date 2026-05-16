# AMR Prediction in *Acinetobacter baumannii*
### Machine Learning Pipeline for Carbapenem Resistance Prediction from Whole-Genome Sequencing

---

## Overview

An end-to-end machine learning pipeline that predicts carbapenem resistance (imipenem and meropenem) in *Acinetobacter baumannii* from whole-genome sequencing data. Using 547 clinical isolates sourced from BV-BRC, I engineer a binary gene presence/absence matrix from AMRFinderPlus outputs and train interpretable ML models, achieving test AUC of **0.93 (imipenem)** and **0.89 (meropenem)**. SHAP-based feature analysis recovers known OXA carbapenemases, AdeABC efflux components, and NDM-1 as top resistance drivers — consistent with published *A. baumannii* resistance literature.

---

## Pipeline

```
Genome Download (BV-BRC)
        ↓
AMRFinderPlus (AMR gene detection)
        ↓
Gene Presence/Absence Matrix (192 genes × 547 genomes)
        ↓
Train/Test Split (80/20, stratified)
        ↓
Feature Selection (Chi² top 100 / RFE top 50)
        ↓
Baseline Modelling (LR, RF, GBM, SVM)
        ↓
Hyperparameter Tuning (RandomizedSearchCV, 5-fold CV)
        ↓
SHAP Feature Importance
```

---

## Key Results

| Antibiotic | Best Model | Test AUC | Recall | MCC |
|---|---|---|---|---|
| Imipenem | Logistic Regression | 0.930 | 0.903 | 0.765 |
| Meropenem | Random Forest | 0.888 | 0.955 | 0.806 |

**Top resistance-driving genes identified:**
- `blaOXA-23`, `blaOXA-58`, `blaOXA-100` — OXA carbapenemases (causal)
- `adeC`, `adeS_G336S` — AdeABC efflux pump components (causal)
- `blaNDM-1` — Class B metallo-β-lactamase (causal)
- `gyrA_S81L`, `sul2` — co-selection MDR markers (not causal)

---

## Repository Structure

```
├── notebooks/
│   ├── 01_data_preparation.ipynb         # Initial EDA and genomes selection
│   ├── 02_amr_gene_modeling.ipynb        # Baseline modelling
├── scripts/
│   ├── 00_install_bvbrc.sh               # Install BV-BRC CLI
│   ├── 01_download_bvbrc_genomes.py      # Download genomes from BV-BRC
│   ├── 02_run_amrfinder.py               # Run AMRFinderPlus
│   ├── 03_build_kmer_matrix.py           # Build k-mer matrix
│   ├── 04_build_amr_gene_matrix.py       # Build gene presence/absence matrix
│   ├── 06_hyperparameter_tuning.py       # Hyperparameter tuning (RF, LR)
│   └── 07_shap_feature_importance.py     # SHAP analysis
├── data/genome_ids.txt                   # Genome fasta files used
├── results/
│   ├── tuning/
│   │   ├── test_set_results.csv          # Final test set metrics
│   │   └── cv_search_results.csv         # Full RandomizedSearch CV log
│   └── shap/                             # SHAP bar, beeswarm, waterfall plots
│       ├── imipenem_shap_importance.csv  # Ranked genes — imipenem
│       └── meropenem_shap_importance.csv # Ranked genes — meropenem
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/akshayonly/amr-acinetobacter-ml.git
cd amr-acinetobacter-ml
pip install -r requirements.txt
```

**External tools required:**
- [AMRFinderPlus](https://github.com/ncbi/amr) — NCBI AMR gene detection
- [BV-BRC CLI](https://www.bv-brc.org/docs/cli_tutorial/) — genome download

---

## Usage

Run scripts in order:

```bash
# 1. Install BV-BRC
bash scripts/00_install_bvbrc.sh

# 2. Download genomes
python scripts/01_download_bvbrc_genomes.py --genome-ids data/genome_ids.txt --out-dir data/genomes

# 3. Run AMRFinderPlus
python scripts/02_run_amrfinder.py --genome-dir data/genomes --out-dir data/amrfinder

# 4. Build gene matrix
python scripts/04_build_amr_gene_matrix.py --amrfinder-dir data/amrfinder --out-dir data/

# 5. Baseline modelling
02_amr_gene_modeling.ipynb

# 6. Hyperparameter tuning
python scripts/06_hyperparameter_tuning.py --modelling-data modelling_data.csv --gene-matrix amr_gene_matrix.csv --out-dir results/tuning

# 7. SHAP analysis
python scripts/07_shap_feature_importance.py --modelling-data modelling_data.csv --gene-matrix amr_gene_matrix.csv --out-dir results/shap
```

---

## Data

Genomes were downloaded from [BV-BRC](https://www.bv-brc.org/) filtering for *Acinetobacter baumannii* with associated MIC phenotype data. Of 579 genomes attempted, **547 were successfully downloaded** and used in analysis. Genome IDs are provided in `data/genome_ids.txt` for full reproducibility.

Phenotype labels (Susceptible / Non-Susceptible) were derived from BV-BRC MIC measurements using EUCAST clinical breakpoints.

---

## Dependencies

```
numpy
pandas
scipy
scikit-learn
shap
matplotlib
seaborn
tqdm
biopython
```

---

## Organism & Clinical Context

*Acinetobacter baumannii* is a WHO Priority 1 Critical pathogen. Carbapenem resistance, driven primarily by OXA-type carbapenemases and AdeABC efflux overexpression, renders infections nearly untreatable. This pipeline provides a genomics-based resistance prediction framework as an alternative to slow culture-based AST, with SHAP interpretability ensuring biological validity of model decisions.

> **Note:** Models achieve test Recall of 0.903 (imipenem) and 0.955 (meropenem). These do not meet FDA VME thresholds (≤1.5%) for standalone clinical deployment and are intended for **research and surveillance use only**.

---

## Author

**Akshay Shirsath**
