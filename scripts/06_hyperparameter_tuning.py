#!/usr/bin/env python3
"""
06_hyperparameter_tuning.py
---------------------------
Hyperparameter tuning of Random Forest and Logistic Regression
for imipenem and meropenem AMR binary classification.

Features:
  - Proper train/test split before any modelling
  - Feature selection inside CV pipeline (no leakage)
      · Imipenem  → Chi2 top 100
      · Meropenem → RFE-LR top 50
  - RandomizedSearchCV on training set only
  - Final evaluation on locked-away test set
  - Results saved to CSV and log file

System: 32-core CPU, 128 GB RAM
Estimated runtime: well within 15-hour session limit

Usage:
    python 06_hyperparameter_tuning.py \
        --modelling-data modelling_data.csv \
        --gene-matrix    gene_matrix.csv \
        --out-dir        results/tuning
"""

import argparse
import logging
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform, loguniform

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
    classification_report, confusion_matrix,
)
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────── CONSTANTS ───────────────────────────────────────

ANTIBIOTICS = ["imipenem", "meropenem"]
LABEL_MAP   = {"Susceptible": 0, "Non-Susceptible": 1}
RANDOM_STATE = 42

# Feature selection per antibiotic (from feature engineering stage)
FEATURE_SELECTION = {
    "imipenem" : {"method": "chi2", "k": 100},
    "meropenem": {"method": "rfe",  "k": 50},
}

# CV and search settings — tuned for 32-core system
N_SPLITS     = 5
N_ITER_RF    = 100   # RandomizedSearch iterations for RF
N_ITER_LR    = 80    # RandomizedSearch iterations for LR
N_JOBS       = 30    # leave 2 cores free for OS


# ─────────────────────────── CLI ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Hyperparameter tuning: RF and LR for imipenem / meropenem"
    )
    p.add_argument("--modelling-data", required=True, type=Path,
                   help="Path to modelling_data CSV (phenotype labels)")
    p.add_argument("--gene-matrix",    required=True, type=Path,
                   help="Path to gene_matrix CSV (192 binary gene features)")
    p.add_argument("--out-dir",        default="results/tuning", type=Path,
                   help="Output directory for results and logs")
    p.add_argument("--test-size",      default=0.20, type=float,
                   help="Fraction of data held out as test set (default: 0.20)")
    return p.parse_args()


# ─────────────────────────── LOGGING ─────────────────────────────────────────

def setup_logging(out_dir: Path) -> logging.Logger:
    log = logging.getLogger("tuning")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(sh)

    fh = logging.FileHandler(out_dir / "tuning.log")
    fh.setFormatter(fmt)
    log.addHandler(fh)

    return log


# ─────────────────────────── DATA LOADING ────────────────────────────────────

def load_data(modelling_path: Path, gene_matrix_path: Path,
              antibiotic: str, test_size: float, log: logging.Logger):
    """
    Load, merge, split for one antibiotic.
    Returns X_train, X_test, y_train, y_test and gene column names.
    """
    log.info(f"Loading data for {antibiotic} ...")

    modelling = pd.read_csv(modelling_path, dtype=str)
    gene_mat   = pd.read_csv(gene_matrix_path, dtype=str)

    # Normalise column names
    modelling.columns = modelling.columns.str.strip()
    gene_mat.columns  = gene_mat.columns.str.strip()

    # Filter to antibiotic
    pheno = (
        modelling[modelling["Antibiotic"].str.lower() == antibiotic]
        [["Genome ID", "Binary Label"]]
        .dropna(subset=["Binary Label"])
        .drop_duplicates(subset="Genome ID")
        .copy()
    )

    # Merge with gene matrix
    merged = pheno.merge(gene_mat, on="Genome ID", how="inner")

    gene_cols = [c for c in merged.columns
                 if c not in ["Genome ID", "Binary Label"]]

    X = merged[gene_cols].astype(float).values
    y = merged["Binary Label"].map(LABEL_MAP).values

    log.info(f"  Merged samples : {len(y)}")
    log.info(f"  Non-Susceptible: {y.sum()}  |  Susceptible: {(y==0).sum()}")
    log.info(f"  Gene features  : {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    log.info(f"  Train : n={len(y_train)}  "
             f"(NS={y_train.sum()}, S={(y_train==0).sum()})")
    log.info(f"  Test  : n={len(y_test)}   "
             f"(NS={y_test.sum()}, S={(y_test==0).sum()})")

    # Save splits
    return X_train, X_test, y_train, y_test, gene_cols


# ─────────────────────────── FEATURE SELECTION ───────────────────────────────

def get_feature_selector(antibiotic: str):
    """
    Return the feature selection step for this antibiotic.
    Must sit INSIDE the pipeline so it only sees training folds.
    """
    cfg = FEATURE_SELECTION[antibiotic]

    if cfg["method"] == "chi2":
        return ("select", SelectKBest(chi2, k=cfg["k"]))

    if cfg["method"] == "rfe":
        return ("select", RFE(
            estimator=LogisticRegression(
                class_weight="balanced", max_iter=1000,
                random_state=RANDOM_STATE
            ),
            n_features_to_select=cfg["k"],
            step=10,
        ))

    raise ValueError(f"Unknown feature selection method: {cfg['method']}")


# ─────────────────────────── SEARCH SPACES ───────────────────────────────────

def rf_search_space():
    return {
        "clf__n_estimators"      : randint(200, 1200),
        "clf__max_depth"         : [None, 5, 10, 15, 20, 30],
        "clf__min_samples_split" : randint(2, 20),
        "clf__min_samples_leaf"  : randint(1, 10),
        "clf__max_features"      : ["sqrt", "log2", 0.3, 0.5, 0.7],
        "clf__class_weight"      : ["balanced", "balanced_subsample"],
        "clf__bootstrap"         : [True, False],
    }


def lr_search_space():
    return {
        "clf__C"              : loguniform(1e-4, 1e2),   # inverse regularisation
        "clf__penalty"        : ["l1", "l2", "elasticnet"],
        "clf__solver"         : ["saga"],                # handles all penalties
        "clf__l1_ratio"       : uniform(0.0, 1.0),       # only used for elasticnet
        "clf__class_weight"   : ["balanced"],
        "clf__max_iter"       : [2000],
    }


# ─────────────────────────── EVALUATION ──────────────────────────────────────

def evaluate_on_test(model, X_test, y_test, label: str, log: logging.Logger):
    """Evaluate a fitted model on the held-out test set."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Label"     : label,
        "AUC"       : round(roc_auc_score(y_test, y_proba), 4),
        "F1"        : round(f1_score(y_test, y_pred), 4),
        "Recall"    : round(recall_score(y_test, y_pred), 4),
        "Precision" : round(precision_score(y_test, y_pred), 4),
        "Accuracy"  : round(accuracy_score(y_test, y_pred), 4),
        "MCC"       : round(matthews_corrcoef(y_test, y_pred), 4),
    }

    log.info(f"\n  ── {label} — TEST SET RESULTS ──────────────────────")
    for k, v in metrics.items():
        if k != "Label":
            log.info(f"    {k:<12}: {v}")

    log.info(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test, y_pred)
    log.info(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    log.info(f"    FN={cm[1,0]}  TP={cm[1,1]}")
    log.info(f"\n{classification_report(y_test, y_pred, target_names=['Susceptible','Non-Susceptible'])}")

    return metrics


# ─────────────────────────── MAIN ────────────────────────────────────────────

def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logging(args.out_dir)
    log.info(f"Output directory : {args.out_dir}")
    log.info(f"n_jobs           : {N_JOBS}  (of 32 cores)")
    log.info(f"RF iterations    : {N_ITER_RF}")
    log.info(f"LR iterations    : {N_ITER_LR}")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                         random_state=RANDOM_STATE)

    scoring_primary = "roc_auc"   # primary metric for search

    all_cv_results  = []
    all_test_results = []

    for antibiotic in ANTIBIOTICS:

        log.info(f"\n{'='*60}")
        log.info(f"  ANTIBIOTIC: {antibiotic.upper()}")
        log.info(f"{'='*60}")

        X_train, X_test, y_train, y_test, gene_cols = load_data(
            args.modelling_data, args.gene_matrix,
            antibiotic, args.test_size, log,
        )

        feat_step = get_feature_selector(antibiotic)
        feat_cfg  = FEATURE_SELECTION[antibiotic]
        log.info(f"  Feature selection: {feat_cfg['method'].upper()} "
                 f"top {feat_cfg['k']} genes (inside CV pipeline)")

        # ── Model definitions ─────────────────────────────────────────────────
        model_configs = {
            "Random Forest": (
                Pipeline([
                    feat_step,
                    ("scaler", StandardScaler()),
                    ("clf", RandomForestClassifier(random_state=RANDOM_STATE,
                                                   n_jobs=N_JOBS)),
                ]),
                rf_search_space(),
                N_ITER_RF,
            ),
            "Logistic Regression": (
                Pipeline([
                    feat_step,
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(random_state=RANDOM_STATE)),
                ]),
                lr_search_space(),
                N_ITER_LR,
            ),
        }

        best_models = {}

        for model_name, (pipeline, param_dist, n_iter) in model_configs.items():

            log.info(f"\n  ── {model_name}  ({n_iter} random search iterations) ──")
            t0 = time.time()

            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_dist,
                n_iter=n_iter,
                scoring=scoring_primary,
                cv=cv,
                refit=True,            # refit best model on full X_train
                n_jobs=N_JOBS,
                random_state=RANDOM_STATE,
                verbose=1,
                return_train_score=False,
            )

            search.fit(X_train, y_train)
            elapsed = round(time.time() - t0, 1)

            log.info(f"    Search complete in {elapsed}s")
            log.info(f"    Best CV AUC : {search.best_score_:.4f}")
            log.info(f"    Best params :")
            for k, v in search.best_params_.items():
                log.info(f"      {k}: {v}")

            best_models[model_name] = search.best_estimator_

            # Save CV results
            cv_df = pd.DataFrame(search.cv_results_)
            cv_df["Antibiotic"] = antibiotic
            cv_df["Model"]      = model_name
            all_cv_results.append(cv_df)

            # Evaluate on locked test set
            test_metrics = evaluate_on_test(
                search.best_estimator_,
                X_test, y_test,
                label=f"{antibiotic} | {model_name}",
                log=log,
            )
            test_metrics["Antibiotic"]   = antibiotic
            test_metrics["Model"]        = model_name
            test_metrics["Best CV AUC"]  = round(search.best_score_, 4)
            test_metrics["Search Time"]  = elapsed
            all_test_results.append(test_metrics)

        # ── Winner per antibiotic ─────────────────────────────────────────────
        ab_results = [r for r in all_test_results if r["Antibiotic"] == antibiotic]
        winner = max(ab_results, key=lambda r: r["AUC"])
        log.info(f"\n  ★ Best model for {antibiotic.upper()}: "
                 f"{winner['Model']}  (Test AUC={winner['AUC']})")

    # ── Save outputs ──────────────────────────────────────────────────────────
    test_df = pd.DataFrame(all_test_results)
    cols    = ["Antibiotic", "Model", "Best CV AUC",
               "AUC", "F1", "Recall", "Precision", "Accuracy", "MCC",
               "Search Time"]
    test_df = test_df[cols].sort_values(["Antibiotic", "AUC"], ascending=[True, False])

    test_path = args.out_dir / "test_set_results.csv"
    test_df.to_csv(test_path, index=False)
    log.info(f"\nTest results saved : {test_path}")

    # Save full CV results for each model
    cv_all = pd.concat(all_cv_results, ignore_index=True)
    cv_path = args.out_dir / "cv_search_results.csv"
    cv_all.to_csv(cv_path, index=False)
    log.info(f"CV search results  : {cv_path}")

    # Final summary
    log.info("\n\n── FINAL SUMMARY ────────────────────────────────────────────────")
    log.info(test_df[["Antibiotic", "Model", "Best CV AUC",
                       "AUC", "F1", "Recall", "MCC"]].to_string(index=False))


if __name__ == "__main__":
    main()
