#!/usr/bin/env python3
"""
07_shap_feature_importance.py
------------------------------
SHAP + native feature importance analysis for the two selected models:
  - Imipenem  → Logistic Regression (best test model)
  - Meropenem → Random Forest       (best test model)

Usage:
    python 07_shap_feature_importance.py \
        --modelling-data modelling_data.csv \
        --gene-matrix    gene_matrix.csv \
        --out-dir        results/shap

Install:
    pip install shap matplotlib seaborn
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import shap

from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# ─────────────────────────── CONFIG ──────────────────────────────────────────

ANTIBIOTICS  = ["imipenem", "meropenem"]
LABEL_MAP    = {"Susceptible": 0, "Non-Susceptible": 1}
RANDOM_STATE = 42
TOP_N        = 20

# Best hyperparameters from 06_hyperparameter_tuning.py
BEST_PARAMS = {
    "imipenem": {
        "model"   : "Logistic Regression",
        "params"  : {
            "C"            : 0.039,
            "penalty"      : "elasticnet",
            "l1_ratio"     : 0.291,
            "solver"       : "saga",
            "class_weight" : "balanced",
            "max_iter"     : 2000,
            "random_state" : RANDOM_STATE,
        },
        "feat_sel": ("chi2", 100),
    },
    "meropenem": {
        "model"   : "Random Forest",
        "params"  : {
            "n_estimators"      : 1096,
            "max_depth"         : 15,
            "max_features"      : 0.5,
            "min_samples_leaf"  : 1,
            "min_samples_split" : 3,
            "bootstrap"         : False,
            "class_weight"      : "balanced",
            "random_state"      : RANDOM_STATE,
            "n_jobs"            : -1,
        },
        "feat_sel": ("rfe", 50),
    },
}


# ─────────────────────────── CLI ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="SHAP feature importance for imipenem (LR) and meropenem (RF)"
    )
    p.add_argument("--modelling-data", required=True, type=Path,
                   help="Path to modelling_data CSV (phenotype labels)")
    p.add_argument("--gene-matrix",    required=True, type=Path,
                   help="Path to gene_matrix CSV (192 binary gene features)")
    p.add_argument("--out-dir",        default="results/shap", type=Path,
                   help="Output directory for plots and CSVs (default: results/shap)")
    p.add_argument("--test-size",      default=0.20, type=float,
                   help="Test split fraction — must match tuning script (default: 0.20)")
    p.add_argument("--top-n",          default=TOP_N, type=int,
                   help="Number of top genes to display in plots (default: 20)")
    return p.parse_args()


# ─────────────────────────── DATA LOADING ────────────────────────────────────

def load_antibiotic_data(modelling_path: Path, gene_matrix_path: Path,
                         antibiotic: str, test_size: float):
    """
    Load modelling_data + gene_matrix, merge on Genome ID,
    filter to one antibiotic, and return train/test splits
    with gene column names.
    """
    modelling = pd.read_csv(modelling_path, dtype=str)
    gene_mat  = pd.read_csv(gene_matrix_path, dtype=str)

    modelling.columns = modelling.columns.str.strip()
    gene_mat.columns  = gene_mat.columns.str.strip()

    # Filter phenotype table to this antibiotic
    pheno = (
        modelling[modelling["Antibiotic"].str.lower() == antibiotic.lower()]
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

    print(f"  Samples : {len(y)}  |  "
          f"Non-Susceptible={y.sum()}  Susceptible={(y==0).sum()}")
    print(f"  Genes   : {len(gene_cols)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(f"  Train   : n={len(y_train)}")
    print(f"  Test    : n={len(y_test)}")

    return X_train, X_test, y_train, y_test, gene_cols


# ─────────────────────────── PIPELINE BUILDER ────────────────────────────────

def build_pipeline(cfg: dict) -> Pipeline:
    """Build and return an unfitted pipeline from config."""
    feat_method, feat_k = cfg["feat_sel"]

    if feat_method == "chi2":
        selector = SelectKBest(chi2, k=feat_k)
    else:  # rfe
        selector = RFE(
            estimator=LogisticRegression(
                class_weight="balanced", max_iter=1000,
                random_state=RANDOM_STATE
            ),
            n_features_to_select=feat_k,
            step=10,
        )

    if cfg["model"] == "Logistic Regression":
        clf = LogisticRegression(**cfg["params"])
    else:
        clf = RandomForestClassifier(**cfg["params"])

    return Pipeline([
        ("select", selector),
        ("scaler", StandardScaler()),
        ("clf",    clf),
    ])


# ─────────────────────────── HELPERS ─────────────────────────────────────────

def get_selected_gene_names(pipeline: Pipeline, gene_cols: list) -> np.ndarray:
    """Return gene names that survived feature selection."""
    mask = pipeline.named_steps["select"].get_support()
    return np.array(gene_cols)[mask]


def transform_data(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    """Apply select + scale steps (exclude clf) to produce SHAP input."""
    X_sel    = pipeline.named_steps["select"].transform(X)
    X_scaled = pipeline.named_steps["scaler"].transform(X_sel)
    return X_scaled


# ─────────────────────────── PLOTS ───────────────────────────────────────────

def plot_shap_bar(shap_vals, gene_names, antibiotic, model_name,
                  out_dir, top_n):
    mean_abs = np.abs(shap_vals).mean(axis=0)
    idx      = np.argsort(mean_abs)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        range(top_n),
        mean_abs[idx][::-1],
        color=sns.color_palette("RdYlGn_r", top_n),
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(gene_names[idx][::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(
        f"{antibiotic.capitalize()} — {model_name}\n"
        f"Top {top_n} genes by mean |SHAP|",
        fontsize=12, fontweight="bold"
    )
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    plt.tight_layout()
    path = out_dir / f"{antibiotic}_shap_bar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_shap_beeswarm(shap_vals, X_transformed, gene_names,
                       antibiotic, model_name, out_dir, top_n):
    mean_abs = np.abs(shap_vals).mean(axis=0)
    idx      = np.argsort(mean_abs)[::-1][:top_n]

    shap.summary_plot(
        shap_vals[:, idx],
        X_transformed[:, idx],
        feature_names=gene_names[idx],
        show=False,
        plot_size=(10, 7),
    )
    plt.title(
        f"{antibiotic.capitalize()} — {model_name}\n"
        f"SHAP Beeswarm (Top {top_n} genes)",
        fontsize=12, fontweight="bold"
    )
    path = out_dir / f"{antibiotic}_shap_beeswarm.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_shap_waterfall(explainer, shap_vals, X_transformed,
                        gene_names, antibiotic, model_name,
                        out_dir, sample_idx, label):
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)) and np.asarray(ev).ndim >= 1:
        base = float(np.asarray(ev).flat[1])  # index 1 = Non-Susceptible class
    else:
        base = float(ev)

    explanation = shap.Explanation(
        values        = shap_vals[sample_idx],
        base_values   = base,
        data          = X_transformed[sample_idx],
        feature_names = list(gene_names),
    )
    shap.waterfall_plot(explanation, show=False, max_display=15)
    plt.title(
        f"{antibiotic.capitalize()} — {model_name}\n"
        f"Waterfall: {label} sample (idx={sample_idx})",
        fontsize=11, fontweight="bold"
    )
    path = out_dir / f"{antibiotic}_shap_waterfall_{label}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def save_importance_csv(shap_vals, gene_names, antibiotic, out_dir):
    mean_abs  = np.abs(shap_vals).mean(axis=0)
    mean_shap = shap_vals.mean(axis=0)
    std_shap  = shap_vals.std(axis=0)

    df = pd.DataFrame({
        "gene"          : gene_names,
        "mean_abs_shap" : mean_abs.round(6),
        "mean_shap"     : mean_shap.round(6),  # positive = drives Non-Susceptible
        "std_shap"      : std_shap.round(6),
        "rank"          : pd.Series(mean_abs).rank(ascending=False).astype(int).values,
    }).sort_values("rank").reset_index(drop=True)

    path = out_dir / f"{antibiotic}_shap_importance.csv"
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")
    print(f"\n  Top 10 genes:\n{df.head(10).to_string(index=False)}")
    return df


# ─────────────────────────── MAIN ────────────────────────────────────────────

def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for antibiotic in ANTIBIOTICS:
        cfg = BEST_PARAMS[antibiotic]

        print(f"\n{'='*60}")
        print(f"  {antibiotic.upper()}  —  {cfg['model']}")
        print(f"{'='*60}")

        # ── Load data ─────────────────────────────────────────────────────────
        X_train, X_test, y_train, y_test, gene_cols = load_antibiotic_data(
            args.modelling_data,
            args.gene_matrix,
            antibiotic,
            args.test_size,
        )

        # ── Build & fit pipeline ──────────────────────────────────────────────
        pipeline = build_pipeline(cfg)
        pipeline.fit(X_train, y_train)
        print(f"  Pipeline fitted.")

        # ── Selected gene names & transformed arrays ──────────────────────────
        selected_genes  = get_selected_gene_names(pipeline, gene_cols)
        X_train_t       = transform_data(pipeline, X_train)
        X_test_t        = transform_data(pipeline, X_test)

        print(f"  Selected genes : {len(selected_genes)}")

        # ── SHAP explainer ────────────────────────────────────────────────────
        if cfg["model"] == "Random Forest":
            explainer   = shap.TreeExplainer(pipeline.named_steps["clf"])
            shap_values = explainer.shap_values(X_test_t)
            # Newer SHAP returns (n_samples, n_features, n_classes) ndarray
            # Older SHAP returns a list [class_0_array, class_1_array]
            if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_vals = shap_values[:, :, 1]   # class 1 = Non-Susceptible
            elif isinstance(shap_values, list):
                shap_vals = shap_values[1]          # class 1 = Non-Susceptible
            else:
                shap_vals = shap_values             # already 2D

        else:  # Logistic Regression
            explainer   = shap.LinearExplainer(
                pipeline.named_steps["clf"],
                X_train_t,
                feature_perturbation="interventional",
            )
            shap_vals = explainer.shap_values(X_test_t)

        print(f"  SHAP values shape: {shap_vals.shape}")

        # ── Plots ─────────────────────────────────────────────────────────────
        plot_shap_bar(shap_vals, selected_genes,
                      antibiotic, cfg["model"], args.out_dir, args.top_n)

        plot_shap_beeswarm(shap_vals, X_test_t, selected_genes,
                           antibiotic, cfg["model"], args.out_dir, args.top_n)

        # Waterfall for one Non-Susceptible and one Susceptible sample
        ns_idx = int(np.where(y_test == 1)[0][0])
        s_idx  = int(np.where(y_test == 0)[0][0])

        plot_shap_waterfall(explainer, shap_vals, X_test_t, selected_genes,
                            antibiotic, cfg["model"], args.out_dir,
                            sample_idx=ns_idx, label="Non-Susceptible")

        plot_shap_waterfall(explainer, shap_vals, X_test_t, selected_genes,
                            antibiotic, cfg["model"], args.out_dir,
                            sample_idx=s_idx,  label="Susceptible")

        # ── CSV ───────────────────────────────────────────────────────────────
        save_importance_csv(shap_vals, selected_genes, antibiotic, args.out_dir)

    print(f"\nAll outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
