#!/usr/bin/env python3
"""
04_build_amr_gene_matrix.py
---------------------------
Build a gene presence/absence matrix from AMRFinderPlus results.

Each row = one genome (sample), each column = one AMR element symbol.
Cell value = 1 if the gene/element was detected in that genome, else 0.

Usage:
    python 04_build_amr_gene_matrix.py \
        --amrfinder-dir data/amrfinder \
        --out-dir data/amr_matrix \
        --scope core \
        --type AMR

Output files:
    amr_gene_matrix.csv          — samples × genes presence/absence matrix
    amr_gene_matrix_metadata.csv — per-gene metadata (Type, Subtype, Class, Subclass)
    amr_gene_matrix.log          — run log

Requirements:
    pip install pandas tqdm
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# ─────────────────────────── CLI ────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Build AMR gene presence/absence matrix from AMRFinderPlus TSVs."
    )
    p.add_argument(
        "--amrfinder-dir",
        required=True,
        type=Path,
        help="Directory containing AMRFinderPlus .tsv output files",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory to write output files",
    )
    p.add_argument(
        "--scope",
        default=None,
        choices=["core", "plus", None],
        help="Filter by Scope column: 'core' or 'plus'. Default: keep all.",
    )
    p.add_argument(
        "--type",
        default=None,
        help=(
            "Filter by Type column (e.g. AMR, STRESS, VIRULENCE). "
            "Comma-separated for multiple. Default: keep all."
        ),
    )
    p.add_argument(
        "--col",
        default="Element symbol",
        help=(
            "Column to use as gene identifier. "
            "Options: 'Element symbol' (default), 'Element name', "
            "'Closest reference accession', 'HMM accession'. "
        ),
    )
    p.add_argument(
        "--ext",
        default=".tsv",
        help="File extension for AMRFinderPlus result files (default: .tsv)",
    )
    return p.parse_args()


# ─────────────────────────── LOGGING ────────────────────────────────────────

def setup_logging(out_dir: Path) -> logging.Logger:
    log = logging.getLogger("build_amr_matrix")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(sh)

    fh = logging.FileHandler(out_dir / "amr_gene_matrix.log")
    fh.setFormatter(fmt)
    log.addHandler(fh)

    return log


# ─────────────────────────── HELPERS ────────────────────────────────────────

EXPECTED_COLS = [
    "Protein id", "Contig id", "Start", "Stop", "Strand",
    "Element symbol", "Element name", "Scope", "Type", "Subtype",
    "Class", "Subclass", "Method", "Target length",
    "Reference sequence length", "% Coverage of reference",
    "% Identity to reference", "Alignment length",
    "Closest reference accession", "Closest reference name",
    "HMM accession", "HMM description",
]

METADATA_COLS = ["Type", "Subtype", "Class", "Subclass"]


def load_tsv(path: Path) -> pd.DataFrame | None:
    """Load one AMRFinderPlus TSV; return None on failure."""
    try:
        df = pd.read_csv(path, sep="\t", dtype=str)
        # Normalize column names (strip whitespace)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        return None


def apply_filters(
    df: pd.DataFrame,
    scope: str | None,
    types: list[str] | None,
) -> pd.DataFrame:
    if df.empty:
        return df
    if scope and "Scope" in df.columns:
        df = df[df["Scope"].str.lower() == scope.lower()]
    if types and "Type" in df.columns:
        types_lower = [t.lower() for t in types]
        df = df[df["Type"].str.lower().isin(types_lower)]
    return df


# ─────────────────────────── MAIN ───────────────────────────────────────────

def main():
    args = parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(args.out_dir)

    tsv_files = sorted(args.amrfinder_dir.glob(f"*{args.ext}"))
    if not tsv_files:
        log.error(f"No {args.ext} files found in {args.amrfinder_dir}")
        return

    log.info(f"TSV files found : {len(tsv_files)}")
    log.info(f"Gene ID column  : {args.col}")
    log.info(f"Scope filter    : {args.scope or 'none (all)'}")
    log.info(f"Type filter     : {args.type or 'none (all)'}")

    types = [t.strip() for t in args.type.split(",")] if args.type else None

    # ── Pass 1: collect all gene symbols + per-gene metadata ────────────────
    log.info("Pass 1: collecting gene universe …")

    gene_meta: dict[str, dict] = {}   # gene_symbol → {Type, Subtype, Class, Subclass}
    failed_files: list[str] = []

    for path in tqdm(tsv_files, desc="Scanning TSVs", unit="file"):
        df = load_tsv(path)
        if df is None:
            failed_files.append(path.name)
            continue

        df = apply_filters(df, args.scope, types)

        if args.col not in df.columns:
            log.warning(f"Column '{args.col}' not in {path.name} — skipping")
            failed_files.append(path.name)
            continue

        for _, row in df.iterrows():
            gene = str(row[args.col]).strip()
            if not gene or gene.lower() == "nan":
                continue
            if gene not in gene_meta:
                gene_meta[gene] = {
                    col: str(row.get(col, "")).strip()
                    for col in METADATA_COLS
                    if col in df.columns
                }

    all_genes = sorted(gene_meta.keys())
    log.info(f"Unique genes detected : {len(all_genes)}")

    if not all_genes:
        log.error("No genes found after filtering. Check --scope / --type flags.")
        return

    gene_index = {g: i for i, g in enumerate(all_genes)}

    # ── Pass 2: build presence/absence matrix ───────────────────────────────
    log.info("Pass 2: building presence/absence matrix …")

    records: list[dict] = []

    for path in tqdm(tsv_files, desc="Building matrix", unit="file"):
        sample = path.stem
        df = load_tsv(path)

        row_dict: dict[str, int] = {"sample_id": sample}

        if df is not None and args.col in df.columns:
            df = apply_filters(df, args.scope, types)
            detected = set(
                str(g).strip()
                for g in df[args.col].dropna()
                if str(g).strip() and str(g).strip().lower() != "nan"
            )
        else:
            detected = set()

        for gene in all_genes:
            row_dict[gene] = 1 if gene in detected else 0

        records.append(row_dict)

    matrix_df = pd.DataFrame(records).set_index("sample_id")
    matrix_df.index.name = "sample_id"

    # ── Summary stats ────────────────────────────────────────────────────────
    n_samples, n_genes = matrix_df.shape
    gene_freq = matrix_df.sum(axis=0)

    log.info(f"Matrix shape    : {n_samples} samples × {n_genes} genes")
    log.info(f"Core genes (≥90% prevalence) : {(gene_freq >= 0.9 * n_samples).sum()}")
    log.info(f"Rare genes (≤5% prevalence)  : {(gene_freq <= 0.05 * n_samples).sum()}")
    log.info(f"Mean genes/sample            : {matrix_df.sum(axis=1).mean():.1f}")

    if failed_files:
        log.warning(f"Files skipped/failed: {len(failed_files)}")
        failed_path = args.out_dir / "failed_tsv_files.txt"
        failed_path.write_text("\n".join(failed_files))
        log.warning(f"Failed file list written to: {failed_path}")

    # ── Save outputs ─────────────────────────────────────────────────────────
    matrix_path = args.out_dir / "amr_gene_matrix.csv"
    matrix_df.to_csv(matrix_path)
    log.info(f"Matrix saved    : {matrix_path}")

    meta_df = pd.DataFrame.from_dict(gene_meta, orient="index")
    meta_df.index.name = "gene"
    meta_df["prevalence"] = gene_freq.values
    meta_df["prevalence_pct"] = (gene_freq.values / n_samples * 100).round(2)
    meta_path = args.out_dir / "amr_gene_matrix_metadata.csv"
    meta_df.to_csv(meta_path)
    log.info(f"Metadata saved  : {meta_path}")

    log.info("Done.")


if __name__ == "__main__":
    main()
