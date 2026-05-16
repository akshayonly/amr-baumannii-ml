#!/usr/bin/env python3
"""
02_run_amrfinder.py
-------------------
Run AMRFinderPlus on genome assemblies in parallel.

Usage:
    python 02_run_amrfinder.py \
        --genomes-dir data/genomes \
        --out-dir data/amrfinder \
        --organism Acinetobacter \
        --workers 6 \
        --threads 4

Requirements:
    - AMRFinderPlus installed
    - amrfinder available in PATH
"""

import argparse
import logging
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


# ─────────────────────────── CLI ────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Batch-run AMRFinderPlus on genome assemblies."
    )

    p.add_argument(
        "--genomes-dir",
        required=True,
        help="Directory containing genome FASTA files",
    )

    p.add_argument(
        "--out-dir",
        required=True,
        help="Directory to store AMRFinderPlus results",
    )

    p.add_argument(
        "--organism",
        default="Acinetobacter",
        help="AMRFinderPlus organism name",
    )

    p.add_argument(
        "--ext",
        default=".fna",
        help="Genome FASTA extension (default: .fna)",
    )

    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel AMRFinder jobs (default: 4)",
    )

    p.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Threads per AMRFinder job (default: 4)",
    )

    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs",
    )

    return p.parse_args()


# ─────────────────────────── LOGGING ────────────────────────────────────────

def setup_logging(out_dir: Path) -> logging.Logger:
    log = logging.getLogger("run_amrfinder")
    log.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    )

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(sh)

    fh = logging.FileHandler(out_dir / "amrfinder.log")
    fh.setFormatter(fmt)
    log.addHandler(fh)

    return log


# ─────────────────────────── VALIDATION ─────────────────────────────────────

def check_dependencies():
    if shutil.which("amrfinder") is None:
        raise SystemExit(
            "ERROR: amrfinder not found in PATH.\n"
            "Install AMRFinderPlus before running."
        )


# ─────────────────────────── RUNNER ─────────────────────────────────────────

def run_amrfinder(
    fasta: Path,
    out_dir: Path,
    organism: str,
    threads: int,
    force: bool = False,
) -> tuple:
    """
    Run AMRFinderPlus on a single genome.

    Returns:
        (sample, status, stderr)
    """

    sample = fasta.stem
    out_path = out_dir / f"{sample}.tsv"

    # Skip completed outputs
    if (
        out_path.exists()
        and out_path.stat().st_size > 0
        and not force
    ):
        return sample, "cached", None

    tmp_out = out_path.with_suffix(".tmp")

    cmd = [
        "amrfinder",
        "-n", str(fasta),
        "--organism", organism,
        "--plus",
        "-o", str(tmp_out),
        "--threads", str(threads),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        if tmp_out.exists():
            tmp_out.unlink()

        return sample, "failed", result.stderr.strip()

    # Validate output
    if not tmp_out.exists() or tmp_out.stat().st_size == 0:
        return sample, "failed", "Empty AMRFinder output"

    # Atomic rename
    tmp_out.rename(out_path)

    return sample, "done", None


# ─────────────────────────── MAIN ───────────────────────────────────────────

def main():
    args = parse_args()

    genomes_dir = Path(args.genomes_dir)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logging(out_dir)

    check_dependencies()

    fasta_files = sorted(
        genomes_dir.glob(f"*{args.ext}")
    )

    if not fasta_files:
        log.error(
            f"No {args.ext} files found in {genomes_dir}"
        )
        return

    log.info(f"Genomes found: {len(fasta_files)}")
    log.info(f"Organism: {args.organism}")
    log.info(f"Workers: {args.workers}")
    log.info(f"Threads/job: {args.threads}")
    log.info(f"Output: {out_dir}")

    counts = {
        "done": 0,
        "cached": 0,
        "failed": 0,
    }

    failed = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:

        futures = {
            executor.submit(
                run_amrfinder,
                fasta=fasta,
                out_dir=out_dir,
                organism=args.organism,
                threads=args.threads,
                force=args.force,
            ): fasta
            for fasta in fasta_files
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Running AMRFinder",
            unit="genome",
        ):

            sample, status, stderr = future.result()

            counts[status] += 1

            if status == "failed":
                failed.append(sample)

                log.error(
                    f"FAILED: {sample}\n{stderr}"
                )

    # Failed sample report
    if failed:
        failed_file = out_dir / "failed_samples.txt"

        failed_file.write_text(
            "\n".join(sorted(failed))
        )

        log.warning(
            f"Failed samples written to {failed_file}"
        )

    # Summary
    log.info(
        f"Complete — done: {counts['done']} | "
        f"cached: {counts['cached']} | "
        f"failed: {counts['failed']} / {len(fasta_files)} total"
    )


if __name__ == "__main__":
    main()
