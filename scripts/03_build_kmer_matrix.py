#!/usr/bin/env python3
"""
03_build_kmer_matrix.py
-----------------------
End-to-end KMC + sparse k-mer matrix generation pipeline.

Pipeline:
    Genome assemblies
        ↓
    KMC counting (parallel, isolated tmp dirs)
        ↓
    k-mer text dumps
        ↓
    Sparse CSR matrix construction (streaming, low-memory)

Key fixes vs prior version:
    - Per-job KMC temp directories (no more bin-file collisions →
      no more 'CThreadCancellationException' / 'Corrupted file' errors).
    - Cleanup of stale .kmc_pre / .kmc_suf from prior failed runs.
    - Streaming sparse matrix construction with periodic COO->CSR
      flushes so RAM doesn't blow up on large vocabularies.
    - Parallel KMC dumps.
    - Robust resume logic: skip a genome only if the FINAL .txt exists.
    - Single-pass vocabulary + matrix build option (saves one full read
      over every dump file).

Example (32-core / 128 GB instance, k=16):
    python 03_build_kmer_matrix.py \\
        --genomes-dir data/genomes \\
        --kmc-dir     data/kmc_k16 \\
        --output-prefix k16_matrix \\
        --k 16 \\
        --workers 8 \\
        --threads 4 \\
        --memory 14
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, save_npz, vstack
from tqdm import tqdm


# ─────────────────────────── CLI ────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="KMC k-mer counting + sparse matrix pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--genomes-dir", required=True,
                   help="Directory containing genome FASTA files (.fna/.fna.gz/.fasta)")
    p.add_argument("--kmc-dir", required=True,
                   help="Directory to store KMC outputs and per-job tmp subdirs")
    p.add_argument("--output-prefix", default="kmer_matrix",
                   help="Output matrix prefix")
    p.add_argument("--k", type=int, default=11, help="k-mer size")
    p.add_argument("--workers", type=int, default=8,
                   help="Number of genomes processed in parallel")
    p.add_argument("--threads", type=int, default=4,
                   help="Threads per KMC job (workers * threads should be <= physical cores)")
    p.add_argument("--memory", type=int, default=14,
                   help="RAM limit per KMC job in GB (workers * memory should be <= total RAM)")
    p.add_argument("--min-count", type=int, default=1,
                   help="Minimum k-mer count threshold (KMC -ci)")
    p.add_argument("--max-count", type=int, default=1_000_000,
                   help="Maximum KMC counter saturation (KMC -cs)")
    p.add_argument("--dtype", default="uint32",
                   choices=["uint16", "uint32", "uint64"],
                   help="Sparse matrix value dtype")
    p.add_argument("--flush-every", type=int, default=64,
                   help="Flush COO buffer to CSR every N samples (memory control)")
    p.add_argument("--force", action="store_true",
                   help="Recompute KMC outputs even if cached")
    p.add_argument("--keep-tmp", action="store_true",
                   help="Don't delete the parent tmp dir at the end (per-job tmp dirs are always cleaned)")

    return p.parse_args()


# ─────────────────────────── LOGGING ────────────────────────────────────────

def setup_logging(out_dir: Path) -> logging.Logger:
    log = logging.getLogger("kmer_pipeline")
    log.setLevel(logging.INFO)
    # Avoid duplicate handlers if main() is re-entered (e.g. notebooks)
    log.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(sh)

    fh = logging.FileHandler(out_dir / "kmer_pipeline.log")
    fh.setFormatter(fmt)
    log.addHandler(fh)

    return log


# ─────────────────────────── VALIDATION ─────────────────────────────────────

def check_dependencies():
    for tool in ("kmc", "kmc_tools"):
        if shutil.which(tool) is None:
            raise SystemExit(f"ERROR: {tool} not found in PATH")


# ─────────────────────────── GENOME DISCOVERY ───────────────────────────────

def discover_genomes(genomes_dir: Path) -> list[Path]:
    genomes = sorted(
        list(genomes_dir.glob("*.fna"))
        + list(genomes_dir.glob("*.fna.gz"))
        + list(genomes_dir.glob("*.fasta"))
        + list(genomes_dir.glob("*.fasta.gz"))
        + list(genomes_dir.glob("*.fa"))
        + list(genomes_dir.glob("*.fa.gz"))
    )
    if not genomes:
        raise FileNotFoundError(f"No genomes found in: {genomes_dir}")
    return genomes


def genome_sample_name(genome: Path) -> str:
    """Strip .gz and extension to derive sample name."""
    name = genome.name
    if name.endswith(".gz"):
        name = name[:-3]
    for ext in (".fna", ".fasta", ".fa"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    return name


# ─────────────────────────── KMC RUNNER ─────────────────────────────────────

def run_kmc(
    genome: Path,
    kmc_dir: Path,
    tmp_parent: Path,
    k: int,
    threads: int,
    memory: int,
    min_count: int,
    max_count: int,
    force: bool = False,
):
    """
    Count k-mers for one genome and dump to text. Each invocation gets
    its OWN tmp directory so KMC's internal bin files (kmc_00000.bin …)
    can't collide between parallel workers.
    """
    name = genome_sample_name(genome)

    out_prefix = kmc_dir / name
    txt_file = kmc_dir / f"{name}.txt"
    log_file = kmc_dir / f"{name}.kmc.log"

    # Resume: only treat as cached if the final dump exists
    if txt_file.exists() and txt_file.stat().st_size > 0 and not force:
        return name, "cached", None

    # Clean up partial state from a previous failed run
    for ext in (".kmc_pre", ".kmc_suf"):
        stale = Path(f"{out_prefix}{ext}")
        if stale.exists():
            try:
                stale.unlink()
            except OSError:
                pass
    if txt_file.exists():
        try:
            txt_file.unlink()
        except OSError:
            pass

    # Per-job tmp dir, auto-cleaned on exit (success OR failure)
    try:
        with tempfile.TemporaryDirectory(prefix=f"{name}_", dir=tmp_parent) as job_tmp:
            cmd = [
                "kmc",
                f"-k{k}",
                "-fm",
                f"-ci{min_count}",
                f"-cs{max_count}",
                f"-t{threads}",
                f"-m{memory}",
                str(genome),
                str(out_prefix),
                job_tmp,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            with open(log_file, "w") as log:
                log.write("=== kmc ===\n")
                log.write(result.stdout)
                log.write(result.stderr)

            if result.returncode != 0:
                return name, "failed", (result.stderr or result.stdout).strip()

            dump_cmd = [
                "kmc_tools",
                "transform",
                str(out_prefix),
                "dump",
                str(txt_file),
            ]
            dump_result = subprocess.run(dump_cmd, capture_output=True, text=True)

            with open(log_file, "a") as log:
                log.write("\n=== kmc_tools dump ===\n")
                log.write(dump_result.stdout)
                log.write(dump_result.stderr)

            if dump_result.returncode != 0:
                return name, "failed", (dump_result.stderr or dump_result.stdout).strip()

        # Remove KMC database files now that we have the .txt dump
        for ext in (".kmc_pre", ".kmc_suf"):
            db = Path(f"{out_prefix}{ext}")
            if db.exists():
                try:
                    db.unlink()
                except OSError:
                    pass

        return name, "done", None

    except Exception as e:
        return name, "failed", f"exception: {e!r}"


# ─────────────────────────── MATRIX CONSTRUCTION ────────────────────────────

def build_matrix_streaming(
    txt_files: list[Path],
    dtype: str,
    flush_every: int,
    log: logging.Logger,
):
    """
    Single-pass build:
      - assigns k-mer IDs on first sight
      - accumulates a COO buffer
      - flushes to CSR every `flush_every` samples to keep RAM bounded
      - finalises by hstack-padding partial CSRs to the full vocab width
        and vstacking them into the final matrix.
    """
    kmer_to_idx: dict[str, int] = {}
    sample_ids: list[str] = []

    # Per-flush buffers
    buf_rows: list[int] = []
    buf_cols: list[int] = []
    buf_data: list[int] = []
    flushed_blocks: list[csr_matrix] = []
    flushed_widths: list[int] = []  # vocab size at the moment of each flush

    samples_in_current_block = 0
    block_row_offset = 0  # row index within the current block (resets each flush)

    np_dtype = np.dtype(dtype)

    def flush_block():
        nonlocal buf_rows, buf_cols, buf_data, samples_in_current_block, block_row_offset
        if samples_in_current_block == 0:
            return
        width = len(kmer_to_idx)
        coo = coo_matrix(
            (
                np.asarray(buf_data, dtype=np_dtype),
                (
                    np.asarray(buf_rows, dtype=np.int64),
                    np.asarray(buf_cols, dtype=np.int64),
                ),
            ),
            shape=(samples_in_current_block, width),
            dtype=np_dtype,
        )
        flushed_blocks.append(coo.tocsr())
        flushed_widths.append(width)
        buf_rows = []
        buf_cols = []
        buf_data = []
        samples_in_current_block = 0
        block_row_offset = 0

    for file in tqdm(txt_files, desc="Reading dumps + matrix", unit="file"):
        sample_ids.append(file.stem)
        with open(file) as fh:
            for line in fh:
                # Each line: <kmer>\t<count>
                tab = line.find("\t")
                if tab < 0:
                    continue
                kmer = line[:tab]
                count_str = line[tab + 1:].rstrip()

                idx = kmer_to_idx.get(kmer)
                if idx is None:
                    idx = len(kmer_to_idx)
                    kmer_to_idx[kmer] = idx

                buf_rows.append(block_row_offset)
                buf_cols.append(idx)
                buf_data.append(int(count_str))

        block_row_offset += 1
        samples_in_current_block += 1

        if samples_in_current_block >= flush_every:
            flush_block()

    flush_block()  # final partial block

    final_width = len(kmer_to_idx)
    log.info(f"Stitching {len(flushed_blocks)} block(s) into final CSR…")

    # Pad each block (which was built with a smaller width snapshot) up
    # to the final vocab width. scipy CSR can be resized cheaply via
    # constructor on the underlying arrays.
    padded_blocks = []
    for block, width in zip(flushed_blocks, flushed_widths):
        if width == final_width:
            padded_blocks.append(block)
        else:
            padded_blocks.append(
                csr_matrix(
                    (block.data, block.indices, block.indptr),
                    shape=(block.shape[0], final_width),
                )
            )

    matrix = vstack(padded_blocks, format="csr", dtype=np_dtype)
    return matrix, np.array(sample_ids), kmer_to_idx


# ─────────────────────────── MAIN ───────────────────────────────────────────

def main():
    args = parse_args()

    genomes_dir = Path(args.genomes_dir)
    kmc_dir = Path(args.kmc_dir)
    kmc_dir.mkdir(parents=True, exist_ok=True)

    tmp_parent = kmc_dir / "tmp"
    tmp_parent.mkdir(exist_ok=True)

    log = setup_logging(kmc_dir)
    check_dependencies()

    genomes = discover_genomes(genomes_dir)
    log.info(f"Genomes discovered: {len(genomes)}")
    log.info(f"k-mer size:         {args.k}")
    log.info(f"Workers:            {args.workers}")
    log.info(f"Threads per job:    {args.threads}  (total CPU budget: {args.workers * args.threads})")
    log.info(f"Memory per job:     {args.memory} GB  (total RAM budget: {args.workers * args.memory} GB)")

    counts = {"done": 0, "cached": 0, "failed": 0}
    failed: list[tuple[str, str]] = []

    # -- KMC counting in parallel ---------------------------------------------
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_kmc,
                genome=genome,
                kmc_dir=kmc_dir,
                tmp_parent=tmp_parent,
                k=args.k,
                threads=args.threads,
                memory=args.memory,
                min_count=args.min_count,
                max_count=args.max_count,
                force=args.force,
            ): genome
            for genome in genomes
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Running KMC",
            unit="genome",
        ):
            name, status, stderr = future.result()
            counts[status] += 1
            if status == "failed":
                failed.append((name, stderr or ""))
                log.error(f"FAILED: {name}\n{stderr}")

    if failed:
        failed_file = kmc_dir / "failed_kmc.txt"
        failed_file.write_text(
            "\n".join(f"{n}\t{(e or '').splitlines()[0] if e else ''}" for n, e in sorted(failed))
        )
        log.warning(f"Failed genomes written to {failed_file}")

    # -- Sparse matrix construction -------------------------------------------
    txt_files = sorted(kmc_dir.glob("*.txt"))
    log.info(f"KMC text dumps to ingest: {len(txt_files)}")

    if not txt_files:
        log.error("No KMC dumps found — aborting before matrix step.")
        return

    matrix, sample_ids, kmer_to_idx = build_matrix_streaming(
        txt_files=txt_files,
        dtype=args.dtype,
        flush_every=args.flush_every,
        log=log,
    )

    log.info(
        f"Sparse matrix shape: {matrix.shape[0]:,} × {matrix.shape[1]:,} "
        f"({matrix.nnz:,} non-zeros, {matrix.data.nbytes / 1e9:.2f} GB data)"
    )

    # -- Save -----------------------------------------------------------------
    prefix = args.output_prefix
    save_npz(f"{prefix}.npz", matrix)
    np.save(f"{prefix}_sample_ids.npy", sample_ids)

    idx_to_kmer = np.empty(len(kmer_to_idx), dtype=object)
    for kmer, idx in kmer_to_idx.items():
        idx_to_kmer[idx] = kmer
    np.save(f"{prefix}_vocab.npy", idx_to_kmer)

    log.info(f"Saved matrix:     {prefix}.npz")
    log.info(f"Saved sample IDs: {prefix}_sample_ids.npy")
    log.info(f"Saved vocabulary: {prefix}_vocab.npy")

    # -- Cleanup --------------------------------------------------------------
    if not args.keep_tmp:
        try:
            shutil.rmtree(tmp_parent)
        except OSError as e:
            log.warning(f"Could not remove {tmp_parent}: {e}")

    log.info(
        f"Complete — done: {counts['done']} | "
        f"cached: {counts['cached']} | "
        f"failed: {counts['failed']}"
    )


if __name__ == "__main__":
    main()
