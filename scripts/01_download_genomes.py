#!/usr/bin/env python3

"""
Download genome FASTA files from BV-BRC using genome IDs.

Example:
    python download_genomes.py \
        --input genome_ids.txt \
        --output carbapenem_baumannii_genomes \
        --threads 4 \
        --archive

Requirements:
    - BV-BRC CLI installed
    - p3-genome-fasta available in PATH
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download genome FASTA files from BV-BRC"
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="Path to genome IDs text file",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="genomes",
        type=Path,
        help="Output directory (default: genomes)",
    )

    parser.add_argument(
        "-t",
        "--threads",
        default=4,
        type=int,
        help="Number of parallel downloads (default: 8)",
    )

    parser.add_argument(
        "--archive",
        action="store_true",
        help="Create compressed tar.gz archive after download",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download existing genomes",
    )

    return parser.parse_args()


def check_dependencies() -> None:
    if shutil.which("p3-genome-fasta") is None:
        raise SystemExit(
            "Error: 'p3-genome-fasta' not found in PATH.\n"
            "Install BV-BRC CLI first."
        )


def load_genome_ids(file_path: Path) -> list[str]:
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with open(file_path) as f:
        genome_ids = sorted(
            {
                line.strip()
                for line in f
                if line.strip()
            }
        )

    if not genome_ids:
        raise ValueError("No genome IDs found in input file.")

    return genome_ids


def download_genome(
    genome_id: str,
    output_dir: Path,
    force: bool = False,
) -> tuple[str, bool, str | None]:
    output_file = output_dir / f"{genome_id}.fna"

    # Skip existing files
    if output_file.exists() and not force:
        return genome_id, True, "skipped"

    cmd = [
        "p3-genome-fasta",
        "--contig",
        genome_id,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode == 0 and result.stdout.strip():
        output_file.write_text(result.stdout)
        return genome_id, True, "downloaded"

    return genome_id, False, result.stderr.strip()


def create_archive(output_dir: Path) -> Path:
    archive_path = output_dir.with_suffix(".tar.gz")

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(output_dir, arcname=output_dir.name)

    return archive_path


def main() -> None:
    args = parse_args()

    check_dependencies()

    args.output.mkdir(parents=True, exist_ok=True)

    genome_ids = load_genome_ids(args.input)

    failed: list[tuple[str, str | None]] = []
    downloaded = 0
    skipped = 0

    print(f"Genomes to process: {len(genome_ids)}")
    print(f"Output directory: {args.output}")
    print(f"Threads: {args.threads}\n")

    with ThreadPoolExecutor(max_workers=args.threads) as executor:

        futures = {
            executor.submit(
                download_genome,
                genome_id=gid,
                output_dir=args.output,
                force=args.force,
            ): gid
            for gid in genome_ids
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Downloading genomes",
        ):
            genome_id, success, message = future.result()

            if success:
                if message == "skipped":
                    skipped += 1
                else:
                    downloaded += 1
            else:
                failed.append((genome_id, message))

    # Save failed downloads
    if failed:
        failed_file = args.output / "failed_genomes.txt"

        with open(failed_file, "w") as f:
            for genome_id, error in failed:
                f.write(f"{genome_id}\t{error}\n")

        print(f"\nFailed genome list written to: {failed_file}")

    # Optional archive creation
    if args.archive:
        print("\nCreating archive...")
        archive_path = create_archive(args.output)
        print(f"Archive created: {archive_path}")

    # Summary
    print("\nSummary")
    print("-" * 40)
    print(f"Total genomes : {len(genome_ids)}")
    print(f"Downloaded    : {downloaded}")
    print(f"Skipped       : {skipped}")
    print(f"Failed        : {len(failed)}")


if __name__ == "__main__":
    main()
