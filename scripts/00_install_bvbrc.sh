#!/usr/bin/env bash

set -euo pipefail

# Update package index
sudo apt-get update

# Install gdebi
sudo apt-get install -y gdebi-core curl

# Download BV-BRC CLI package
curl -O -L https://github.com/BV-BRC/BV-BRC-CLI/releases/download/1.040/bvbrc-cli-1.040.deb

# Install the package
sudo dpkg -i bvbrc-cli-1.040.deb || true

# Fix missing dependencies if any
sudo apt-get -f install -y

# Verify installation
p3-genome-fasta --contig --help
