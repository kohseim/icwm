#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
config="${1:-configs/paper.json}"
data_root="${2:-data/paper}"
output_root="${3:-outputs/paper}"
python -m graph_icl.data --config "$config" --out "$data_root"
for model in 4b 8b 14b; do
    for topology in lattice_4x4 lattice_5x5 lattice_3x5 triangle_4x4 torus_4x4; do
        python -m graph_icl.probe --config "$config" --model "$model" \
            --topology "$topology" --data "$data_root" --out "$output_root"
    done
done
