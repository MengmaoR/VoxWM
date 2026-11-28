#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the GPU IDs for this run}"
python_bin="${VAE_PYTHON:-.venv-vae/bin/python}"
IFS=, read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
if [[ "${#devices[@]}" -eq 1 ]]; then
    exec "$python_bin" -u -m voxel_vae.curated_vae run --device cuda:0 "$@"
fi
exec "$python_bin" -u -m torch.distributed.run --standalone \
    --nproc_per_node="${#devices[@]}" --module voxel_vae.curated_vae run "$@"
