#!/bin/bash
#SBATCH --job-name=neuromechfly
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b300:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/home/zny/flygym/outputs/slurm-%j.log
set -euo pipefail
cd /mnt/home/zny/flygym
hostname > outputs/compute-node
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
export OMP_NUM_THREADS=8
export MPLCONFIGDIR="$PWD/outputs/matplotlib"
export CUPY_CACHE_DIR="$PWD/outputs/cupy-cache"
export PYTHONUNBUFFERED=1
# Deployment preparation can proceed while Slurm assigns the GPU.
for attempt in $(seq 1 900); do
  if test -f deploy/ready; then
    exec .venv/bin/python scripts/serve_local.py --backend cuda --port 8000 \
      --allowed-origin https://cw-login-zny.alpaca-elnath.ts.net:8443
  fi
  sleep 2
done
echo 'Deployment files were not marked ready within 30 minutes.' >&2
exit 1
