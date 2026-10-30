#!/usr/bin/env bash
set -eu
cd /workspace/ecg-code
mkdir -p /workspace/outputs
python3 --version
nvidia-smi
if ! git lfs version >/dev/null 2>&1
then
  apt-get update
  apt-get install -y git-lfs
fi
python3 -m venv --system-site-packages /opt/ecg-reproduction
/opt/ecg-reproduction/bin/python -m pip install --disable-pip-version-check --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
/opt/ecg-reproduction/bin/python -m pip install --disable-pip-version-check --no-cache-dir -r reproductions/requirements-common.txt wfdb==4.3.0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export MPLBACKEND=Agg
export HF_HOME=/workspace/hf-cache
worker_module=$(/opt/ecg-reproduction/bin/python -c "import json; print(json.load(open('/workspace/execution.json'))['plan'].get('worker_module', 'reproductions.runpod.worker'))")
/opt/ecg-reproduction/bin/python -u -m "$worker_module" --workspace /workspace
