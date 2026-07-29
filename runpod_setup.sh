#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

python3 -m ipykernel install \
  --user \
  --name sarcasm-runpod \
  --display-name "Sarcasm RunPod GPU"
