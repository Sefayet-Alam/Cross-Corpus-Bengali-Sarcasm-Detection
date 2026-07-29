# rerunPod.md — RunPod Restart / New Pod Setup Guide

Use this when the old RunPod GPU is unavailable, migration fails, or you create a fresh new GPU pod.

**Important:** Your GitHub repo is private now, so GitHub will ask for authentication when cloning/pulling/pushing.

Never paste your GitHub token into ChatGPT, notebooks, markdown files, or Git commits.

---

## 0. Recommended GPU

Use any of these if available:

```text
A40 48GB       Best
RTX 4090 24GB  Recommended
RTX A5000 24GB Okay
RTX A4500 20GB Okay
RTX A4000 16GB Tight but usable for smaller runs
```

Avoid CPU unless you only need to inspect files or push/pull GitHub changes.

---

## 1. Open Jupyter Terminal

After pod starts:

```text
RunPod Dashboard → Pod → Connect → Jupyter Lab → Terminal
```

---

## 2. Check GPU

Run:

```bash
nvidia-smi

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expected important output:

```text
True
```

If `torch.cuda.is_available()` is `False`, do not start training.

---

## 3. Set cache paths to `/workspace`

Run this before downloading models:

```bash
mkdir -p /workspace/.cache/huggingface
mkdir -p /workspace/.cache/pip

export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export PIP_CACHE_DIR=/workspace/.cache/pip

echo 'export HF_HOME=/workspace/.cache/huggingface' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=/workspace/.cache/huggingface' >> ~/.bashrc
echo 'export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets' >> ~/.bashrc
echo 'export PIP_CACHE_DIR=/workspace/.cache/pip' >> ~/.bashrc
```

This keeps HuggingFace and pip downloads in `/workspace`, not the smaller root disk.

---

## 4. Configure Git

```bash
git config --global user.name "Sefayet-Alam"
git config --global user.email "sefayetalam14@gmail.com"
git config --global credential.helper "cache --timeout=86400"
```

This caches your GitHub token for 24 hours.

---

## 5. Clone private GitHub repo

Go to workspace:

```bash
cd /workspace
```

If this is a completely new pod and the repo is not already there:

```bash
git clone https://github.com/Sefayet-Alam/Sarcasm_detection.git
cd Sarcasm_detection
```

When GitHub asks:

```text
Username: Sefayet-Alam
Password: paste your GitHub token
```

The password field may look blank while pasting. That is normal.

If you attached the same network volume and the repo already exists:

```bash
cd /workspace/Sarcasm_detection
git pull origin main
```

Again, if asked, use your GitHub token as the password.

---

## 6. Install Python packages

```bash
pip install transformers==4.40.0 \
            datasets==2.19.0 \
            accelerate==0.29.3 \
            scikit-learn==1.4.2 \
            pandas==2.2.2 \
            numpy==1.26.4 \
            openpyxl==3.1.2 \
            matplotlib==3.8.4 \
            seaborn==0.13.2 \
            scipy==1.13.0 \
            tqdm==4.66.2 \
            ipykernel==6.29.4 \
            jupyter==1.0.0 \
            --quiet
```

---

## 7. Create a clear Jupyter kernel

```bash
python -m ipykernel install --user --name sarcasm-runpod --display-name "Sarcasm RunPod GPU"
```

In Jupyter notebooks, choose:

```text
Kernel → Change Kernel → Sarcasm RunPod GPU
```

---

## 8. Verify environment

```bash
cd /workspace/Sarcasm_detection

python -c "import torch, transformers, pandas, sklearn; \
print('cuda:', torch.cuda.is_available()); \
print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'); \
print('transformers:', transformers.__version__); \
print('pandas:', pandas.__version__); \
print('sklearn:', sklearn.__version__)"
```

Expected:

```text
cuda: True
gpu: <your GPU name>
transformers: 4.40.0
pandas: 2.2.2
sklearn: 1.4.2
```

---

## 9. Check raw data files

```bash
ls 01_data/raw/ben_sarc/
ls 01_data/raw/banglasarc/
ls 01_data/raw/banglasarc3/
```

Expected files:

```text
Ben-Sarc.xlsx
SarcasDetection.csv
BanglaSarc3.xlsx
```

If any are missing, upload them or pull the correct repo state.

---

## 10. Optional: pre-download HuggingFace models

Run this in a notebook cell or Python script:

```python
from transformers import AutoTokenizer, AutoModel

models = [
    "csebuetnlp/banglabert",
    "google/muril-base-cased",
    "xlm-roberta-base",
    "bert-base-multilingual-cased",
    "sagorsarker/bangla-bert-base",
    "l3cube-pune/bengali-bert",
]

for m in models:
    print(f"Downloading {m}...")
    AutoTokenizer.from_pretrained(m)
    AutoModel.from_pretrained(m)
    print(f"✅ {m} cached")
```

---

# Normal Workflow

## A. On Mac / VSCode

After changing notebooks or code locally:

```bash
cd ~/path/to/Sarcasm_detection

git status
git add 01_data 02_notebooks 04_outputs
git commit -m "update notebook/code"
git push origin main
```

---

## B. On RunPod before running notebook

```bash
cd /workspace/Sarcasm_detection
git pull origin main
```

Then open the notebook in Jupyter and choose:

```text
Sarcasm RunPod GPU
```

Run the notebook.

---

## C. On RunPod after running notebook

Check status:

```bash
cd /workspace/Sarcasm_detection
git status
```

Push small generated files:

```bash
git add 01_data 02_notebooks 04_outputs
git commit -m "NBxx: completed"
git push origin main
```

Do **not** push checkpoints or model weights.

Ignored by `.gitignore`:

```text
03_checkpoints/
03_models/
*.pt
*.bin
*.pth
*.safetensors
*.ckpt
*.h5
```

---

## D. On Mac after RunPod push

```bash
cd ~/path/to/Sarcasm_detection
git pull origin main
```

Then inspect outputs in VSCode.

---

# Storage and Cost Checks

## Check disk space

```bash
df -h /workspace
df -h /
du -h --max-depth=1 /workspace/Sarcasm_detection | sort -hr
du -sh /workspace/.cache/huggingface /workspace/Sarcasm_detection/03_checkpoints 2>/dev/null
```

Inside a notebook cell:

```python
!df -h /workspace
!df -h /
!du -h --max-depth=1 /workspace/Sarcasm_detection | sort -hr
```

---

## Check GPU usage

```bash
nvidia-smi
```

Live monitoring:

```bash
watch -n 2 nvidia-smi
```

Stop live monitoring:

```text
Ctrl + C
```

If you see:

```text
GPU-Util 0%
No running processes found
```

and you are not actively running a notebook, the GPU is idle.

---

# Before Taking a Break

Run:

```bash
cd /workspace/Sarcasm_detection
git status
df -h /workspace
nvidia-smi
```

Then:

1. Push important notebooks, CSVs, logs, and outputs to GitHub.
2. Do not push checkpoints.
3. Stop the pod if you are not actively working.
4. Expect the expensive GPU cost to stop after stopping the pod.
5. Small storage/volume cost may continue while data is preserved.

---

# If Pod GPU Is No Longer Available

If RunPod shows:

```text
Your Pod's GPUs are no longer available.
```

You have three options:

## Option 1 — Do nothing

Choose this if you are done for now and want to avoid restarting compute.

## Option 2 — Automatically migrate pod data

Choose this if you want to continue working now and RunPod has a similar GPU available.

If it says:

```text
There are no instances currently available
```

wait 5–20 minutes and try again, or create a new pod with another GPU.

## Option 3 — Start with CPU

Only use CPU if you just need to inspect files, push/pull GitHub, or copy data.

Do not use CPU for transformer training.

---

# Quick New Pod Command Block

Use this block when starting a fresh pod:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

mkdir -p /workspace/.cache/huggingface
mkdir -p /workspace/.cache/pip

export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export PIP_CACHE_DIR=/workspace/.cache/pip

echo 'export HF_HOME=/workspace/.cache/huggingface' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=/workspace/.cache/huggingface' >> ~/.bashrc
echo 'export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets' >> ~/.bashrc
echo 'export PIP_CACHE_DIR=/workspace/.cache/pip' >> ~/.bashrc

git config --global user.name "Sefayet-Alam"
git config --global user.email "sefayetalam14@gmail.com"
git config --global credential.helper "cache --timeout=86400"

cd /workspace
git clone https://github.com/Sefayet-Alam/Sarcasm_detection.git
cd Sarcasm_detection

pip install transformers==4.40.0 \
            datasets==2.19.0 \
            accelerate==0.29.3 \
            scikit-learn==1.4.2 \
            pandas==2.2.2 \
            numpy==1.26.4 \
            openpyxl==3.1.2 \
            matplotlib==3.8.4 \
            seaborn==0.13.2 \
            scipy==1.13.0 \
            tqdm==4.66.2 \
            ipykernel==6.29.4 \
            jupyter==1.0.0 \
            --quiet

python -m ipykernel install --user --name sarcasm-runpod --display-name "Sarcasm RunPod GPU"

python -c "import torch, transformers, pandas, sklearn; \
print('cuda:', torch.cuda.is_available()); \
print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'); \
print('transformers:', transformers.__version__); \
print('pandas:', pandas.__version__); \
print('sklearn:', sklearn.__version__)"
```

If `git clone` says the folder already exists, use:

```bash
cd /workspace/Sarcasm_detection
git pull origin main
```

