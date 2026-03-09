# LSQ Reproduction on ImageNet (Pre-Activation ResNet-18)

This repository is my implementation-focused reproduction of the LSQ paper:

- **Paper:** *Learned Step Size Quantization* (arXiv: `1902.08153v3`)
- **Local copy in this repo:** [`1902.08153v3.pdf`](./1902.08153v3.pdf)
- **Goal:** reproduce the core training protocol and quantization behavior for ImageNet classification

I built this project as part of my graduate coursework and as a practical, engineering-oriented research reproduction for my portfolio.

## Why This Project

Quantization-aware training is a key technique for deploying deep models on resource-constrained hardware. This project reproduces LSQ with a clean PyTorch codebase and paper-aligned defaults, focusing on:

- implementation correctness
- reproducible training workflow
- practical CLI tooling for training and evaluation

## What Is Implemented

- **Model:** pre-activation ResNet-18
- **Two-stage pipeline:**
  1. Full-precision (FP) training
  2. LSQ quantized fine-tuning initialized from FP checkpoint
- **Quantized targets:** Conv/Linear weights + input activations
- **First/last layer policy:** optional higher precision (default 8-bit)
- **Optimizer:** SGD + momentum (`0.9`)
- **Scheduler:** cosine annealing (no restart)
- **Default schedule by bit-width (paper-aligned):**
  - 2/3/4-bit: 90 epochs, LR 0.01
  - 8-bit: 1 epoch, LR 0.001
- **Weight decay defaults (ResNet-18, Table 2):**
  - 2-bit: `0.25e-4`
  - 3-bit: `0.5e-4`
  - 4/8-bit: `1e-4`

## Repository Layout

- `train_fp.py`: full-precision training
- `train.py`: LSQ quantized fine-tuning
- `eval.py`: checkpoint evaluation (FP or LSQ)
- `lsq/quant/lsq.py`: LSQ quantizer implementation (`gradscale`, `roundpass`, `quantize`)
- `lsq/models/preact_resnet.py`: pre-activation ResNet-18 + LSQ wrapping
- `lsq/data/imagenet.py`: ImageNet loaders and transforms
- `export_hf_imagenet.py`: export HF ImageNet cache to `ImageFolder` layout
- `split_dataset.py`: random class-folder train/val split helper

## Environment

```bash
python -m pip install -r requirements.txt
```

## Dataset Format

`--data-root` should contain:

```text
<data-root>/
  train/
    class_x/xxx.jpg
  val/
    class_x/yyy.jpg
```

## Data Preparation

### Option A: Export from Hugging Face cache

```bash
python export_hf_imagenet.py \
  --dataset-id ILSVRC/imagenet-1k \
  --cache-dir ./Dataset \
  --out-root ./data_imagenet1k
```

### Option B: Split an existing class-folder dataset

```bash
python split_dataset.py \
  --src /path/to/class_folder_dataset \
  --dst ./data \
  --val-ratio 0.2 \
  --seed 42
```

## Training Workflow

### 1) Train full-precision baseline

```bash
python train_fp.py \
  --data-root data_imagenet1k \
  --output-dir runs/fp_preact18
```

### 2) Fine-tune with LSQ (example: W4A4)

```bash
python train.py \
  --data-root data_imagenet1k \
  --fp-ckpt runs/fp_preact18/best.pth \
  --output-dir runs/lsq4_preact18 \
  --w-bits 4 \
  --a-bits 4 \
  --first-last-bits 8
```

## Evaluation

```bash
python eval.py \
  --data-root data_imagenet1k \
  --ckpt runs/lsq4_preact18/best.pth \
  --w-bits 4 \
  --a-bits 4
```

## Engineering Highlights (Portfolio)

- Implemented LSQ from paper pseudocode in modular PyTorch layers
- Built end-to-end reproducible training/evaluation CLI workflow
- Encoded paper-specific hyperparameter defaults directly in code
- Added dataset utility scripts for practical experiment setup

## Notes

- This repo focuses on faithful implementation and reproducible workflow.
- Final accuracy may vary with hardware, data pipeline details, and training budget.
