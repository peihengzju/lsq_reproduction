# LSQ Reproduction on ImageNet (Pre-Activation ResNet-18)

Implementation-focused reproduction of **Learned Step Size Quantization (LSQ)** from arXiv `1902.08153v3`.

- Paper copy in this repo: [`1902.08153v3.pdf`](./1902.08153v3.pdf)
- Core objective: reproduce LSQ training behavior with a clean, reusable PyTorch pipeline
- Scope: full-precision baseline + LSQ quantized fine-tuning on ImageNet-1k

## What Is Implemented

- Model: pre-activation ResNet-18
- Two-stage training:
  1. Full-precision (FP) baseline training
  2. LSQ fine-tuning from FP checkpoint
- Quantized modules: Conv/Linear weights and input activations
- First/last layer policy: optional higher precision (default 8-bit)
- Optimizer: SGD + momentum (0.9)
- LR schedule: cosine annealing

Paper-aligned defaults (ResNet-18):

- Epoch/LR by bit-width:
  - 2/3/4-bit: 90 epochs, LR=0.01
  - 8-bit: 1 epoch, LR=0.001
- Weight decay (Table 2):
  - 2-bit: `0.25e-4`
  - 3-bit: `0.5e-4`
  - 4/8-bit: `1e-4`

## Reproduction Results (ImageNet-1k)

The following are run results from this repo (March 2026):

| Setting | Epochs | Val Top-1 (%) | Val Top-5 (%) | Top-1 Drop vs FP |
|---|---:|---:|---:|---:|
| FP (baseline) | 90 | **69.67** | **89.04** | 0.00 |
| LSQ W8A8 (first/last 8-bit) | 1 | 68.52 | 88.50 | -1.15 |
| LSQ W4A4 (first/last 8-bit) | 90 | 68.64 | 88.28 | -1.03 |

Key takeaway: the 4-bit quantized model is within about **1.0 top-1** of FP in this setup.

## Repository Structure

- `train_fp.py`: full-precision training
- `train.py`: LSQ quantized fine-tuning
- `eval.py`: evaluation for FP/LSQ checkpoints
- `lsq/quant/lsq.py`: LSQ quantizer (`grad_scale`, `round_pass`, learnable step size)
- `lsq/models/preact_resnet.py`: pre-activation ResNet-18 + quantization wrapping
- `lsq/data/imagenet.py`: ImageNet data loading and transforms
- `export_hf_imagenet.py`: export HF ImageNet cache to `ImageFolder`
- `split_dataset.py`: split class-folder dataset into train/val

## Quick Start

### 1) Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2) Prepare dataset

Expected layout for `--data-root`:

```text
<data-root>/
  train/
    class_x/xxx.jpg
  val/
    class_x/yyy.jpg
```

Option A (Hugging Face cache export):

```bash
python export_hf_imagenet.py \
  --dataset-id ILSVRC/imagenet-1k \
  --cache-dir ./Dataset \
  --out-root ./data_imagenet1k
```

Option B (split existing class-folder dataset):

```bash
python split_dataset.py \
  --src /path/to/class_folder_dataset \
  --dst ./data \
  --val-ratio 0.2 \
  --seed 42
```

### 3) Run paper-style training recipe

```bash
# Set your data root first
DATA=data_imagenet1k

# FP baseline (90 epochs)
python train_fp.py \
  --data-root "$DATA" \
  --epochs 90 \
  --lr 0.1 \
  --momentum 0.9 \
  --weight-decay 1e-4 \
  --batch-size 256 \
  --num-workers 8 \
  --output-dir runs/fp_paper

# LSQ 8-bit (paper: 1 epoch)
python train.py \
  --data-root "$DATA" \
  --fp-ckpt runs/fp_paper/best.pth \
  --w-bits 8 \
  --a-bits 8 \
  --first-last-bits 8 \
  --epochs 1 \
  --lr 0.001 \
  --momentum 0.9 \
  --weight-decay 1e-4 \
  --batch-size 256 \
  --num-workers 8 \
  --output-dir runs/lsq8_paper

# LSQ 4-bit (paper: 90 epochs)
python train.py \
  --data-root "$DATA" \
  --fp-ckpt runs/fp_paper/best.pth \
  --w-bits 4 \
  --a-bits 4 \
  --first-last-bits 8 \
  --epochs 90 \
  --lr 0.01 \
  --momentum 0.9 \
  --weight-decay 1e-4 \
  --batch-size 256 \
  --num-workers 8 \
  --output-dir runs/lsq4_paper
```

### 4) Evaluate checkpoints

```bash
python eval.py \
  --data-root "$DATA" \
  --ckpt runs/lsq4_paper/best.pth \
  --w-bits 4 \
  --a-bits 4
```

## Notes

- This repository emphasizes implementation fidelity and workflow reproducibility.
- Final numbers can vary with hardware, preprocessing, augmentation details, and random seed.
