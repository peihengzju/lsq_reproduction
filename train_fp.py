from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from lsq.data import build_imagenet_loaders, infer_num_classes
from lsq.engine import run_training
from lsq.models import preact_resnet18


def parse_args():
    p = argparse.ArgumentParser("Full precision pre-activation ResNet-18 training (paper-aligned)")
    p.add_argument("--data-root", type=str, default="data_imagenet1k")
    p.add_argument("--output-dir", type=str, default="runs/fp_preact18")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)

    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.9)

    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_classes = args.num_classes if args.num_classes is not None else infer_num_classes(args.data_root)
    print(f"Using num_classes: {num_classes}")

    model = preact_resnet18(num_classes=num_classes).to(device)

    train_loader, val_loader = build_imagenet_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "run_args.txt", "w", encoding="utf-8") as f:
        run_args = vars(args).copy()
        run_args["num_classes"] = num_classes
        f.write(str(run_args))

    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        device=device,
        out_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
