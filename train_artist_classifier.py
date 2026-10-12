"""
Artist image classifier — tuned to reduce underfitting.

Why underfitting often happens on this dataset:
- training a tiny CNN from scratch
- freezing the whole backbone
- too few epochs / LR too small
- too many rare classes

This script counters that with:
- ImageNet-pretrained ResNet50 (more capacity + prior knowledge)
- full fine-tuning (backbone unfrozen)
- differential LR (head higher, backbone lower)
- enough epochs + ReduceLROnPlateau
- top artists only (min samples) so the model can actually fit
- light augmentation (heavy aug can look like underfitting)
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from tqdm import tqdm


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ArtDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, label_to_idx: dict[str, int], transform):
        self.paths = frame["image_path"].tolist()
        self.labels = [label_to_idx[a] for a in frame["artist_name"].tolist()]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


def build_model(num_classes: int, backbone: str = "resnet50", pretrained: bool = True) -> nn.Module:
    backbone = backbone.lower()
    if backbone == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        head_hidden = 256
    elif backbone == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        head_hidden = 512
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    # Slightly larger head than default Linear — helps capacity / underfitting
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, head_hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.1),
        nn.Linear(head_hidden, num_classes),
    )
    return model


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    n = len(loader.dataset)
    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / n, float(acc), float(macro_f1), all_labels, all_preds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="final.csv")
    parser.add_argument("--backbone", choices=["resnet18", "resnet50"], default="resnet50")
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = df[df["image_path"].map(os.path.exists)].copy()
    df = df[df["artist_name"].str.lower() != "unknown"].copy()

    counts = df["artist_name"].value_counts()
    keep = counts[counts >= args.min_samples].index
    df = df[df["artist_name"].isin(keep)].copy()

    labels = sorted(df["artist_name"].unique())
    label_to_idx = {a: i for i, a in enumerate(labels)}
    idx_to_label = {i: a for a, i in label_to_idx.items()}

    print(f"Device: {device}")
    print(f"Classes: {len(labels)} | Samples: {len(df)} | min_samples={args.min_samples}")

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=args.seed,
        stratify=df["artist_name"],
    )

    # Light aug — aggressive aug can look like underfitting on small art sets
    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_ds = ArtDataset(train_df, label_to_idx, train_tf)
    val_ds = ArtDataset(val_df, label_to_idx, val_tf)

    # Class-balanced sampling helps rare artists get learned (reduces underfit on minority classes)
    train_labels = train_ds.labels
    class_count = np.bincount(train_labels, minlength=len(labels))
    class_weight = 1.0 / np.maximum(class_count, 1)
    sample_weights = [class_weight[y] for y in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(len(labels), backbone=args.backbone, pretrained=True).to(device)

    # Differential LR: unfreeze all layers (anti-underfitting) but keep backbone slower
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    head_params = [p for n, p in model.named_parameters() if n.startswith("fc.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": head_params, "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    history = []
    best_f1 = -1.0
    best_path = outdir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, labels_batch in pbar:
            images = images.to(device)
            labels_batch = labels_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels_batch).sum().item()
            total += images.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{correct / total:.3f}")

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc, val_f1, y_true, y_pred = evaluate(
            model, val_loader, device, criterion
        )
        scheduler.step(val_f1)

        gap = train_acc - val_acc
        status = "ok"
        if train_acc < 0.55 and val_acc < 0.55:
            status = "underfitting?"
        elif gap > 0.20:
            status = "overfitting?"

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
            "train_val_gap": gap,
            "status_hint": status,
        }
        history.append(row)
        print(
            f"Epoch {epoch}: train_acc={train_acc:.3f} val_acc={val_acc:.3f} "
            f"val_f1={val_f1:.3f} gap={gap:.3f} [{status}]"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "label_to_idx": label_to_idx,
                    "idx_to_label": idx_to_label,
                    "args": vars(args),
                    "val_macro_f1": best_f1,
                    "val_acc": val_acc,
                },
                best_path,
            )

    # Final report from best checkpoint
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    _, val_acc, val_f1, y_true, y_pred = evaluate(model, val_loader, device, criterion)
    report = classification_report(
        y_true,
        y_pred,
        target_names=[idx_to_label[i] for i in range(len(idx_to_label))],
        zero_division=0,
    )
    print("\n=== Best model validation report ===")
    print(report)

    with open(outdir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(outdir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    with open(outdir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(label_to_idx, f, indent=2, ensure_ascii=False)

    print(f"\nSaved best checkpoint -> {best_path}")
    print(f"Best val macro-F1={best_f1:.3f} | val_acc={val_acc:.3f}")
    print(
        "Tip: if status stays 'underfitting?', raise --epochs, raise --lr-head, "
        "or lower --min-samples carefully."
    )


if __name__ == "__main__":
    main()
