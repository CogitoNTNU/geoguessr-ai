"""
Fine-tune TinyViT (timm) on the prepared country-labeled dataset.

Key points:
- Uses timm tiny_vit_5m_224 pretrain by default.
- Simple PyTorch loop with mixed precision and cosine schedule.
- Collate function stacks images and labels, keeps optional metadata.
- Computes top-1 (and top-5 if applicable) accuracy for validation.

Run:
  python -m finetune_tinyvit.train_tinyvit_timm --train_csv finetune_tinyvit/manifests/train.csv --val_csv finetune_tinyvit/manifests/val.csv --out_dir finetune_tinyvit/outputs
"""

from __future__ import annotations
import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from timm.data import create_transform
from timm.utils import accuracy
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class ImageCSVDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        class_map: Dict[str, int] | None = None,
        img_size: int = 224,
    ):
        self.df = pd.read_csv(csv_path)
        self.img_size = img_size

        # build class map if not provided
        classes = sorted(self.df["country"].astype(str).unique())
        if class_map is None:
            self.class_to_id = {c: i for i, c in enumerate(classes)}
        else:
            self.class_to_id = class_map
        self.id_to_class = {v: k for k, v in self.class_to_id.items()}

        # timm transform
        self.transform = create_transform(
            input_size=(3, img_size, img_size),
            is_training=True,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            auto_augment="rand-m9-mstd0.5-inc1",
            interpolation="bicubic",
        )

        self.eval_transform = create_transform(
            input_size=(3, img_size, img_size),
            is_training=False,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            interpolation="bicubic",
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row["filepath"]
        label_name = str(row["country"])  # e.g., country string
        y = self.class_to_id.get(label_name, self.class_to_id.setdefault("UNKNOWN", 0))

        with Image.open(path) as img:
            img = img.convert("RGB")
            # training/eval transform will be selected by DataLoader via attribute
            x = img

        meta = {
            "location_id": row.get("location_id"),
            "lat": row.get("lat"),
            "lon": row.get("lon"),
            "country": label_name,
        }
        return x, int(y), meta


@dataclass
class CollateWithTransform:
    transform_train: object
    transform_eval: object
    is_train: bool = True

    def __call__(self, batch):
        images, labels, metas = [], [], []
        tfm = self.transform_train if self.is_train else self.transform_eval
        for img, y, m in batch:
            images.append(tfm(img))
            labels.append(y)
            metas.append(m)
        images = torch.stack(images, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)
        return {"pixel_values": images, "labels": labels, "metas": metas}


def build_dataloaders(
    train_csv: str, val_csv: str, img_size: int, batch_size: int, num_workers: int
):
    train_ds = ImageCSVDataset(train_csv, img_size=img_size)
    # share class map with val
    val_ds = ImageCSVDataset(val_csv, class_map=train_ds.class_to_id, img_size=img_size)

    collate_train = CollateWithTransform(
        train_ds.transform, train_ds.eval_transform, is_train=True
    )
    collate_val = CollateWithTransform(
        train_ds.transform, train_ds.eval_transform, is_train=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_train,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_val,
    )
    return train_loader, val_loader, train_ds


def create_model(
    num_classes: int, model_name: str = "tiny_vit_5m_224", pretrained: bool = True
):
    model = timm.create_model(
        model_name, pretrained=pretrained, num_classes=num_classes
    )
    return model


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Dict[str, float]:
    model.eval()
    top1_meter = 0.0
    top5_meter = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["pixel_values"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            logits = model(x)
            bsz = y.size(0)
            acc1, acc5 = accuracy(logits, y, topk=(1, min(5, logits.size(1))))
            top1_meter += acc1.item() * bsz
            top5_meter += acc5.item() * bsz
            n += bsz
    return {"val_top1": top1_meter / max(1, n), "val_top5": top5_meter / max(1, n)}


def train():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--val_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="finetune_tinyvit/outputs")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--model_name", type=str, default="tiny_vit_5m_224")
    ap.add_argument("--pretrained", action="store_true", help="Use pretrained weights")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, train_ds = build_dataloaders(
        args.train_csv, args.val_csv, args.img_size, args.batch_size, args.num_workers
    )
    num_classes = len(train_ds.class_to_id)
    model = create_model(num_classes, args.model_name, pretrained=args.pretrained).to(
        device
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # Cosine schedule
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    loss_fn = nn.CrossEntropyLoss()

    best_top1 = -1.0
    best_ckpt = Path(args.out_dir) / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch in train_loader:
            x = batch["pixel_values"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * y.size(0)
            seen += y.size(0)

        scheduler.step()

        metrics = evaluate(model, val_loader, device)
        avg_loss = running_loss / max(1, seen)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | loss {avg_loss:.4f} | val@1 {metrics['val_top1']:.3f} | val@5 {metrics['val_top5']:.3f}"
        )

        if metrics["val_top1"] > best_top1:
            best_top1 = metrics["val_top1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "class_to_id": train_ds.class_to_id,
                    "args": vars(args),
                },
                best_ckpt,
            )
            print(f"Saved best checkpoint to {best_ckpt}")


if __name__ == "__main__":
    train()
