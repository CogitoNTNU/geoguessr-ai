from __future__ import annotations

"""
Extract TinyViT embeddings for images listed in a CSV manifest.

Loads the best checkpoint from train_tinyvit_timm.py and runs forward_features
to obtain per-image embeddings. Saves to Parquet with columns:
  location_id, filepath, lat, lon, country, emb_0..emb_{D-1}
"""

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd
import timm
import torch
from timm.data import create_transform
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class ImageCSVNoLabel(Dataset):
    def __init__(self, csv_path: str, img_size: int = 224):
        self.df = pd.read_csv(csv_path)
        self.transform = create_transform(
            input_size=(3, img_size, img_size),
            is_training=False,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            interpolation='bicubic',
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        with Image.open(row["filepath"]) as img:
            img = img.convert("RGB")
            x = self.transform(img)
        meta = {
            "location_id": row.get("location_id"),
            "lat": row.get("lat"),
            "lon": row.get("lon"),
            "country": row.get("country"),
            "filepath": row.get("filepath"),
        }
        return x, meta


def load_model_for_features(ckpt_path: str, model_name: str = "tiny_vit_5m_224"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    class_to_id: Dict[str, int] = ckpt.get("class_to_id", {})
    num_classes = len(class_to_id) if class_to_id else 1
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model, class_to_id


@torch.no_grad()
def forward_to_emb(model, x: torch.Tensor) -> torch.Tensor:
    # Prefer model.forward_features if available
    if hasattr(model, "forward_features"):
        feats = model.forward_features(x)
        # Some models return (B, C) after pooling; some return dicts
        if isinstance(feats, dict):
            # common key names
            for k in ["x", "feat", "features", "pool"]:
                if k in feats:
                    feats = feats[k]
                    break
        # if still 4D (B, C, H, W), pool
        if feats.ndim == 4:
            feats = torch.nn.functional.adaptive_avg_pool2d(feats, 1).flatten(1)
        return feats
    # Fallback: remove classifier by taking everything except last layer
    h = model.global_pool(model.forward_features(x)) if hasattr(model, "global_pool") else model(x)
    if h.ndim > 2:
        h = torch.nn.functional.adaptive_avg_pool2d(h, 1).flatten(1)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="CSV with image rows (from prepare_dataset)")
    ap.add_argument("--ckpt", type=str, required=True, help="Trained checkpoint .pt file")
    ap.add_argument("--out_parquet", type=str, default="finetune_tinyvit/embeddings.parquet")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--model_name", type=str, default="tiny_vit_5m_224")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = ImageCSVNoLabel(args.csv, img_size=args.img_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model, _ = load_model_for_features(args.ckpt, model_name=args.model_name)
    model = model.to(device)

    all_rows = []
    for x, metas in loader:
        x = x.to(device, non_blocking=True)
        feats = forward_to_emb(model, x).cpu().numpy()
        for i in range(feats.shape[0]):
            m = {k: (metas[k][i] if isinstance(metas[k], list) else metas[k]) for k in metas}
            row = {**m}
            for j, val in enumerate(feats[i].tolist()):
                row[f"emb_{j}"] = val
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_parquet, index=False)
    print(f"Wrote embeddings to {args.out_parquet} with shape {df.shape}")


if __name__ == "__main__":
    main()
