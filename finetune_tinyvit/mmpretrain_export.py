from __future__ import annotations

"""
Export CSV manifests to MMPretrain JSON annotation format.

Produces train.json and val.json with entries like:
  {"img_path": "/abs/path.jpg", "gt_label": 3}

Also writes label_map.json mapping class name -> id.
"""

import argparse
import json
from pathlib import Path
import pandas as pd


def export(csv_path: str, out_json: str, label_map: dict[str, int] | None = None) -> dict[str, int]:
    df = pd.read_csv(csv_path)
    if label_map is None:
        classes = sorted(df["country"].astype(str).unique())
        label_map = {c: i for i, c in enumerate(classes)}

    records = []
    for _, row in df.iterrows():
        c = str(row["country"])
        records.append({
            "img_path": str(Path(row["filepath"]).absolute()),
            "gt_label": int(label_map[c]),
        })
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(records, f)
    return label_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--val_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="finetune_tinyvit/mmpretrain")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_map = export(args.train_csv, str(out_dir / "train.json"))
    export(args.val_csv, str(out_dir / "val.json"), label_map=label_map)
    with open(out_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"Wrote MMPretrain annotations to {out_dir}")


if __name__ == "__main__":
    main()
