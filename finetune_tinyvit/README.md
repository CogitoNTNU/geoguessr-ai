# TinyViT finetuning (lightweight alternative to CLIP)

This folder provides a minimal pipeline to finetune TinyViT on your S3 Street View dataset and extract embeddings for downstream tasks.

- Model: timm tiny_vit_5m_224 (pretrained) – fits on modest GPUs/CPUs.
- Task: country classification derived from GADM polygons (can swap labels later).
- Outputs: best checkpoint and an embeddings Parquet for any CSV of images.

## Steps

1) Prepare the dataset (download + label + split)

```bash
python -m finetune_tinyvit.prepare_dataset --out_dir ./out --limit 2000 --val_frac 0.1 --manifest_dir finetune_tinyvit/manifests
```

2) Train TinyViT

```bash
python -m finetune_tinyvit.train_tinyvit_timm \
  --train_csv finetune_tinyvit/manifests/train.csv \
  --val_csv finetune_tinyvit/manifests/val.csv \
  --out_dir finetune_tinyvit/outputs \
  --epochs 5 --batch_size 64 --pretrained
```

3) Extract embeddings for further processing

```bash
python -m finetune_tinyvit.extract_embeddings \
  --csv finetune_tinyvit/manifests/all.csv \
  --ckpt finetune_tinyvit/outputs/best.pt \
  --out_parquet finetune_tinyvit/embeddings.parquet
```

## What is a collate function?

A collate function tells the DataLoader how to merge a list of samples into one batch. Here we:
- Apply transforms to each image.
- Stack images into a tensor (B, 3, H, W).
- Build a labels tensor and keep metadata (location_id, lat/lon) as a list.

This helps when samples are not simple (image, label) pairs or you need custom preprocessing.

## Metrics and evaluation

- We compute top-1 and top-5 accuracy on the validation set.
- For geographic tasks you can also compute median Haversine distance (km) between predicted class centroids and true coords if doing regression or geocell classification.

## Hugging Face notes

- `compute_metrics`: Return a dict with your scalar metrics, e.g., `{\"accuracy\": acc}`; for geolocation, you can return `{\"median_km\": value}`. With Transformers `Trainer`, `compute_metrics` takes `EvalPrediction(predictions, label_ids)`.
- `remove_unused_columns=False`: Recommended for vision tasks with custom datasets so Trainer doesn't drop columns your `collate_fn`/model rely on.

## MMPretrain (optional)

MMPretrain supports TinyViT via timm. If you prefer that framework:
- Export manifests to MMPretrain's JSON annotation format.
- Use a TinyViT config (backbone=TinyViT-5M, head=LinearClsHead, pretrained timm weights).
- To get embeddings, call `model.backbone.forward_features(x)` or set head to Identity during inference.

This repo's timm path is simpler and already supports embedding extraction.
