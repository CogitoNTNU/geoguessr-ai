import argparse
import os
from typing import List

import torch
import pandas as pd
from PIL import Image
from torchvision import transforms as T

from backend.s3bucket import (
    download_latest_model_checkpoint,
    download_random_panorama,
)
from config import CLIP_MODEL, TINYVIT_MODEL
from models.super_guessr import SuperGuessr
from models.tinyvit import TinyViTAdapter
from models.proto_refiner import ProtoRefiner


_BASE_DIR = os.path.dirname(__file__)
TINYVIT_DEFAULT_CHECKPOINT = os.path.join(
    _BASE_DIR,
    "inference",
    "checkpoints",
    "tinyvit_panorama_1",
    "epoch_0006_5.110418.pt",
)

_GEOCELL_META_CACHE: dict[int, tuple[str, str]] | None = None


def _load_geocell_metadata() -> dict[int, tuple[str, str]]:
    """Map geocell_index -> (country, admin1) from proto_df.csv."""
    global _GEOCELL_META_CACHE
    if _GEOCELL_META_CACHE is not None:
        return _GEOCELL_META_CACHE

    proto_path = os.path.join(_BASE_DIR, "data", "geocells", "proto_df.csv")
    df = pd.read_csv(proto_path, usecols=["geocell_index", "country", "admin1"])
    df = df.drop_duplicates(subset=["geocell_index"], keep="first")
    mapping: dict[int, tuple[str, str]] = {}
    for _, row in df.iterrows():
        mapping[int(row["geocell_index"])] = (str(row["country"]), str(row["admin1"]))
    _GEOCELL_META_CACHE = mapping
    return mapping


def _load_backbone(name: str):
    if name == "clip":
        from transformers import CLIPVisionModel

        dest = os.path.join("checkpoints", "clip_backbone_s3")
        try:
            ckpt_dir = download_latest_model_checkpoint(dest)
        except FileNotFoundError:
            ckpt_dir = CLIP_MODEL
        return CLIPVisionModel.from_pretrained(ckpt_dir)
    else:
        # For TinyViT we rely on the SuperGuessr checkpoint for weights.
        # Do NOT load the timm ImageNet-pretrained TinyViT weights here.
        return TinyViTAdapter(model_name=TINYVIT_MODEL, pretrained=False)


def _build_transform(name: str):
    if name == "clip":
        size = 336
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
    else:
        size = 512
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    return T.Compose(
        [T.Resize(size), T.CenterCrop(size), T.ToTensor(), T.Normalize(mean, std)]
    )


def _load_images(paths: List[str], transform, device):
    imgs = [transform(Image.open(p).convert("RGB")) for p in paths]
    x = torch.stack(imgs, dim=0)
    if len(paths) == 4:
        x = x.unsqueeze(0)  # (1, 4, C, H, W)
    return x.to(device)


def run_inference(
    image_paths: List[str],
    backbone: str = "tinyvit",
    use_refiner: bool = False,
    checkpoint: str | None = None,
    device: str | None = None,
) -> tuple[float, float, list[int], list[float]]:
    if len(image_paths) not in (1, 4):
        raise ValueError("Inference expects either 1 image or 4 images as input.")

    panorama = len(image_paths) == 4
    device = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # If TinyViT is selected and no checkpoint was provided, fall back to the
    # trained TinyViT panorama checkpoint if it exists.
    if (
        backbone == "tinyvit"
        and not checkpoint
        and os.path.isfile(TINYVIT_DEFAULT_CHECKPOINT)
    ):
        checkpoint = TINYVIT_DEFAULT_CHECKPOINT

    backbone_model = _load_backbone(backbone).to(device)
    model = SuperGuessr(
        base_model=backbone_model,
        panorama=panorama,
        serving=True,
        should_smooth_labels=False,
    ).to(device)

    if checkpoint:
        raw_state = torch.load(checkpoint, map_location=device)
        # Unwrap common training format {"model_state_dict": ..., ...}
        state_dict = (
            raw_state.get("model_state_dict", raw_state)
            if isinstance(raw_state, dict)
            else raw_state
        )
        model_state = model.state_dict()
        filtered_state = {}
        for name, param in state_dict.items():
            if name not in model_state:
                print("skip")
                continue
            if model_state[name].shape != param.shape:
                print("skip")
                continue
            filtered_state[name] = param
        print(f"Loaded {len(filtered_state)}/{len(model_state)} parameters from checkpoint.")
        model.load_state_dict(filtered_state, strict=False)

    model.eval()
    transform = _build_transform(backbone)
    pixel_values = _load_images(image_paths, transform, device)

    with torch.no_grad():
        # SuperGuessr's forward currently expects a non-None labels_clf, even
        # in serving mode, so pass a dummy tensor that is ignored in eval.
        dummy_labels = torch.zeros(1, dtype=torch.long, device=device)
        if not panorama and pixel_values.dim() == 5:
            pixel_values = pixel_values.squeeze(1)
        pred_llh, topk, embedding = model(
            pixel_values=pixel_values, labels_clf=dummy_labels
        )

    # Extract top-k geocell indices and probabilities from SuperGuessr output
    top_indices = topk.indices[0].detach().cpu().tolist()
    top_probs = topk.values[0].detach().cpu().tolist()

    if use_refiner:
        refiner = ProtoRefiner(topk=topk.indices.size(1), protos="load").to(device)
        refiner.eval()
        with torch.no_grad():
            _, refined_llh, _ = refiner(
                embedding=embedding,
                initial_preds=pred_llh,
                candidate_cells=topk.indices,
                candidate_probs=topk.values,
            )
        lon, lat = refined_llh[0].tolist()
    else:
        lon, lat = pred_llh[0].tolist()

    # Coordinates are (lon, lat); return (lat, lon) plus top-k geocell predictions
    return lat, lon, top_indices, top_probs


def main():
    parser = argparse.ArgumentParser(
        description="Run SuperGuessr inference on 1 or 4 images."
    )
    parser.add_argument(
        "images",
        nargs="*",
        help=(
            "Path(s) to input image(s). Use 1 or 4 paths. "
            "If omitted, uses the bundled test image."
        ),
    )
    parser.add_argument(
        "--backbone",
        choices=["clip", "tinyvit"],
        default="tinyvit",
        help="Vision backbone to use (default: tinyvit with trained panorama checkpoint).",
    )
    parser.add_argument(
        "--useRefiner",
        action="store_true",
        help="If set, refine SuperGuessr candidates with ProtoRefiner.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Optional path to a SuperGuessr checkpoint (.pt). "
            "If it contains a 'model_state_dict' key, that sub-dict is used."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string, e.g. 'cuda' or 'cpu'. Defaults to CUDA if available.",
    )
    args = parser.parse_args()

    # Optional ground-truth coordinate (only available when sampling from training set)
    gt_lat: float | None = None
    gt_lon: float | None = None

    # If no images were provided, sample a random panorama from the training snapshot.
    if not args.images:
        pano = download_random_panorama(
            dest_dir=os.path.join(_BASE_DIR, "inference", "tmp_panorama"),
            overwrite=False,
        )
        args.images = pano["paths"]
        gt_lat = float(pano["lat"])
        gt_lon = float(pano["lon"])

    lat, lon, top_ids, top_probs = run_inference(
        image_paths=args.images,
        backbone=args.backbone,
        use_refiner=False,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    print(f"The model guessed: {lat:.6f},{lon:.6f}")
    if gt_lat is not None and gt_lon is not None:
        print(f"The ground truth is: {gt_lat:.6f},{gt_lon:.6f}")

        # Clean up downloaded panorama images after use
        for p in args.images:
            try:
                os.remove(p)
            except OSError:
                pass

    meta = _load_geocell_metadata()
    print("Top geocells (id, probability, country, admin1):")
    for gid, prob in zip(top_ids, top_probs):
        country, admin1 = meta.get(gid, ("?", "?"))
        print(f"{gid}\t{prob:.4f}\t{country}\t{admin1}")


if __name__ == "__main__":
    main()
