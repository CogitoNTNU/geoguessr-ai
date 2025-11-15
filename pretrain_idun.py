import io
import os
import random
import numpy as np
import torch
from typing import Dict
from loguru import logger
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    CLIPModel,
    CLIPProcessor,
    Trainer,
    TrainingArguments,
)
from dotenv import load_dotenv
import wandb
from config import (
    CLIP_MODEL,
    PRETRAIN_ARGS,  # TrainingArguments or kwargs dict
)
from backend.s3bucket import download_climate_file
from backend.metadata import sample_koppen, CLIMATE_DICT, geocell_mgr, MONTHS
from pretrain.leftdrive_countries import left_list
from training.load_sqlite_dataset import load_sqlite_dataset
from random import shuffle

THE_LIST = [
    "Bahamas",
    "British Virgin Islands",
    "Cayman Islands",
    "Cocos Islands",
    "Comoros",
    "Cook Islands",
    "Falkland Islands",
    "Faroe Islands",
    "French Southern Territories",
    "Maldives",
    "Marshall Islands",
    "Netherlands",
    "Northern Mariana Islands",
    "Paracel Islands",
    "Philippines",
    "Pitcairn Islands",
    "Seychelles",
    "Solomon Islands",
    "Spratly Islands",
    "Turks and Caicos Islands",
    "United Arab Emirates",
    "United States",
]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _is_valid(x) -> bool:
    return (
        (x is not None)
        and (not (isinstance(x, float) and pd.isna(x)))
        and (str(x).strip() != "")
    )


def select_caption(df: pd.DataFrame, index: int) -> str:
    """
    Caption using geo + climate + driving side + month info, with some randomness
    so CLIP sees multiple textual descriptions for similar images.
    """
    s = df.iloc[index]

    # ----- Basic fields (with fallbacks) -----
    lat = getattr(s, "lat", None)
    lon = getattr(s, "lon", None)
    capture_date = getattr(s, "capture_date", None)

    # From geocell / climate enrichment
    country = getattr(s, "country", None)
    region = getattr(s, "region", None)
    town = (
        getattr(s, "cell", None) if "town" in df.columns else None
    )  # only if you have it
    climate_zone = getattr(s, "climate_zone", None)
    driving_right = getattr(s, "drive_right")
    month = getattr(
        s, "month", None
    )  # should be something like "January" after mapping

    # ----- Country name normalization -----
    # if your pipeline sometimes uses "United States Of America"
    if country == "United States Of America":
        country = "United States"

    # Add "the" in front for a small set of countries / territories
    if _is_valid(country) and country in THE_LIST:
        country_str = f"the {country}"
    else:
        country_str = country

    # ----- Location string (country / region / town) -----
    if _is_valid(country_str):
        if _is_valid(region) and random.random() > 0.4:
            region_string = f"in the region of {region} "
        else:
            region_string = ""

        if _is_valid(town) and random.random() > 0.6:
            town_string = f"close to the town of {town} "
        else:
            town_string = ""

        location_caption = (
            f"A Street View photo {town_string}{region_string}in {country_str}."
        )
    else:
        # Fallback: use coordinates if we don't have country
        if _is_valid(lat) and _is_valid(lon):
            location_caption = (
                f"A Street View photo taken around latitude {float(lat):.3f}, "
                f"longitude {float(lon):.3f}."
            )
        else:
            location_caption = "A Street View photo."

    # ----- Climate caption -----
    if _is_valid(climate_zone) and random.random() > 0.6:
        climate_caption = f" This location has {str(climate_zone).lower()}."
    else:
        climate_caption = ""

    # ----- Driving side caption -----
    if (
        driving_right is not None
        and _is_valid(country_str)
        and climate_caption == ""
        and random.random() > 0.7
    ):
        direction = "right" if bool(driving_right) else "left"
        driving_caption = (
            f" In this location, people drive on the {direction} side of the road."
        )
    else:
        driving_caption = ""

    # ----- Month / season caption -----
    month_caption = ""
    if _is_valid(month) and random.random() > 0.7:
        # If df["month"] already holds the name ("January", etc.)
        month_caption = f" The photo was taken in {month}."
    elif _is_valid(capture_date) and random.random() > 0.7:
        # Parse month from capture_date string if you want
        try:
            month_code = str(capture_date)[5:7]
            month_name = MONTHS.get(month_code, month_code)
            month_caption = f" The photo was taken in {month_name}."
        except Exception:
            pass

    # ----- Combine pieces with some shuffling for variety -----
    other_components = [
        climate_caption,
        driving_caption,
        month_caption,
    ]
    shuffle(other_components)

    caption = (location_caption + "".join(other_components)).strip()
    return caption


class PretrainDataset(Dataset):
    """
    (image, text) dataset where `image` is read from SQLite BLOB bytes.
    Expects the DataFrame to have an 'image' column with JPEG bytes.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_bytes = row["image"]
        if img_bytes is None:
            raise ValueError("Row has no image bytes")

        # Decode JPEG from bytes
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Build text caption
        caption = select_caption(self.df, idx)

        return image, caption


def collate_fn(examples):
    images = [ex[0] for ex in examples]
    text = [ex[1] for ex in examples]

    inputs = clip_processor(
        images=images,
        text=text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs["return_loss"] = True
    return inputs


def freeze_backbone_keep_head(model: CLIPModel):
    """
    Freeze CLIP backbone and only keep:
      - visual_projection  (image head)
      - optionally text_projection (text head)
      - logit_scale        (temperature)
    trainable.
    """
    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze logit scale
    if hasattr(model, "logit_scale"):
        model.logit_scale.requires_grad = True

    # Unfreeze image projection head
    for name, p in model.named_parameters():
        if "visual_projection" in name:
            p.requires_grad = True


def pretrain(
    model_name: str,
    dataset_dict: Dict[str, PretrainDataset],
    train_args: TrainingArguments,
    resume: bool = False,
    output_dir: str = "./clip_pretrain_out",
) -> CLIPModel:
    """
    Pretrains CLIP model on (image, caption) pairs, only fine-tuning the head.
    """
    # 1) Load CLIP
    model = CLIPModel.from_pretrained(model_name)

    # 2) Freeze backbone, keep head trainable
    freeze_backbone_keep_head(model)

    # 2b) Watch model in W&B
    if getattr(wandb, "run", None) is not None:
        wandb.watch(model, log="all")

    # 2c) Ensure TrainingArguments report to W&B
    # `report_to` can be None, str, or list
    if train_args.report_to is None:
        train_args.report_to = ["wandb"]
    elif isinstance(train_args.report_to, str):
        if train_args.report_to != "wandb":
            train_args.report_to = [train_args.report_to, "wandb"]
    else:
        if "wandb" not in train_args.report_to:
            train_args.report_to = list(train_args.report_to) + ["wandb"]

    # 3) Trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["val"],
        data_collator=collate_fn,
    )

    # 4) Train
    if resume and train_args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)
    else:
        trainer.train()

    # 5) Save full CLIP
    os.makedirs(output_dir, exist_ok=True)
    full_out = os.path.join(output_dir, "clip_full_head_finetuned")
    model.save_pretrained(full_out)
    clip_processor.save_pretrained(full_out)

    # 6) Save only vision tower as CLIPVisionModel
    vision_out = os.path.join(output_dir, "clip_vision_head_finetuned")
    model.vision_model.save_pretrained(vision_out)

    logger.info(f"Saved full CLIP to: {full_out}")
    logger.info(f"Saved vision tower to: {vision_out}")
    return model


if __name__ == "__main__":
    set_seed()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
    repo_parent_dir = os.path.abspath(os.path.join(repo_root, ".."))
    candidates = []
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    for name in os.listdir(repo_parent_dir):
        if name.startswith("dataset_sqlite_") and name.endswith(".sqlite"):
            full = os.path.join(repo_parent_dir, name)
            try:
                mtime = os.path.getmtime(full)
            except Exception:
                continue
            candidates.append((full, mtime))
    if not candidates:
        raise FileNotFoundError(
            f"No local SQLite dataset found next to project root at '{repo_parent_dir}'. "
            f"Expected files named like 'dataset_sqlite_<run_ts=...>.sqlite'."
        )
    candidates.sort(key=lambda x: x[1], reverse=True)
    sqlite_path = candidates[0][0]
    logger.info(f"Using local SQLite dataset: {sqlite_path}")

    df = load_sqlite_dataset(sqlite_path)
    raster_path = download_climate_file()
    df["month_code"] = df["batch_date"].str[5:7]  # "01", "02", ...
    df["month"] = df["month_code"].map(MONTHS)  # "January", ...
    df["latitude"] = df["lat"]
    df["longitude"] = df["lon"]
    out = df.apply(lambda r: geocell_mgr.get_geocell_id(r), axis=1)
    df[["cell", "country", "region"]] = pd.DataFrame(out.tolist(), index=df.index)
    df = sample_koppen(df, raster_path, CLIMATE_DICT)
    df["drive_right"] = df["country"].apply(
        lambda c: (pd.notna(c)) and (c not in left_list)
    )

    # Simple train/val split
    frac = 0.9
    n_train = int(len(df) * frac)
    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_val = df.iloc[n_train:].reset_index(drop=True)

    ds_train = PretrainDataset(df_train)
    ds_val = PretrainDataset(df_val)
    dataset_dict = {"train": ds_train, "val": ds_val}

    if isinstance(PRETRAIN_ARGS, TrainingArguments):
        train_args = PRETRAIN_ARGS
    else:
        train_args = TrainingArguments(**PRETRAIN_ARGS)

        load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")
    try:
        if api_key:
            wandb.login(key=api_key)
        else:
            wandb.login()
    except Exception as e:
        logger.warning(f"W&B login failed, proceeding with W&B disabled: {e}")

    try:
        wandb.init(
            project="geoguessr-ai",
            config=train_args.to_dict()
            if isinstance(train_args, TrainingArguments)
            else PRETRAIN_ARGS,
            mode="online",
        )
    except Exception as e:
        logger.warning(f"W&B init failed, falling back to disabled mode: {e}")
        wandb.init(
            project="geoguessr-ai",
            config=train_args.to_dict()
            if isinstance(train_args, TrainingArguments)
            else PRETRAIN_ARGS,
            mode="disabled",
        )

    pretrain(
        model_name=CLIP_MODEL,
        dataset_dict=dataset_dict,
        train_args=train_args,
        resume=False,
        output_dir="./clip_pretrain_out_sqlite",
    )
