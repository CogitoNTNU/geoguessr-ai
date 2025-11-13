import random
import torch
import numpy as np
import pandas as pd
from random import shuffle
from PIL import Image
from typing import Tuple, Any
from datasets import DatasetDict
from transformers import Trainer, TrainingArguments, CLIPModel, CLIPProcessor
from torchvision.transforms import RandomCrop
from config import (
    CLIP_MODEL,
    PRETRAIN_METADATA_PATH,
    PRETAIN_ARGS,
)
from backend.s3bucket import load_climate_file, load_latest_snapshot_df
from backend.metadata import sample_koppen, CLIMATE_DICT, geocell_mgr, MONTHS
from pretrain.leftdrive_countries import left_list

# Initialize CLIP image processor
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

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


class PretrainDataset(torch.utils.data.Dataset):
    """Dataset used for pretraining CLIP with geolocalization data."""

    def __init__(self, split: str, df: pd.DataFrame):
        """Initializes a PretrainDataset used for pretraining CLIP.

        Args:
            split (str): dataset split to load.
            metadata (str, optional): path to metadata csv file containing image filepaths
                and auxiliary information. Defaults to METADATA_PATH.
            auxiliary (bool, optional): whether to use auxiliary information for pretraining.
                Defaults to True.
        """
        assert split in ["train", "val", "test"]
        self.df = self.get_metadata(df)
        self.df = self.df.reset_index(drop=True)

    def get_metadata(df):
        raster_path = load_climate_file()
        df = load_latest_snapshot_df()
        df["month"] = df["batch_date"].str[5:7]
        df["latitude"] = df["lat"]
        df["longitude"] = df["lon"]
        out = df.apply(lambda r: geocell_mgr.get_geocell_id(r), axis=1)
        df[["cell", "country", "region"]] = pd.DataFrame(out.tolist(), index=df.index)
        df = sample_koppen(df, raster_path, CLIMATE_DICT)
        df["drive_right"] = df["country"].apply(
            lambda c: (pd.notna(c)) and (c not in left_list)
        )

    def _is_valid(self, value: Any):
        """Checks whether the provided value is valid.

        Args:
            value (Any): any value.
        """
        return type(value) is str or not np.isnan(value)

    def _select_caption(self, index: int, heading_offset: int) -> str:
        """Generates a random caption for the given image using auxiliary data.

        Args:
            index (int): row index to generate caption for.
            heading_offset (int): heading offset in angles from north.

        Returns:
            str: randomly generated caption.
        """
        s = self.df.iloc[index]
        # Country, region, town
        country = s.country_name
        if country == "United States Of America":
            country = "United States"

        country = country if country not in THE_LIST else f"the {country}"

        if self._is_valid(s.geo_area) and random.random() > 0.4:
            region_string = f"in the region of {s.geo_area} "
        else:
            region_string = ""

        if self._is_valid(s.town) and random.random() > 0.6:
            town_string = f"close to the town of {s.town} "
        else:
            town_string = ""

        # Climate zone
        if self._is_valid(s.climate_zone) and random.random() > 0.6:
            climate_caption = f" This location has {s.climate_zone.lower()}."
        else:
            climate_caption = ""

        # Location
        if random.random() > 0.3 or climate_caption == "" or not self.auxiliary:
            location_caption = (
                f"A Street View photo {town_string}{region_string}in {country}."
            )
            if not self.auxiliary:
                return location_caption
        else:
            location_caption = ""

        # Driving right or left
        if (
            self._is_valid(s.driving_right)
            and climate_caption == ""
            and random.random() > 0.7
        ):
            direction = "right" if s.driving_right else "left"
            driving_right_caption = (
                f" In this location, people drive on the {direction} side of the road."
            )
        else:
            driving_right_caption = ""

        # Month (because of seasons)
        if self._is_valid(s.month) and random.random() > 0.7:
            month_caption = f" The photo was taken in {MONTHS[s.month]}."
        else:
            month_caption = ""

        other_components = [
            climate_caption,
            driving_right_caption,
            month_caption,
        ]
        shuffle(other_components)
        components = [location_caption] + other_components
        caption = "".join(components).strip()
        return caption

    def _random_transform(self, image: Image) -> Image:
        """Randomly transforms the image on data load.

        Args:
            image (Image): image.

        Returns:
            Image: transformed image.
        """
        side_length, _ = image.size
        cropped_length = random.uniform(0.8, 1) * side_length
        cropper = RandomCrop(cropped_length)
        return cropper(image)

    def __getitem__(self, index: int) -> Tuple:
        """Retrieves item in dataset for given index.

        Args:
            index (int): sample index.

        Returns:
            Dict: sample model input
        """
        row_index, image_col = self._convert_to_row_index(index)

        # Randomly select one of the four images
        image, heading_offset = self._select_image(row_index, image_col)
        caption = self._select_caption(row_index, heading_offset)
        return image, caption

    def __len__(self):
        return self.cutoff_3

    @classmethod
    def generate(
        cls, metadata: str = PRETRAIN_METADATA_PATH, auxiliary: bool = True
    ) -> DatasetDict:
        """Generates a DatasetDict with PretrainedDatasets.

        Args:
            split (str): dataset split to load.
            metadata (str, optional): path to metadata csv file containing image filepaths
                and auxiliary information. Defaults to METADATA_PATH.
            auxiliary (bool, optional): whether to use auxiliary information for pretraining.
                Defaults to True.

        Returns:
            DatasetDict: dataset dictionary from train, val, and test.
        """
        return DatasetDict(
            train=cls("train", metadata, auxiliary),
            val=cls("val", metadata, auxiliary),
            test=cls("test", metadata, auxiliary),
        )

    def accuracy(self, model: Any, batch_size: int, trials: int = 30) -> float:
        """Computes the accuracy of a given mode on the current dataset.

        Args:
            model (Any): pretrained CLIP model.
            batch_size (int): batch size of model
            trials (int, optional): Number of runs for the Monte-Carlo estimation
                of accuracy. Defaults to 30.

        Returns:
            float: accuracys
        """
        accs = []
        for t in range(trials):
            inputs = [self[(t * batch_size) + i] for i in range(batch_size)]
            images, captions, _ = zip(*inputs)
            images = list(images)
            captions = list(captions)

            inputs = clip_processor(
                text=captions,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            for key in inputs:
                inputs[key] = inputs[key].to("cuda")

            inputs["return_loss"] = True
            outputs = model(**inputs)
            predictions = outputs.logits_per_image.softmax(dim=1).argmax(dim=1)
            accuracy = (predictions == torch.arange(batch_size, device="cuda")).sum()
            accs.append(accuracy / batch_size)

        acc = sum(accs) / trials
        return acc


# Initialize CLIP image processor
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)


def collate_fn(examples):
    images = [example[0] for example in examples]
    text = [example[1] for example in examples]
    inputs = clip_processor(
        images=images, text=text, return_tensors="pt", padding=True, truncation=True
    )
    inputs["return_loss"] = True
    return inputs


def pretrain(
    model: str,
    dataset: PretrainDataset,
    train_args: TrainingArguments = PRETAIN_ARGS,
    resume: bool = False,
) -> CLIPModel:
    """Pretrains a CLIP model on the given dataset.

    Args:
        model (str): Name of Huggingface model or trainable object.
        dataset (PretrainDataset): Dataset to be used for contrasrive pretraining.
        train_args (TrainingArguments, optional): Pretraining arguments. Defaults to PRETAIN_ARGS.
        resume (bool, optional): Whether to resume model training from checkpoint.

    Returns:
        CLIPModel: Pretrained CLIP model.
    """
    model = CLIPModel.from_pretrained(model)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        data_collator=collate_fn,
    )

    # Before training
    before_acc = dataset["val"].accuracy(model, batch_size=16)
    print("Before traing: Accuracy on batch size of 16 is", before_acc)

    # Train
    if resume:
        print("Resuming training from checkpoint ...")
        trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)

    # After training
    after_acc = dataset["val"].accuracy(model, batch_size=16)
    print("After training: Accuracy on batch size of 16 is", after_acc)
