import logging
import torch
from torchsummary import summary
from typing import Any
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForImageClassification,
    CLIPVisionModel,
    CLIPModel,
    CLIPProcessor,
)
import timm
from models import SuperGuessr, load_state_dict
from datasets import DatasetDict
from dataset_creation.pretrain import PretrainDataset
from evaluation.evaluate import compute_geoguessr_metrics
from training import train_model
from config import TRAIN_ARGS, PRETRAINED_CLIP, PRETAIN_ARGS, CLIP_MODEL

# Initialize Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train")

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


def finetune_model(
    model: Any,
    dataset: DatasetDict,
    early_stopping: int = None,
    train_args: TrainingArguments = TRAIN_ARGS,
) -> AutoModelForImageClassification:
    """Finetunes a given model.

    Args:
        model (Any): name of Huggingface model or trainable object.
        dataset (DatasetDict): dataset.
        early_stopping (int, optional): early stopping patience. Defaults to None.
        train_args (TrainingArguments, optional): training arguments. Defaults to DEFAULT_TRAIN_ARGS.

    Returns:
        AutoModel: finetuned model.
    """
    logger.warning(f"Downloading model: {model}.")

    # Loaded pre-loaded
    if type(model) is not str:
        loaded_model = model

    else:
        if "clip-vit" in model:
            loaded_model = CLIPVisionModel.from_pretrained(model)
            state_dict = torch.load(PRETRAINED_CLIP, map_location=torch.device("cuda"))
            load_state_dict(loaded_model, state_dict)
            print(f"Initialized base model with weights from: {PRETRAINED_CLIP}")
        elif "tiny" in model:
            loaded_model = timm.create_model(
                "tiny_vit_21m_512.dist_in22k_ft_in1k", pretrained=True, num_classes=0
            )
        else:
            raise Exception("Not a clip-vit or tiny-vit model.")

        model = SuperGuessr(
            loaded_model.base_model,
            panorama=True,
            hierarchical=False,
            freeze_base=False,
            should_smooth_labels=True,
        )

        print(loaded_model)

    loaded_model = train_model(
        model, dataset, False, train_args, compute_geoguessr_metrics, early_stopping
    )
    return loaded_model


def finetune_on_embeddings(
    dataset: DatasetDict,
    early_stopping: int = None,
    train_args: TrainingArguments = TRAIN_ARGS,
) -> AutoModelForImageClassification:
    """Finetunes a model on embeddings.

    Args:
        dataset (DatasetDict): dataset.
        early_stopping (int, optional): early stopping patience. Defaults to None.
        train_args (TrainingArguments, optional): training arguments. Defaults to DEFAULT_TRAIN_ARGS.

    Returns:
        AutoModel: finetuned model.
    """
    model = SuperGuessr(
        base_model=None,
        panorama=True,
        hierarchical=False,
        freeze_base=True,
        should_smooth_labels=True,
    )
    summary(model)
    print(model)

    loaded_model = train_model(
        model, dataset, True, train_args, compute_geoguessr_metrics, early_stopping
    )
    return loaded_model
