from dataclasses import dataclass, asdict
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import Module
import torch.nn.functional as F
from tqdm import tqdm
from backend.data import GeoImageIterableDataset
from backend.s3bucket import load_latest_snapshot_df, load_latest_holdout_snapshot_df
from loguru import logger
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from data.geocells.geocell_manager import GeocellManager
from dotenv import load_dotenv
import wandb
from models.super_guessr import SuperGuessr
from models.tinyvit import TinyViTAdapter
from transformers import CLIPVisionModel
from config import CLIP_MODEL, TINYVIT_MODEL
from models.utils import haversine_matrix


def main(config):
    # Overall-modus for å velge mellom training og inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fetch main dataset from s3-bucket
    df = load_latest_snapshot_df()

    train_test_split = 0.9
    num_training_samples = int(len(df) * train_test_split)
    df_train = df.iloc[:num_training_samples]
    df_test = df.iloc[num_training_samples:]

    # Fetch holdout dataset (validation set) from s3-bucket
    df_val = load_latest_holdout_snapshot_df()

    train_dataset = GeoImageIterableDataset(df_train)
    test_dataset = GeoImageIterableDataset(df_test)
    val_dataset = GeoImageIterableDataset(df_val)

    logger.info(
        f"Dataset loaded with {len(train_dataset)} training samples, {len(test_dataset)} test samples, {len(val_dataset)} validation samples"
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=64, num_workers=4, pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=64, num_workers=4, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=64, num_workers=4, pin_memory=True
    )

    # Initialize model and set it to train
    geocell_manager = GeocellManager("data/geocells/finished_geocells")
    num_geocells = geocell_manager.get_num_geocells()

    embeddingModelUsed = "TINYVIT"  # Possible values are "CLIP" or "TINYVIT"

    embedding_model = 0
    if embeddingModelUsed == "CLIP":
        embedding_model = CLIPVisionModel.from_pretrained(CLIP_MODEL)
    elif embeddingModelUsed == "TINYVIT":
        embedding_model = TinyViTAdapter(model_name=TINYVIT_MODEL, pretrained=True)

    # Select target input resolution per embedding model
    if embeddingModelUsed == "CLIP":
        target_dimensions = (336, 336)
        norm_mean = (0.48145466, 0.4578275, 0.40821073)
        norm_std = (0.26862954, 0.26130258, 0.27577711)
    elif embeddingModelUsed == "TINYVIT":
        target_dimensions = (512, 512)
        norm_mean = (0.485, 0.456, 0.406)
        norm_std = (0.229, 0.224, 0.225)
    else:
        target_dimensions = None
        norm_mean = None
        norm_std = None

    # Instantiate SuperGuessr with CLIP embedding model
    model = SuperGuessr(
        base_model=embedding_model,
        panorama=False,
        serving=False,
        should_smooth_labels=True,
    ).to(device)
    wandb.watch(model, log="all")
    model.train()

    train(
        model,
        train_dataloader,
        val_dataloader,
        device,
        config,
        target_dimensions=target_dimensions,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )


@dataclass
class Configuration:
    # Optimizer
    betas: tuple[float] = (0.9, 0.999)
    lr: float = 5e-5
    weight_decay: float = 0.01
    epochs: int = 5000
    # Scheduler
    T_0: int = 10
    T_mult: int = 2
    eta_min: int = 1e-6


def train(
    model: Module,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    device,
    config: Configuration,
    target_dimensions=None,
    norm_mean=None,
    norm_std=None,
):
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, config.T_0, config.T_mult, config.eta_min
    )

    # Prepare geocell centroids from model (ordered by proto_df geocell_index)
    centroids = model.geocell_centroid_coords.to(device)  # (num_cells, 2) in (lng, lat)

    global_step = 0
    for epoch in range(config.epochs):
        model.train()
        running_loss, running_top1, running_topk = 0.0, 0.0, 0.0
        num_batches = 0

        for batch_idx, (images, targets) in enumerate(
            tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config.epochs}")
        ):
            # Resize images to match embedding model input resolution (CLIP or Tiny-ViT)
            if target_dimensions is not None:
                if images.dim() == 5:  # (B, V, C, H, W) panorama batches
                    b, v, c, h, w = images.shape
                    images = images.view(b * v, c, h, w)
                    images = F.interpolate(
                        images,
                        size=target_dimensions,
                        mode="bilinear",
                        align_corners=False,
                    )
                    images = images.view(
                        b, v, c, target_dimensions[0], target_dimensions[1]
                    )
                elif images.dim() == 4:  # (B, C, H, W)
                    images = F.interpolate(
                        images,
                        size=target_dimensions,
                        mode="bilinear",
                        align_corners=False,
                    )

            images = images.to(device, non_blocking=True)

            # Normalize input
            if images.dtype == torch.uint8:
                images = images.float().div_(255.0)
            # Normalize per embedding model if provided
            if norm_mean is not None and norm_std is not None:
                if images.dim() == 5:
                    mean_t = torch.tensor(norm_mean, device=images.device).view(
                        1, 1, 3, 1, 1
                    )
                    std_t = torch.tensor(norm_std, device=images.device).view(
                        1, 1, 3, 1, 1
                    )
                else:
                    mean_t = torch.tensor(norm_mean, device=images.device).view(
                        1, 3, 1, 1
                    )
                    std_t = torch.tensor(norm_std, device=images.device).view(
                        1, 3, 1, 1
                    )
                images = (images - mean_t) / std_t

            # Build coordinate labels and derive class indices by nearest centroid (proto_df ordering)
            lat = targets["lat"]
            lon = targets["lon"]
            lat_t = torch.as_tensor(lat, dtype=torch.float32, device=device)
            lon_t = torch.as_tensor(lon, dtype=torch.float32, device=device)
            coord_labels = torch.stack([lon_t, lat_t], dim=1)  # (B, 2) in (lng, lat)
            # distances: (B, num_cells), centroids.t(): (2, num_cells)
            distances = haversine_matrix(coord_labels, centroids.t())
            targets = torch.argmin(distances, dim=-1).to(device, dtype=torch.long)

            optimizer.zero_grad()
            output = model(pixel_values=images, labels_clf=targets, labels=coord_labels)
            loss = output.loss
            geocell_topk = output.top5_geocells

            # Compute metrics
            with torch.no_grad():
                top1_pred = geocell_topk.indices[:, 0]
                top1_acc = (top1_pred == targets).float().mean().item()
                topk_acc = (
                    (geocell_topk.indices == targets.unsqueeze(1))
                    .any(dim=1)
                    .float()
                    .mean()
                    .item()
                )

            # Log per batch
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/top1_acc": top1_acc,
                    "train/top5_acc": topk_acc,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                },
                step=global_step,
            )

            # Backward + optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_top1 += top1_acc
            running_topk += topk_acc
            num_batches += 1
            global_step += 1

            tqdm.write(
                f"[Epoch {epoch + 1} | Batch {batch_idx}] Loss={loss.item():.4f} "
                f"Top1={top1_acc:.3f} Top5={topk_acc:.3f}"
            )

        # Scheduler step at the end of each epoch
        scheduler.step(epoch)

        # Aggregate training metrics per epoch
        epoch_loss = running_loss / num_batches
        epoch_top1 = running_top1 / num_batches
        epoch_topk = running_topk / num_batches

        # Log validation metrics
        wandb.log(
            {
                "epoch": epoch,
                "train/epoch_loss": epoch_loss,
                "train/epoch_top1": epoch_top1,
                "train/epoch_top5": epoch_topk,
            },
            step=global_step,
        )

        """
        * Hente 4 bilder av gangen
        * Lage variabel for lat, long hentet fra batch_metadata
        * Kalle på modellen og mate de 4 bildene inn her, få ut loss og topk geocells (geocell_topk variabel)
        * batch_images = batch_images.to(device)  # [B, 3, H, W]
        * For hver batch lager vi en liste med alle (lat, long)-par, altså fasit-koordinatene.
        * Vi lager så en liste med geocell_id-ene (eller noe annet, f.eks. centroiden) til geocellene disse koordinatene faller inni, kall den labels_ground_truth
        * Vi må kjøre optimizer.zero_grad(set_to_none=True) for å fjerne tidligere gradient-buffers
        * Vi kjører forward-pass gjennom modellen og setter labels_clf=labels_ground_truth, og får ut en loss.
        * Vi kjører loss.backward() for å oppdatere vektene til modellen
        * Optimizer.step()
        * Repeat
        """


if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")
    config = Configuration()

    wandb.login(key=api_key)
    wandb.init(
        project="geoguessr-ai",  # Your project name
        # entity="cogito-geoguessr-ai",  # Your team name
        config=asdict(config),
        mode="online" if api_key else "disabled",
    )

    main(config)
