from dataclasses import dataclass, asdict
import os
import io
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import Module
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from torchvision import transforms as T
from training.load_sqlite_dataset import load_sqlite_dataset
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
from typing import Optional


class LocalGeoMapDataset(torch.utils.data.Dataset):
    """
    Map-style dataset that reads JPEG bytes from a pandas DataFrame produced by load_sqlite_dataset().
    Ensures a consistent spatial size to allow DataLoader collation.
    """

    def __init__(self, df, required_size: int = 336, transform: Optional[torch.nn.Module] = None):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.required_size = required_size
        if transform is None:
            ops = []
            if self.required_size:
                ops.append(T.Resize(self.required_size))
            ops.append(T.ToTensor())
            transform = T.Compose(ops)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_bytes = row["image"]
        if img_bytes is None:
            # Fallback placeholder to keep batch shape valid
            c, h, w = 3, self.required_size or 224, self.required_size or 224
            tensor = torch.zeros(c, h, w)
        else:
            pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            tensor = self.transform(pil) if self.transform else pil
            if isinstance(tensor, Image.Image):
                tensor = T.ToTensor()(tensor)
        target = {
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
        }
        return tensor, target


def main(config):
    # Overall-modus for å velge mellom training og inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fetch dataset from local SQLite (via training.load_sqlite_dataset)
    # Hardcoded: find the latest dataset_sqlite_*.sqlite next to project root (parent of geoguessr-ai)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
    repo_parent_dir = os.path.abspath(os.path.join(repo_root, ".."))
    candidates = []
    for name in os.listdir(repo_parent_dir):
        if (
            name.startswith("dataset_sqlite_2")
            and name.endswith(".sqlite")
            and "clip_embeddings" not in name
            and "tinyvit_embeddings" not in name
        ):
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

    train_test_split = 0.9
    num_training_samples = int(len(df) * train_test_split)
    df_train = df.iloc[:num_training_samples]
    df_test = df.iloc[num_training_samples:]

    # Use test split as validation set for local workflow
    df_val = df_test

    train_dataset = LocalGeoMapDataset(df_train)
    test_dataset = LocalGeoMapDataset(df_test)
    val_dataset = LocalGeoMapDataset(df_val)

    logger.info(
        f"Dataset loaded with {len(train_dataset)} training samples, {len(test_dataset)} test samples, {len(val_dataset)} validation samples"
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=32, num_workers=4, pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=32, num_workers=4, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=32, num_workers=4, pin_memory=True
    )

    # Initialize model and set it to train
    geocell_manager = GeocellManager("data/geocells/finished_geocells")
    num_geocells = geocell_manager.get_num_geocells()

    embeddingModelUsed = "TINYIT"  # Possible values are "CLIP" or "TINYVIT"

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
    epochs: int = 3
    # Early stopping (defaults approximate common built-ins)
    early_stopping_patience: int = 2
    # Scheduler
    T_0: int = 10
    T_mult: int = 2
    eta_min: int = 1e-6
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_epochs: int = 1
    keep_last_n: int = 3
    monitor_mode: str = "min"  # "min" uses epoch_loss, "max" uses epoch_top1
    resume_path: Optional[str] = None


def train(
    model: Module,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    device,
    config: Configuration,
    target_dimensions=None,
    norm_mean=None,
    norm_std=None,
    checkpoint_dir: Optional[str] = None,
):
    # Prepare checkpoint directory
    if checkpoint_dir is None:
        run_id = getattr(getattr(wandb, "run", None), "id", None)
        run_suffix = run_id if run_id else "default"
        checkpoint_dir = os.path.join(config.checkpoint_dir, run_suffix)
    # Ensure checkpoint directory exists before any file operations
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create checkpoint directory '{checkpoint_dir}': {e}")

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

    is_min_mode = (config.monitor_mode or "min").lower() == "min"
    best_value = float("inf") if is_min_mode else float("-inf")

    # Resume if provided
    start_epoch = 0
    global_step = 0
    # Early stopping state
    patience_counter = 0
    patience = getattr(config, "early_stopping_patience", None)
    if config.resume_path:
        try:
            checkpoint = torch.load(config.resume_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            # Scheduler state may not always be present (older checkpoints)
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = int(checkpoint.get("epoch", -1)) + 1
            global_step = int(checkpoint.get("global_step", 0))
            best_value = checkpoint.get("best_value", best_value)
            logger.info(
                f"Resumed from '{config.resume_path}' at epoch {start_epoch}, global_step {global_step}"
            )
        except Exception as e:
            logger.error(f"Failed to resume from '{config.resume_path}': {e}")

    for epoch in range(start_epoch, config.epochs):
    # for epoch in range(3):
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

            # Log per batch (training)
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

        # Validation loop: compute and log validation loss per batch
        if validation_dataloader is not None:
            model.eval()
            with torch.no_grad():
                for val_batch_idx, (images, targets) in enumerate(validation_dataloader):
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
                    coord_labels = torch.stack(
                        [lon_t, lat_t], dim=1
                    )  # (B, 2) in (lng, lat)
                    distances = haversine_matrix(
                        coord_labels, centroids.t()
                    )  # (B, num_cells)
                    targets_idx = torch.argmin(distances, dim=-1).to(
                        device, dtype=torch.long
                    )

                    output = model(
                        pixel_values=images,
                        labels_clf=targets_idx,
                        labels=coord_labels,
                    )
                    val_loss = output.loss

                    # Log per batch (validation)
                    wandb.log(
                        {
                            "val/loss": val_loss.item(),
                            "epoch": epoch,
                        },
                        step=global_step,
                    )

            model.train()

        # Log training metrics per epoch
        wandb.log(
            {
                "epoch": epoch,
                "train/epoch_loss": epoch_loss,
                "train/epoch_top1": epoch_top1,
                "train/epoch_top5": epoch_topk,
            },
            step=global_step,
        )

        # Checkpoint saving
        current_value = epoch_loss if is_min_mode else epoch_top1
        state = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_value": best_value,
            "monitored_value": current_value,
            "config": asdict(config),
        }

        # Always update 'last' checkpoint
        last_path = os.path.join(checkpoint_dir, "last.pt")
        try:
            # Re-create directory defensively in case it was removed mid-run
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(state, last_path)
        except Exception as e:
            logger.error(f"Failed to save last checkpoint: {e}")

        # Save per-epoch checkpoints only if among top-K best
        if (epoch + 1) % max(1, int(config.save_every_epochs)) == 0:
            # Helper to parse value from filename "epoch_XXXX_VALUE.pt"
            def parse_value_from_filename(filename: str) -> Optional[float]:
                try:
                    stem = filename[:-3]  # remove '.pt'
                    parts = stem.split("_")
                    # last token should be numeric value if present
                    value_str = parts[-1]
                    return float(value_str)
                except Exception:
                    return None

            # Gather existing epoch checkpoints with values
            existing: list[tuple[str, float]] = []
            try:
                for f in os.listdir(checkpoint_dir):
                    if not (f.startswith("epoch_") and f.endswith(".pt")):
                        continue
                    value = parse_value_from_filename(f)
                    if value is None:
                        # Unknown value; push to worst side so it gets pruned first
                        value = float("inf") if is_min_mode else float("-inf")
                    existing.append((f, value))
            except FileNotFoundError:
                # Directory vanished between epochs; recreate it and continue
                try:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                except Exception as e:
                    logger.error(f"Failed to recreate checkpoint directory '{checkpoint_dir}': {e}")

            k = max(0, int(config.keep_last_n))
            should_save = False
            if k == 0:
                should_save = False
            elif len(existing) < k:
                should_save = True
            else:
                # Determine worst among current kept set
                worst_value = max(v for _, v in existing) if is_min_mode else min(v for _, v in existing)
                should_save = (current_value < worst_value) if is_min_mode else (current_value > worst_value)

            if should_save:
                # Encode metric in filename for efficient pruning/sorting
                epoch_path = os.path.join(checkpoint_dir, f"epoch_{epoch:04d}_{current_value:.6f}.pt")
                try:
                    # Ensure checkpoint directory exists before saving
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save(state, epoch_path)
                    # Upload to Weights & Biases Artifacts
                    try:
                        if getattr(wandb, "run", None) is not None:
                            artifact_name = f"{wandb.run.id}-epoch-{epoch:04d}"
                            artifact = wandb.Artifact(name=artifact_name, type="model", metadata={
                                "epoch": epoch,
                                "monitored_value": current_value,
                                "monitor_mode": "min" if is_min_mode else "max",
                            })
                            artifact.add_file(epoch_path, name=os.path.basename(epoch_path))
                            wandb.log_artifact(artifact, aliases=[f"epoch-{epoch:04d}", "topk"])
                    except Exception as e:
                        logger.warning(f"Failed to upload epoch artifact to W&B: {e}")
                except Exception as e:
                    logger.error(f"Failed to save epoch checkpoint: {e}")

                # Recompute including the newly saved file
                existing = []
                try:
                    for f in os.listdir(checkpoint_dir):
                        if not (f.startswith("epoch_") and f.endswith(".pt")):
                            continue
                        value = parse_value_from_filename(f)
                        if value is None:
                            value = float("inf") if is_min_mode else float("-inf")
                        existing.append((f, value))
                except FileNotFoundError:
                    # If directory was removed, rebuild it and treat as no existing checkpoints
                    try:
                        os.makedirs(checkpoint_dir, exist_ok=True)
                    except Exception as e:
                        logger.error(f"Failed to recreate checkpoint directory '{checkpoint_dir}' during cleanup: {e}")

                # Sort by best first
                existing.sort(key=lambda t: t[1], reverse=not is_min_mode)
                # Keep top-k, remove the rest
                for f, _ in existing[k:]:
                    try:
                        os.remove(os.path.join(checkpoint_dir, f))
                    except FileNotFoundError:
                        pass
                    except Exception as e:
                        logger.warning(f"Failed to remove old checkpoint '{f}': {e}")

        # Save best checkpoint
        improved = (current_value < best_value) if is_min_mode else (current_value > best_value)
        if improved:
            best_value = current_value
            best_path = os.path.join(checkpoint_dir, "best.pt")
            try:
                state["best_value"] = best_value
                torch.save(state, best_path)
                wandb.summary["best_value"] = best_value
            except Exception as e:
                logger.error(f"Failed to save best checkpoint: {e}")

        # Early stopping check (uses monitored metric according to monitor_mode)
        if patience is not None and patience > 0:
            if improved:
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.warning(
                        f"Early stopping triggered after {epoch + 1} epochs "
                        f"with patience={patience} on metric "
                        f"{'epoch_loss' if is_min_mode else 'epoch_top1'}."
                    )
                    break

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
    config = Configuration()

    # Prefer explicit API key if provided, otherwise rely on prior `wandb login` (CLI) credentials.
    api_key = os.getenv("WANDB_API_KEY")
    try:
        if api_key:
            wandb.login(key=api_key)
        else:
            # Uses stored credentials from `wandb login` in the CLI (e.g. ~/.netrc or wandb config)
            wandb.login()
    except Exception as e:
        logger.warning(f"W&B login failed, proceeding with W&B disabled: {e}")

    # Try to run online; if that fails, fall back to disabled mode so training still works.
    try:
        wandb.init(
            project="geoguessr-ai",  # Your project name
            # entity="cogito-geoguessr-ai",  # Your team name
            config=asdict(config),
            mode="online",
        )
    except Exception as e:
        logger.warning(f"W&B init failed, falling back to disabled mode: {e}")
        wandb.init(
            project="geoguessr-ai",
            config=asdict(config),
            mode="disabled",
        )

    main(config)
