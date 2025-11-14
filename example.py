import os
import yaml
import argparse
import json
from sklearn.metrics import roc_auc_score
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from tqdm import tqdm

from monai.transforms import Compose, RandFlipd, RandGaussianNoised, ToTensord
from monai.networks.nets import resnet as monai_resnet


# ----------------------------
# Simple Dataset for processed .npz unilateral files
# ----------------------------
class OdeliaNPZDataset(Dataset):
    """
    Expects items list of dicts: {'npz': path, 'label': int (0-normal,1-benign,2-malignant), 'study_id': str}
    """

    def __init__(self, items, transforms=None):
        self.items = items
        self.transforms = transforms

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        data = np.load(item["npz"], allow_pickle=True)
        arr = data["arr"]  # (C,Z,Y,X)
        # MONAI normally expects (C, Z, Y, X) and ToTensor will convert to torch
        sample = {"image": arr.astype(np.float32), "label": np.int64(item["label"])}
        if self.transforms:
            sample = self.transforms(sample)
        # ensure dtype
        return sample["image"], sample["label"]


# ----------------------------
# Model wrapper
# ----------------------------
class ResNet3DClassifier(torch.nn.Module):
    def __init__(self, in_channels, n_classes=3, pretrained=False):
        super().__init__()
        # Use MONAI's resnet implementations; pick a lightweight variant (18)
        self.backbone = monai_resnet.resnet18(
            spatial_dims=3, n_input_channels=in_channels, num_classes=n_classes
        )
        # MONAI's resnet already ends with linear -> we can keep it

    def forward(self, x):
        return self.backbone(x)


# ----------------------------
# Training & validation functions
# ----------------------------
def compute_auc_ytrue_ypred(y_true, y_pred_probs):
    # y_true: (N,) values 0/1/2 - compute malignant vs non-malignant AUC
    y_true_bin = (np.array(y_true) == 2).astype(int)
    y_scores = np.array(y_pred_probs)[:, 2]  # predicted malignant prob
    if len(np.unique(y_true_bin)) < 2:
        return float("nan")
    return roc_auc_score(y_true_bin, y_scores)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images)
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.shape[0]
    return running_loss / len(loader.dataset)


def valid_epoch(model, loader, device):
    model.eval()
    all_labels = []
    all_probs = []
    val_loss = 0.0
    criterion = CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="valid", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            loss = criterion(logits, labels).item()
            val_loss += loss * images.shape[0]
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.tolist())
    auc = compute_auc_ytrue_ypred(all_labels, all_probs)
    return val_loss / len(loader.dataset), auc, all_labels, all_probs


# ----------------------------
# Simple config loader & dataset builder
# ----------------------------
def load_items_from_manifest(manifest_path: str):
    """
    Expect a manifest CSV/TSV or JSON lines file mapping npz -> label.
    For simplicity accept a JSON list of dicts: [{'npz': '/path', 'label': 2, 'side': 'left'}, ...]
    """
    with open(manifest_path, "r") as f:
        items = json.load(f)
    return items


def build_transforms(train=True):
    # Minimal transforms: convert to float tensor and maybe augment
    train_trans = Compose(
        [
            RandFlipd(keys=["image"], spatial_axis=1, prob=0.5),
            RandGaussianNoised(keys=["image"], prob=0.2),
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_trans = Compose(
        [
            ToTensord(keys=["image", "label"]),
        ]
    )
    return train_trans, val_trans


# ----------------------------
# Main training script
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="yaml config file")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load manifests
    train_items = load_items_from_manifest(cfg["train_manifest"])
    test_items = load_items_from_manifest(cfg["test_manifest"])
    val_items = load_items_from_manifest(cfg["val_manifest"])

    # Merge train and test for more data
    print(f"Len of train items before merge: {len(train_items)}")
    train_items += test_items
    print(f"Len of train items after merge: {len(train_items)}")

    train_trans, val_trans = build_transforms(cfg.get("crop_size", [32, 256, 256]))
    train_ds = OdeliaNPZDataset(train_items, transforms=train_trans)
    val_ds = OdeliaNPZDataset(val_items, transforms=val_trans)

    # exit(0) # Debugging line to prevent further execution

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.get("batch_size", 2),
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.get("batch_size", 2),
        shuffle=False,
        num_workers=cfg.get("num_workers", 2),
        pin_memory=True,
    )

    # determine input channels from first item
    sample_arr = np.load(train_items[0]["npz"])["arr"]
    in_channels = sample_arr.shape[0]

    model = ResNet3DClassifier(in_channels=in_channels, n_classes=3).to(device)
    criterion = CrossEntropyLoss(weight=None)  # optionally set weights
    optimizer = AdamW(
        model.parameters(), lr=float(cfg.get("lr", 1e-4)), weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_val_auc = -1.0
    ckpt_dir = cfg.get("checkpoint_dir", "./checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, cfg.get("max_epochs", 100) + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc, val_labels, val_probs = valid_epoch(
            model, val_loader, device
        )

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_auc={val_auc:.4f}"
        )
        scheduler.step(val_auc if not np.isnan(val_auc) else 0.0)

        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "val_auc": val_auc,
        }
        torch.save(ckpt, os.path.join(ckpt_dir, f"ckpt_epoch{epoch:03d}.pt"))

        # best model
        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(ckpt, os.path.join(ckpt_dir, "best_model.pt"))
            # save predictions for inspection
            np.savez_compressed(
                os.path.join(ckpt_dir, "best_val_preds.npz"),
                labels=np.array(val_labels),
                probs=np.array(val_probs),
            )

    print("Training finished. Best val AUC:", best_val_auc)


if __name__ == "__main__":
    main()
