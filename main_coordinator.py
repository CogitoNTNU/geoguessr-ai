from dataclasses import dataclass, asdict
import os
from pyarrow import null
import yaml
import argparse
import json
from sklearn.metrics import roc_auc_score
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss, Module
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from backend.data import GeoImageIterableDataset
from backend.s3bucket import load_latest_snapshot_df, load_latest_holdout_snapshot_df
from loguru import logger
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from data.geocells.geocell_manager import GeocellManager    
from dotenv import load_dotenv
import os
import wandb
from models.super_guessr import SuperGuessr
from models.tinyvit import TinyViTAdapter
from transformers import CLIPVisionModel
from config import CLIP_MODEL, TINYVIT_MODEL


def main(config):
    # Overall-modus for å velge mellom training og inference 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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

    logger.info(f"Dataset loaded with {len(train_dataset)} training samples, {len(test_dataset)} test samples, {len(val_dataset)} validation samples")

    train_dataloader = DataLoader(train_dataset, batch_size=64, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, num_workers=4, pin_memory=True)
    

    # Initialize model and set it to train
    geocell_manager = GeocellManager("data/geocells/finished_geocells")
    num_geocells = geocell_manager.get_num_geocells()

    # class Net(torch.nn.Module):
    #     def __init__(self, num_geocells: int):
    #         super(Net, self).__init__()
    #         self.conv1 = torch.nn.Conv2d(3, 16, 3, 1)
    #         self.conv2 = torch.nn.Conv2d(16, 32, 3, 1)
    #         self.fc1 = torch.nn.Linear(508032 , 128)
    #         self.fc2 = torch.nn.Linear(128, num_geocells) # get_num_geocells() in data/geocells/geocell_manager.py

    #     def forward(self, x):
    #         x = self.conv1(x)
    #         x = F.relu(x)

    #         x = self.conv2(x)
    #         x = F.relu(x)

    #         x = F.max_pool2d(x, 2)
    #         x = torch.flatten(x, 1)
    #         x = self.fc1(x)
    #         x = F.relu(x)
    #         logits = self.fc2(x)

    #         class_probabilities = F.softmax(logits, dim=1)
    #         return logits, class_probabilities
    # model = Net(num_geocells).to(device)       
    # train(model=model, train_dataloader=train_dataloader, validation_dataloader=val_dataloader, device=device, config=config)@

    embeddingModelUsed = "TINYVIT" # Possible values are "CLIP" or "TINYVIT"
    
    embedding_model = 0
    if embeddingModelUsed == "CLIP":
        embedding_model = CLIPVisionModel.from_pretrained(CLIP_MODEL)
    elif embeddingModelUsed == "TINYVIT":
        embedding_model = TinyViTAdapter(model_name=TINYVIT_MODEL, pretrained=True)

    # Select target input resolution per embedding model
    if embeddingModelUsed == "CLIP":
        target_dimensions = (336, 336)
    elif embeddingModelUsed == "TINYVIT":
        target_dimensions = (512, 512)
    else:
        target_dimensions = None

    # Instantiate SuperGuessr with CLIP backbone
    model = SuperGuessr(base_model=embedding_model, panorama=True, serving=False).to(device)
    model.train()

    


    """
    Hva som må gjøres (roughly)
    * Sette scriptet til å kjøre på GPU: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    * Initialisere SuperGuessr-modellen
    * Sette modellen i treningsmodus ved å kalle model.train()
    * Initialisere en optimizer 
    """ 

    # =========================
    # MAIN TRAINING LOOP
    # =========================
    """
    Train the model
    Data = bilder
    Labels = tilhørende metadata med følgende felter:
    lat: float
    lon: float
    heading: int
    location_id: str
    pano_id: str
    capture_date: str
    batch_date: str
    """

@dataclass
class Configuration:
    #Optimizer
    betas: tuple[float] = (0.9, 0.999)
    lr: float = 5e-5
    weight_decay: float = 0.01
    epochs: int = 5
    #Scheduler
    T_0: int = 10
    T_mult: int = 2
    eta_min: int = 1e-6


def train(model: Module, train_dataloader: DataLoader, validation_dataloader: DataLoader, device, config: Configuration, target_dimensions=None):
    optimizer = AdamW(
        model.parameters(), 
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, config.T_0, config.T_mult, config.eta_min) 
    criterion = CrossEntropyLoss()
    # Build index mapping aligned with SuperGuessr's internal ordering
    geocell_manager = getattr(model, "_geocell_mgr", None)
    if geocell_manager is None:
        geocell_manager = GeocellManager("data/geocells/finished_geocells")
    rows = []
    for country in geocell_manager.geocells:
        for admin1 in geocell_manager.geocells[country]:
            for cell in geocell_manager.geocells[country][admin1]:
                rows.append((str(country), str(admin1), cell.id))
    rows.sort(key=lambda r: (r[0], r[1], str(r[2])))
    geocell_index_map = { (country, admin1, cell_id): idx for idx, (country, admin1, cell_id) in enumerate(rows) }

    for epoch in range(config.epochs):
        for batch_idx, (images, targets) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")):
            # Resize images to match backbone input resolution (CLIP or Tiny-Vit)
            if target_dimensions is not None:
                if images.dim() == 5:  # (B, V, C, H, W) panorama batches
                    b, v, c, h, w = images.shape
                    images = images.view(b * v, c, h, w)
                    images = F.interpolate(images, size=target_dimensions, mode='bilinear', align_corners=False)
                    images = images.view(b, v, c, target_dimensions[0], target_dimensions[1])
                elif images.dim() == 4:  # (B, C, H, W)
                    images = F.interpolate(images, size=target_dimensions, mode='bilinear', align_corners=False)

            # Move tensors to device for training
            images = images.to(device, non_blocking=True)

            # Find target geocell labels from lat, lon
            lat = targets['lat']
            lon = targets['lon']
            list_of_geocell_indices = []
            
            for lat_item, lon_item in zip(lat, lon):
                point = {'latitude': lat_item.item(), 'longitude': lon_item.item()}
                info = geocell_manager.get_geocell_id(point)
                # info -> (geocell_id, country, admin1)
                geocell_id, country, admin1 = info
                idx = geocell_index_map[(str(country), str(admin1), geocell_id)]
                list_of_geocell_indices.append(idx)
            
            targets = torch.tensor(list_of_geocell_indices, dtype=torch.long).to(device) # Now the indices of the geocells here will match the ordering inside SuperGuessr. 
            # They are sorted according to the geocell-id, which is (country, admin1, str(cell.id))

            # Zero your gradients for every batch!           
            optimizer.zero_grad()
            # Make predictions for this batch
            output = model(pixel_values=images, labels_clf=targets)

            # Compute the loss and its gradients
            loss = output.loss
            geocell_topk = output.geocell_topk
            # Metrics: top-1 and top-k accuracy
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
            # Update progress bar
            tqdm.write(f"batch={batch_idx} loss={loss.item():.4f} top1={top1_acc:.3f} topk={topk_acc:.3f}")
            # Find gradients
            loss.backward()
            # Adjust learning weights
            optimizer.step()

        scheduler.step(epoch)

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

 
if __name__ == '__main__':
    

    load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")
    config = Configuration()

    wandb.login(key=api_key)
    run = wandb.init(
            project="geoguessr-ai",        # Your project name
            entity="cogito-geoguessr-ai", # Your team name
            config=asdict(config),
            mode="online" if api_key else "disabled"
            )
    
    main(config)
