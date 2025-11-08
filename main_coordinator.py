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
from data.geocells.geocell_manager import get_geocell_id
from dotenv import load_dotenv
import os
import wandb


# ----------------------------
# Main training script
# ----------------------------
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
    test_dataloader = DataLoader(train_dataset, batch_size=64, num_workers=4, pin_memory=True)
    val_dataset = DataLoader(train_dataset, batch_size=64, num_workers=4, pin_memory=True)

    # Initialize model and set it to train
    train(model=None, train_dataloader=train_dataloader, validation_dataloader=val_dataset, device=device, config=config)

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
    weight_decay: float =0.01,
    epocs: int = 5
    #Scheduler
    T_0: int = 10
    T_mult: int = 2
    eta_min: int = 1e-6




def train(model: Module, train_dataloader: DataLoader, validation_dataloader: DataLoader, device, config: Configuration):
    optimizer = AdamW(
        model.parameters(), 
        betas=config.betas,
        lr=config.lr,
        weight_decay=config.weight_decay,
        )
    scheduler = CosineAnnealingWarmRestarts(optimizer, config.T_0, config.T_mult, config.eta_min) 
    criterion = CrossEntropyLoss()

    for epoch in range(config.epocs):
        for batch_idx, (images, targets) in enumerate(train_dataloader):
            print(batch_idx, images, targets)
            
            # Find target geocell labels from lat, lon
            lat = targets['lat']
            lon = targets['lon']
            geocell_info = get_geocell_id({'latitude': lat, 'longitude': lon})
            geocell_id = 0
            targets = geocell_info[geocell_id]

            # Zero your gradients for every batch!           
            optimizer.zero_grad()
            # Make predictions for this batch
            outputs = model(images)

            # Compute the loss and its gradients
            loss = criterion(outputs, targets)
            geocell_topk = None  # Placeholder for model output
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

    wandb.login(api_key)
    run = wandb.init(
            project="geoguessr-ai",        # Your project name
            entity="cogito-geoguessr-ai", # Your team name
            config=config.asdict(),
            mode="online" if api_key else "disabled"
            )
    
    main(config)