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
from torch.utils.data import DataLoader
from data import GeoImageIterableDataset
from s3bucket import load_latest_snapshot_df




# ----------------------------
# Main training script
# ----------------------------
def main():

    # Overall-modus for å velge mellom training og inference 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load entire dataset from s3-bucket
    df = load_latest_snapshot_df()
    df = df.head(10)

    dataset = GeoImageIterableDataset(df)
    train_dataloader = DataLoader(dataset, batch_size=64, num_workers=4)

    # Make variables  with train, test and eval dataset, fetched from datalaoder. 
    # Make train and test dataset by splitting the already fetched dataset


    # Initialize model and set it to train


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

    num_epochs = 5

    for epoch in range(num_epochs):
        for batch_idx, (images, targets) in enumerate(train_dataloader):
            print(batch_idx, images, targets)
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


if __name__ == '__main__':
    main()
