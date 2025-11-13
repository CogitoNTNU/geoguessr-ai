from transformers import Trainer
from datasets import Dataset
from typing import Any, Tuple, Dict
from torch.nn.parameter import Parameter
from collections import namedtuple
import torch
from torch import Tensor
from config import LABEL_SMOOTHING_CONSTANT

ModelOutput = namedtuple('ModelOutput', 'loss loss_clf \
                         preds_LLH preds_geocell \
                         top5_geocells embedding')

def smooth_labels(distances: Tensor) -> Tensor:
    """Haversine smooths labels for shared representation learning across geocells.

    Args:
        distances (Tensor): distance (km) matrix of size (batch_size, num_geocells)

    Returns:
        Tensor: smoothed labels
    """
    adj_distances = distances - distances.min(dim=-1, keepdim=True)[0]
    smoothed_labels = torch.exp(-adj_distances / LABEL_SMOOTHING_CONSTANT)
    smoothed_labels = torch.nan_to_num(smoothed_labels, nan=0.0, posinf=0.0, neginf=0.0)
    return smoothed_labels


rad_torch = torch.tensor(6378137.0, dtype=torch.float64)

# Implementation to calculate all possible combinations of distances in parallel
def haversine_matrix(x: Tensor, y: Tensor) -> Tensor:
    """Computes the haversine distance between two matrices of points

    Args:
        x (Tensor): matrix 1 (lon, lat) -> shape (N, 2)
        y (Tensor): matrix 2 (lon, lat) -> shape (2, M)

    Returns:
        Tensor: haversine distance in km -> shape (N, M)
    """
    x_rad, y_rad = torch.deg2rad(x), torch.deg2rad(y)
    delta = x_rad.unsqueeze(2) - y_rad
    p = torch.cos(x_rad[:, 1]).unsqueeze(1) * torch.cos(y_rad[1, :]).unsqueeze(0)
    a = torch.sin(delta[:, 1, :] / 2)**2 + p * torch.sin(delta[:, 0, :] / 2)**2
    c = 2 * torch.arcsin(torch.sqrt(a))
    # ensure radius constant matches tensor device and dtype
    rad = torch.tensor(6378137.0, dtype=c.dtype, device=c.device)
    km = (rad * c) / 1000
    return km

def predict(model: Any, dataset: Dataset) -> Tuple:
    """Makes predictions given a Huggingface model.

    Args:
        model (Any): trained model.
        dataset (Dataset): dataset.

    Returns:
        Tuple: prediction tuple.
    """
    trainer = Trainer(model=model)
    return trainer.predict(dataset)

def load_state_dict(self, state_dict: Dict, embedder: bool=False):
    """Loads parameters in state_dict into model wherever possible

    Args:
        state_dict (Dict): model parameter dict
        embedder (bool, optional): whether loading state for the CLIP emebdder.
            Defaults to False.
    """
    own_state = self.state_dict()
    for name, param in state_dict.items():
        if embedder and 'base_model' in name:
            name = '.'.join(name.split('.')[1:])

        if name not in own_state:
            print(f'Parameter {name} not in model\'s state.')
            continue

        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data

        own_state[name].copy_(param)