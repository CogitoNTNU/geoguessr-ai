import torch
import pandas as pd
from torch import nn, Tensor
from torch.nn.parameter import Parameter
from collections import namedtuple
from preprocessing import haversine_matrix, smooth_labels
from models.layers import PositionalEncoder
from models.utils import ModelOutput
from config import *
from backend.s3bucket import *

def get_snapshot_metadata(): 
    df = load_latest_snapshot_df()

    meta_data = 0 #stuff

    return meta_data




class geo_guessr(nn.Module):
    def __init__(self):
        # inkluder CLIP(ene) som backbone og geocellene og andre nødvendige parameter(super_guessr?)
        pass

    ### Data collection ###
    def data_collection(self):

        pass

    ### Preproccesing ###
    def preproccesing(self):

        pass

    ### Pretrain ### (gjøres utenom)
    def pretrain(self):

        pass

    ### Finetuneing ###
    def finetuneing(self):
        
        pass

    ### Evaluate ### 
    def evaluate(self):

        pass

    def forward(self):

        pass




