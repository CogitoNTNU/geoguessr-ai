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
    """
    Parquet structure: (?)
    location_id
    lat
    lon
    heading
    image_path 
    batch_date 
    """
    df = load_latest_snapshot_df()

    remove = ("image", "img", "path", "picture", "pixel") #hva må fjernes, eventuelt hva som burde beholdes
    cols_to_drop = []
    for col in df.columns:
         for remove in remove:
             if remove in col:
                 cols_to_drop.append(col)
                 break

    meta_data = df.drop(columns=cols_to_drop).reset_index(drop=True)

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




