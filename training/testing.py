import logging
import torch
from models.super_guessr import SuperGuessr
from models.tinyvit import TinyViTAdapter
from backend.data import PanoramaIterableDataset
from torch.utils.data import DataLoader
from backend.s3bucket import load_latest_snapshot_df

# from training.train_eval_loop import train_model
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

# TinyViT-21M-512 config
TINYVIT_SIZE = (512, 512)
TINYVIT_MEAN = [0.485, 0.456, 0.406]
TINYVIT_STD = [0.229, 0.224, 0.225]

transform = T.Compose(
    [
        T.Resize((512, 512), interpolation=InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),  # <-- must come before Normalize
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

df = load_latest_snapshot_df()

dataset = PanoramaIterableDataset(
    df=df.head(32),
    transform=transform,  # <— apply per-sample on CPU
)
loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
    drop_last=False,
)
device = torch.device("cpu")

# Initialize Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train")

if __name__ == "__main__":
    loaded_model = TinyViTAdapter()
    model = SuperGuessr(
        loaded_model,
        panorama=True,
        hierarchical=False,
        freeze_base=False,
        # should_smooth_labels=True,
    ).to(device)
    model.eval()

    with torch.no_grad():
        images, targets = next(iter(loader))  # images: (B, 4, C, H, W)
        images = images.to(device, non_blocking=True)
        pred_coords, geocell_topk, embeds = model(pixel_values=images)

    print(images.shape)  # (B, 4, 3, 512, 512)
    print(pred_coords.shape)  # (B, 2) if your centroids are (lng,lat)
    print(geocell_topk.indices.shape)  # (B, K)
    print(embeds.shape)  # (B, hidden)
