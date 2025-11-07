from torch.utils.data import DataLoader
from data import ImageDataset


def train_model(data, labels, num_epochs: int):
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
    dataset = ImageDataset(data, labels)
    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    for epoch in range(num_epochs):
        for batch_images, batch_metadata in train_dataloader:
            pass
