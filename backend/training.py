from torch.utils.data import DataLoader
from data import ImageDataset


def train_model(data, labels, num_epochs: int):
    """
    Train the model
    Not finished
    """
    dataset = ImageDataset(data, labels)
    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    for epoch in range(num_epochs):
        for batch_features, batch_labels in train_dataloader:
            pass
