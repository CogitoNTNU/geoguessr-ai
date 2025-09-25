from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        sample = self.data[id]
        label = self.labels[id]

        return sample, label
