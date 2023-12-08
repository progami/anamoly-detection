import torch
import numpy as np
import pandas as pd
from config.autoencoder_params import batch_size


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.data = dataset.drop(["gt"], axis=1)
        self.targets = dataset["gt"]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        data = torch.from_numpy(np.array(row)).float().reshape(1, -1)
        target = self.targets.iloc[idx]
        # normalization 
        # data = (data - data.min()) / (data.max() - data.min() if (data.max()-data.min()) else 1) * 10
        # data = data - data.mean() + data.max() - data.min()
        # data = (data - data.mean()) / (data.std() if data.std() else 1)
        data = data - data.mean()
        return data, target


def get_data_loaders(train_dataset_path, test_dataset_path):
    clean_dataset = pd.read_csv(train_dataset_path, index_col=False)
    contam_dataset = pd.read_csv(test_dataset_path, index_col=False)

    train_dataset = Dataset(clean_dataset)
    train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                pin_memory=True
            )
    val_size = int(0.5 * len(contam_dataset))
    val_dataset = Dataset(contam_dataset[:val_size])
    val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=1,
                pin_memory=True
            )
    test_dataset = Dataset(contam_dataset[val_size+1:])
    test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=1,
                pin_memory=True
            )
    return train_loader, val_loader, test_loader, len(train_dataset)