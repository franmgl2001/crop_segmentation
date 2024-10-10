import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class CustomDataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        # List all folders inside the dataset directory
        self.data_paths = [
            os.path.join(self.dataset_dir, d)
            for d in os.listdir(self.dataset_dir)
            if os.path.isdir(os.path.join(self.dataset_dir, d))
        ]

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        folder_path = self.data_paths[idx]
        # Load image data from 'data.npy'
        image_path = os.path.join(folder_path, "data.npy")
        image = np.load(image_path)

        # Load label data from 'gt.npy'
        label_path = os.path.join(folder_path, "gt.npy")
        label = np.load(label_path)
        label = np.squeeze(label)

        # Convert to PyTorch tensors
        image_tensor = torch.tensor(image, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor
