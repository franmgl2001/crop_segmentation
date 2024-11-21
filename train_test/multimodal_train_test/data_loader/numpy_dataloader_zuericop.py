import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, txt_file, root_dir):
        """
        Initialize the dataset by reading folder paths from a text file and appending the root directory.
        """
        self.root_dir = root_dir

        # Load folder paths from the text file and append the root directory
        with open(txt_file, "r") as f:
            self.data_paths = [os.path.join(root_dir, line.strip()) for line in f]

        print(f"Loaded {len(self.data_paths)} folders from {txt_file}")

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        """
        Get the image and label tensors for the given index.
        """
        folder_path = self.data_paths[idx]

        try:
            # Load image data from 'data.npy'
            image_path = os.path.join(folder_path, "data.npy")
            image = np.load(image_path)

            # Load label data from 'gt_relabeled.npy'
            label_path = os.path.join(folder_path, "gt_relabeled_5.npy")
            label = np.load(label_path)
            label = np.squeeze(label)

            # Convert to PyTorch tensors
            image_tensor = torch.tensor(image, dtype=torch.float32)
            print(image_tensor.shape)
            image_tensor = image_tensor[1::2]
            print(image_tensor.shape)
            label_tensor = torch.tensor(label, dtype=torch.long)

            # Apply rescaling to the image tensor
            image_tensor = image_tensor * 0.0001  # Rescaling

            return image_tensor, label_tensor

        except Exception as e:
            print(f"Error loading data from {folder_path}: {e}")
            # Return empty tensors if there's an error
            return torch.tensor([]), torch.tensor([])


def load_data(txt_file, root_dir, batch_size=4, shuffle=True):
    """
    Create a DataLoader from the given text file and root directory.
    """
    dataset = CustomDataset(txt_file, root_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader
