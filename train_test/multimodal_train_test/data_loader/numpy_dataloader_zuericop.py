import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class HVFlip(object):
    """
    Random horizontal and vertical flip.
    Applies flips to specified inputs and ground truths.
    """

    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, ground_truths=[]):
        assert isinstance(hflip_prob, (float,))
        assert isinstance(vflip_prob, (float,))
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.ground_truths = ground_truths

    def __call__(self, sample):
        if random.random() < self.hflip_prob:
            sample["inputs"] = torch.flip(sample["inputs"], (2,))
            if "inputs_backward" in sample:
                sample["inputs_backward"] = torch.flip(sample["inputs_backward"], (2,))
            for gt in self.ground_truths:
                sample[gt] = torch.flip(sample[gt], (1,))

        if random.random() < self.vflip_prob:
            sample["inputs"] = torch.flip(sample["inputs"], (1,))
            if "inputs_backward" in sample:
                sample["inputs_backward"] = torch.flip(sample["inputs_backward"], (1,))
            for gt in self.ground_truths:
                sample[gt] = torch.flip(sample[gt], (0,))
        return sample


class CustomDataset(Dataset):
    def __init__(self, txt_file, root_dir, augmentations=None):
        """
        Initialize the dataset by reading folder paths and setting augmentations.
        """
        self.root_dir = root_dir
        self.augmentations = augmentations

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
            label_path = os.path.join(folder_path, "gt_relabeled_7.npy")
            label = np.load(label_path)
            label = np.squeeze(label)

            # Convert to PyTorch tensors
            image_tensor = torch.tensor(image, dtype=torch.float32)
            image_tensor = image_tensor[1::2]
            label_tensor = torch.tensor(label, dtype=torch.long)

            # Apply rescaling to the image tensor
            image_tensor = image_tensor * 0.0001  # Rescaling

            # Create the sample dictionary
            sample = {
                "inputs": image_tensor,
                "labels": label_tensor,
            }

            # Apply augmentations if provided
            if self.augmentations:
                sample = self.augmentations(sample)

            return sample["inputs"], sample["labels"]

        except Exception as e:
            print(f"Error loading data from {folder_path}: {e}")
            # Return empty tensors if there's an error
            return torch.tensor([]), torch.tensor([])


def load_data(txt_file, root_dir, batch_size=4, shuffle=True, augmentations=None):
    """
    Create a DataLoader from the given text file and root directory.
    """
    dataset = CustomDataset(txt_file, root_dir, augmentations)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader
