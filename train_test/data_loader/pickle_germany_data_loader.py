import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pickle

# Simplified version of the remap label dictionary
original_label_dict = {
    0: "unknown",
    1: "sugar_beet",
    2: "summer_oat",
    3: "meadow",
    5: "rape",
    8: "hop",
    9: "winter_spelt",
    12: "winter_triticale",
    13: "beans",
    15: "peas",
    16: "potatoes",
    17: "soybeans",
    19: "asparagus",
    22: "winter_wheat",
    23: "winter_barley",
    24: "winter_rye",
    25: "summer_barley",
    26: "maize",
}
remap_label_dict = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    5: 4,
    8: 5,
    9: 6,
    12: 7,
    13: 8,
    15: 9,
    16: 10,
    17: 11,
    19: 12,
    22: 13,
    23: 14,
    24: 15,
    25: 16,
    26: 17,
}


# Preprocessing Classes
class Normalize:
    """Normalize image bands and day/year."""

    def __call__(self, sample):
        sample["x10"] = sample["x10"] * 1e-4
        sample["x20"] = sample["x20"] * 1e-4
        sample["x60"] = sample["x60"] * 1e-4
        # sample["day"] = sample["day"] / 365.0001
        sample["year"] = sample["year"] - 2016
        return sample


class Rescale:
    """Rescale the image to a given square size."""

    def __init__(self, output_size):
        assert isinstance(output_size, (tuple,))
        self.output_size = output_size

    def __call__(self, sample):
        sample["x10"] = self.rescale(sample["x10"])
        sample["x20"] = self.rescale(sample["x20"])
        sample["x60"] = self.rescale(sample["x60"])
        return sample

    def rescale(self, image):
        img = np.moveaxis(image, -1, 0)  # Move channels to the front for resizing
        img = F.interpolate(
            torch.tensor(img), size=self.output_size, mode="bilinear"
        ).numpy()
        return np.moveaxis(img, 0, -1)  # Move channels back to the last axis


class RemapLabel:
    """Remap labels to consecutive integers using remap_label_dict."""

    def __init__(self, labels_dict):
        self.labels_dict = labels_dict

    def __call__(self, sample):
        labels = sample["labels"]
        remapped_labels = np.copy(labels)
        for original_label, remapped_label in self.labels_dict.items():
            remapped_labels[labels == original_label] = remapped_label
        sample["labels"] = remapped_labels
        return sample


# Compose the transformations
def create_transform(img_res):
    transform_list = [
        Normalize(),
        Rescale(output_size=(img_res, img_res)),
        RemapLabel(remap_label_dict),
    ]
    return transform_list


def preprocess_data(pickle_file_path, img_res=24):
    """
    Loads the pickle file, applies transformations, selects useful bands, and returns the first label,
    combined bands, and day information.

    Args:
        pickle_file_path (str): Path to the pickle file.
        img_res (int): Image resolution to apply during rescaling.

    Returns:
        tuple: (first_label, combined_bands, days)
    """

    # Load the pickle file
    with open(pickle_file_path, "rb") as file:
        pickle_file = pickle.load(file)

    # Apply the transformations
    transformations = create_transform(img_res=img_res)
    sample_data = pickle_file
    for transform in transformations:
        sample_data = transform(sample_data)

    # Extract useful bands from x10 and x20
    x10_selected = sample_data["x10"][:, :, :, [0, 1, 2, 3]]  # B04, B03, B02, B08
    x20_selected = sample_data["x20"][
        :, :, :, [0, 1, 2, 3, 4, 5]
    ]  # B05, B06, B07, B8A, B11, B12

    # Combine x10 and x20 bands
    combined_bands = np.concatenate((x10_selected, x20_selected), axis=-1)

    # Extract the first label
    labels_array = np.array(sample_data["labels"])
    first_label = labels_array[0, :, :]

    # Extract the day information
    days = sample_data["day"]

    return first_label, combined_bands, days


class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_paths = pd.read_csv(csv_file)  # Load CSV with file paths
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        # Get file path from the CSV for the given index
        data_path = self.data_paths.iloc[idx, 0]

        # Load data from the file
        label, bands, days = preprocess_data(f"../../datsets/{data_path}", img_res=24)

        # Convert to PyTorch tensors
        bands_tensor = torch.tensor(bands, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        days_tensor = torch.tensor(days, dtype=torch.float32)

        return bands_tensor, label_tensor, days_tensor
