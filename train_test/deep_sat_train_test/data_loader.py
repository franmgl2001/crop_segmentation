import torch
import os
import pickle
import pandas as pd
from torch.utils.data import Dataset
from transforms import PASTIS_segmentation_transform
from torch.utils.data import DataLoader


def get_dataloader(csv_file, root_dir, config):
    test_dataset = SatImDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        transform=PASTIS_segmentation_transform(config, True),
    )

    dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
    )

    return dataloader


class SatImDataset(Dataset):
    """Satellite Images dataset."""

    def __init__(
        self, csv_file, root_dir, transform=None, multilabel=False, return_paths=False
    ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if type(csv_file) == str:
            self.data_paths = pd.read_csv(csv_file, header=None)
        elif type(csv_file) in [list, tuple]:
            self.data_paths = pd.concat(
                [pd.read_csv(csv_file_, header=None) for csv_file_ in csv_file], axis=0
            ).reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.multilabel = multilabel
        self.return_paths = return_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.data_paths.iloc[idx, 0])

        with open(img_name, "rb") as handle:
            sample = pickle.load(handle, encoding="latin1")

        if self.transform:
            sample = self.transform(sample)

        if self.return_paths:
            return sample, img_name

        return sample

    def read(self, idx, abs=False):
        """
        read single dataset sample corresponding to idx (index number) without any data transform applied
        """
        if type(idx) == int:
            img_name = os.path.join(self.root_dir, self.data_paths.iloc[idx, 0])
        if type(idx) == str:
            if abs:
                img_name = idx
            else:
                img_name = os.path.join(self.root_dir, idx)
        with open(img_name, "rb") as handle:
            sample = pickle.load(handle, encoding="latin1")
        return sample


def my_collate(batch):
    "Filter out sample where mask is zero everywhere"
    idx = [b["unk_masks"].sum(dim=(0, 1, 2)) != 0 for b in batch]
    batch = [b for i, b in enumerate(batch) if idx[i]]
    return torch.utils.data.dataloader.default_collate(batch)
