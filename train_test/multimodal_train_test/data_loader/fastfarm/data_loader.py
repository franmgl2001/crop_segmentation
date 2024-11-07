import os
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, txt_file, root_dir, max_seq_len=73):
        """
        Initialize the dataset by reading file paths and setting max sequence length.
        """
        self.root_dir = root_dir
        self.max_seq_len = max_seq_len
        self.cut = Cut(seq_len=max_seq_len)  # Initialize Cut with the max sequence length

        # Load file paths from the text file and append the root directory
        with open(txt_file, "r") as f:
            self.data_paths = [os.path.join(root_dir, line.strip()) for line in f]

        print(f"Loaded {len(self.data_paths)} files from {txt_file}")

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        """
        Get the input and mask tensors for the given index.
        """
        file_path = self.data_paths[idx]

        try:
            # Load data from the .pickle file
            with open(file_path, "rb") as f:
                data = pickle.load(f)

            # Extract the 'input' and 'mask' keys
            input_data = data['input']
            mask_data = data['mask']

            # Convert to PyTorch tensors
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
            mask_tensor = torch.tensor(mask_data, dtype=torch.long)

            # Apply Cut if the input length exceeds max_seq_len
            if input_tensor.shape[0] > self.max_seq_len:
                sample = {"inputs": input_tensor, "doy": data.get("doy", None)}  # Include DOY if present
                sample = self.cut(sample)
                input_tensor = sample["inputs"]
                if sample.get("doy") is not None:
                    data["doy"] = sample["doy"]

            # Optional rescaling
            input_tensor = input_tensor * 0.0001  # Adjust scaling as needed

            return input_tensor, mask_tensor

        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")
            # Return empty tensors if there's an error
            return torch.tensor([]), torch.tensor([])

def load_data(txt_file, root_dir, batch_size=4, shuffle=True, max_seq_len=73):
    """
    Create a DataLoader from the given text file and root directory.
    """
    dataset = CustomDataset(txt_file, root_dir, max_seq_len=max_seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


class Cut:
    """
    Randomly selects `seq_len` points from inputs and DOY arrays if seq_len exceeds the threshold.
    """

    def __init__(self, seq_len):
        self.seq_len = seq_len

    def __call__(self, sample):
        inputs = sample["inputs"]
        doy = sample.get("doy", None)  # Get DOY if present

        total_len = inputs.shape[0]

        # Ensure the requested seq_len does not exceed available length
        if self.seq_len > total_len:
            raise ValueError(
                f"Requested seq_len ({self.seq_len}) exceeds available length ({total_len})."
            )

        # Randomly select and sort the indices for temporal consistency
        indices = torch.randperm(total_len)[: self.seq_len].sort()[0]

        # Cut the inputs (and DOY if present) using the selected indices
        cut_inputs = inputs[indices]
        sample["inputs"] = cut_inputs
        if doy is not None:
            cut_doy = doy[indices]
            sample["doy"] = cut_doy

        return sample