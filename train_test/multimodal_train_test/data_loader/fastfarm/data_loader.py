import os
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, txt_file, root_dir, max_seq_len=73, doy_range=None):
        """
        Initialize the dataset by reading file paths and setting max sequence length.
        """
        self.root_dir = root_dir
        self.max_seq_len = max_seq_len
        # Choose between the original Cut class or the new CutWithDoyRange
        if doy_range is not None:
            self.cut = CutWithDoyRange(seq_len=max_seq_len, doy_range=doy_range)
        else:
            self.cut = Cut(seq_len=max_seq_len)

        # Load file paths from the text file and append the root directory
        with open(txt_file, "r") as f:
            self.data_paths = [os.path.join(root_dir, line.strip()) for line in f]

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        """
        Get the image and mask tensors for the given index.
        """
        file_path = self.data_paths[idx]

        # Load data from the .pickle file
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        # Extract the 'image' and 'mask' keys
        input_data = data["image"]
        mask_data = data["mask"]

        print(f"Loaded data from {file_path}")
        print(f"Image shape: {input_data.shape}, Mask shape: {mask_data.shape}")
        # Convert to PyTorch tensors
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        mask_tensor = torch.tensor(mask_data, dtype=torch.long)

        # Apply Cut if the input length exceeds max_seq_len
        if input_tensor.shape[0] > self.max_seq_len:
            sample = {
                "image": input_tensor,
                "doy": data.get("doy", None),
            }  # Include DOY if present
            sample = self.cut(sample)
            input_tensor = sample["image"]
            if sample.get("doy") is not None:
                data["doy"] = sample["doy"]

        # Optional rescaling
        input_tensor = input_tensor * 0.0001  # Adjust scaling as needed

        # Return the input and mask tensors
        input_tensor = input_tensor.permute(0, 2, 3, 1)

        return input_tensor, mask_tensor


def load_data(
    txt_file,
    root_dir,
    batch_size=4,
    shuffle=True,
    max_seq_len=73,
    doy_range=None,
):
    """
    Create a DataLoader from the given text file and root directory.

    Parameters:
    - txt_file (str): Path to the text file containing file paths.
    - root_dir (str): Root directory for the data.
    - batch_size (int): Batch size for the DataLoader.
    - shuffle (bool): Whether to shuffle the data.
    - max_seq_len (int): Maximum sequence length.
    - doy_range (tuple, optional): A tuple (min_doy, max_doy) to filter DOY values.
    - use_doy_range (bool): Whether to use the CutWithDoyRange class.

    Returns:
    - DataLoader: A PyTorch DataLoader for the dataset.
    """
    dataset = CustomDataset(
        txt_file,
        root_dir,
        max_seq_len=max_seq_len,
        doy_range=doy_range,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


class Cut:
    """
    Randomly selects `seq_len` points from image and DOY arrays if seq_len exceeds the threshold.
    """

    def __init__(self, seq_len):
        self.seq_len = seq_len

    def __call__(self, sample):
        image = sample["image"]
        # doy = sample.get("doy", None)  # Get DOY if present

        total_len = image.shape[0]

        # Ensure the requested seq_len does not exceed available length
        if self.seq_len > total_len:
            raise ValueError(
                f"Requested seq_len ({self.seq_len}) exceeds available length ({total_len})."
            )

        # Randomly select and sort the indices for temporal consistency
        indices = torch.randperm(total_len)[: self.seq_len].sort()[0]

        # Cut the image (and DOY if present) using the selected indices
        cut_image = image[indices]
        sample["image"] = cut_image
        # if doy is not None:
        # cut_doy = doy[indices]
        # sample["doy"] = cut_doy

        return sample


class CutWithDoyRange:
    """
    Randomly selects `seq_len` points from image and DOY arrays if seq_len exceeds the threshold.
    Optionally filters indices based on a given range of DOY values.
    """

    def __init__(self, seq_len, doy_range=None):
        """
        Initializes the CutWithDoyRange class with the desired sequence length and an optional DOY range.

        Parameters:
        - seq_len (int): The maximum sequence length to select.
        - doy_range (tuple, optional): A tuple (min_doy, max_doy) specifying the range of DOY values to include.
        """
        self.seq_len = seq_len
        self.doy_range = doy_range  # Tuple: (min_doy, max_doy)

    def __call__(self, sample):
        image = sample["image"]
        doy = sample.get("doy", None)  # Get DOY if present

        # Ensure image and doy arrays are torch tensors
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image)
        if doy is not None and not isinstance(doy, torch.Tensor):
            doy = torch.tensor(doy)

        total_len = image.shape[0]

        # Ensure the requested seq_len does not exceed available length
        if self.seq_len > total_len:
            raise ValueError(
                f"Requested seq_len ({self.seq_len}) exceeds available length ({total_len})."
            )

        # Filter indices based on DOY range if provided
        if doy is not None and self.doy_range:
            min_doy, max_doy = self.doy_range
            print(f"Filtering DOY values between {min_doy} and {max_doy}...", doy)
            valid_indices = (doy >= min_doy) & (doy <= max_doy)
            valid_indices = torch.where(valid_indices)[0]

            if len(valid_indices) < self.seq_len:
                raise ValueError(
                    f"Insufficient valid indices ({len(valid_indices)}) within DOY range ({self.doy_range}) "
                    f"to satisfy the requested seq_len ({self.seq_len})."
                )
        else:
            valid_indices = torch.arange(total_len)

        # Randomly select and sort the indices for temporal consistency
        selected_indices = valid_indices[
            torch.randperm(len(valid_indices))[: self.seq_len]
        ].sort()[0]

        # Cut the image (and DOY if present) using the selected indices
        sample["image"] = image[selected_indices]
        if doy is not None:
            sample["doy"] = doy[selected_indices]

        print(
            f"Cut image shape: {sample['image'].shape}, Cut DOY shape: {len(sample.get('doy', None))}"
        )
        return sample
