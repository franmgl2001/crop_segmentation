"""
By: Francisco Martinez Gallardo.
This script is a playground to test the windowing technique with multiple labels in the same season.
It plots the average probabilities as padding increases.
"""

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from pathlib import Path

# Add the 'playground' directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now import TSViT
from models.TSViT import TSViT
import gc  # Import garbage collection module


class_names_ff = {
    0: "Background",
    1: "Sorhgum",
    2: "Maize",
    3: "Barley",
    4: "Wheat",
}

class_names_zuericrop = {
    0: "Background",
}

# Current band order in the input tensor
input_order = [
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B11",
    "B12",
    "SCL",
]

# Desired band order
target_order = ["B08", "B03", "B02", "B04", "B05", "B06", "B07", "B11", "B12"]


def zero_pad_time(tensor, start_time, end_time):
    """
    Sets values to zero for a specified range of time steps within the tensor.

    Parameters:
    tensor (torch.Tensor): A tensor of shape (T, B, H, W).
    start_time (int): The start time step (inclusive) for zero padding.
    end_time (int): The end time step (inclusive) for zero padding.

    Returns:
    torch.Tensor: A tensor with values set to zero for the specified time steps.
    """
    print(
        f"Tensor shape: {tensor.shape}, Start time: {start_time}, End time: {end_time}"
    )

    # Ensure tensor dimensions match the expected shape
    if len(tensor.shape) != 4:
        raise ValueError("Input tensor must have shape (T, B, H, W).")

    T, B, H, W = tensor.shape

    # Validate start_time and end_time
    if start_time < 0 or start_time > end_time:
        raise ValueError("Invalid start_time or end_time range.")

    # Create a clone of the tensor to avoid in-place modification
    padded_tensor = tensor.clone()

    # Apply zero padding outside the specified time window
    padded_tensor[:start_time, :, :, :] = 0  # Zero-pad before the start_time
    if end_time < T - 1:
        padded_tensor[end_time + 1 :, :, :, :] = 0  # Zero-pad after the end_time

    return padded_tensor


def get_masked_probabilities(tensor, mask, num_classes=3, value=0):
    """
    Computes the mean probabilities for each class in the masked regions.

    Parameters:
    tensor (torch.Tensor): A tensor of shape (C, H, W) with class probabilities.
    mask (torch.Tensor): A tensor of shape (H, W) with the mask values.
    num_classes (int): The number of classes.
    value (int): The value that is not being masked.

    Returns:
    torch.Tensor: A tensor of mean probabilities per class in the masked regions.
    """
    mask = mask != value  # Create a boolean mask
    masked_means = []

    # Calculate mean probabilities for each class in the masked region
    for c in range(num_classes):
        class_probabilities = tensor[0][c]  # Select probabilities for class `c`
        masked_class_probabilities = class_probabilities[mask]  # Apply mask
        masked_mean = (
            masked_class_probabilities.mean()
            if masked_class_probabilities.numel() > 0
            else torch.tensor(0.0)
        )
        masked_means.append(masked_mean)

    return masked_means


# Load the pickle file
with open("dev_5943_2019.pkl", "rb") as f:
    data = pickle.load(f)

# Model configuration and initialization
num_classes = 10
MAX_SEQ_LEN = 71
patch_size = 2
window_size = 40

# Model configuration and initialization
num_classes = 10
config = {
    "patch_size": 2,
    "patch_size_time": 1,
    "patch_time": 4,
    "dim": 128,
    "temporal_depth": 6,
    "spatial_depth": 2,
    "channel_depth": 4,
    "heads": 4,
    "dim_head": 64,
    "dropout": 0.0,
    "emb_dropout": 0.0,
    "scale_dim": 4,
    "depth": 4,
}

model = TSViT(
    config,
    img_res=24,
    num_channels=[9],
    num_classes=num_classes,
    max_seq_len=MAX_SEQ_LEN,
    patch_embedding="Channel Encoding",
)


def main(pickle_file, model_path, output_plot, max_seq_len=71, window_size=40):
    """
    Main function to process .pkl files, test the windowing technique,
    and plot the average probabilities.

    Parameters:
    - pickle_file (str): Path to the .pkl file.
    - model_path (str): Path to the pre-trained TSViT model.
    - output_plot (str): Path to save the output plot.
    - max_seq_len (int): Maximum sequence length. Default is 71.
    - window_size (int): Window size for padding. Default is 40.
    """
    # Load the pickle file
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)

    # Load the model weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()  # Set model to evaluation mode
    # Loop through padding steps

    input_data = np.array(data["image"], dtype=np.float32)

    # Reorder tensor channels to match the target order
    indices = [input_order.index(band) for band in target_order]
    input_data = input_data[:, indices, :, :]

    # If the time dimension is less then remove return and clean the code

    # Convert to PyTorch tensors
    input_tensor = torch.tensor(input_data, dtype=torch.float32) * 0.0001
    input_tensor = input_tensor[:max_seq_len].permute(0, 2, 3, 1)

    input_tensor = input_tensor.unsqueeze(0)
    # Add time channel
    B, T, H, W, C = input_tensor.shape
    if T < max_seq_len:
        return

    time_points = torch.linspace(0, 364, steps=max_seq_len)
    time_channel = time_points.repeat(B, H, W, 1).permute(0, 3, 1, 2)
    print(time_channel.shape)
    inputs = torch.cat((input_tensor, time_channel[:, :, :, :, None]), dim=4)
    inputs = inputs.permute(0, 1, 4, 2, 3)

    # Run the model
    output = model(inputs)
    probabilities = torch.softmax(output, dim=1)

    # Get an argmax of the probabilities
    predictions = torch.argmax(probabilities, dim=1)

    print(predictions.shape)

    # Plot image of predictions
    plt.imshow(predictions[0].cpu().numpy())
    # Plot colorbar
    plt.colorbar()
    # Save the plot on
    plt.savefig(output_plot)

    # Plot the RGB image
    plt.imshow(input_data[0, 0:3, :, :].transpose(1, 2, 0))
    plt.show()

    # Free memory
    del input_tensor, time_channel, inputs, output, probabilities
    gc.collect()


def get_all_pkl_files(root_folder):
    """
    Get all .pkl files from the specified root folder, including subdirectories.

    Parameters:
        root_folder (str): The root folder to search for .pkl files.

    Returns:
        list: A list of file paths to all .pkl files found.
    """
    root_path = Path(root_folder)
    if not root_path.is_dir():
        raise ValueError(
            f"The specified path '{root_folder}' is not a valid directory."
        )

    # Collect all .pkl files from the root folder and subdirectories
    pkl_files = [str(file) for file in root_path.rglob("*.pkl")]

    return pkl_files


# Run the main function
if __name__ == "__main__":

    for file in get_all_pkl_files(
        "../../../datasets/FASTFARM/main_transforms/pickle_crops_yearly/"
    ):
        main(
            file,
            "../models/zuericrop11.pth",
            f"fastfarm_plots/results_{file.split('/')[-1]}.png",
        )
