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

# Add the 'playground' directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now import TSViT
from models.TSViT import TSViT
import gc  # Import garbage collection module


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
    print(tensor.shape, start_time, end_time)
    T, B, H, W = tensor.shape
    if start_time < 0 or end_time >= T or start_time > end_time:
        raise ValueError("Invalid start_time or end_time range.")

    padded_tensor = tensor.clone()
    padded_tensor[start_time : end_time + 1, :, :, :] = 0
    return padded_tensor


def get_masked_probabilities(tensor, mask, num_classes=3, value=2):
    """
    Computes the mean probabilities for each class in the masked regions.

    Parameters:
    tensor (torch.Tensor): A tensor of shape (C, H, W) with class probabilities.
    mask (torch.Tensor): A tensor of shape (H, W) with the mask values.
    num_classes (int): The number of classes.
    value (int): The value in the mask to use for selecting regions.

    Returns:
    torch.Tensor: A tensor of mean probabilities per class in the masked regions.
    """
    mask = mask == value  # Create a boolean mask
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
with open("3226_2023.pkl", "rb") as f:
    data = pickle.load(f)

# Model configuration and initialization
num_classes = 2
MAX_SEQ_LEN = 73
patch_size = 2

config = {
    "patch_size": patch_size,
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
    num_channels=[10],
    num_classes=3,
    max_seq_len=MAX_SEQ_LEN,
    patch_embedding="Channel Encoding",
)

# Load the model weights
model.load_state_dict(
    torch.load("../models/binary_window.pth", map_location=torch.device("cpu"))
)
model.eval()  # Set model to evaluation mode

# Placeholder for average probabilities
average_probabilities = []

# Loop through each padding level and calculate average probability
for padding_step in range(0, MAX_SEQ_LEN, 5):
    print(f"Padding Step: {padding_step}")
    input_data = np.array(data["image"], dtype=np.float32)
    mask_data = data["mask"]

    # Convert to PyTorch tensors
    input_tensor = torch.tensor(input_data, dtype=torch.float32) * 0.0001
    input_tensor = input_tensor.permute(0, 2, 3, 1)  # Reshape to (T, H, W, C)

    padded_tensor = zero_pad_time(
        input_tensor, 0, padding_step
    )  # Add batch dimension (1, T, H, W, C)

    # Add the batch dimension
    padded_tensor = padded_tensor.unsqueeze(0)
    # Prepare inputs with time channel
    B, T, H, W, C = padded_tensor.shape
    time_points = torch.linspace(0, 364, steps=MAX_SEQ_LEN)
    time_channel = time_points.repeat(B, H, W, 1).permute(0, 3, 1, 2)
    inputs = torch.cat((padded_tensor, time_channel[:, :, :, :, None]), dim=4)
    inputs = inputs.permute(0, 1, 4, 2, 3)

    # Run the model and get the probabilities
    output = model(inputs)
    probabilities = torch.softmax(output, dim=1)

    # Get the masked probabilities
    average_probabilities.append(
        get_masked_probabilities(probabilities, mask_data, value=2)
    )
    print(average_probabilities[1])
    # Delete variables to free memory
    del input_tensor, padded_tensor, time_channel, inputs, output, probabilities
    gc.collect()  # Run garbage collection

# Class one probabilities
class_one_probabilities = [p[1].item() for p in average_probabilities]
class_two_probabilities = [p[2].item() for p in average_probabilities]


# Plot the average probabilities for class one, and class two
plt.figure(figsize=(12, 6))
plt.plot(range(0, MAX_SEQ_LEN, 5), class_one_probabilities, label="Single label")
plt.plot(range(0, MAX_SEQ_LEN, 5), class_two_probabilities, label="Multi label")
plt.xlabel("Padding Steps")
plt.legend()
plt.ylabel("Average Probability")
plt.title("Average Probabilities for Class 1 and Class 2")
plt.savefig("average_probabilities_3226.png")
