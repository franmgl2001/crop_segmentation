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
    print(tensor.shape, start_time, end_time)
    T, B, H, W = tensor.shape
    if start_time < 0 or end_time >= T or start_time > end_time:
        raise ValueError("Invalid start_time or end_time range.")

    padded_tensor = tensor.clone()
    padded_tensor[start_time : end_time + 1, :, :, :] = 0
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
    num_channels=[9],
    num_classes=num_classes,
    max_seq_len=MAX_SEQ_LEN,
    patch_embedding="Channel Encoding",
)

# Load the model weights
model.load_state_dict(
    torch.load("../models/zuericrop11.pth", map_location=torch.device("cpu"))
)
model.eval()  # Set model to evaluation mode

# Placeholder for average probabilities
average_probabilities = []


padding_steps = list(range(0, MAX_SEQ_LEN, 5))  # Adjusted to step size used in the loop
# Loop through each padding level and calculate average probability
for padding_step in padding_steps:
    # Remove the
    input_data = np.array(data["image"], dtype=np.float32)
    mask_data = data["mask"]

    # Get indices of the target order from the current order
    indices = [input_order.index(band) for band in target_order]

    # Reorder tensor channels
    input_data = input_data[:, indices, :, :]

    # Convert to PyTorch tensors
    input_tensor = torch.tensor(input_data, dtype=torch.float32) * 0.0001
    # Cut the input tensor
    input_tensor = input_tensor[:MAX_SEQ_LEN]
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
        get_masked_probabilities(
            probabilities, mask_data, value=2, num_classes=num_classes
        )
    )
    print(average_probabilities[-1])
    # Delete variables to free memory
    del input_tensor, padded_tensor, time_channel, inputs, output, probabilities
    gc.collect()  # Run garbage collection

# Class one probabilities
# Plot the average probabilities for all classes
plt.figure(figsize=(12, 6))

# Transpose `average_probabilities` for easier plotting (class-wise separation)
average_probabilities_transposed = list(zip(*average_probabilities))

# Iterate through classes (ignoring Background if needed)
for class_idx in range(1, num_classes):  # Skipping class 0 (Background)
    print(class_idx)
    if class_idx - 1 < len(average_probabilities_transposed):  # Avoid index errors
        print(class_idx)
        # Detach tensor and convert to numpy for plotting
        class_probabilities = [
            prob.detach().numpy() if isinstance(prob, torch.Tensor) else prob
            for prob in average_probabilities_transposed[class_idx - 1]
        ]
        plt.plot(
            padding_steps,
            class_probabilities,  # Class-wise averages
            label=class_idx,
        )

plt.xlabel("Padding Steps")
plt.ylabel("Average Probability")
plt.title("Average Probabilities for Crop Classes as Padding Increases")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save and show the plot
plt.savefig("plots/average_probabilities_crops_5975.png")
