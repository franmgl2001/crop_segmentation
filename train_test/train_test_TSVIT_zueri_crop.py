import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from models.TSViT import TSViT
from data_loader.numpy_dataloader_zuericop import CustomDataset
import numpy as np
import os
import csv


def export_results_to_csv(results, output_csv_path):
    csv_columns = ["Class", "Overall Accuracy", "MIoU"]
    with open(output_csv_path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=csv_columns)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    print(f"Results saved to {output_csv_path}")


# Simplified Training Loop
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    iteration = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            unique_labels = torch.unique(labels)
            print(f"Unique labels in this batch: {unique_labels}")

            print(iteration)
            iteration += 1
            inputs = inputs * 0.0001  # Rescaling here

            B, T, H, W, C = inputs.shape
            # Add channel that contains time steps
            time_points = torch.linspace(0, 364, steps=142).to(device)
            time_channel = (
                time_points.repeat(B, H, W, 1).permute(0, 3, 1, 2).to(device)
            )  # BxTxHxW

            inputs = torch.cat(
                (inputs, time_channel[:, :, :, :, None]), dim=4
            )  # Add time channel
            inputs = inputs.permute(0, 1, 4, 2, 3)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()

            print(
                f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}",
                end="\r",
            )

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}"
        )
    print("Finished Training")


# Function to calculate IoU for each class
def compute_iou_per_class(preds, labels, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_cls = preds == cls
        true_cls = labels == cls

        intersection = np.logical_and(pred_cls, true_cls).sum()
        union = np.logical_or(pred_cls, true_cls).sum()

        if union == 0:
            iou = float("nan")  # Ignore classes with no presence in both true and pred
        else:
            iou = intersection / union

        ious.append(iou)
    return np.array(ious)


# Evaluation Loop with MIoU and mean accuracy calculation
def evaluate_model(model, test_loader, criterion, num_classes):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    # Initialize counters for each class
    correct_per_class = torch.zeros(num_classes).to(device)
    total_per_class = torch.zeros(num_classes).to(device)

    # To store predictions and labels for MIoU and mean accuracy
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            B, T, H, W, C = inputs.shape
            time_points = torch.linspace(0, 364, steps=142).to(device)
            time_channel = time_points.repeat(B, H, W, 1).permute(0, 3, 1, 2).to(device)

            inputs = torch.cat((inputs, time_channel[:, :, :, :, None]), dim=4)
            inputs = inputs.permute(0, 1, 4, 2, 3)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Get predicted classes
            _, predicted = torch.max(outputs, 1)

            # Store predictions and labels for MIoU and mean accuracy
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # Update global accuracy
            correct += (predicted == labels).sum().item()
            total += labels.numel()

            # Update per-class accuracy
            for label in range(num_classes):
                correct_per_class[label] += (
                    ((predicted == labels) & (labels == label)).sum().item()
                )
                total_per_class[label] += (labels == label).sum().item()

    # Convert all predictions and labels to numpy arrays for MIoU and mean accuracy calculation
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate per-class IoU
    ious = compute_iou_per_class(all_preds, all_labels, num_classes)
    mean_iou = np.nanmean(ious)
    print(f"Mean Intersection over Union (MIoU): {mean_iou:.4f}")

    # Calculate per-class accuracy
    class_accuracies = 100 * correct_per_class / total_per_class
    mean_accuracy = torch.nanmean(class_accuracies).item()

    # Calculate and print overall accuracy
    avg_loss = total_loss / len(test_loader)
    overall_accuracy = 100 * correct / total
    print(f"Test Loss: {avg_loss:.4f}, Overall Accuracy: {overall_accuracy:.2f}%")

    # Prepare data for CSV export
    results = []
    for i in range(num_classes):
        class_iou = ious[i]
        class_accuracy = (
            class_accuracies[i].item() if total_per_class[i] > 0 else float("nan")
        )
        results.append(
            {"Class": i + 1, "Overall Accuracy": class_accuracy, "MIoU": class_iou}
        )

    # Print per-class IoU and accuracy
    for i in range(num_classes):
        print(f"Class {i}: Accuracy = {class_accuracies[i]:.2f}%, MIoU = {ious[i]:.4f}")

    export_results_to_csv(results, "Prueba.csv")

    model.train()  # Switch back to training mode


# Create Dataset and Split into Train and Test Sets
dataset = CustomDataset("../../datasets/zuericrop/dataset")
num_classes = 13
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# Model Configuration
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

# Initialize the TSViT Model
model = TSViT(
    config,
    img_res=24,
    num_channels=[9],
    num_classes=num_classes,
    max_seq_len=142,
    patch_embedding="Channel Encoding",
)
print("Done creating model")

# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore the background class
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Train the Model
train_model(model, train_loader, criterion, optimizer)

# Evaluate the Model on Test Set
evaluate_model(model, test_loader, criterion, num_classes)

# Save the Model
torch.save(model.state_dict(), "tsvit_model.pth")
