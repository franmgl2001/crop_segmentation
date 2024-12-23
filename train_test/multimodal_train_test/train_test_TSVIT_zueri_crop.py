import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.TSViT import TSViT
from data_loader.numpy_dataloader_zuericop import CustomDataset, HVFlip
import numpy as np
import os
import csv
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def export_results_to_csv(results, output_csv_path):
    csv_columns = ["Class", "Overall Accuracy", "MIoU"]
    with open(output_csv_path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=csv_columns)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    print(f"Results saved to {output_csv_path}")


# Simplified Training Loop


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=200):
    model.train()  # Set model to training mode

    for epoch in range(num_epochs):
        running_train_loss = 0.0
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        writer = SummaryWriter(log_dir="./logs")  # Create a TensorBoard writer

        # Training Loop
        with tqdm(total=len(train_loader), desc=f"Training Epoch {epoch + 1}") as pbar:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                B, T, H, W, C = inputs.shape

                # Add time channel
                time_points = torch.linspace(0, 364, steps=MAX_SEQ_LEN).to(device)
                time_channel = (
                    time_points.repeat(B, H, W, 1).permute(0, 3, 1, 2).to(device)
                )

                inputs = torch.cat((inputs, time_channel[:, :, :, :, None]), dim=4)
                inputs = inputs.permute(0, 1, 4, 2, 3)

                optimizer.zero_grad()  # Zero the gradients

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Update running loss
                running_train_loss += loss.item()
                pbar.update(1)

            # Update progress bar description
            train_loss = running_train_loss / len(train_loader)
            pbar.set_postfix(train_loss=train_loss)

        # Validation Loop
        model.eval()  # Set model to evaluation mode
        running_val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                B, T, H, W, C = inputs.shape

                # Add time channel
                time_points = torch.linspace(0, 364, steps=MAX_SEQ_LEN).to(device)
                time_channel = (
                    time_points.repeat(B, H, W, 1).permute(0, 3, 1, 2).to(device)
                )

                inputs = torch.cat((inputs, time_channel[:, :, :, :, None]), dim=4)
                inputs = inputs.permute(0, 1, 4, 2, 3)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Accumulate validation loss
                running_val_loss += loss.item()

        val_loss = running_val_loss / len(val_loader)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
        )

        model.train()  # Switch back to training mode after validation
        # Log losses to TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch + 1)
        writer.add_scalar("Loss/Validation", val_loss, epoch + 1)
        checkpoint_path = os.path.join(
            checkpoint_dir, f"checkpoint_epoch_drop_0.1_{epoch + 1}.pth"
        )
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved at {checkpoint_path}")

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


def evaluate_model(
    model, test_loader, criterion, num_classes, csv_filename="predictions_log.csv"
):
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

    # Open CSV file to log predictions and labels
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Index", "True Label", "Predicted Label"])  # CSV headers

        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                print(inputs.shape, labels.shape)

                B, T, H, W, C = inputs.shape
                time_points = torch.linspace(0, 364, steps=MAX_SEQ_LEN).to(device)
                time_channel = (
                    time_points.repeat(B, H, W, 1).permute(0, 3, 1, 2).to(device)
                )

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

                # Flatten the labels and predictions before logging them
                for i in range(labels.size(0)):  # Iterate through batch size
                    flattened_labels = labels[i].flatten().cpu().numpy()
                    flattened_predictions = predicted[i].flatten().cpu().numpy()

                    # Log all pixels for this image
                    for j in range(len(flattened_labels)):
                        writer.writerow(
                            [idx * B + i, flattened_labels[j], flattened_predictions[j]]
                        )

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

    # Prepare data for CSV export of accuracy and IoU results
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

    if csv_filename:
        export_results_to_csv(results, "results_5.csv")

    model.train()  # Switch back to training mode


augmentations = HVFlip(hflip_prob=0.5, vflip_prob=0.5, ground_truths=["labels"])


# Create Dataset and Split into Train and Test Sets
train_dataset = CustomDataset(
    "csvs/train_zuericrop_11.txt",
    "../../../datasets/zuericrop/dataset",
    augmentations=augmentations,
)
test_dataset = CustomDataset(
    "csvs/test_zuericrop_11.txt", "../../../datasets/zuericrop/dataset"
)

num_classes = 10
MAX_SEQ_LEN = 71
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_dir = "check_points"

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=24)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

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
    "dropout": 0.1,
    "emb_dropout": 0.1,
    "scale_dim": 4,
    "depth": 4,
}

# Initialize the TSViT Model
model = TSViT(
    config,
    img_res=24,
    num_channels=[9],
    num_classes=num_classes,
    max_seq_len=MAX_SEQ_LEN,
    patch_embedding="Channel Encoding",
)
print("Done creating model")


# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()  # Ignore the background class
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Set device

model.to(device)

# Train the Model
train_model(model, train_loader, test_loader, criterion, optimizer)

# Evaluate the Model on Test Set
evaluate_model(model, test_loader, criterion, num_classes)

# Save the Model
torch.save(model.state_dict(), "tsvit_model.pth")
