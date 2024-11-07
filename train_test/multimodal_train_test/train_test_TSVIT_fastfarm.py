import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.TSViT import TSViT
from data_loader.fastfarm.data_loader import CustomDataset
import numpy as np
import os
import csv

MAX_SEQ_LEN = 72

def export_results_to_csv(results, output_csv_path):
    csv_columns = ["Class", "Overall Accuracy", "MIoU"]
    with open(output_csv_path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=csv_columns)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    print(f"Results saved to {output_csv_path}")

def train_model(model, train_loader, criterion, optimizer, num_epochs=50):
    model.train()
    iteration = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            iteration += 1

            B, T, H, W, C = inputs.shape
            time_points = torch.linspace(0, 364, steps=MAX_SEQ_LEN).to(device)
            time_channel = (
                time_points.repeat(B, H, W, 1).permute(0, 3, 1, 2).to(device)
            )

            inputs = torch.cat(
                (inputs, time_channel[:, :, :, :, None]), dim=4
            )
            inputs = inputs.permute(0, 1, 4, 2, 3)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            print(
                f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}",
                end="\r",
            )
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}"
        )
    print("Finished Training")

def compute_iou_per_class(preds, labels, num_classes):
    ious = []
    for cls in range(1, num_classes):  # Skip the background class (0)
        pred_cls = preds == cls
        true_cls = labels == cls
        intersection = np.logical_and(pred_cls, true_cls).sum()
        union = np.logical_or(pred_cls, true_cls).sum()

        if union == 0:
            iou = float("nan")
        else:
            iou = intersection / union
        ious.append(iou)
    return np.array(ious)

def evaluate_model(model, test_loader, criterion, num_classes, csv_filename="predictions_log.csv"):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    correct_per_class = torch.zeros(num_classes).to(device)
    total_per_class = torch.zeros(num_classes).to(device)

    all_preds = []
    all_labels = []

    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Index", "True Label", "Predicted Label"])

        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
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

                _, predicted = torch.max(outputs, 1)
                all_preds.append(predicted.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

                for i in range(labels.size(0)):
                    flattened_labels = labels[i].flatten().cpu().numpy()
                    flattened_predictions = predicted[i].flatten().cpu().numpy()
                    for j in range(len(flattened_labels)):
                        writer.writerow([idx * B + i, flattened_labels[j], flattened_predictions[j]])

                correct += ((predicted == labels) & (labels != 0)).sum().item()
                total += (labels != 0).sum().item()

                for label in range(1, num_classes):  # Skip background class 0
                    correct_per_class[label] += ((predicted == labels) & (labels == label)).sum().item()
                    total_per_class[label] += (labels == label).sum().item()

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    ious = compute_iou_per_class(all_preds, all_labels, num_classes)
    mean_iou = np.nanmean(ious)
    print(f"Mean Intersection over Union (MIoU): {mean_iou:.4f}")

    class_accuracies = 100 * correct_per_class / total_per_class
    mean_accuracy = torch.nanmean(class_accuracies[1:]).item()

    avg_loss = total_loss / len(test_loader)
    overall_accuracy = 100 * correct / total
    print(f"Test Loss: {avg_loss:.4f}, Overall Accuracy: {overall_accuracy:.2f}%")

    results = []
    for i in range(1, num_classes):  # Skip background class 0
        class_iou = ious[i-1]  # Adjust index since we skip 0
        class_accuracy = (
            class_accuracies[i].item() if total_per_class[i] > 0 else float("nan")
        )
        results.append({"Class": i, "Overall Accuracy": class_accuracy, "MIoU": class_iou})

    for i in range(1, num_classes):
        print(f"Class {i}: Accuracy = {class_accuracies[i]:.2f}%, MIoU = {ious[i-1]:.4f}")

    export_results_to_csv(results, "results_5.csv")
    model.train()  # Switch back to training mode

train_dataset = CustomDataset("csvs/fastfarm/train.txt", "../../../datasets/FASTFARM/main_transforms/pickles")
test_dataset = CustomDataset("csvs/fastfarm/test.txt", "../../../datasets/FASTFARM/main_transforms/pickles")

num_classes = 11
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

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
    num_channels=[num_classes],
    num_classes=num_classes,
    max_seq_len=MAX_SEQ_LEN,
    patch_embedding="Channel Encoding",
)
model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore the background class
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, criterion, optimizer)
evaluate_model(model, test_loader, criterion, num_classes)
torch.save(model.state_dict(), "tsvit_model.pth")
