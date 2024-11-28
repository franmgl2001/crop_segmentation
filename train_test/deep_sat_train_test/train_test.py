import torch
from loss.loss import MaskedCrossEntropyLoss
from torch.optim import Adam
from models.TSViTdense import TSViT
from data_loader.data_loader import get_dataloader
from configs.config_1 import config
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os


train_csv_file = "pickle_paths.csv"
test_csv_file = "pickle_paths.csv"
root_dir = "./"
epochs = 10
log_dir = "./tensorboard_logs"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)


def evaluate_model(model, eval_loader, criterion, device):
    """
    Evaluate the model on the evaluation dataset.

    Args:
        model: The trained model to evaluate.
        eval_loader: DataLoader for the evaluation dataset.
        criterion: Loss function to compute evaluation loss.
        device: Device to run the evaluation on (CPU or GPU).

    Returns:
        avg_eval_loss: Average evaluation loss over the dataset.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0

    with torch.no_grad():  # Disable gradient calculation for evaluation
        pbar = tqdm(eval_loader, desc="Evaluating")
        for batch in pbar:
            # Retrieve data from the batch
            data = samples["inputs"].to(device)
            target = samples["labels"].to(device)
            mask = samples["unk_masks"].to(device)

            # Forward pass
            logits = model(data)
            logits = logits.permute(0, 2, 3, 1)

            # Compute loss
            loss = criterion(logits, (target, mask.to(device)))

            # Accumulate loss
            total_loss += loss.item()
            pbar.set_postfix({"Loss": loss.item()})

    # Compute average loss
    avg_eval_loss = total_loss / len(eval_loader)
    print(f"Average Evaluation Loss: {avg_eval_loss:.4f}")
    writer.add_scalar("Average Validation Loss", avg_eval_loss, epoch)

    return avg_eval_loss


criterion = MaskedCrossEntropyLoss(mean=True)
model = TSViT(config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = Adam(
    model.parameters(),
    lr=config["lr_base"],
    weight_decay=config["weight_decay"],
)

train_loader = get_dataloader(train_csv_file, root_dir, config)
test_loader = get_dataloader(test_csv_file, root_dir, config)

model.to(device)
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")

    for samples in train_loader:

        data = samples["inputs"].to(device)
        target = samples["labels"].to(device)
        mask = samples["unk_masks"].to(device)

        # Forward pass
        logits = model(data)
        logits = logits.permute(0, 2, 3, 1)
        # Compute loss
        loss = criterion(logits, (target, mask))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Average Training Loss: {avg_train_loss:.4f}")
    writer.add_scalar("Average Training Loss", avg_train_loss, epoch)

    # Evaluate the model
    evaluate_model(model, test_loader, criterion, device)
