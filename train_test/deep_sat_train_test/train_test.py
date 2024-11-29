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
epochs = 100
log_dir = "./tensorboard_logs"
os.makedirs(log_dir, exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)


def save_checkpoint(epoch, model, optimizer, loss, path):
    """
    Save the model and optimizer state.

    Args:
        epoch (int): Current epoch.
        model: The model to save.
        optimizer: The optimizer to save.
        loss (float): Current evaluation loss.
        path (str): Path to save the checkpoint.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at {path}")


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
        with tqdm(total=len(eval_loader), desc=f"Evaluating Epoch {epoch + 1}") as pbar:
            for samples in eval_loader:
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
                pbar.update(1)
                pbar.set_postfix(train_loss=total_loss)

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
    with tqdm(total=len(train_loader), desc=f"Training Epoch {epoch + 1}") as pbar:
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
            pbar.update(1)
            pbar.set_postfix(train_loss=total_loss)

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Average Training Loss: {avg_train_loss:.4f}")
    writer.add_scalar("Average Training Loss", avg_train_loss, epoch)
    # Save the model checkpoint
    save_checkpoint(
        epoch, model, optimizer, avg_train_loss, f"checkpoints/model_{epoch}.pth"
    )

    # Evaluate the model
    evaluate_model(model, test_loader, criterion, device)
