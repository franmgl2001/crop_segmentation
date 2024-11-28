import torch
from loss import MaskedCrossEntropyLoss
from torch.optim import Adam
from models.TSViTdense import TSViT
from data_loader import get_dataloader
from configs.config_1 import config
import tqdm

train_csv_file = "pickle_paths.csv"
test_csv_file = "pickle_paths.csv"
root_dir = "./"
epochs = 10

import torch


class MaskedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, mean=True):
        """
        mean: return mean loss vs per element loss
        """
        super(MaskedCrossEntropyLoss, self).__init__()
        self.mean = mean

    def forward(self, logits, ground_truth):
        """
        Args:
            logits: (N,T,H,W,...,NumClasses)A Variable containing a FloatTensor of size
                (batch, max_len, num_classes) which contains the
                unnormalized probability for each class.
            target: A Variable containing a LongTensor of size
                (batch, max_len) which contains the index of the true
                class for each corresponding step.
            length: A Variable containing a LongTensor of size (batch,)
                which contains the length of each data in a batch.
        Returns:
            loss: An average loss value masked by the length.
        """
        if type(ground_truth) == torch.Tensor:
            target = ground_truth
            mask = None
        elif len(ground_truth) == 1:
            target = ground_truth[0]
            mask = None
        elif len(ground_truth) == 2:
            target, mask = ground_truth
        else:
            raise ValueError(
                "ground_truth parameter for MaskedCrossEntropyLoss is either (target, mask) or (target)"
            )

        if mask is not None:

            mask_flat = mask.reshape(-1, 1)  # (N*H*W x 1)
            nclasses = logits.shape[-1]
            logits_flat = logits.reshape(-1, logits.size(-1))  # (N*H*W x Nclasses)
            print(mask_flat.shape, logits_flat.shape)
            masked_logits_flat = logits_flat[mask_flat.repeat(1, nclasses)].view(
                -1, nclasses
            )

            target_flat = target.reshape(-1, 1)  # (N*H*W x 1)
            masked_target_flat = (
                target_flat[mask_flat].unsqueeze(dim=-1).to(torch.int64)
            )
        else:
            masked_logits_flat = logits.reshape(
                -1, logits.size(-1)
            )  # (N*H*W x Nclasses)
            masked_target_flat = target.reshape(-1, 1).to(torch.int64)  # (N*H*W x 1)
        masked_log_probs_flat = torch.nn.functional.log_softmax(
            masked_logits_flat
        )  # (N*H*W x Nclasses)
        masked_losses_flat = -torch.gather(
            masked_log_probs_flat, dim=1, index=masked_target_flat
        )  # (N*H*W x 1)
        if self.mean:
            return masked_losses_flat.mean()
        return masked_losses_flat


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
            data = batch["data"].to(device)  # Input data
            target = batch["target"].to(device)  # Ground truth labels
            mask = batch.get("mask", None)  # Optional mask (if used)

            # Forward pass
            logits = model(data)

            # Compute loss
            if mask is not None:
                loss = criterion(logits, (target, mask.to(device)))
            else:
                loss = criterion(logits, target)

            # Accumulate loss
            total_loss += loss.item()
            pbar.set_postfix({"Loss": loss.item()})

    # Compute average loss
    avg_eval_loss = total_loss / len(eval_loader)
    print(f"Average Evaluation Loss: {avg_eval_loss:.4f}")
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

    for samples in train_loader:
        print(samples.keys())

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

    # Evaluate the model
    evaluate_model(model, test_loader, criterion, device)
