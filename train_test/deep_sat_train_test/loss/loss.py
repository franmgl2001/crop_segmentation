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
