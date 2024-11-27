import torch.optim.lr_scheduler as lr_scheduler
from loss import MaskedCrossEntropyLoss
from torch.optim import Adam
from models.TSViTdense import TSViT

training_config = {
    "num_epochs": 100,
    "num_warmup_epochs": 10,
    "steps": (0, 80000),
    "loss_function": "masked_cross_entropy",
    "lr_scheduler": "cosine",
    "lr_base": 1e-3,
    "lr_min": 5e-6,
    "lr_start": 1e-8,
    "num_cycles": 1,
    "reset_lr": True,
    "weight_decay": 0.000,
}

transforms_config = {
    "img_res": 24,
    "labels": 19,
    "max_seq_len": 60,
}
model_config = {
    "img_res": 24,
    "patch_size": 2,
    "patch_size_time": 1,
    "patch_time": 4,
    "num_classes": 19,
    "max_seq_len": 60,
    "dim": 128,
    "temporal_depth": 4,
    "spatial_depth": 4,
    "depth": 4,
    "heads": 4,
    "pool": "cls",
    "num_channels": 11,
    "dim_head": 32,
    "dropout": 0.0,
    "emb_dropout": 0.0,
    "scale_dim": 4,
}

criterion = MaskedCrossEntropyLoss(mean=True)
model = TSViT(model_config)


optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=0.000)


scheduler = lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=config["steps"][-1], eta_min=config["lr_min"]
)
