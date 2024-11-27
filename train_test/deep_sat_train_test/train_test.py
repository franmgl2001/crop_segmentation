import torch.optim.lr_scheduler as lr_scheduler
from loss import MaskedCrossEntropyLoss
from torch.optim import Adam
from models.TSViTdense import TSViT
import data_loader
from torch.utils.data import DataLoader
from configs.config_1 import config


criterion = MaskedCrossEntropyLoss(mean=True)
model = TSViT(config)

train_dataset = data_loader.get_dataset(split="train", config=config)
val_dataset = data_loader.get_dataset(split="val", config=config)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)


optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=0.000)


scheduler = lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=config["steps"][-1], eta_min=config["lr_min"]
)
