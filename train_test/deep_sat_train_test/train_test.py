import torch.optim.lr_scheduler as lr_scheduler
from loss import MaskedCrossEntropyLoss
from torch.optim import Adam
from models.TSViTdense import TSViT
import data_loader
from torch.utils.data import DataLoader


criterion = MaskedCrossEntropyLoss(mean=True)
model = TSViT(model_config)

train_dataset = data_loader.get_dataset(split="train", config=model_config)
val_dataset = data_loader.get_dataset(split="val", config=model_config)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)


optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=0.000)


scheduler = lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=model_config["steps"][-1], eta_min=model_config["lr_min"]
)
