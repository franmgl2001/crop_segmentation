from data_loader import SatImDataset
from torch.utils.data import DataLoader
from transforms import PASTIS_segmentation_transform
from configs.config_1 import config


test_dataset = SatImDataset(
    csv_file="pickle_paths.csv",
    root_dir="./",
    transform=PASTIS_segmentation_transform(config, True),
)

dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=config["batch_size"],
    shuffle=True,
)

for i, data in enumerate(dataloader):
    print(data["inputs"].shape)
