from data_loader import SatImDataset
from torch.utils.data import DataLoader


test_dataset = SatImDataset(
    csv_file=csv_file,
    root_dir=root_dir,
    transform=test_transforms,
    multilabel=multilabel,
    return_paths=return_paths,
)

dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    collate_fn=my_collate,  # Custom collate function
)

for i, data in enumerate(dataloader):
    print(data)
    if i == 0:
        break
