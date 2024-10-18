import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from cheap_transforms_pastis import process_image


class CropDataset(Dataset):
    def __init__(self, image_ids, seq_len=37):
        self.image_ids = image_ids
        self.seq_len = seq_len


    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        return process_image(image_id, self.seq_len, "../../../../datasets/PASTIS/PASTIS-R")

def load_dataset(json_path, batch_size=8, shuffle=True, seq_len=37):
    with open(json_path, 'r') as f:
        data_split = json.load(f)
    
    train_dataset = CropDataset(data_split["train"], seq_len=seq_len)
    test_dataset = CropDataset(data_split["test"],  seq_len=seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Example usage
# metadata = load_your_metadata_function()  # Load your metadata here
#train_loader, test_loader = load_dataset('train_test_split.json')
