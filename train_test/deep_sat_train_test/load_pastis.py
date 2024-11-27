from models.TSViTdense import TSViT
import torch
from transforms import PASTIS_segmentation_transform
import pickle


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


model = TSViT(model_config)

model.load_state_dict(torch.load("best.pth", map_location=torch.device("cpu")))


transform_pipeline = PASTIS_segmentation_transform(model_config, True)

# Get pickle file from the dataset
sample = pickle.load(open("pickles/10110_15.pickle", "rb"))

transformed_sample = transform_pipeline(sample)


transformed_sample["inputs"] = transformed_sample["inputs"].unsqueeze(0)
model.eval()
with torch.no_grad():
    output = model(transformed_sample["inputs"])

print(output.shape)


# Make an arg max to get the class
output = torch.argmax(output, dim=2)

print(output.shape)
