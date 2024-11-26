from models.TSViTdense import TSViT
import torch

model_config ={
    'img_res': 24,
    'patch_size': 2,
    'patch_size_time': 1,
    'patch_time': 4,
    'num_classes': 19,
    'max_seq_len': 60,
    'dim': 64,
    'temporal_depth': 4,
    'spatial_depth': 4,
    'depth': 4,
    'heads': 4,
    'pool': 'cls',
    'num_channels': 11,
    'dim_head': 16,
    'dropout': 0.,
    'emb_dropout': 0.,
    'scale_dim': 4
}


model = TSViT(model_config)

model.load_state_dict(
    torch.load("best.pth", map_location=torch.device("cpu"))
)


