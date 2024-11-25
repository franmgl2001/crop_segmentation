from models.TSViTcls import TSViTcls

model_config = {
        'img_res': 24,
        'patch_size': 3,
        'patch_size_time': 1,
        'patch_time': 4,
        'num_classes': 20,
        'max_seq_len': 16,
        'dim': 128,
        'temporal_depth': 10,
        'spatial_depth': 4,
        'depth': 4,
        'heads': 3,
        'pool': 'cls',
        'num_channels': 14,
        'dim_head': 64,
        'dropout': 0.,
        'emb_dropout': 0.,
        'scale_dim': 4
    }

model = TSViTcls(model_config)