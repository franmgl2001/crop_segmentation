"""
TSViT Implementation
Adapted from: https://github.com/michaeltrs/DeepSatModels
Authors: Michail Tarasiou and Erik Chavez and Stefanos Zafeiriou
License:  Apache License 2.0
"""

import sys

sys.path.append("..")
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
import pytorch_lightning as pl
from train_test.models.patch_embeddings import StandardPatchEmbedding


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)
        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)

        return out


class TSViT(nn.Module):
    def __init__(
        self,
        model_config,
        img_res=80,
        num_channels=[4],
        num_classes=16,
        max_seq_len=37,
        patch_embedding="Standard",
    ):  # model config must contain: patch_size=2, d=128, temporal_depth=6, spatial_depth=2, n_heads=4, dim_head=64, dropout=0., emb_dropout=0., scale_dim=4
        """ """
        super().__init__()
        self.name = "TSViT"

        self.patch_size = model_config["patch_size"]  # p1=p2
        self.num_patches_1d = img_res // (self.patch_size)  # h=w
        self.num_classes = num_classes  # K
        self.dim = model_config["dim"]  # d

        if "temporal_depth" in model_config:
            self.temporal_depth = model_config["temporal_depth"]
        else:
            self.temporal_depth = model_config["depth"]
        if "spatial_depth" in model_config:
            self.spatial_depth = model_config["spatial_depth"]
        else:
            self.spatial_depth = model_config["depth"]

        # transformer encoder parameters
        self.heads = model_config["heads"]
        self.dim_head = model_config["dim_head"]
        self.dropout = model_config["dropout"]
        self.emb_dropout = model_config["emb_dropout"]
        self.scale_dim = model_config["scale_dim"]

        self.patch_embedding = patch_embedding
        self.num_modalities = len(num_channels)
        self.to_patch_embedding = StandardPatchEmbedding(
            self.dim, num_channels, self.patch_size
        )

        # temporal position encoding: project temp position one hot encoding [0,365] to d, this is then added to the tokens
        # learn d-dim vector for each time point
        self.to_temporal_embedding_input = nn.Linear(366, self.dim)
        # temporal class token 1xKxd
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        # temporal encoder
        self.temporal_transformer = Transformer(
            self.dim,
            self.temporal_depth,
            self.heads,
            self.dim_head,
            self.dim * self.scale_dim,
            self.dropout,
        )

        # spatial position embedding: 1x(h*w)xd
        self.space_pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches_1d**2, self.dim)
        )
        # spatial encoder
        self.space_transformer = Transformer(
            self.dim,
            self.spatial_depth,
            self.heads,
            self.dim_head,
            self.dim * self.scale_dim,
            self.dropout,
        )

        self.dropout = nn.Dropout(self.emb_dropout)
        # project back from d to p1xp2 2 dimensional patch
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim), nn.Linear(self.dim, self.patch_size**2)
        )

    def forward(self, x):
        B, T, _, H, W = (
            x.shape
        )  # B=batch size, T=temporal dimension, C=channel dimension, H,W=spatial dimensions

        xt = x[
            :, :, -1, 0, 0
        ]  # BxTx1 = time step: last band in channel dimension encodes the timestep as value in [0,365], e.g. if image has 10 spectral bands, there is an 11th band with only the timestep value everywhere
        x = x[:, :, :-1]  # BxTxCxHxW C=only bands

        # 1. PATCH EMBEDDING
        x = self.to_patch_embedding(x)

        # 2. TEMPORAL ENCODING
        print(xt)
        xt = xt.to(torch.int64)
        print(xt)
        xt = F.one_hot(xt, num_classes=366).to(
            torch.float32
        )  # make into one hot encoding BxTx366
        print(xt.shape)
        xt = xt.reshape(-1, 366)  # (B*T)x365
        temporal_pos_embedding = self.to_temporal_embedding_input(xt).reshape(
            B, T, self.dim
        )
        if self.patch_embedding == "Modality Concatenation":
            temporal_pos_embedding = temporal_pos_embedding.repeat(
                1, self.num_modalities, 1
            )  # repeat for each modality

        num_temporal_tokens = temporal_pos_embedding.shape[1]
        x = x.reshape(B, -1, num_temporal_tokens, self.dim)

        x += temporal_pos_embedding.unsqueeze(1)
        x = x.reshape(-1, num_temporal_tokens, self.dim)

        cls_temporal_tokens = repeat(
            self.temporal_token, "() K d -> b K d", b=B * self.num_patches_1d**2
        )
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x = self.temporal_transformer(x)
        x = x[:, : self.num_classes]

        # 3. SPATIAL ENCODER
        x = (
            x.reshape(B, self.num_patches_1d**2, self.num_classes, self.dim)
            .permute(0, 2, 1, 3)
            .reshape(B * self.num_classes, self.num_patches_1d**2, self.dim)
        )
        x += self.space_pos_embedding
        x = self.space_transformer(self.dropout(x))

        # 4. TO PIXEL PROBABILITIES
        x = rearrange(
            x,
            "(b k) (h w) d-> (b k h w) d",
            h=self.num_patches_1d,
            k=self.num_classes,
            w=self.num_patches_1d,
        )
        x = self.mlp_head(x)

        # assemble original image extent HxW from amount of patches (hxw) and patch size (p1xp2)
        x = rearrange(
            x,
            "(b k h w) (p1 p2) -> b k (h p1) (w p2)",
            h=self.num_patches_1d,
            k=self.num_classes,
            w=self.num_patches_1d,
            p1=self.patch_size,
        )
        return x


import os

if __name__ == "__main__":
    pl.seed_everything(35)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    res = 24  # has to be divisible by patch_size
    max_seq_len = 37  # has to be divisible by patch_size_time
    channels = [8]
    num_classes = 16
    batch_size = 10
    patch_size = 2
    print(
        f"creating {(res/patch_size)} * {(res/patch_size)} = {(res/patch_size)**2} patches of size {patch_size} x {patch_size}"
    )

    x = torch.rand((batch_size, max_seq_len, res, res, sum(channels)))  # BxTxHxWxC
    B, T, H, W, C = x.shape

    print("Shape of x :", x.shape)

    # add channel that contains time steps
    time_points = torch.randint(low=0, high=365, size=(max_seq_len,))
    print("time points", time_points)
    time_channel = time_points.repeat(B, H, W, 1).permute(0, 3, 1, 2)  # BxTxHxW
    x = torch.cat(
        (x, time_channel[:, :, :, :, None]), dim=4
    )  # BxTxHxWxC + BxTxHxWx1 = BxTx(C+1)xHxW
    # last layer should contain only the value of the timestep for fixed T
    for t in range(T):
        assert (
            int(np.unique(x[:, t, :, :, -1].numpy(), return_counts=True)[0][0])
            == time_points.numpy()[t]
        )

    model_config = {
        "patch_size": patch_size,
        "patch_size_time": 1,
        "patch_time": 4,
        "dim": 128,
        "temporal_depth": 6,
        "spatial_depth": 2,
        "channel_depth": 4,
        "heads": 4,
        "dim_head": 64,
        "dropout": 0.0,
        "emb_dropout": 0.0,
        "scale_dim": 4,
        "depth": 4,
    }
    print(f"model configuration: {channels} input dim, {max_seq_len} sequencelength")
    print(channels)
    model = TSViT(
        img_res=res,
        num_channels=channels,
        model_config=model_config,
        num_classes=num_classes,
        patch_embedding="Channel Encoding",
    )

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print("Trainable Parameters: %.3fM" % parameters)

    x = x.permute(0, 1, 4, 2, 3)

    print("\ninput", x.shape, "\n")
    out = model(x)

    print("total memory usage", torch.cuda.memory_allocated(x.device) * 1e-06, "Mb")

    print("Shape of out :", out.shape)  # [B, num_classes, H, W]
    print("Trainable Parameters: %.3fM" % parameters)
