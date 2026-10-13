import torch
from torch import nn
from .iql import build_mlp

ARCH = dict(
    version="three_branch_cnn_v1",
    channels=[32, 32, 64],
    fusion_dim=128,
    head_hidden=[256, 256],
    independent_iql_encoders=True,
    spatial_stride=2,
    temporal_stride=1,
    dropout=0.0,
)
GROUPS = {"G1": 3}


def dimension(group):
    return GROUPS[group] * 6144 + 7


class CNN(nn.Module):

    def __init__(self, group, output_dim):
        super().__init__()
        self.r = GROUPS[group]
        self.n3 = 16 * self.r * 256
        self.n2 = 32 * self.r * 64
        self.out = output_dim
        self.c3 = nn.Sequential(
            nn.Conv3d(16 * self.r, 32, 1),
            nn.SiLU(),
            nn.Conv3d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(32, 64, 3, stride=(1, 2, 2), padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d((2, 2, 2)),
            nn.Flatten(),
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(32 * self.r, 32, 1),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )
        self.fuse = nn.Sequential(nn.Linear(768, 128), nn.SiLU())
        self.head = build_mlp(135, output_dim, 256, 2, 0.0)

    def forward(self, x):
        a = self.c3(x[:, : self.n3].reshape(-1, 16 * self.r, 4, 8, 8))
        b = self.c2(x[:, self.n3 : self.n3 + self.n2].reshape(-1, 32 * self.r, 8, 8))
        y = self.head(torch.cat((self.fuse(torch.cat((a, b), 1)), x[:, -7:]), 1))
        return y.squeeze(-1) if self.out == 1 else y


def encode_normalized(raw, normalizer):
    z = ((raw.float() - normalizer["mean"]) / normalizer["std"]).half()
    return z
