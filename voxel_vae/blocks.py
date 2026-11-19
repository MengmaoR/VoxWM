"""Small 3D residual building blocks used by E00."""
import torch.nn as nn


def _groups(channels):
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = int(out_channels or in_channels)
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.projection = (nn.Identity() if in_channels == out_channels else
                           nn.Conv3d(in_channels, out_channels, 1))

    def forward(self, value):
        residual = self.projection(value)
        value = self.conv1(self.act1(self.norm1(value)))
        value = self.conv2(self.act2(self.norm2(value)))
        return value + residual
