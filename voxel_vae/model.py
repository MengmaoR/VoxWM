"""Protocol-frozen E00 3D ResNet VAE."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResidualBlock3D


class VoxelVAE(nn.Module):
    ARCHITECTURE = "voxel-resnet3d-vae-e00-v1"

    def __init__(self, vocabulary_size, embedding_dim=32, latent_channels=16,
                 base_channels=96, hidden_channels=192, logvar_min=-10.0,
                 logvar_max=10.0):
        super().__init__()
        if embedding_dim != 32 or latent_channels != 16:
            raise ValueError("E00 fixes embedding_dim=32 and latent_channels=16")
        self.vocabulary_size = int(vocabulary_size)
        self.embedding_dim = int(embedding_dim)
        self.latent_channels = int(latent_channels)
        self.base_channels = int(base_channels)
        self.hidden_channels = int(hidden_channels)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)

        self.embedding = nn.Embedding(self.vocabulary_size, self.embedding_dim)
        self.encoder_input = nn.Conv3d(32, base_channels, 3, padding=1)
        self.encoder_level16 = nn.Sequential(
            ResidualBlock3D(base_channels), ResidualBlock3D(base_channels))
        self.downsample16_8 = nn.Conv3d(base_channels, hidden_channels, 4, 2, 1)
        self.encoder_level8 = nn.Sequential(
            ResidualBlock3D(hidden_channels), ResidualBlock3D(hidden_channels))
        self.downsample8_4 = nn.Conv3d(hidden_channels, hidden_channels, 4, 2, 1)
        self.encoder_level4 = nn.Sequential(
            ResidualBlock3D(hidden_channels), ResidualBlock3D(hidden_channels))
        self.moments_hidden = nn.Conv3d(hidden_channels, 64, 1)
        self.moments_output = nn.Conv3d(64, 2 * latent_channels, 1)

        self.decoder_input = nn.Conv3d(latent_channels, 64, 1)
        self.decoder_hidden = nn.Conv3d(64, hidden_channels, 3, padding=1)
        self.decoder_level4 = nn.Sequential(
            ResidualBlock3D(hidden_channels), ResidualBlock3D(hidden_channels))
        self.decoder_up8 = nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1)
        self.decoder_level8 = nn.Sequential(
            ResidualBlock3D(hidden_channels), ResidualBlock3D(hidden_channels))
        self.decoder_up16 = nn.Conv3d(hidden_channels, base_channels, 3, padding=1)
        self.decoder_level16 = nn.Sequential(
            ResidualBlock3D(base_channels), ResidualBlock3D(base_channels))
        self.decoder_features = nn.Conv3d(base_channels, 32, 3, padding=1)
        self.decoder_logits = nn.Conv3d(32, self.vocabulary_size, 1)

    def architecture_config(self):
        return {
            "name": self.ARCHITECTURE,
            "vocabulary_size": self.vocabulary_size,
            "embedding_dim": self.embedding_dim,
            "encoder_channels": [self.base_channels, self.hidden_channels,
                                  self.hidden_channels],
            "latent_shape": [self.latent_channels, 4, 4, 4],
            "decoder_channels": [self.hidden_channels, self.hidden_channels,
                                  self.base_channels, 32],
            "logvar_clamp": [self.logvar_min, self.logvar_max],
            "skip_connections": False,
        }

    def embed(self, tokens):
        if tokens.ndim != 4 or tuple(tokens.shape[1:]) != (16, 16, 16):
            raise ValueError("Expected token IDs [B,16,16,16]")
        if tokens.dtype != torch.long:
            tokens = tokens.long()
        if torch.any(tokens < 0) or torch.any(tokens >= self.vocabulary_size):
            raise ValueError("Token outside canonical vocabulary")
        return self.embedding(tokens).permute(0, 4, 1, 2, 3).contiguous()

    def encode(self, tokens):
        value = self.encoder_input(self.embed(tokens))
        value = self.encoder_level16(value)
        value = self.encoder_level8(self.downsample16_8(value))
        value = self.encoder_level4(self.downsample8_4(value))
        moments = self.moments_output(F.silu(self.moments_hidden(value)))
        mu, logvar = moments.chunk(2, dim=1)
        return mu, logvar.clamp(self.logvar_min, self.logvar_max)

    def encode_mean(self, tokens):
        return self.encode(tokens)[0]

    def sample(self, mu, logvar):
        if not self.training:
            return mu
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, latent):
        if latent.ndim != 5 or tuple(latent.shape[1:]) != (16, 4, 4, 4):
            raise ValueError("Expected latent [B,16,4,4,4]")
        value = self.decoder_hidden(self.decoder_input(latent))
        value = self.decoder_level4(value)
        value = F.interpolate(value, scale_factor=2, mode="nearest")
        value = self.decoder_level8(self.decoder_up8(value))
        value = F.interpolate(value, scale_factor=2, mode="nearest")
        value = self.decoder_level16(self.decoder_up16(value))
        value = self.decoder_features(value)
        return self.decoder_logits(value)

    def forward(self, tokens):
        mu, logvar = self.encode(tokens)
        latent = self.sample(mu, logvar)
        return {"logits": self.decode(latent), "mu": mu,
                "logvar": logvar, "z": latent}
