"""CVPR23 EarthVision-style CNN-LSTM (fastai-compatible nn.Module).

Canonical implementation module. On Google Colab with Drive, prefer:

    from pipeline.cnn_lstm_cvpr23 import CNNLSTM

if `pipeline/cvpr23_cnn_lstm.py` is an empty placeholder (unsynced file).

Input:  (B, nb_features * nb_time_steps, H, W)
Output: (B, H, W) in [0, 1].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CNNLSTM"]


def _norm2d(num_channels: int, norm: str) -> nn.Module:
    """``batch`` = BatchNorm (needs batch size > 1 for stable training). ``group`` = GroupNorm(1, C) (works with batch size 1)."""
    n = norm.lower().strip()
    if n == "batch":
        return nn.BatchNorm2d(num_channels)
    if n == "group":
        return nn.GroupNorm(1, num_channels)
    raise ValueError("norm must be 'batch' or 'group'")


class CNNLSTM(nn.Module):
    def __init__(
        self,
        nb_features: int = 8,
        init_size: int = 24,
        nb_layers: int = 1,
        nb_time_steps: int = 10,
        img_hw: int = 64,
        lstm_hidden: int = 128,
        spatial_output_size: int = 4,
        norm: str = "batch",
    ):
        super().__init__()
        self.nb_features = int(nb_features)
        self.nb_time_steps = int(nb_time_steps)
        self.img_hw = int(img_hw)
        self.norm = norm

        c_in = self.nb_features
        c_out = int(init_size)
        blocks: list[nn.Module] = []
        n_layers = max(0, int(nb_layers))
        for ly in range(n_layers + 1):
            blocks += [
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, padding_mode="reflect"),
                _norm2d(c_out, norm),
                nn.ReLU(inplace=True),
            ]
            if ly < n_layers:
                blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
            c_in = c_out
            c_out = min(c_out * 2, 256)
        blocks.append(nn.AdaptiveAvgPool2d((spatial_output_size, spatial_output_size)))
        self.cnn = nn.Sequential(*blocks)

        cnn_flat = c_in * spatial_output_size * spatial_output_size
        self.lstm = nn.LSTM(
            input_size=cnn_flat,
            hidden_size=int(lstm_hidden),
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.spatial_output_size = spatial_output_size

        self.regressor_linear = nn.Sequential(
            nn.Linear(int(lstm_hidden), int(lstm_hidden) * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(int(lstm_hidden) * 2, spatial_output_size * spatial_output_size),
        )
        self.regressor_conv = nn.Sequential(
            nn.ConvTranspose2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            _norm2d(32, norm),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            _norm2d(16, norm),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        expected_c = self.nb_features * self.nb_time_steps
        if c != expected_c:
            raise ValueError(
                f"Expected {expected_c} input channels (nb_features * nb_time_steps), got {c}"
            )

        t = self.nb_time_steps
        x = x.view(b * t, self.nb_features, h, w)
        feat = self.cnn(x)
        feat = feat.view(b, t, -1)
        lstm_out, _ = self.lstm(feat)
        last = lstm_out[:, -1, :]
        z = self.regressor_linear(last)
        z = z.view(b, 1, self.spatial_output_size, self.spatial_output_size)
        y = self.regressor_conv(z).squeeze(1)

        if y.shape[-2:] != (h, w):
            y = F.interpolate(y.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False).squeeze(
                1
            )
        return y.clamp(0.0, 1.0)
