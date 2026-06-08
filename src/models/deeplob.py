"""Faithful DeepLOB implementation (Zhang, Zohren & Roberts, IEEE T-SP 2019).

Input  : (batch, 1, 100, 40)  -- 100 time steps, 40 LOB features (10 levels x [ask p, ask v, bid p, bid v]).
Output : (batch, 3)           -- softmax over {down, stationary, up}.

Notes
-----
- The (1,2)-stride convolutions compress the 40-wide feature axis 40 -> 20 -> 10, and a final (1,10) conv
  collapses it to width 1, mirroring the paper.
- The six (4,1) convolutions (no time padding) shrink the time axis; the LSTM handles the resulting length.
- The Inception module uses time-preserving padding so its branches concatenate cleanly along channels.
Exact paddings can be tuned; this version is dimensionally consistent for width-40 input.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, width_kernel: int, width_stride: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(1, width_kernel), stride=(1, width_stride)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Inception(nn.Module):
    """Multi-scale temporal module; time dimension preserved via padding."""

    def __init__(self, in_ch: int, out_ch: int = 64):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (1, 1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, (3, 1), padding=(1, 0)), nn.LeakyReLU(0.01), nn.BatchNorm2d(out_ch),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (1, 1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, (5, 1), padding=(2, 0)), nn.LeakyReLU(0.01), nn.BatchNorm2d(out_ch),
        )
        self.b3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_ch, out_ch, (1, 1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)  # channels concat


class DeepLOB(nn.Module):
    def __init__(self, n_classes: int = 3, conv_ch: int = 32, lstm_hidden: int = 64, inception_ch: int = 64):
        super().__init__()
        self.block1 = _ConvBlock(1, conv_ch, width_kernel=2, width_stride=2)        # 40 -> 20
        self.block2 = _ConvBlock(conv_ch, conv_ch, width_kernel=2, width_stride=2)  # 20 -> 10
        self.block3 = _ConvBlock(conv_ch, conv_ch, width_kernel=10, width_stride=1) # 10 -> 1
        self.inception = _Inception(conv_ch, inception_ch)
        self.lstm = nn.LSTM(input_size=inception_ch * 3, hidden_size=lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T, 40)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)                 # (B, C, T', 1)
        x = self.inception(x)              # (B, 3*inception_ch, T', 1)
        x = x.squeeze(-1)                  # (B, 3*inception_ch, T')
        x = x.permute(0, 2, 1)             # (B, T', 3*inception_ch)
        out, _ = self.lstm(x)              # (B, T', H)
        last = out[:, -1, :]               # last time step
        return self.fc(last)              # (B, n_classes)


if __name__ == "__main__":
    model = DeepLOB()
    dummy = torch.randn(8, 1, 100, 40)
    y = model(dummy)
    print("output shape:", tuple(y.shape))  # expect (8, 3)
