"""
models/lcnn_detector.py

Light CNN (LCNN) binary classifier for audio anti-spoofing / voice-clone
detection, built around the Max-Feature-Map (MFM) activation function.

Input:  Log-Mel Spectrogram tensor of shape [batch, 1, 80, time_steps]
         (the output of processing/feature_extractor.py)
Output: A single probability per batch item in [0.0, 1.0], where values
        near 1.0 indicate a cloned/spoofed voice and values near 0.0
        indicate a genuine voice.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MaxFeatureMap2D(nn.Module):
    """
    Max-Feature-Map (MFM) activation for 2D conv feature maps.

    Splits the channel dimension in half and takes the element-wise
    maximum between the two halves, halving the channel count while
    acting as a competitive, sparsity-inducing activation. This is the
    core building block of the Light CNN (LCNN) architecture widely
    used in anti-spoofing / speaker-verification systems.

    Input:  [batch, 2*C, H, W]
    Output: [batch, C, H, W]
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] % 2 != 0:
            raise ValueError(
                f"MaxFeatureMap2D requires an even channel count, got {x.shape[1]}"
            )
        left, right = torch.chunk(x, 2, dim=1)
        return torch.maximum(left, right)


class _ConvMFMBlock(nn.Module):
    """
    A single Conv2D -> MFM -> BatchNorm -> MaxPool2D block.

    The convolution outputs 2x the target channel count so that the
    subsequent MFM activation halves it back down to `out_channels`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        pool_kernel: int = 2,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * 2,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        self.mfm = MaxFeatureMap2D()
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.mfm(x)
        x = self.bn(x)
        x = self.pool(x)
        return x


class LCNNClassifier(nn.Module):
    """
    Lightweight LCNN binary classifier for spoof/clone detection.

    Architecture: 3 Conv2D + MFM + MaxPool2D blocks, followed by an
    adaptive pooling layer (to make the network robust to variable
    `time_steps` input lengths), a flatten, and a final linear layer
    with sigmoid activation producing a single spoof-probability score
    per batch item.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        adaptive_pool_size: tuple[int, int] = (5, 5),
        dropout: float = 0.3,
    ) -> None:
        """
        Args:
            in_channels: Number of input channels (1 for a single
                Log-Mel Spectrogram channel).
            base_channels: Channel width of the first conv block; each
                subsequent block doubles it.
            adaptive_pool_size: Fixed (H, W) the feature map is pooled to
                before flattening, so the classifier head works regardless
                of the input's `time_steps` dimension.
            dropout: Dropout probability applied before the final linear
                layer.
        """
        super().__init__()

        self.block1 = _ConvMFMBlock(in_channels, base_channels)  # 80 -> 40
        self.block2 = _ConvMFMBlock(base_channels, base_channels * 2)  # 40 -> 20
        self.block3 = _ConvMFMBlock(base_channels * 2, base_channels * 4)  # 20 -> 10

        self.adaptive_pool = nn.AdaptiveAvgPool2d(adaptive_pool_size)

        flattened_dim = base_channels * 4 * adaptive_pool_size[0] * adaptive_pool_size[1]

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(flattened_dim, 1)
        self.sigmoid = nn.Sigmoid()

        logger.info(
            "LCNNClassifier initialized (base_channels=%d, adaptive_pool_size=%s)",
            base_channels,
            adaptive_pool_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Float tensor of shape [batch_size, 1, 80, time_steps].

        Returns:
            Float tensor of shape [batch_size, 1] with values in [0.0, 1.0].
            1.0 = predicted cloned/spoofed, 0.0 = predicted genuine.
        """
        if x.dim() != 4:
            raise ValueError(
                f"Expected input of shape [batch, channels, mels, time], got {tuple(x.shape)}"
            )

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.adaptive_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    torch.manual_seed(0)

    model = LCNNClassifier()
    model.eval()

    # Dummy input matching the SpectrogramExtractor output: [1, 1, 80, 201]
    dummy_input = torch.randn(1, 1, 80, 201)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Input shape:  {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Output value: {output.item():.4f}")

    assert output.shape == (1, 1), f"Expected shape (1, 1), got {tuple(output.shape)}"
    assert 0.0 <= output.item() <= 1.0, "Output value out of expected [0, 1] range"
    print("OK — output shape and value range are as expected")