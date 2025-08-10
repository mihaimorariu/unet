"""Implementation of the UNet architecture: https://arxiv.org/abs/1505.04597"""

import torch
from torch import Tensor
from torch.nn import (
    Conv2d,
    ConvTranspose2d,
    Module,
    ModuleList,
)
from torch.nn.functional import interpolate, max_pool2d, relu
from torchvision.transforms import CenterCrop

# pylint: disable=missing-function-docstring, invalid-name


class Block(Module):
    """A basic block for the UNet architecture consisting of two convolutional
    layers with ReLU activation."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3)

    def forward(self, x: Tensor) -> Tensor:
        h = relu(self.conv1(x))
        h = relu(self.conv2(h))

        return h


class Encoder(Module):
    """Encoder block for the UNet architecture."""

    def __init__(self, channels: list[int] | None = None) -> None:
        super().__init__()

        if channels is None:
            channels = [64, 128, 256, 512, 1024]

        channels = [3] + channels  # Input channel is 3 for RGB images

        self.blocks = ModuleList(
            Block(c_prev, c_next) for c_prev, c_next in zip(channels[:-1], channels[1:])
        )

    def forward(self, x: Tensor) -> list[Tensor]:
        features = []
        h = x

        for block in self.blocks:
            h = block(h)
            features.append(h)
            h = max_pool2d(h, kernel_size=2)

        return features


class Decoder(Module):
    """Decoder block for the UNet architecture."""

    def __init__(self, channels: list[int] | None = None) -> None:
        super().__init__()

        if channels is None:
            channels = [1024, 512, 256, 128, 64]

        self.upconv = ModuleList(
            ConvTranspose2d(c_prev, c_next, kernel_size=2, stride=2)
            for c_prev, c_next in zip(channels[:-1], channels[1:])
        )
        self.blocks = ModuleList(
            Block(c_prev, c_next) for c_prev, c_next in zip(channels[:-1], channels[1:])
        )

    def forward(self, features: list[Tensor]) -> Tensor:
        if not features:
            raise ValueError("Features list is empty!")

        x = features[0]

        for upconv, block, feature in zip(self.upconv, self.blocks, features[1:]):
            x = upconv(x)
            height, width = x.shape[-2:]
            x = torch.cat([x, self.crop(feature, (height, width))], dim=1)
            x = block(x)

        return x

    def crop(self, x: Tensor, target_size: tuple[int, int]) -> Tensor:
        target_height, target_width = target_size
        _, _, height, width = x.shape

        if height == target_height and width == target_width:
            return x

        # Center crop the feature map
        crop = CenterCrop((target_height, target_width))
        return crop(x)


class UNet(Module):
    """UNet architecture for image segmentation."""

    def __init__(self, num_classes: int, channels: list[int] | None = None) -> None:
        super().__init__()

        if channels is None:
            channels = [64, 128, 256, 512, 1024]

        self.encoder = Encoder(channels)
        self.decoder = Decoder(channels[::-1])
        self.head = Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]

        features = self.encoder(x)
        decoded = self.decoder(features[::-1])

        out = self.head(decoded)
        out = interpolate(out, size=(height, width))

        return out
