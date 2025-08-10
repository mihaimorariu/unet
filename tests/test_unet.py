# pylint: disable=missing-function-docstring, missing-module-docstring

import torch
from torch import Tensor
from unet import Encoder, Decoder, UNet


def test_encoder():
    encoder = Encoder()

    x = torch.randn(1, 3, 572, 572)
    features = encoder(x)

    assert isinstance(features, list)
    assert len(features) == 5

    expected_shapes = [
        (1, 64, 568, 568),
        (1, 128, 280, 280),
        (1, 256, 136, 136),
        (1, 512, 64, 64),
        (1, 1024, 28, 28),
    ]

    for shape, feature in zip(expected_shapes, features):
        assert isinstance(feature, Tensor)
        assert feature.shape == shape


def test_decoder():
    decoder = Decoder()

    features = [
        torch.randn(1, 1024, 28, 28),
        torch.randn(1, 512, 64, 64),
        torch.randn(1, 256, 136, 136),
        torch.randn(1, 128, 280, 280),
        torch.randn(1, 64, 568, 568),
    ]
    x = decoder(features)

    assert isinstance(x, Tensor)
    assert x.shape == (1, 64, 388, 388)


def test_unet():
    unet = UNet(num_classes=10)

    x = torch.rand(1, 3, 572, 572)
    y = unet(x)

    assert isinstance(y, Tensor)
    assert y.shape == (1, 10, 572, 572)
