from typing import Optional

import torch
from torchvision.transforms.functional import gaussian_blur

from ..emotions import Emotion


def transform_img_for_disc(img_tensor: torch.Tensor) -> torch.Tensor:
    return gaussian_blur(img_tensor, kernel_size=3)


class Discriminator(torch.nn.Module):

    def __init__(self, canvas_size: int, audio_embedding_dim: int,
                 num_conv_layers: int, num_linear_layers: int):
        super(Discriminator, self).__init__()

        layers = []
        in_channels = 3  # RGB
        out_channels = in_channels * 8
        double_channels = True
        conv_kernel_size = 6
        conv_stride = 4
        conv_padding = 1
        ds_size = canvas_size

        for i in range(num_conv_layers):
            # Dimensions of the downsampled image
            ds_size = (ds_size + 2 * conv_padding - conv_kernel_size) // conv_stride + 1

            layers += [
                torch.nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=conv_kernel_size,
                    stride=conv_stride,
                    padding=conv_padding,
                    bias=False
                ),
                torch.nn.LayerNorm([ds_size, ds_size]),
                # torch.nn.LeakyReLU(0.1),
                torch.nn.ELU(),
            ]

            in_channels = out_channels
            if double_channels:
                out_channels *= 2
                double_channels = False
            else:
                double_channels = True
            conv_kernel_size = max(conv_kernel_size - 1, 3)
            conv_stride = max(conv_stride - 1, 2)

        self.model = torch.nn.Sequential(*layers)

        out_channels //= 2  # Output channels of the last Conv2d
        img_dim = out_channels * (ds_size ** 2)

        in_features = img_dim + audio_embedding_dim
        layers = []
        for i in range(num_linear_layers - 1):
            out_features = in_features // 64
            layers += [
                torch.nn.Linear(in_features=in_features, out_features=out_features),
                # torch.nn.LeakyReLU(0.2),
                torch.nn.ELU(),
                # torch.nn.Dropout2d(0.2)
            ]
            in_features = out_features
        layers += [
            torch.nn.Linear(in_features=in_features, out_features=1)
        ]
        self.adv_layer = torch.nn.Sequential(*layers)

    def forward(self, img: torch.Tensor, audio_embedding: torch.Tensor) -> torch.Tensor:
        transformed_img = transform_img_for_disc(img)
        output = self.model(transformed_img)
        output = output.reshape(output.shape[0], output.shape[1]*output.shape[2]*output.shape[3])  # Flatten elements in the batch
        inp = torch.cat((output, audio_embedding), dim=1)
        validity = self.adv_layer(inp)
        assert not torch.any(torch.isnan(validity))
        return validity
