from typing import Optional
import torch.nn as nn
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

        self.emb = nn.Embedding(audio_embedding_dim, audio_embedding_dim)

        layers = [
            nn.Conv2d(3, 32, 3, 2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, 2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 2, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 3, 2, padding=1),
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Sigmoid()
        ]

        self.model = torch.nn.Sequential(*layers)

        # out_channels //= 2  # Output channels of the last Conv2d
        # img_dim = out_channels * (ds_size ** 2)

        # in_features = layers[-3].out_channels + audio_embedding_dim
        layers = [
            nn.Linear(in_features=32+audio_embedding_dim, out_features=1),
            nn.Sigmoid()
        ]
        self.adv_layer = torch.nn.Sequential(*layers)

    def forward(self, img: torch.Tensor, audio_embedding: torch.Tensor) -> torch.Tensor:
        transformed_img = transform_img_for_disc(img)
        output = self.model(transformed_img)
        output = output.reshape(output.shape[0], output.shape[1]*output.shape[2]*output.shape[3])  # Flatten elements in the batch
        audio_embedding = self.emb(audio_embedding.argmax(dim=1))
        inp = torch.cat((output, audio_embedding), dim=1)
        validity = self.adv_layer(inp)
        assert not torch.any(torch.isnan(validity))
        return validity
