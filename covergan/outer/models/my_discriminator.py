from typing import Optional

import torch
import torch.nn as nn
from torchvision.transforms.functional import gaussian_blur

from ..emotions import Emotion


def transform_img_for_disc(img_tensor: torch.Tensor) -> torch.Tensor:
    return gaussian_blur(img_tensor, kernel_size=29)


class MyDiscriminator(torch.nn.Module):
    def __init__(self, canvas_size: int, audio_embedding_dim: int, has_emotions: bool,
                 num_conv_layers: int, num_linear_layers: int):
        super(MyDiscriminator, self).__init__()

        layers = []

        self.emb = nn.Embedding(audio_embedding_dim, audio_embedding_dim)

        layers = [
            nn.Conv2d(3, 32, 3, 2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(),
            nn.Conv2d(32, 64, 3, 2, padding=1),
            nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.Sigmoid(),
        ]

        # for i in range(num_conv_layers):
        #     # Dimensions of the downsampled image
        #     ds_size = (ds_size + 2 * conv_padding - conv_kernel_size) // conv_stride + 1

        #     layers += [
        #         torch.nn.Conv2d(
        #             in_channels=in_channels, out_channels=out_channels,
        #             kernel_size=conv_kernel_size,
        #             stride=conv_stride,
        #             padding=conv_padding,
        #             bias=False
        #         ),
        #         torch.nn.LayerNorm([ds_size, ds_size]),
        #         torch.nn.LeakyReLU(0.1)
        #     ]

        #     in_channels = out_channels
        #     if double_channels:
        #         out_channels *= 2
        #         double_channels = False
        #     else:
        #         double_channels = True
        #     conv_kernel_size = max(conv_kernel_size - 1, 3)
        #     conv_stride = max(conv_stride - 1, 2)

        self.model = torch.nn.Sequential(*layers)

        # if has_emotions:
        #     in_features += len(Emotion)
        # for i in range(num_linear_layers - 1):
        #     out_features = in_features // 64
        #     layers += [
        #         torch.nn.Linear(in_features=in_features, out_features=out_features),
        #         torch.nn.LeakyReLU(0.2),
        #         torch.nn.Dropout2d(0.2)
        #     ]
        #     in_features = out_features
        # layers += [
        #     torch.nn.Linear(in_features=in_features, out_features=1)
        # ]
        # in_features = layers[-3].out_channels + audio_embedding_dim
        layers = [
            nn.Linear(in_features=32+audio_embedding_dim, out_features=1)
        ]
        self.adv_layer = torch.nn.Sequential(*layers)

    def forward(self, img: torch.Tensor, audio_embedding: torch.Tensor,
                emotions: Optional[torch.Tensor]) -> torch.Tensor:
        transformed_img = transform_img_for_disc(img)
        output = self.model(transformed_img)
        # output = output.reshape(output.shape[0], -1)  # Flatten elements in the batch
        if emotions is None:
            validity = self.adv_layer(torch.cat((output, self.emb(audio_embedding.argmax(dim=1))), dim=1))
        else:
            validity = self.adv_layer(torch.cat((output, self.emb(audio_embedding.argmax(dim=1)), emotions), dim=1))
        assert not torch.any(torch.isnan(validity))
        return validity
