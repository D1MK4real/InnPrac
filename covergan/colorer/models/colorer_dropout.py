from typing import Optional, List

import numpy as np
import torch
from numpy import ndarray
from torch import nn

from outer.emotions import Emotion


class Colorer2(nn.Module):
    def __init__(self, z_dim: int, audio_embedding_dim: int, num_layers: int,
                 colors_count: int = 6):
        super(Colorer2, self).__init__()
        colors_count = colors_count
        self.colors_count = colors_count
        in_features = z_dim + audio_embedding_dim
        out_features = colors_count

        feature_step = (in_features - out_features) // num_layers

        layers = []
        for i in range(num_layers - 1):
            out_features = in_features - feature_step
            layers += [
                torch.nn.Linear(in_features=in_features, out_features=out_features),
                torch.nn.Dropout(0.2),
                torch.nn.BatchNorm1d(num_features=out_features),
                torch.nn.LeakyReLU(0.2)
            ]
            in_features = out_features
        layers += [
            torch.nn.Linear(in_features=in_features, out_features=colors_count * 3),  # 3 = RGB
            torch.nn.Sigmoid()
        ]

        self.model_ = torch.nn.Sequential(*layers)

    def forward(self, noise: torch.Tensor, audio_embedding: torch.Tensor) -> List[torch.Tensor]:

        inp = torch.cat((noise, audio_embedding), dim=1)

        return self.model_(inp)

    def predict(self, noise: torch.Tensor, audio_embedding: torch.Tensor) -> ndarray:
        palette = self.forward(noise, audio_embedding)
        palette = palette[0].detach().cpu().numpy() * 255
        palette = palette.reshape(-1, 3)
        palette = np.around(palette, 0).astype(int)
        return palette
