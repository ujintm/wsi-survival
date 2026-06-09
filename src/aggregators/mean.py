import torch
import torch.nn as nn


class MeanPoolingAggregator(nn.Module):
    def __init__(self, input_dim=1024, output_dim=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

    def forward(self, h):
        # h: [N, 1024]
        h = h.mean(dim=0)      # [1024]
        h = self.fc(h)         # [512]
        return h