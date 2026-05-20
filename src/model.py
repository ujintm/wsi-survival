import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from aggregators.attention  import AttentionAggregator
from aggregators.transmil   import TransMILAggregator
from aggregators.mambamil   import MambaMIL


class SurvivalModel(nn.Module):
    def __init__(self, aggregator, clinical_dim=9):
        super().__init__()
        self.aggregator = aggregator
        
        # Clinical feature 전처리
        self.clinical_net = nn.Sequential( 
            nn.Linear(clinical_dim, 32),
            nn.ReLU()
        )
        self.survival_head = nn.Linear(512 + 32, 1)

    def forward(self, h, clinical):
        slide_feat = self.aggregator(h)   # [512]
        c_feat = self.clinical_net(clinical)
        combined = torch.cat([slide_feat, c_feat], dim=0)
        return self.survival_head(combined).squeeze(-1)



