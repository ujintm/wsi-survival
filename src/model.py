import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SurvivalAttentionModel(nn.Module):
    def __init__(self, embed_dim=1024, clinical_dim=9, dropout=True):
        super().__init__()

        self.feature_proj = nn.Linear(embed_dim, 512)

        # Gated Attention 구조
        self.attention_a = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh()
        )
        self.attention_b = nn.Sequential(
            nn.Linear(512, 256),
            nn.Sigmoid()
        )

        if dropout:
            self.attention_a.add_module('dropout', nn.Dropout(0.25))
            self.attention_b.add_module('dropout', nn.Dropout(0.25))

        self.attention_c = nn.Linear(256, 1)

        # Clinical feature 전처리
        self.clinical_net = nn.Sequential(
            nn.Linear(clinical_dim, 32),
            nn.ReLU()
        )

        self.survival_head = nn.Linear(512 + 32, 1)


    def forward(self, h, clinical, return_attention=False):

        h = self.feature_proj(h)  # [N,512]

        # Gated Attention
        a = self.attention_a(h)
        b = self.attention_b(h)
        A = a * b
        A = self.attention_c(A)   # [N,1]

        A = torch.softmax(A, dim=0)

        # 가중 합 (Aggregation)
        M = torch.sum(A * h, dim=0)

        c_feat = self.clinical_net(clinical)
        combined = torch.cat([M, c_feat], dim=0)

        risk = self.survival_head(combined)

        if return_attention:
            return risk.squeeze(-1), A.squeeze(-1)

        return risk.squeeze(-1)