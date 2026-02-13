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



    def forward(self, h, clinical):
        # h: [N, 1024]
        h = self.feature_proj(h)  # [N,512]

        # Gated Attention 적용
        a = self.attention_a(h) # [N, 256]
        b = self.attention_b(h) # [N, 256]
        A = a * b               # Element-wise product
        A = self.attention_c(A) # [N, 1]
        
        A = torch.softmax(A, dim=0) # 패치별 가중치 합이 1이 되도록

        # 가중 합 (Aggregation)
        M = torch.sum(A * h, dim=0) # [512]

        # Clinical 데이터와 결합
        c_feat = self.clinical_net(clinical) # [32]
        combined = torch.cat([M, c_feat], dim=0) # [544]

        risk = self.survival_head(combined)
        return risk.squeeze(-1) 
    


