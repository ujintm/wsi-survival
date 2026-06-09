import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from aggregators.attention import Attn_Net_Gated


class SlideAttentionAggregator(nn.Module):
    def __init__(self, dim=512, attn_dim=256, dropout=True):
        super().__init__()

        self.attention_net = Attn_Net_Gated(
            L=dim,
            D=attn_dim,
            dropout=dropout,
            n_classes=1
        )

    def forward(self, slide_feats, return_attention=False):
        # slide_feats: [num_slides, 512]
        A, slide_feats = self.attention_net(slide_feats) # [num_slides, 1]
        A = torch.softmax(A, dim=0)

        patient_feat = torch.sum(A * slide_feats, dim=0)  # [512]

        if return_attention:
            return patient_feat, A.squeeze(-1)

        return patient_feat
    
    

# class SurvivalModel(nn.Module):
#     def __init__(self, aggregator, clinical_dim=9):
#         super().__init__()
#         self.aggregator = aggregator
        
#         # Clinical feature 전처리
#         self.clinical_net = nn.Sequential( 
#             nn.Linear(clinical_dim, 32),
#             nn.ReLU()
#         )
#         self.survival_head = nn.Linear(512 + 32, 1)

#     def forward(self, h, clinical):
#         slide_feat = self.aggregator(h)   # [512]
#         c_feat = self.clinical_net(clinical)
#         combined = torch.cat([slide_feat, c_feat], dim=0)
#         return self.survival_head(combined).squeeze(-1)


class SurvivalModel(nn.Module):
    def __init__(self, aggregator, clinical_dim=9):
        super().__init__()
        self.patch_aggregator = aggregator
        self.slide_aggregator = SlideAttentionAggregator(dim=512)

        self.clinical_net = nn.Sequential(
            nn.Linear(clinical_dim, 32),
            nn.ReLU()
        )

        self.survival_head = nn.Linear(512 + 32, 1)

    # def forward(self, slides, clinical):
    #     # slides: list of slide patch features
    #     # each slide_h: [num_patches, feat_dim]

    #     slide_feats = []

    #     for slide_h in slides:
    #         slide_feat = self.patch_aggregator(slide_h)  # [512]
    #         slide_feats.append(slide_feat)

    #     slide_feats = torch.stack(slide_feats, dim=0)    # [num_slides, 512]
    #     patient_feat = self.slide_aggregator(slide_feats)  # [512]

    #     c_feat = self.clinical_net(clinical)               # [32]
    #     combined = torch.cat([patient_feat, c_feat], dim=0)
    #     risk = self.survival_head(combined).squeeze(-1)

    #     return risk

    def forward(self, slides, clinical):
        # slides가 단일 slide tensor [num_patches, 1024]인 경우
        if torch.is_tensor(slides):
            slides = [slides]

        slide_feats = []

        for slide_h in slides:
            slide_feat = self.patch_aggregator(slide_h)  # [512] or [1,512]
            slide_feat = slide_feat.squeeze()
            slide_feats.append(slide_feat)

        slide_feats = torch.stack(slide_feats, dim=0)      # [num_slides, 512]
        patient_feat = self.slide_aggregator(slide_feats)  # [512]

        c_feat = self.clinical_net(clinical)
        combined = torch.cat([patient_feat, c_feat], dim=0)
        risk = self.survival_head(combined).squeeze(-1)

        return risk