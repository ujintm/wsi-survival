import torch
import torch.nn as nn


class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 512, D = 256, dropout = False, n_classes = 1):
        super().__init__()
        
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x): # x: [N, L]
        a = self.attention_a(x) # [N, D]
        b = self.attention_b(x) # [N, D]
        A = a.mul(b) # [N, D]
        A = self.attention_c(A)  # [N, 1]
        return A, x
    
    
    
class AttentionAggregator(nn.Module):
    def __init__(self, embed_dim=1024, out_dim=512, attn_dim=256, dropout=True):
        super().__init__()

        self.feature_proj = nn.Linear(embed_dim, out_dim)

        self.attention_net = Attn_Net_Gated(
            L=out_dim,
            D=attn_dim,
            dropout=dropout,
            n_classes=1
        )

    def forward(self, h, return_attention=False):
        # print(f"input h: {h.shape}")  
        # h: [N, 1024]
        h = self.feature_proj(h)             # [N, 512]

        A, h = self.attention_net(h)         # A: [N, 1], h: [N, 512]
        A = torch.softmax(A, dim=0)          # [N, 1]

        M = torch.sum(A * h, dim=0)          # weighted sum → [512]

        if return_attention:
            return M, A.squeeze(-1)          # [512], [N]

        return M                             # [512]
    