import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import torch
import h5py
import openslide
import numpy as np
import matplotlib.pyplot as plt

from src.model import SurvivalAttentionModel

import inspect
print(inspect.getfile(SurvivalAttentionModel))


device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt_path = "../src/best_survival_attn.pt"
h5_path = "../CLAM/results/features/h5_files/TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF-D8FB-45CF-B4BD-C6B76294C291.h5"
wsi_path = "../wsi_files/TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF-D8FB-45CF-B4BD-C6B76294C291.svs"


FEATURE_COLS = [
    'demographic.age_at_index',
    'stage_numeric',
    'gender_numeric',
    'race_asian',
    'race_black or african american',
    'race_not reported',
    'race_white',
    'dx_Lobular',
    'dx_Other'
]

# 모델 로드
model = SurvivalAttentionModel(
    embed_dim=1024,
    clinical_dim=len(FEATURE_COLS)
).to(device)
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.to(device).eval()

# h5 로드
with h5py.File(h5_path, "r") as f:
    h = torch.tensor(f["features"][:]).to(device)
    coords = f["coords"][:]

clinical = torch.zeros(9).to(device)  # 실제 값 있으면 교체

with torch.no_grad():
    risk, A = model(h, clinical, return_attention=True)

A = A.cpu().numpy()
A = (A - A.min()) / (A.max() - A.min() + 1e-8)

slide = openslide.OpenSlide(wsi_path)
thumb = slide.get_thumbnail((2000,2000))
thumb_np = np.array(thumb)

downsample_x = slide.dimensions[0] / 2000
downsample_y = slide.dimensions[1] / 2000

scaled_coords = coords.astype(np.float32)
scaled_coords[:,0] /= downsample_x
scaled_coords[:,1] /= downsample_y

plt.figure(figsize=(8,8))
plt.imshow(thumb_np)
plt.scatter(scaled_coords[:,0],
            scaled_coords[:,1],
            c=A,
            cmap="jet",
            s=6,
            alpha=0.6)
plt.axis("off")
plt.tight_layout()
plt.savefig("survival_attention_heatmap.png", dpi=300)
print("saved survival_attention_heatmap.png")
