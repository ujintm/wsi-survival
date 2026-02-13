import torch
import h5py

features = torch.load('features/pt_files/TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF-D8FB-45CF-B4BD-C6B76294C291.pt')

print(type(features))
if isinstance(features, torch.Tensor):
    print("Shape:", features.shape) 
    # 결과: [N, 1024] -> N은 패치 개수, 1024는 임베딩 차원
    
with h5py.File('features/h5_files/TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF-D8FB-45CF-B4BD-C6B76294C291.h5', 'r') as f:
    print("Keys:", list(f.keys())) 
    
    data = f['features'] 
    print("Shape:", data.shape)