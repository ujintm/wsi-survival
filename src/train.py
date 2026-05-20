import random
import numpy as np
import pandas as pd
import argparse
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from model import SurvivalModel
from aggregators.attention  import AttentionAggregator
from aggregators.transmil   import TransMILAggregator
from aggregators.mambamil   import MambaMIL


def concordance_index_simple(times, risks, events):
    times = np.asarray(times, dtype=float)
    risks = np.asarray(risks, dtype=float)
    events = np.asarray(events, dtype=int)

    concordant = permissible = tied = 0.0
    n = len(times)

    for i in range(n):
        if events[i] != 1:
            continue
        for j in range(n):
            # i가 더 빨리 사건 발생(time이 더 작음)한 경우만 비교
            if times[i] >= times[j]:
                continue
            permissible += 1.0
            # 더 빨리 죽은 i가 risk가 더 커야 concordant
            if risks[i] > risks[j]:
                concordant += 1.0
            elif risks[i] == risks[j]:
                tied += 1.0

    if permissible == 0:
        return 0.5
    return (concordant + 0.5 * tied) / permissible

# =========================
# Config
# =========================
CSV_PATH = "../clinical/clinical_survival_processed.csv"

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

EPOCHS = 50
LR = 1e-4
SEED = 42

# CLAM 스타일: bag 1개씩 로드
BAG_BATCH_SIZE = 1

# Cox loss를 계산할 “누적 bag 개수”
# (예: 8이면, bag 8개 risk/time/event 모아 loss 한번 계산)
ACCUM_BAGS = 8

NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ===== Regularization / Early stopping =====
WEIGHT_DECAY = 1e-4          # 1e-4 ~ 1e-3 사이에서 조절
PATIENCE = 5                 # 개선 없으면 몇 epoch 기다릴지
MIN_DELTA = 1e-4             # 이 정도 이상 좋아져야 "개선"으로 인정
# CKPT_PATH = "best_survival_attn.pt"


class EarlyStopping:
    """
    mode='max' : metric이 클수록 좋음 (C-index)
    """
    def __init__(self, patience=5, min_delta=1e-4, mode="max", ckpt_path="best.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.ckpt_path = ckpt_path

        self.best = None
        self.num_bad_epochs = 0

    def _is_improved(self, metric):
        if self.best is None:
            return True
        if self.mode == "max":
            return metric > (self.best + self.min_delta)
        else:  # mode == "min"
            return metric < (self.best - self.min_delta)

    def step(self, metric, model):
        """
        Returns:
            improved (bool), should_stop (bool)
        """
        if self._is_improved(metric):
            self.best = metric
            self.num_bad_epochs = 0
            torch.save(model.state_dict(), self.ckpt_path)
            return True, False
        else:
            self.num_bad_epochs += 1
            should_stop = self.num_bad_epochs >= self.patience
            return False, should_stop


# =========================
# Dataset: bag 1개씩 반환
# =========================
class SurvivalDataset(Dataset):
    def __init__(self, csv_path: str):
        self.data = pd.read_csv(csv_path)
        if "feature_path" not in self.data.columns:
            raise ValueError("CSV에 'feature_path' 컬럼이 없습니다.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # WSI bag feature: [N, 1024] (N 가변)
        h = torch.load(row["feature_path"]).float()

        # clinical: [9]
        clinical_np = pd.to_numeric(row[FEATURE_COLS], errors="coerce").to_numpy(dtype=np.float32)
        clinical_np = np.nan_to_num(clinical_np, nan=0.0, posinf=0.0, neginf=0.0)
        clinical = torch.from_numpy(clinical_np)

        # survival
        time = torch.tensor(float(row["survival_time"]), dtype=torch.float32)
        event = torch.tensor(float(row["event"]), dtype=torch.float32)

        return h, clinical, time, event


# =========================
# Cox loss (mini-batch)
# =========================
def cox_loss(risk, time, event):
    if torch.sum(event) == 0:
        # 그래프에 연결된 0 loss (backward 가능)
        return risk.sum() * 0.0

    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order]

    log_cumsum = torch.logcumsumexp(risk, dim=0)
    loss = -torch.sum((risk - log_cumsum) * event) / torch.sum(event)
    return loss

@torch.no_grad()
def eval_c_index(model, loader):
    model.eval()
    risks, times, events = [], [], []

    for (h, clinical, time, event) in loader:
        # bag_batch_size=1이라서 h는 shape [1, N, 1024] 처럼 들어올 수 있음
        # -> squeeze로 [N,1024] 맞춰줌
        h = h.squeeze(0).to(DEVICE)
        clinical = clinical.squeeze(0).to(DEVICE)

        r = model(h, clinical)
        risks.append(float(r.item()))
        times.append(float(time.item()))
        events.append(int(event.item()))

    # risk 높을수록 위험↑ (생존↓) => 부호 반전해서 concordance_index
    return concordance_index_simple(times, np.array(risks), events)



def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    updates = 0

    # 누적용 버퍼
    buf_risk, buf_time, buf_event = [], [], []

    optimizer.zero_grad(set_to_none=True)

    for step, (h, clinical, time, event) in enumerate(loader, start=1):
        h = h.squeeze(0).to(DEVICE)
        clinical = clinical.squeeze(0).to(DEVICE)
        time = time.to(DEVICE)
        event = event.to(DEVICE)

        risk = model(h, clinical)  # scalar
        buf_risk.append(risk)
        buf_time.append(time.squeeze())
        buf_event.append(event.squeeze())

        # ACCUM_BAGS개 모이면 Cox loss 한 번 계산해서 업데이트
        if len(buf_risk) == ACCUM_BAGS:
            risks = torch.stack(buf_risk)      # [B]
            times = torch.stack(buf_time)      # [B]
            events = torch.stack(buf_event)    # [B]

            loss = cox_loss(risks, times, events)

            if torch.isfinite(loss):
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                total_loss += float(loss.item())
                updates += 1

            buf_risk, buf_time, buf_event = [], [], []

    # epoch 끝에 버퍼에 남은 게 있으면 마지막 업데이트(선택)
    if len(buf_risk) > 1:
        risks = torch.stack(buf_risk)
        times = torch.stack(buf_time)
        events = torch.stack(buf_event)
        loss = cox_loss(risks, times, events)

        if torch.isfinite(loss):
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.item())
            updates += 1

    return total_loss / max(updates, 1)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    default="attention",
    choices=["attention", "transmil", "mamba"]
)




def main():
    set_seed(SEED)

    dataset = SurvivalDataset(CSV_PATH)

    # stratified train/val split (by event)
    indices = np.arange(len(dataset))
    events = dataset.data["event"].values  # SurvivalDataset 안에서 self.data로 읽어둔 df 사용

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=SEED,
        stratify=events
    )

    train_set = Subset(dataset, train_idx)
    val_set   = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=BAG_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BAG_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    args = parser.parse_args()

    AGGREGATOR_NAME = args.model

    if AGGREGATOR_NAME == "attention":
        aggregator = AttentionAggregator(embed_dim=1024, out_dim=512)
        CKPT_PATH = "best_survival_attention.pt"

    elif AGGREGATOR_NAME == "transmil":
        aggregator = TransMILAggregator(embed_dim=1024, out_dim=512)
        CKPT_PATH = "best_survival_transmil.pt"

    elif AGGREGATOR_NAME == "mamba":
        aggregator = MambaMIL(embed_dim=1024, out_dim=512)
        CKPT_PATH = "best_survival_mamba.pt"
    
    
    # aggregator = TransMILAggregator(embed_dim=1024, out_dim=512)
    model = SurvivalModel(aggregator).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY) # AdamW + weight decay

    early_stopper = EarlyStopping(
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        mode="max",
        ckpt_path=CKPT_PATH,
    )

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_c = eval_c_index(model, val_loader)

        improved, should_stop = early_stopper.step(val_c, model)

        print(
            f"[Epoch {epoch:03d}] train_loss={train_loss:.4f}  "
            f"val_cindex={val_c:.4f}  best={early_stopper.best:.4f}  "
            f"{'(improved)' if improved else ''}"
        )

        if should_stop:
            print(f"Early stopping triggered (patience={PATIENCE}).")
            break

    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    print("Loaded best checkpoint:", CKPT_PATH)


if __name__ == "__main__":
    main()
