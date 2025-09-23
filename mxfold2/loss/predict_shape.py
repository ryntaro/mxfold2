import torch
import torch.nn as nn
import torch.nn.functional as F
from .embedding import OneHotEmbedding

class ShapeMLP(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.embed = OneHotEmbedding()
        self.fc1 = nn.Linear(5, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, seq: list[str], paired: list[torch.Tensor], targets: list[torch.Tensor]):
        losses = []
        for s, p, t in zip(seq, paired, targets):
            # --- 入力特徴を作成 ---
            x = self.embed([s])           # (1,4,N)
            x = x.transpose(1, 2)         # (1,N,4)
            x = torch.cat([x, p.unsqueeze(0).unsqueeze(-1)], dim=-1)  # (1,N,5)

            # --- MLP で予測 ---
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            pred = self.fc3(h).squeeze(0).squeeze(-1)  # (N,)

            # --- mask MSE ---
            mask = t >= -1
            if mask.sum() > 0:
                losses.append(torch.mean((pred[mask] - t[mask]) ** 2))

        return torch.stack(losses).mean()
