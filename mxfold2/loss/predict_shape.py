import torch
import torch.nn as nn
import torch.nn.functional as F
# from sklearn.metrics import r2_score, mean_absolute_error
from ..fold.embedding import OneHotEmbedding

class ShapeMLP(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.embed = OneHotEmbedding()
        # self.fc1 = nn.Linear(5, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(5, 1)

    def _encode(self, seq: list[str], paired: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        ネットワーク本体の計算をまとめる（位置1..N に対する予測値を返す）。
        戻り値: list of tensor (各 tensor は長さ N, index0 を含まない)
        """
        device = next(self.parameters()).device
        preds = []
        for s, p in zip(seq, paired):
            x = self.embed([s]).to(device)    # (1,4,N')
            p_trim = p[1:].to(device)

            x = x.transpose(1, 2)             # (1,N',4)
            x = torch.cat([x, p_trim.unsqueeze(0).unsqueeze(-1)], dim=-1)  # (1,N',5)

            # h = F.relu(self.fc1(x))
            # h = F.relu(self.fc2(h))
            # pred = self.fc3(h).squeeze(0).squeeze(-1)  # (N',)

            pred = self.fc(x).squeeze(0).squeeze(-1)  # (N',)

            # pred = 2*(1-p)

            preds.append(pred)
        return preds

    def forward(self, seq: list[str], paired: list[torch.Tensor], targets: list[torch.Tensor]) -> torch.Tensor:
        """
        学習用 forward: 常に autograd に対応した scalar loss (torch.Tensor) を返す。
        ネットワーク計算は _encode に委譲する。
        """
        device = next(self.parameters()).device
        preds = self._encode(seq, paired)

        losses = []
        for pred, t in zip(preds, targets):
            t_trim = t[1:].to(device)
            mask = t_trim >= 0
            if mask.sum() > 0:
                losses.append(torch.mean((pred[mask] - t_trim[mask]) ** 2))
        loss_tensor = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)
        return loss_tensor

    def predict(self, seq: list[str], paired: list[torch.Tensor]) -> list[torch.Tensor]:
        
        device = next(self.parameters()).device
        preds_trim = self._encode(seq, paired)
        # そのまま長さ L の予測を返す（index0 を含めない）
        return preds_trim
