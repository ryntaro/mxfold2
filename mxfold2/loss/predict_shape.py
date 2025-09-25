import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_absolute_error
from ..fold.embedding import OneHotEmbedding

class ShapeMLP(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.embed = OneHotEmbedding()
        # self.fc1 = nn.Linear(5, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(5, 1)

    def forward(self, seq: list[str], paired: list[torch.Tensor], targets: list[torch.Tensor]):
        """学習用: MSE loss を返す (autograd 有効)"""
        device = next(self.parameters()).device
        losses = []

        for s, p, t in zip(seq, paired, targets):
            x = self.embed([s]).to(device)    # (1,4,N)
            p = p[1:].to(device)
            t = t[1:].to(device)

            x = x.transpose(1, 2)             # (1,N,4)
            x = torch.cat([x, p.unsqueeze(0).unsqueeze(-1)], dim=-1)  # (1,N,5)

            # h = F.relu(self.fc1(x))
            # h = F.relu(self.fc2(h))
            # pred = self.fc3(h).squeeze(0).squeeze(-1)  # (N,)
            pred = self.fc(x).squeeze(0).squeeze(-1)  # (N,)

            mask = t >= -1
            if mask.sum() > 0:
                losses.append(torch.mean((pred[mask] - t[mask]) ** 2))

        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)

    @torch.no_grad()
    def predict(self, seq: list[str], paired: list[torch.Tensor], targets: list[torch.Tensor]):
        """評価用: loss + metrics を返す (autograd 無効)"""
        device = next(self.parameters()).device
        losses, maes, r2s = [], [], []

        for s, p, t in zip(seq, paired, targets):
            x = self.embed([s]).to(device)
            p = p[1:].to(device)
            t = t[1:].to(device)

            x = x.transpose(1, 2)
            x = torch.cat([x, p.unsqueeze(0).unsqueeze(-1)], dim=-1)

            # h = F.relu(self.fc1(x))
            # h = F.relu(self.fc2(h))
            # pred = self.fc3(h).squeeze(0).squeeze(-1)
            pred = self.fc(x).squeeze(0).squeeze(-1)  # (N,)

            mask = t >= -1
            if mask.sum() > 0:
                y_true = t[mask].cpu().numpy()
                y_pred = pred[mask].cpu().numpy()

                losses.append(torch.mean((pred[mask] - t[mask]) ** 2).item())
                maes.append(mean_absolute_error(y_true, y_pred))
                if len(y_true) > 1:
                    r2s.append(r2_score(y_true, y_pred))

        loss = sum(losses) / len(losses) if losses else 0.0
        mae = sum(maes) / len(maes) if maes else float("nan")
        r2 = sum(r2s) / len(r2s) if r2s else float("nan")

        return loss, {"MAE": mae, "R2": r2}
