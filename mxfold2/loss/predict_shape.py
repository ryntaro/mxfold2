import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_absolute_error
from ..fold.embedding import OneHotEmbedding

class ShapeMLP(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.embed = OneHotEmbedding()
        self.fc1 = nn.Linear(5, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        # self.fc = nn.Linear(5, 1)

    def forward(self, seq: list[str], paired: list[torch.Tensor], targets: list[torch.Tensor], return_metrics: bool = False):
        """
        ネットワーク本体は一度だけ計算する。
        - デフォルト: 学習用に autograd 対応の loss (torch.Tensor scalar) を返す。
        - return_metrics=True: (loss_float, metrics_dict) を返す（予測評価用）。
        """
        device = next(self.parameters()).device

        preds_and_targets = []
        for s, p, t in zip(seq, paired, targets):
            x = self.embed([s]).to(device)    # (1,4,N)
            p = p[1:].to(device)
            t = t[1:].to(device)

            x = x.transpose(1, 2)             # (1,N,4)
            x = torch.cat([x, p.unsqueeze(0).unsqueeze(-1)], dim=-1)  # (1,N,5)

            # ネットワークはここで一回だけ実行
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            pred = self.fc3(h).squeeze(0).squeeze(-1)  # (N,)

            preds_and_targets.append((pred, p, t))

        # 常に differentiable な loss(tensor) を計算して返す（train 用互換）
        losses = []
        for pred, p, t in preds_and_targets:
            mask = t >= 0
            if mask.sum() > 0:
                losses.append(torch.mean((pred[mask] - t[mask]) ** 2))
        loss_tensor = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)

        if not return_metrics:
            return loss_tensor

        # 評価指標を追加で計算 (autograd に影響しないよう detach() を使う)
        losses_f, maes, r2s = [], [], []
        for pred, p, t in preds_and_targets:
            mask = t >= 0
            if mask.sum() > 0:
                y_true = t[mask].cpu().numpy()
                y_pred = pred[mask].detach().cpu().numpy()
                losses_f.append(((pred[mask] - t[mask]) ** 2).mean().item())
                maes.append(mean_absolute_error(y_true, y_pred))
                if len(y_true) > 1:
                    r2s.append(r2_score(y_true, y_pred))

        loss_f = sum(losses_f) / len(losses_f) if losses_f else 0.0
        mae = sum(maes) / len(maes) if maes else float("nan")
        r2 = sum(r2s) / len(r2s) if r2s else float("nan")
        return loss_f, {"MAE": mae, "R2": r2}
