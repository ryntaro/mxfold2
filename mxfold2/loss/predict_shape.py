import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..fold.embedding import OneHotEmbedding

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        L = x.size(1)
        x = x + self.pe[:, :L, :].to(x.device)
        return self.dropout(x)

class ShapeMLP(nn.Module):
    def __init__(self, d_model: int = 64, nhead: int = 8, nlayers: int = 2, dim_feedforward: int = 256, dropout: float = 0.1, max_len: int = 10000):
        super().__init__()
        self.embed = OneHotEmbedding()   # produces (B,4,L)
        self.input_proj = nn.Linear(5, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.out_proj = nn.Linear(d_model, 1)

    def _encode(self, seq: list[str], paired: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        ネットワーク本体の計算をまとめる（位置1..N に対する予測値を返す）。
        戻り値: list of tensor (各 tensor は長さ N, index0 を含まない)
        """
        device = next(self.parameters()).device
        preds = []
        for s, p in zip(seq, paired):
            x = self.embed([s]).to(device)    # (1,4,N')
            # p は構造モデルが出す位置ごとの確率（連続値）を想定
            p_trim = p[1:].to(device).float()  # (N',) — float で detach しないこと
            x = x.transpose(1, 2)               # (1,N',4)
            x = torch.cat([x, p_trim.unsqueeze(0).unsqueeze(-1)], dim=-1)  # (1,N',5)
            # project -> transformer expects (B, L, D)
            x = self.input_proj(x)              # (1,N',d_model)
            x = self.pos_enc(x)                 # (1,N',d_model)
            h = self.transformer(x)             # (1,N',d_model)
            pred = self.out_proj(h).squeeze(0).squeeze(-1)  # (N',)
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
                # losses.append(torch.mean(torch.abs(pred[mask] - t_trim[mask])))
                losses.append(torch.mean((pred[mask] - t_trim[mask]) ** 2))
        loss_tensor = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)
        return loss_tensor

    def predict(self, seq: list[str], paired: list[torch.Tensor]) -> list[torch.Tensor]:
        device = next(self.parameters()).device
        preds_trim = self._encode(seq, paired)
        # そのまま長さ L の予測を返す（index0 を含めない）
        return preds_trim
