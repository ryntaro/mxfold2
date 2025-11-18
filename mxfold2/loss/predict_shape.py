import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..fold.embedding import OneHotEmbedding

class RelativePositionalEncoding(nn.Module):
    """
    シンプルな相対位置エンコーディング。
    各位置に対して、他トークンとの相対距離に基づく埋め込みを平均化して加える。
    """
    def __init__(self, d_model: int, max_distance: int = 256):
        super().__init__()
        self.max_distance = max_distance
        self.embedding = nn.Embedding(2 * max_distance + 1, d_model)

    def forward(self, seq_len: int) -> torch.Tensor:
        device = self.embedding.weight.device
        pos = torch.arange(seq_len, device=device)
        rel = pos[None, :] - pos[:, None]  # (L, L)
        rel = rel.clamp(-self.max_distance, self.max_distance) + self.max_distance
        enc = self.embedding(rel)  # (L, L, d_model)
        enc = enc.mean(dim=1)      # 各位置ごとに平均化 → (L, d_model)
        return enc


class ShapeTransformer(nn.Module):
    """
    SHAPE reactivity predictor with relative positional encoding.
    出力は連続値、損失は「区間所属の正誤」（ソフト区間分類ロス）で学習。
    forward(seq, paired, targets) -> scalar loss
    predict(seq, paired) -> list[Tensor] (reactivity予測)
    """
    def __init__(self,
                 d_model: int = 32,
                 nhead: int = 4,
                 num_layers: int = 1,
                 dim_feedforward: int = 64,
                 dropout: float = 0.1,
                 max_len: int = 2000,
                 boundaries=(0.25, 0.5),
                 tau: float = 10.0):
        super().__init__()
        self.embed = OneHotEmbedding()
        self.input_proj = nn.Linear(5, d_model)
        self.rel_pos_enc = RelativePositionalEncoding(d_model, max_distance=128)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, 1)  # ← 出力は実数値のまま

        # 区間境界とスムージング係数
        self.boundaries = boundaries
        self.tau = tau

    # --- ソフト区間所属確率を計算する関数 ---
    def smooth_bin_probs(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: (...,) 実数出力
        return: (..., 3) 各区間の所属確率
        区間は boundaries=(b0,b1) で定義
        """
        b0, b1 = self.boundaries
        tau = self.tau
        sig = lambda x: torch.sigmoid(tau * x)

        # y は (...,) を想定し，各要素ごとに3クラスの確率ベクトルを返す (...,3)
        p0 = sig(b0 - y)
        p2 = 1 - sig(b1 - y)
        p1 = 1 - p0 - p2
        probs = torch.stack([p0, p1, p2], dim=-1)
        probs = probs / probs.sum(-1, keepdim=True)
        return probs

    # --- Transformer符号化 ---
    def _encode(self, seq: list[str], paired: list[torch.Tensor]) -> list[torch.Tensor]:
        device = next(self.parameters()).device
        preds = []
        for s, p in zip(seq, paired):
            # one-hot embedding
            x = self.embed([s]).to(device)      # (1,4,L)
            p_trim = p[1:].to(device).float()   # remove index0
            x = x.transpose(1, 2)               # (1,L,4)
            x = torch.cat([x, p_trim.unsqueeze(0).unsqueeze(-1)], dim=-1)  # (1,L,5)
            x = self.input_proj(x)              # (1,L,d_model)

            # 相対位置エンコーディングを加算
            L = x.size(1)
            rel_enc = self.rel_pos_enc(L).unsqueeze(0)  # (1,L,d_model)
            x = x + rel_enc

            # Transformer → 出力
            h = self.transformer(x)    # (1,L,d_model)
            pred = self.out_proj(h).squeeze(0).squeeze(-1)  # (L,)
            preds.append(pred)
        return preds

    # --- 損失計算（ソフト区間分類ロス） ---
    def forward(self, seq: list[str], paired: list[torch.Tensor], targets: list[torch.Tensor]) -> torch.Tensor:
        device = next(self.parameters()).device
        preds = self._encode(seq, paired)
        losses = []

        b0, b1 = self.boundaries  # ← ここで引き受けて使う

        for pred, t in zip(preds, targets):
            # 位置ごとに処理して無効塩基はスキップ（mask 不要）
            t_trim = t[1:].to(device)
            L = t_trim.size(0)
            for i in range(L):
                ti = t_trim[i]
                # 無効を負値で表している場合（データに合わせて条件を調整）
                if ti < -10:
                    continue
                ti = ti.clamp(0.0, 1.0)

                # pred[i] はスカラー，確率ベクトルに変換
                probs_i = self.smooth_bin_probs(pred[i])       # (3,)
                # 連続値からクラスインデックスを決定
                if ti < b0:
                    yi = 0
                elif ti < b1:
                    yi = 1
                else:
                    yi = 2
                loss_i = F.nll_loss(probs_i.unsqueeze(0).log(),
                                    torch.tensor([yi], dtype=torch.long, device=device))
                losses.append(loss_i)

        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)

    def predict(self, seq: list[str], paired: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        予測値は連続値（SHAPE reactivity）を返す。
        区間確率が欲しい場合は smooth_bin_probs() を適用する。
        """
        return self._encode(seq, paired)


class ShapeConv(nn.Module):
    """
    Lightweight ShapeConv:
    - initial depthwise per-channel conv (in_channels=5) + pointwise projection
    - smaller hidden_dim, kernel_size=3
    - BatchNorm を除去してパラメータとランニングステートを削減
    """
    def __init__(self,
                 hidden_dim: int = 32,
                 kernel_size: int = 3,
                 dropout: float = 0.0,
                 boundaries=(0.25, 0.5),
                 tau: float = 10.0):
        super().__init__()
        self.embed = OneHotEmbedding()
        # depthwise per-input-channel conv (out_channels = in_channels)
        self.dw = nn.Conv1d(in_channels=5, out_channels=5,
                            kernel_size=kernel_size, padding=kernel_size // 2,
                            groups=5, bias=False)
        # pointwise projection to hidden_dim (cheap)
        self.pw = nn.Conv1d(in_channels=5, out_channels=hidden_dim, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Identity() if dropout == 0.0 else nn.Dropout(dropout)
        # lightweight output projection
        self.out_proj = nn.Conv1d(hidden_dim, 1, kernel_size=1)

        self.boundaries = boundaries
        self.tau = tau

    def smooth_bin_probs(self, y: torch.Tensor) -> torch.Tensor:
        b0, b1 = self.boundaries
        tau = self.tau
        sig = lambda x: torch.sigmoid(tau * x)
        p0 = sig(b0 - y)
        p2 = 1 - sig(b1 - y)
        p1 = 1 - p0 - p2
        probs = torch.stack([p0, p1, p2], dim=-1)
        probs = probs / probs.sum(-1, keepdim=True)
        return probs

    def _encode(self, seq: list[str], paired: list[torch.Tensor]) -> list[torch.Tensor]:
        device = next(self.parameters()).device
        preds = []
        for s, p in zip(seq, paired):
            x = self.embed([s]).to(device)       # (1,4,L)
            p_trim = p[1:].to(device).float()    # (L,)
            x = x.transpose(1, 2)                # (1,L,4)
            x = torch.cat([x, p_trim.unsqueeze(0).unsqueeze(-1)], dim=-1)  # (1,L,5)
            x = x.transpose(1, 2)                # (1,5,L)
            h = self.dw(x)                       # (1,5,L)
            h = self.pw(h)                       # (1,hidden_dim,L)
            h = self.act(h)
            h = self.dropout(h)
            out = self.out_proj(h)               # (1,1,L)
            out = out.squeeze(0).squeeze(0)      # (L,)
            preds.append(out)
        return preds

    # forward/predict は元の実装と同様
    def forward(self, seq: list[str], paired: list[torch.Tensor], targets: list[torch.Tensor]) -> torch.Tensor:
        device = next(self.parameters()).device
        preds = self._encode(seq, paired)
        losses = []
        b0, b1 = self.boundaries
        for pred, t in zip(preds, targets):
            t_trim = t[1:].to(device)
            L = t_trim.size(0)
            for i in range(L):
                ti = t_trim[i]
                if ti < -10:
                    continue
                ti = ti.clamp(0.0, 1.0)
                probs_i = self.smooth_bin_probs(pred[i])
                if ti < b0:
                    yi = 0
                elif ti < b1:
                    yi = 1
                else:
                    yi = 2
                losses.append(F.nll_loss(probs_i.unsqueeze(0).log(),
                                         torch.tensor([yi], dtype=torch.long, device=device)))
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)

    def predict(self, seq: list[str], paired: list[torch.Tensor]) -> list[torch.Tensor]:
        return self._encode(seq, paired)

