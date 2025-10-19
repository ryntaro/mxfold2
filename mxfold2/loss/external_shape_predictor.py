# -*- coding: utf-8 -*-
"""
external_shape_predictor.py
ShapeTransformer を外部から呼び出すためのモジュール。
構成は predict_shape.py と同じ（forward/predict 両対応）。

学習済みの .full.pth モデルをそのままロードして使用。
"""

import torch
import torch.nn as nn
from shapetransformer.train import BASE_VOCAB, PAD_IDX


class ExternalShapePredictor(nn.Module):
    def __init__(self, model_path: str = "/gs/bs/tga-satolab-gtex/yamauchi/SHAPEtransformer/logs/5127328/model.full.pth"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- 学習済みモデルオブジェクトをそのままロード ---
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()

    # -----------------------------
    # ShapeMLP と同じ構造
    # -----------------------------
    def _encode(self, seqs, partners):
        preds = []
        with torch.no_grad():
            for seq, partner in zip(seqs, partners):
                base = torch.tensor(
                    [[BASE_VOCAB.get(b, PAD_IDX) for b in seq]],
                    dtype=torch.long,
                    device=self.device,
                )
                partner = partner.unsqueeze(0).to(self.device)
                mask = torch.ones_like(partner, dtype=torch.bool, device=self.device)
                pred = self.model(base, partner, mask)[0].detach().cpu()
                preds.append(pred)
        return preds

    def forward(self, seqs, partners, targets):
        """
        学習時の loss 計算用。
        seqs: list[str]
        partners: list[Tensor(L,)]
        targets: list[Tensor(L,)]
        """
        preds = self._encode(seqs, partners)
        losses = []
        for pred, t in zip(preds, targets):
            t_trim = t[1:].to(self.device)
            mask = t_trim >= 0
            if mask.sum() > 0:
                losses.append(torch.mean((pred[mask] - t_trim[mask]) ** 2))
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=self.device)

    def predict(self, seqs, partners):
        """
        推論用。ShapeMLP と同インターフェース。
        """
        return self._encode(seqs, partners)
