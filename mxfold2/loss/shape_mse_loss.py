from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.autograd

from ..fold.fold import AbstractFold


class ShapeMSELoss(nn.Module):
    def __init__(self, model: AbstractFold,
                 perturb: float = 0., nu: float = 0.1,
                 l1_weight: float = 0., l2_weight: float = 0.,
                 sl_weight: float = 0.) -> None:
        super(ShapeMSELoss, self).__init__()
        self.model = model
        self.perturb = perturb
        self.nu = nu
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.sl_weight = sl_weight
        if sl_weight > 0.0:
            from .. import param_turner2004
            from ..fold.rnafold import RNAFold
            self.turner = RNAFold(param_turner2004).to(next(self.model.parameters()).device)

    def forward(self, seq: list[str], targets: list[torch.Tensor],
                fname: Optional[list[str]] = None,
                dataset_id: Optional[list[int]] = None) -> torch.Tensor:
        pred: torch.Tensor
        pred_s: list[str]
        pred_bps: list[list[int]]
        pred, pred_s, pred_bps, param, _ = self.model(
            seq, return_param=True, return_count=True, perturb=self.perturb
        )

        # --- score / count を集める ---
        pred_params, pred_counts = [], []
        for k in sorted(param[0].keys()):
            if k.startswith("score_"):
                pred_params.append(torch.vstack([param[i][k] for i in range(len(seq))]))
            elif k.startswith("count_"):
                pred_counts.append(torch.vstack([param[i][k] for i in range(len(seq))]))
            elif isinstance(param[0][k], dict):
                for kk in sorted(param[0][k].keys()):
                    if kk.startswith("score_"):
                        pred_params.append(torch.vstack([param[i][k][kk] for i in range(len(seq))]))
                    elif kk.startswith("count_"):
                        pred_counts.append(torch.vstack([param[i][k][kk] for i in range(len(seq))]))

        # --- paired ベクトル作成 ---
        paired = []
        for pred_bp in pred_bps:
            p = [1 if v > 0 else 0 for v in pred_bp]
            p = torch.tensor(p, dtype=torch.float32, requires_grad=True, device=pred.device)
            paired.append(p)
        targets = [t.to(pred.device) for t in targets]

        # --- SHAPE proxy = 2*(1 - paired) ---
        proxies = [2.0 * (1.0 - p) for p in paired]

        # --- MSE を計算 ---
        mses = 0.0
        for proxy, t in zip(proxies, targets):
            mses = mses + torch.mean((proxy - t) ** 2)
        mses = mses / len(proxies)

        # --- 勾配計算 ---
        mses.backward(retain_graph=True)
        grads = [p.grad for p in paired]

        # --- pseudoenergy を付与して再fold ---
        ref: torch.Tensor
        ref_s: list[str]
        ref, ref_s, _, param, _ = self.model(
            seq, param=param, return_param=True, return_count=True,
            pseudoenergy=[self.nu * g for g in grads]
        )

        ref_counts = []
        for k in sorted(param[0].keys()):
            if k.startswith("count_"):
                ref_counts.append(torch.vstack([param[i][k] for i in range(len(seq))]))
            elif isinstance(param[0][k], dict):
                for kk in sorted(param[0][k].keys()):
                    if kk.startswith("count_"):
                        ref_counts.append(torch.vstack([param[i][k][kk] for i in range(len(seq))]))

        # --- ADwrapper ---
        class ADwrapper(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *input):
                return mses

            @staticmethod
            def backward(ctx, grad_output):
                return tuple(p - r for p, r in zip(pred_counts, ref_counts))

        loss = ADwrapper.apply(*pred_params)

        # --- オプション: Turner 正則化 ---
        l = torch.tensor([len(s) for s in seq], device=pred.device)
        if self.sl_weight > 0.0:
            with torch.no_grad():
                ref2: torch.Tensor
                ref2_s: list[str]
                ref2, ref2_s, _ = self.turner(seq)
            loss += self.sl_weight * (ref - ref2) ** 2 / l

        # --- ログ出力 ---
        logging.debug(f"Loss = {loss.item()} (MSE)")
        logging.debug(seq)
        logging.debug(pred_s)
        logging.debug(ref_s)
        if float(loss.item()) > 1e10 or torch.isnan(loss):
            logging.error(fname)
            logging.error(f"{loss.item()}, {pred.item()}, {ref.item()}")
            logging.error(seq)

        # --- L1 正則化 ---
        if self.l1_weight > 0.0:
            for p in self.model.parameters():
                loss += self.l1_weight * torch.sum(torch.abs(p))

        # --- L2 正則化 ---
        if self.l2_weight > 0.0:
            l2_reg = 0.0
            for p in self.model.parameters():
                l2_reg += torch.sum(p ** 2)
            loss += self.l2_weight * l2_reg

        return loss
