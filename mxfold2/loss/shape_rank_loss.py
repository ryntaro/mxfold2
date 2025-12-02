from __future__ import annotations

import logging
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd

from ..fold.fold import AbstractFold
from ..fold.shape_layers import _mask_and_clip_targets


class ShapeRankLoss(nn.Module):
    """
    Non-parametric, rank-based SHAPE loss using the implicit-MLE framework.

    - 1回目の fold で予測構造 pred_bps を得る
    - pred_bps から paired ベクトル (0/1, requires_grad=True) を作る
    - paired と SHAPE targets から，ペアワイズな順位ロス（Mann–Whitney U / AUC の
      logistic 近似）を計算する
    - rank loss を paired で backward して dL/dpaired を得る
    - その勾配を pseudoenergy (= nu * grad / max_abs) に変換して 2 回目の fold
    - 1 回目と 2 回目の count 差分を，構造モデルパラメータへの勾配として流す
    """

    def __init__(
        self,
        model: AbstractFold,
        perturb: float = 0.0,
        nu: float = 0.1,
        tau: float = 0.2,
        l1_weight: float = 0.0,
        l2_weight: float = 0.0,
        sl_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.perturb = perturb
        self.nu = nu          # pseudoenergy のスケール
        self.tau = tau        # 順位ロスの温度
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.sl_weight = sl_weight
        if sl_weight > 0.0:
            from .. import param_turner2004
            from ..fold.rnafold import RNAFold

            self.turner = RNAFold(param_turner2004).to(
                next(self.model.parameters()).device
            )

    def _per_sequence_rank_loss(
        self,
        shape_vec: torch.Tensor,
        paired_vec: torch.Tensor,
    ) -> torch.Tensor:
        """
        1本の配列についての順位ロスを計算する。

        shape_vec: [L_all]  (raw SHAPE targets, 1D)
        paired_vec: [L_all]  (0/1, same length as pred_bps / sequence)
        """
        device = paired_vec.device

        # 対象外 (-1 以下) をマスクし，[min,max] に clip
        valid, t_all = _mask_and_clip_targets(shape_vec, min_val=0.01, max_val=1.0)
        if not valid.any():
            return torch.tensor(0.0, device=device)

        t_valid = t_all[valid].to(device)
        p_mask = paired_vec[valid].to(device)

        # paired / unpaired weight
        w_p = p_mask          # [M]
        w_u = 1.0 - p_mask    # [M]

        # どちらも存在しない場合は loss=0
        if (w_p.sum() <= 0) or (w_u.sum() <= 0):
            return torch.tensor(0.0, device=device)

        # s_j - s_i の差分行列 (i: paired, j: unpaired を想定)
        # diff[i,j] = t_valid[j] - t_valid[i]
        diff = t_valid[None, :] - t_valid[:, None]   # [M,M]
        # 重み行列: i 側が paired, j 側が unpaired のときに効く
        w = w_p[:, None] * w_u[None, :]              # [M,M]

        # 自分自身とのペア (i==j) は無視したいので対角成分を 0 に
        M = t_valid.shape[0]
        if M > 0:
            w = w * (1.0 - torch.eye(M, device=device))

        # logistic 近似: phi(s) = log(1 + exp(-(s/tau)))
        #   s = t_j - t_i が大きいほど（unpaired が高いほど） loss は小さい
        phi = F.softplus(-diff / self.tau)   # [M,M]

        weighted = w * phi
        denom = w.sum()
        if denom <= 0:
            return torch.tensor(0.0, device=device)

        return weighted.sum() / (denom + 1e-8)

    def forward(
        self,
        seq: List[str],
        targets: List[torch.Tensor],
        fname: Optional[List[str]] = None,
        dataset_id: Optional[int] = None,
    ) -> torch.Tensor:
        # dataset_id は ShapeNLLLoss との互換性のために受け取るが，ここでは使わない
        del dataset_id

        # --- 1st fold: 現在のパラメータで通常の予測 ---
        pred: torch.Tensor
        pred_s: List[str]
        pred_bps: List[List[int]]
        pred, pred_s, pred_bps, param, _ = self.model(
            seq, return_param=True, return_count=True, perturb=self.perturb
        )

        pred_params, pred_counts = [], []
        for k in sorted(param[0].keys()):
            if k.startswith("score_"):
                pred_params.append(
                    torch.vstack([param[i][k] for i in range(len(seq))])
                )
            elif k.startswith("count_"):
                pred_counts.append(
                    torch.vstack([param[i][k] for i in range(len(seq))])
                )
            elif isinstance(param[0][k], dict):
                for kk in sorted(param[0][k].keys()):
                    if kk.startswith("score_"):
                        pred_params.append(
                            torch.vstack([param[i][k][kk] for i in range(len(seq))])
                        )
                    elif kk.startswith("count_"):
                        pred_counts.append(
                            torch.vstack([param[i][k][kk] for i in range(len(seq))])
                        )

        # pred_bps -> paired (0/1, requires_grad=True)
        paired: List[torch.Tensor] = []
        for pred_bp in pred_bps:
            # pred_bp は [L+1] or [L] のペア情報 (0: unpaired, >0: paired相手index)
            p = [1.0 if v > 0 else 0.0 for v in pred_bp]
            p = torch.tensor(
                p,
                dtype=torch.float32,
                requires_grad=True,
                device=pred.device,
            )
            paired.append(p)

        # --- rank-based SHAPE loss (per sequence) ---
        per_seq_losses = []
        for p_vec, t_vec in zip(paired, targets):
            t_vec_dev = t_vec.to(pred.device)
            per_seq_losses.append(self._per_sequence_rank_loss(t_vec_dev, p_vec))

        # [B] テンソルにまとめる
        rank_losses = torch.stack(per_seq_losses)

        # rank loss の勾配を paired に対して計算し，その勾配だけを残す
        rank_loss_sum = rank_losses.sum()
        rank_loss_sum.backward()
        grads = [
            (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
            for p in paired
        ]
        for p in paired:
            p.grad = None

        # grads を絶対値で正規化: max_abs = max_i max(|grads[i]|)
        valid_grads = [
            g for g in grads if isinstance(g, torch.Tensor) and g.numel() > 0
        ]
        if len(valid_grads) > 0:
            max_vals = [g.abs().max() for g in valid_grads]
            max_abs = torch.stack(max_vals).max()
            if max_abs.item() == 0.0:
                max_abs = torch.tensor(1.0, device=max_abs.device, dtype=max_abs.dtype)
        else:
            max_abs = torch.tensor(1.0, device=pred.device, dtype=torch.float32)

        # pseudoenergy = nu * g / max_abs
        pseudo_list = [(self.nu * g / max_abs) for g in grads]

        # --- 2nd fold: pseudoenergy を加えた参照構造 ---
        ref: torch.Tensor
        ref_s: List[str]
        ref, ref_s, _, param, _ = self.model(
            seq,
            param=param,
            return_param=True,
            return_count=True,
            pseudoenergy=pseudo_list,
        )

        ref_counts = []
        for k in sorted(param[0].keys()):
            if k.startswith("count_"):
                ref_counts.append(
                    torch.vstack([param[i][k] for i in range(len(seq))])
                )
            elif isinstance(param[0][k], dict):
                for kk in sorted(param[0][k].keys()):
                    if kk.startswith("count_"):
                        ref_counts.append(
                            torch.vstack([param[i][k][kk] for i in range(len(seq))])
                        )

        # pred_counts / ref_counts の差分を勾配として利用する
        diffs = [(pc - rc).detach().clone() for pc, rc in zip(pred_counts, ref_counts)]
        detached_rank = rank_losses.detach().clone()

        class ADwrapper(torch.autograd.Function):
            @staticmethod
            def forward(ctx, detached_rank_tensor, *inputs):
                # inputs = pred_params + diffs
                n_inputs = len(inputs)
                n_pred = n_inputs // 2
                ctx.n_pred = n_pred
                diffs_to_save = inputs[n_pred:]
                ctx.save_for_backward(*diffs_to_save)
                return detached_rank_tensor

            @staticmethod
            def backward(ctx, grad_output):
                saved = ctx.saved_tensors  # diffs
                n_pred = ctx.n_pred

                grads_for_pred = []
                for d in saved:
                    if grad_output.dim() == 0:
                        g = d * grad_output
                    else:
                        shape = [grad_output.shape[0]] + [1] * (d.dim() - 1)
                        g = d * grad_output.view(*shape)
                    grads_for_pred.append(g)

                # (grad_for_detached_rank, *grads_for_pred, *None_for_diffs)
                return (None,) + tuple(grads_for_pred) + tuple([None] * n_pred)

        loss = ADwrapper.apply(detached_rank, *pred_params, *diffs)
        diffs = None
        detached_rank = None

        l = torch.tensor([len(s) for s in seq], device=pred.device)
        if self.sl_weight > 0.0:
            with torch.no_grad():
                ref2: torch.Tensor
                ref2_s: List[str]
                ref2, ref2_s, _ = self.turner(seq)
            loss = loss + self.sl_weight * ((ref - ref2) ** 2).sum() / l

        # logging は scalar 前提なので，バッチサイズ >1 のときは mean を取っておく
        scalar_loss = loss.mean()
        logging.debug(f"ShapeRankLoss = {scalar_loss.item()} = ({pred.item()} - {ref.item()})")
        logging.debug(seq)
        logging.debug(pred_s)
        logging.debug(ref_s)
        if float(scalar_loss.item()) > 1e10 or torch.isnan(scalar_loss):
            logging.error(fname)
            logging.error(f"{scalar_loss.item()}, {pred.item()}, {ref.item()}")
            logging.error(seq)

        # l1/l2 正則化が欲しければここで追加（デフォルトは 0）
        if self.l1_weight > 0.0:
            for p in self.model.parameters():
                scalar_loss = scalar_loss + self.l1_weight * torch.sum(torch.abs(p))

        # if self.l2_weight > 0.0:
        #     l2_reg = 0.0
        #     for p in self.model.parameters():
        #         l2_reg = l2_reg + torch.sum(p ** 2)
        #     scalar_loss = scalar_loss + self.l2_weight * l2_reg

        return scalar_loss
