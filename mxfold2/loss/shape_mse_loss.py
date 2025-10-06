from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.autograd

from ..fold.fold import AbstractFold
from .predict_shape import ShapeMLP


class ShapeMSELoss(nn.Module):
    def __init__(self, model: AbstractFold,
                 shape_model: list[nn.Module],
                 perturb: float = 0., nu: float = 0.1,
                 l1_weight: float = 0., l2_weight: float = 0.,
                 sl_weight: float = 0.) -> None:
        super(ShapeMSELoss, self).__init__()
        self.model = model
        self.shape_model = shape_model
        self.perturb = perturb
        self.nu = nu
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.sl_weight = sl_weight

        # unused, kept for compatibility
        self.shape_predictor = ShapeMLP(hidden_dim=64)

        if sl_weight > 0.0:
            from .. import param_turner2004
            from ..fold.rnafold import RNAFold
            self.turner = RNAFold(param_turner2004).to(next(self.model.parameters()).device)

    def forward(self, seq: list[str], targets: list[torch.Tensor],
                fname: Optional[list[str]] = None,
                dataset_id: Optional[list[int]] = None) -> torch.Tensor:
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

        # --- ShapeMLP による MSE ---
        mses = self.shape_model[dataset_id](seq, paired, targets)

        # # --- paired に対する勾配を取得 (グラフを壊さない) ---
        # grads = torch.autograd.grad(mses, paired, create_graph=True)

        # --- paired に対する勾配を取得 (高階グラフを作らない) ---
        # create_graph=False にして shape_model 側の計算グラフを保持しない
        grads = torch.autograd.grad(mses, paired, create_graph=False)
        # 明示的に detach して model に渡す（不要な参照を残さない）
        grads = tuple((g.detach().clone() if g is not None else torch.zeros_like(p))
                      for g, p in zip(grads, paired))

        # --- pseudoenergy を付与して再fold ---
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

        # # --- ADwrapper ---
        # class ADwrapper(torch.autograd.Function):
        #     @staticmethod
        #     def forward(ctx, *input):
        #         return mses

        #     @staticmethod
        #     def backward(ctx, grad_output):
        #         return tuple(p - r for p, r in zip(pred_counts, ref_counts))

        # loss = ADwrapper.apply(*pred_params)

        # --- ADwrapper: mses は detach() して渡し、diffs を保存して backward で人工勾配を返す ---
        diffs = [ (pc - rc).detach().clone() for pc, rc in zip(pred_counts, ref_counts) ]
        detached_mses = mses.detach().clone()

        class ADwrapper(torch.autograd.Function):
            @staticmethod
            def forward(ctx, detached_mses_tensor, *inputs):
                # inputs = pred_params + diffs
                n_inputs = len(inputs)
                # pred_params と diffs は同数で渡す想定
                n_pred = n_inputs // 2
                ctx.n_pred = n_pred
                diffs_to_save = inputs[n_pred:]
                ctx.save_for_backward(*diffs_to_save)
                return detached_mses_tensor

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

                # 戻り値の順序: (detached_mses_grad) + pred_params_grads + diffs_grads
                # detached_mses は None、diffs 側には勾配を返さない => None
                return (None, ) + tuple(grads_for_pred) + tuple([None] * n_pred)

        loss = ADwrapper.apply(detached_mses, *pred_params, *diffs)
        # 参照を切る
        diffs = None
        detached_mses = None

        # --- オプション: Turner 正則化 ---
        l = torch.tensor([len(s) for s in seq], device=pred.device)
        if self.sl_weight > 0.0:
            with torch.no_grad():
                ref2, ref2_s, _ = self.turner(seq)
            # loss += self.sl_weight * (ref - ref2) ** 2 / l
            loss = loss + self.sl_weight * ((ref - ref2) ** 2).sum() / l

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
        # if self.l1_weight > 0.0:
        #     for p in self.model.parameters():
        #         loss += self.l1_weight * torch.sum(torch.abs(p))

        # optimizerのweight decayでl2正則はできてる
        # --- L2 正則化 ---
        # if self.l2_weight > 0.0:
        #     l2_reg = 0.0
        #     for p in self.model.parameters():
        #         l2_reg += torch.sum(p ** 2)
        #     loss += self.l2_weight * l2_reg

        return loss
