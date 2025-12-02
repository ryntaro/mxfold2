from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd

from ..fold.fold import AbstractFold


class ShapeNLLLoss(nn.Module):
    def __init__(self, model: AbstractFold, 
            shape_model: list[nn.Module],
            perturb: float = 0., nu: float = 0.1, l1_weight: float = 0., l2_weight: float = 0.,
            sl_weight: float = 0.) -> None:
        super(ShapeNLLLoss, self).__init__()
        self.model = model
        self.shape_model = shape_model
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
                dataset_id: Optional[int] = None) -> torch.Tensor:
        
        pred: torch.Tensor
        pred_s: list[str]
        pred_bps: list[list[int]]
        pred, pred_s, pred_bps, param, _ = self.model(seq, return_param=True, return_count=True, perturb=self.perturb)

        pred_params, pred_counts = [], []
        for k in sorted(param[0].keys()):
            if k.startswith('score_'):
                pred_params.append(torch.vstack([param[i][k] for i in range(len(seq))]))
            elif k.startswith('count_'):
                pred_counts.append(torch.vstack([param[i][k] for i in range(len(seq))]))
            elif isinstance(param[0][k], dict):
                for kk in sorted(param[0][k].keys()):
                    if kk.startswith('score_'):
                        pred_params.append(torch.vstack([param[i][k][kk] for i in range(len(seq))]))
                    elif kk.startswith('count_'):
                        pred_counts.append(torch.vstack([param[i][k][kk] for i in range(len(seq))]))

        paired = []
        for pred_bp in pred_bps:
            p = [ 1 if v > 0 else 0 for v in pred_bp ]
            p = torch.tensor(p, dtype=torch.float32, requires_grad=True, device=pred.device)
            paired.append(p)
        targets = [ t.to(pred.device) for t in targets ]
        
        nlls = self.shape_model[dataset_id](seq, paired, targets)
        
        # nlls.backward()
        # grads = [ p.grad for p in paired ]

        # nlls の勾配を計算し、勾配テンソルだけを detach して保持する
        nlls_sum = nlls.sum()
        nlls_sum.backward()  # create_graph=False, retain_graph=False（デフォルト）でグラフを消費
        grads = [ (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for p in paired ]
        # 不要な参照を切る
        for p in paired:
            p.grad = None

        # grads を絶対値で正規化: max_abs = max_i max(|grads[i]|)
        valid_grads = [g for g in grads if isinstance(g, torch.Tensor) and g.numel() > 0]
        if len(valid_grads) > 0:
            max_vals = [g.abs().max() for g in valid_grads]
            max_abs = torch.stack(max_vals).max()
            # 安全策：ゼロに近い場合は 1.0 を使う
            if max_abs.item() == 0.0:
                max_abs = torch.tensor(1.0, device=max_abs.device, dtype=max_abs.dtype)
        else:
            max_abs = torch.tensor(1.0, device=pred.device, dtype=torch.float32)

        # pseudoenergy = nu * g / max_abs
        pseudo_list = [ (self.nu * g / max_abs) for g in grads ]

        # pseudo_list = [ (self.nu * g) for g in grads ]

        ref: torch.Tensor
        ref_s: list[str]
        ref, ref_s, _, param, _ = self.model(seq, param=param, return_param=True, return_count=True,
                                    pseudoenergy=pseudo_list)

        ref_counts = []
        for k in sorted(param[0].keys()):
            if k.startswith('count_'):
                ref_counts.append(torch.vstack([param[i][k] for i in range(len(seq))]))
            elif isinstance(param[0][k], dict):
                for kk in sorted(param[0][k].keys()):
                    if kk.startswith('count_'):
                        ref_counts.append(torch.vstack([param[i][k][kk] for i in range(len(seq))]))

        # class ADwrapper(torch.autograd.Function):
        #     @staticmethod
        #     def forward(ctx, *input):
        #         return nlls
        
        #         # nlls.detach() の値だけ返す（グラフは切る）
        #         # return nlls.detach()
        #         # return nll.detach().clone().reshape(())   # 0-dim & 非view

        #     @staticmethod
        #     def backward(ctx, grad_output):
        #         return tuple( p-r for p, r in zip(pred_counts, ref_counts) )

        # loss = ADwrapper.apply(*pred_params)

        # --- 変更点: pred_counts/ref_counts の差を detach して保持し、
        #              nlls は detach して ADwrapper に渡す ---
        # diffs をローカルで作り、apply に渡して外側で参照を残さない
        diffs = [ (pc - rc).detach().clone() for pc, rc in zip(pred_counts, ref_counts) ]
        detached_nlls = nlls.detach().clone()

        class ADwrapper(torch.autograd.Function):
            @staticmethod
            def forward(ctx, detached_nlls_tensor, *inputs):
                # inputs = pred_params  + diffs
                n_inputs = len(inputs)
                # pred_params と diffs は同数で渡す想定
                n_pred = n_inputs // 2
                ctx.n_pred = n_pred
                # diffs を saved_tensors として保存（autograd が適切に解放する）
                diffs_to_save = inputs[n_pred:]
                ctx.save_for_backward(*diffs_to_save)
                return detached_nlls_tensor

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

                # 戻り値の長さ: (detached_nlls_grad) + pred_params_grads + diffs_grads
                # detached_nlls は None、diffs 自体には勾配を返さない => None
                return (None, ) + tuple(grads_for_pred) + tuple([None] * n_pred)

        # ADwrapper に pred_params と diffs を続けて渡す（外側の diffs 参照はここで切る）
        loss = ADwrapper.apply(detached_nlls, *pred_params, *diffs)
        # 参照を切る
        diffs = None
        detached_nlls = None

        l = torch.tensor([len(s) for s in seq], device=pred.device)
        if self.sl_weight > 0.0:
            with torch.no_grad():
                ref2: torch.Tensor
                ref2_s: list[str]
                ref2, ref2_s, _ = self.turner(seq)
            
            # loss += self.sl_weight * (ref-ref2)**2 / l
            loss = loss + self.sl_weight * ((ref - ref2) ** 2).sum() / l

        logging.debug(f"Loss = {loss.item()} = ({pred.item()} - {ref.item()})")
        logging.debug(seq)
        logging.debug(pred_s)
        logging.debug(ref_s)
        if float(loss.item())> 1e10 or torch.isnan(loss):
            logging.error(fname)
            logging.error(f"{loss.item()}, {pred.item()}, {ref.item()}")
            logging.error(seq)

        # l1, 入れるとnanになる（まだ修正できない）
        # if self.l1_weight > 0.0:
        #     for p in self.model.parameters():
        #         loss += self.l1_weight * torch.nansum(torch.abs(p))

        # optimizerのweight decayでl2正則はできてる
        # if self.l2_weight > 0.0:
        #     l2_reg = 0.0
        #     for p in self.model.parameters():
        #         l2_reg += torch.nansum(p ** 2)
        #     loss += self.l2_weight * l2_reg

        return loss

