from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.autograd

from ..fold.fold import AbstractFold
from .predict_shape import ShapeTransformer, ShapeConv
from .external_shape_predictor import ExternalShapePredictor

class ShapeCLSLoss(nn.Module):
    def __init__(self, model: AbstractFold,
                 shape_model: list[nn.Module],
                 perturb: float = 0., nu: float = 0.1,
                 l1_weight: float = 0., l2_weight: float = 0.,
                 sl_weight: float = 0.) -> None:
        super(ShapeCLSLoss, self).__init__()
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

    def _select_shape_model(self, dataset_id):
        # dataset_id may be int or list/tuple or None
        if isinstance(self.shape_model, (list, tuple)):
            if isinstance(dataset_id, int):
                idx = dataset_id
            elif isinstance(dataset_id, (list, tuple)) and len(dataset_id) > 0:
                idx = dataset_id[0]
            else:
                idx = 0
            return self.shape_model[idx]
        else:
            return self.shape_model

    def forward(self, seq: list[str], targets: list[torch.Tensor],
                fname: Optional[list[str]] = None,
                dataset_id: Optional[list[int]] = None) -> torch.Tensor:
        # 1) fold once to get params / counts / paired predictions
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

        # --- paired ベクトル作成 (requires_grad=True) ---
        paired = []
        for pred_bp in pred_bps:
            p = [1 if v > 0 else 0 for v in pred_bp]
            p = torch.tensor(p, dtype=torch.float32, requires_grad=True, device=pred.device)
            paired.append(p)

        # --- targets を device に移し、valid な要素を [0,1] に clamped しておく ---
        targets = [t.to(pred.device) for t in targets]
        targets_clipped = []
        for t in targets:
            tt = t.float()
            mask = tt > -10.0
            if mask.any():
                tt[mask] = tt[mask].clamp(0.0, 1.0)
            targets_clipped.append(tt)

        # --- Shape model 呼び出し（external か internal かを判定） ---
        idx = dataset_id if isinstance(dataset_id, int) else (dataset_id[0] if dataset_id else 0)
        shape_mod = self._select_shape_model(idx)

        if isinstance(shape_mod, ExternalShapePredictor):
            prior_loss = shape_mod(seq, paired, targets_clipped, paired_requires_grad=True)
        else:
            prior_loss = shape_mod(seq, paired, targets_clipped)

        # --- DEBUG: check whether conv outputs connect to conv params (robust call) ---
        try:
            sm = shape_mod
            preds = None
            # prefer _encode if available
            if hasattr(sm, "_encode"):
                preds = sm._encode(seq, paired)  # list of tensors
            else:
                # try typical forward without keyword 'targets' (ExternalShapePredictor may not accept it)
                try:
                    preds = sm(seq, paired)
                except TypeError:
                    # fallback: cannot obtain preds for this shape_model safely -> skip connectivity check
                    logging.info("[CONNCHK] shape_model.forward signature incompatible, skipping conv->pred connectivity check")
                    preds = None

            if preds is None:
                # nothing to check
                pass
            else:
                if isinstance(preds, torch.Tensor):
                    preds = [preds]
                # make single scalar from preds to test grad flow
                pred_sum = sum(p.sum() for p in preds)
                params_to_check = []
                for n, p in sm.named_parameters():
                    if any(k in n for k in ("dw", "pw", "out_proj")):
                        params_to_check.append((n, p))
                if params_to_check:
                    grads = torch.autograd.grad(pred_sum, [p for _, p in params_to_check], allow_unused=True, retain_graph=True)
                    for (n, p), g in zip(params_to_check, grads):
                        logging.info(f"[CONNCHK] {n} requires_grad={p.requires_grad} grad_is_None={g is None} grad_norm={None if g is None else float(g.norm().item())}")
                else:
                    logging.info("[CONNCHK] no conv-like params found in shape_model")
        except Exception:
            logging.exception("[CONNCHK] failed to check conv->pred connectivity")

        # grads wrt paired
        grads = torch.autograd.grad(prior_loss, paired, create_graph=False, retain_graph=True)
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

        # --- ADwrapper: prior_loss のグラフを切らずに渡し、pred_params 用の人工勾配は保持する ---
        # diffs / prior_loss は detach しない（shape_model 側へ勾配を流すため）
        diffs = [ (pc - rc) for pc, rc in zip(pred_counts, ref_counts) ]

        class ADwrapper(torch.autograd.Function):
            @staticmethod
            def forward(ctx, loss_tensor, *inputs):
                # inputs = pred_params + diffs
                n_inputs = len(inputs)
                n_pred = n_inputs // 2
                ctx.n_pred = n_pred
                diffs_to_save = inputs[n_pred:]
                ctx.save_for_backward(*diffs_to_save)
                return loss_tensor

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

                # 先頭（loss_tensor）に対して grad_output を返すことで
                # prior_loss -> shape_model の勾配が伝搬するようにする
                return (grad_output, ) + tuple(grads_for_pred) + tuple([None] * n_pred)

        # prior_loss をそのまま渡して ADwrapper を適用（グラフを切らない）
        loss = ADwrapper.apply(prior_loss, *pred_params, *diffs)
        # no explicit cleanup of diffs/prior_loss (GC will handle)

        # --- オプション: Turner 正則化 ---
        l = torch.tensor([len(s) for s in seq], device=pred.device)
        if self.sl_weight > 0.0:
            with torch.no_grad():
                ref2, ref2_s, _ = self.turner(seq)
            loss = loss + self.sl_weight * ((ref - ref2) ** 2).sum() / l

        # --- ログ出力 ---
        logging.debug(f"Loss = {loss.item()} (CLS)")
        logging.debug(seq)
        logging.debug(pred_s)
        logging.debug(ref_s)
        if float(loss.item()) > 1e10 or torch.isnan(loss):
            logging.error(fname)
            logging.error(f"{loss.item()}, {pred.item()}, {ref.item()}")
            logging.error(seq)

        return loss
