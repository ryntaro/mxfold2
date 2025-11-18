from __future__ import annotations

import logging
from typing import Optional

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Gamma


class GeneralizedExtremeValue(torch.autograd.Function):
    def __init__(self, 
                xi: float | torch.tensor,
                mu: float | torch.tensor, 
                sigma: float | torch.tensor):
        self.xi = xi
        self.mu = mu
        self.sigma = sigma
    
    def log_prob(self, x: torch.tensor) -> torch.tensor:
        x = self.xi / self.sigma * (x - self.mu)
        v = 1 / self.sigma * (1+x) ** (-(1+1/self.xi)) * torch.exp(- (1 + x) ** (-1/self.xi))
        return torch.log(v.clip(min=1e-5))


class Wu(nn.Module):
    def __init__(self,  
            xi: float = 0.774, 
            mu: float = 0.078, 
            sigma: float = 0.083,
            alpha: float = 1.006, 
            beta: float = 1.404,
            ) -> None:
        super(Wu, self).__init__()
        self.xi = nn.Parameter(torch.tensor(xi))
        self.mu = nn.Parameter(torch.tensor(mu))
        self.sigma = nn.Parameter(torch.tensor(sigma))
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))
        self.paired_dist = GeneralizedExtremeValue(self.xi, self.mu, self.sigma)
        self.unpaired_dist = Gamma(self.alpha, self.beta)


    def forward(self, seq: list[str], paired: list[torch.tensor], targets: list[torch.Tensor]):

        self.xi.data.clamp_(min=1e-2, max=2.0)
        self.sigma.data.clamp_(min=1e-2, max=2.0)
        self.mu.data.clamp_(min=1e-2, max=2.0)
        self.alpha.data.clamp_(min=1e-2, max=5.0)
        self.beta.data.clamp_(min=1e-2, max=5.0)

        nlls = []
        for i in range(len(seq)):
            # valid = targets[i] > -1 # to ignore missing values (-999)
            valid = targets[i] > 0 # to ignore missing values (-999)
            t = targets[i][valid].clip(min=1e-2, max=3.)
            p = paired[i][valid]

            nll = -torch.mean(self.paired_dist.log_prob(t) * p 
                            + self.unpaired_dist.log_prob(t) * (1-p))
            # nll1 = self.paired_dist.log_prob(t) * p
            # print('nll1', nll1)
            # nll2 =self.unpaired_dist.log_prob(t) * (1-p)
            # print('nll2', nll2)
            # nll = -torch.mean(nll1 + nll2)
            
            nlls.append(nll)
        
        # --- 配列単位で NaN チェック ---
        if torch.isnan(torch.stack(nlls)).any():
            logging.error("[NaN detected in batch]")
            logging.error(f"xi={self.xi.item():.4f}, mu={self.mu.item():.4f}, "
                            f"sigma={self.sigma.item():.4f}, alpha={self.alpha.item():.4f}, beta={self.beta.item():.4f}")
            # logging.error(f"nlls={nlls.detach().cpu().numpy()}")

        return torch.stack(nlls)


class Foo(nn.Module):
    def __init__(self,  
            p_alpha: float = 0.540,
            p_beta: float = 1.390,
            u_alpha: float = 1.006, 
            u_beta: float = 1.404,
            ) -> None:
        super(Foo, self).__init__()
        self.p_alpha = nn.Parameter(torch.tensor(p_alpha))
        self.p_beta = nn.Parameter(torch.tensor(p_beta))
        self.u_alpha = nn.Parameter(torch.tensor(u_alpha))
        self.u_beta = nn.Parameter(torch.tensor(u_beta))
        self.paired_dist = Gamma(self.p_alpha, self.p_beta)
        self.unpaired_dist = Gamma(self.u_alpha, self.u_beta)


    def forward(self, seq: list[str], paired: list[torch.tensor], targets: list[torch.Tensor]):
        # self.p_alpha.data.clamp_(min=1e-2)
        # self.p_beta.data.clamp_(min=1e-2)
        # self.u_alpha.data.clamp_(min=1e-2)
        # self.u_beta.data.clamp_(min=1e-2)
        nlls = []
        for i in range(len(seq)):
            valid = targets[i] > -1 # to ignore missing values (-999)
            t = targets[i][valid].clip(min=1e-2, max=3.)
            p = paired[i][valid]
            nll = -torch.mean(self.paired_dist.log_prob(t) * p 
                            + self.unpaired_dist.log_prob(t) * (1-p))
            nlls.append(nll)
        return torch.stack(nlls)


class RiboEM(nn.Module):
    """
    log1p 空間での 2 成分ガウス（paired / unpaired）から尤度を計算するクラス。
    Wu と同じ forward(seq, paired, targets) シグネチャを持ち、初期値をここに直接指定します。
    """
    def __init__(self,
                 mu_u: float = 0.36248604585583205,
                 sig_u: float = 0.3004844655699528,
                 mu_p: float = 0.0,
                 sig_p: float = 0.10) -> None:
        super(RiboEM, self).__init__()
        # Wu と同じくパラメータを nn.Parameter として保持（必要に応じて学習可能に）
        self.mu_u = nn.Parameter(torch.tensor(mu_u))
        self.sig_u = nn.Parameter(torch.tensor(sig_u))
        self.mu_p = nn.Parameter(torch.tensor(mu_p))
        self.sig_p = nn.Parameter(torch.tensor(sig_p))

    def forward(self, seq: list[str], paired: list[torch.tensor], targets: list[torch.Tensor]):
        # 安定化のため clamp（Wu と同様の扱い）
        self.sig_u.data.clamp_(min=1e-6, max=10.0)
        self.sig_p.data.clamp_(min=1e-6, max=10.0)
        self.mu_u.data.clamp_(min=-10.0, max=10.0)
        self.mu_p.data.clamp_(min=-10.0, max=10.0)

        nlls = []
        two_pi = torch.tensor(2.0 * np.pi)

        for i in range(len(seq)):
            # Wu と同じ基準で無効値を除外
            valid = targets[i] > -2
            t = targets[i][valid]
            if t.numel() == 0:
                nlls.append(torch.tensor(0.0, device=self.mu_u.device))
                continue
            p = paired[i][valid].to(t.dtype)

            device = t.device
            mu_u = self.mu_u.to(device)
            sig_u = (self.sig_u.to(device) + 1e-12)
            mu_p = self.mu_p.to(device)
            sig_p = (self.sig_p.to(device) + 1e-12)

            z = torch.log1p(t)

            log_const_u = -0.5 * torch.log(two_pi.to(device)) - torch.log(sig_u)
            log_pdf_u = log_const_u - 0.5 * ((z - mu_u) / sig_u) ** 2

            log_const_p = -0.5 * torch.log(two_pi.to(device)) - torch.log(sig_p)
            log_pdf_p = log_const_p - 0.5 * ((z - mu_p) / sig_p) ** 2

            nll = -torch.mean(log_pdf_u * (1 - p) + log_pdf_p * p)
            nlls.append(nll)

        if torch.isnan(torch.stack(nlls)).any():
            logging.error("[NaN detected in RiboEM batch]")
            logging.error(f"mu_u={self.mu_u.item():.4f}, sig_u={self.sig_u.item():.4f}, "
                          f"mu_p={self.mu_p.item():.4f}, sig_p={self.sig_p.item():.4f}")

        return torch.stack(nlls)
