import torch
import torch.nn as nn
import sys
from pathlib import Path

# --- 外部SHAPEtransformerをパスに追加 ---
sys.path.append("/gs/bs/tga-satolab-gtex/yamauchi/SHAPEtransformer")
from train import ShapeTransformer

class ExternalShapePredictor(nn.Module):
    """
    外部 SHAPEtransformer モデルを読み込み、
    - predict(): shape reactivity 予測 (list[str], list[Tensor]) -> list[Tensor]
    - forward(): loss 計算 (list[str], list[Tensor], list[Tensor]) -> scalar loss
    """

    def __init__(self, model_path: str | Path = None, device: str | None = None, loss_type: str = "mse"):
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.loss_type = loss_type

        # --- デフォルトモデルパス ---
        if model_path is None:
            model_path = Path("/gs/bs/tga-satolab-gtex/yamauchi/SHAPEtransformer/logs/5136586/train/epoch-20.pth")

        # --- チェックポイント読み込み（安全モード優先） ---
        try:
            ckpt = torch.load(model_path, map_location=self.device, weights_only=True)
        except Exception:
            # フォールバック: 信頼できる ckpt のみで使うこと
            ckpt = torch.load(model_path, map_location=self.device, weights_only=False)

        # normalize checkpoint -> state and args
        if isinstance(ckpt, dict):
            ck_args = ckpt.get("args", {}) or {}
            state = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
        else:
            ck_args = {}
            state = ckpt

        # instantiate model using saved args (fallback defaults)
        self.model = ShapeTransformer(
            d_model=int(ck_args.get("dim", 128)),
            nhead=int(ck_args.get("nhead", 8)),
            num_layers=int(ck_args.get("layers", 2)),
            dim_feedforward=int(ck_args.get("dim_feedforward", 256)),
            dropout=float(ck_args.get("dropout", 0.1)),
            max_len=int(ck_args.get("max_len", 10000)),
        ).to(self.device)

        # load parameters (support nested formats)
        try:
            self.model.load_state_dict(state)
        except Exception:
            if isinstance(state, dict) and "model_state_dict" in state:
                self.model.load_state_dict(state["model_state_dict"])
            elif isinstance(state, dict) and "model_state" in state:
                self.model.load_state_dict(state["model_state"])
            else:
                # let exception surface for debugging
                raise

        self.model.eval()

    # helper: convert tensor of base indices -> sequence string (if needed)
    def _tensor_to_seq(self, t: torch.Tensor) -> str:
        t = t.detach().cpu().squeeze()
        idx_to_base = {0: "A", 1: "C", 2: "G", 3: "U", 4: "N", 5: "N"}
        return "".join(idx_to_base.get(int(x), "N") for x in t.tolist())

    # ------------------------------------------------------------
    # 前処理: 現在は seq は文字列期待、partner/target は index0 を含む1次元テンソル期待
    # ------------------------------------------------------------
    def _prepare_single(self, seq, paired, target=None):
        # seq: accept str or 1d-tensor-of-indices
        if isinstance(seq, str):
            seq_str = seq
        elif torch.is_tensor(seq):
            seq_str = self._tensor_to_seq(seq)
        else:
            raise TypeError("seq must be str or Tensor of base indices")

        # paired: expect 1D tensor length L+1 (index0 included) or convertible
        if torch.is_tensor(paired):
            # do not detach here — keep graph connectivity; move to device
            paired_t = paired.to(self.device).squeeze()
        else:
            paired_t = torch.tensor(paired, device=self.device)

        # target: if provided, keep index0 included
        if target is not None:
            if torch.is_tensor(target):
                target_t = target.detach().to(self.device).squeeze()
            else:
                target_t = torch.tensor(target, device=self.device)
        else:
            target_t = None

        return seq_str, paired_t, target_t

    # ------------------------------------------------------------
    # 推論モード: shape reactivity 予測
    # ------------------------------------------------------------
    @torch.no_grad()
    def predict(self, seq_list, paired_list):
        # seq_list: iterable of str or 1D-tensor
        # paired_list: iterable of 1D-tensor (index0 included)
        seqs = []
        paireds = []
        for seq, paired in zip(seq_list, paired_list):
            seq_s, paired_t, _ = self._prepare_single(seq, paired)
            seqs.append(seq_s)
            paireds.append(paired_t)

        preds = self.model.predict(seqs, paireds)  # list[Tensor] on device
        # move to cpu tensors
        return [p.detach().cpu() for p in preds]

    # ------------------------------------------------------------
    # 学習モード: loss 計算（MSE または MAE）
    # ------------------------------------------------------------
    def forward(self, seq_list, paired_list, target_list, paired_requires_grad: bool = False):
        # prepare lists
        seqs = []
        paireds = []
        targets = []
        for seq, paired, target in zip(seq_list, paired_list, target_list):
            seq_s, paired_t, target_t = self._prepare_single(seq, paired, target)
            if paired_requires_grad:
                # if we need grad wrt paired, ensure tensor requires grad on device
                paired_t = paired_t.clone().to(self.device)
                paired_t.requires_grad_(True)
            paireds.append(paired_t)
            seqs.append(seq_s)
            targets.append(target_t if target_t is not None else torch.full((paireds[-1].shape[0]-1,), float("nan"), device=self.device))

        # delegate to model.forward which returns autograd-enabled scalar loss
        loss = self.model(seqs, paireds, targets)
        return loss
