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
    - predict(): shape reactivity 予測
    - forward(): loss 計算
    """

    def __init__(self, model_path: str | Path = None, device: str | None = None, loss_type: str = "mse"):
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.loss_type = loss_type

        # --- モデルパス固定 ---
        if model_path is None:
            model_path = Path("/gs/bs/tga-satolab-gtex/yamauchi/SHAPEtransformer/logs/5127328/model.pth")

        # --- モデルロード ---
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)

        args = ckpt.get("args", {})
        self.model = ShapeTransformer(
            d_model=int(args.get("dim", 128)),
            nhead=int(args.get("nhead", 8)),
            num_layers=int(args.get("layers", 4)),
            partner_bins=int(args.get("partner_bins", 33))
        ).to(self.device)

        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        # 塩基→インデックス変換表
        self.BASE_VOCAB = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3, "N": 4}
        self.PAD_IDX = 5

    # ------------------------------------------------------------
    # 前処理: 文字列→Tensor変換 & ダミー列削除
    # ------------------------------------------------------------
    def _prepare_inputs(self, seq, partner, target=None):
        # seq: 文字列なら Tensor に変換
        if isinstance(seq, str):
            seq = torch.tensor(
                [[self.BASE_VOCAB.get(b, self.PAD_IDX) for b in seq]],
                dtype=torch.long, device=self.device
            )
        else:
            seq = seq.unsqueeze(0).to(self.device)

        # partner: Tensor化 & 先頭のダミー削除
        partner = partner.unsqueeze(0).to(self.device)[:, 1:]

        # target: あればダミー削除
        if target is not None:
            target = target.unsqueeze(0).float().to(self.device)[:, 1:]

        # mask: 全位置有効（パディングがないため全True）
        mask = torch.ones_like(partner, dtype=torch.bool, device=self.device)

        return seq, partner, target, mask

    # ------------------------------------------------------------
    # 推論モード: shape reactivity 予測
    # ------------------------------------------------------------
    @torch.no_grad()
    def predict(self, seq_list, partner_list):
        preds = []
        for seq, partner in zip(seq_list, partner_list):
            seq, partner, _, mask = self._prepare_inputs(seq, partner)
            pred = self.model(seq, partner, mask)
            preds.append(pred.squeeze(0).detach().cpu())
        return preds

    # ------------------------------------------------------------
    # 学習モード: loss 計算（MSE または MAE）
    # ------------------------------------------------------------
    def forward(self, seq_list, partner_list, target_list):
        losses = []
        for seq, partner, target in zip(seq_list, partner_list, target_list):
            seq, partner, target, mask = self._prepare_inputs(seq, partner, target)
            pred = self.model(seq, partner, mask)

            # target の欠損位置を除外
            valid = torch.isfinite(target) & (target > -999)
            if valid.sum() == 0:
                continue

            if self.loss_type == "mae":
                loss = torch.mean(torch.abs(pred[valid] - target[valid]))
            else:
                loss = torch.mean((pred[valid] - target[valid]) ** 2)
            losses.append(loss)

        if not losses:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(losses).mean()
