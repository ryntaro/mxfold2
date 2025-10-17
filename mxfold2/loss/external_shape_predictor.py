import torch
import torch.nn as nn
from typing import Callable, Optional, List
from .predict_shape import ShapeMLP

class ExternalShapePredictor(nn.Module):
    """
    外部に置く shape 予測器のラッパー。

    - predictor: 任意の callable / nn.Module。呼び出し可能なら predictor(seq, paired) を呼ぶ。
                 予測の返り値は list[Tensor] を期待する。predict メソッドがあればそれを使う。
    - predictor が None の場合は既存の ShapeMLP を内部に作ってフォールバックする。
    """
    def __init__(self, predictor: Optional[Callable] = None):
        super().__init__()
        self.external = predictor
        # デフォルト実装として既存の ShapeMLP を保持（必要なら置き換えられる）
        self._default = ShapeMLP()

    def predict(self, seq: List[str], paired: List[torch.Tensor]) -> List[torch.Tensor]:
        # まず外部モデルに委譲
        if self.external is not None:
            # 外部モデルが nn.Module で predict を持つ場合
            if hasattr(self.external, "predict"):
                return self.external.predict(seq, paired)
            # 外部モデルが直接呼び出し可能な場合
            return self.external(seq, paired)

        # フォールバック: internal ShapeMLP の predict を使う
        return self._default.predict(seq, paired)

    # 互換のため forward も用意（loss 側で forward を期待する場合に対応）
    def forward(self, seq: List[str], paired: List[torch.Tensor]):
        return self.predict(seq, paired)