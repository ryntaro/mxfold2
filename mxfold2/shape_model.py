from argparse import Namespace
import torch.nn as nn

def build_shape_model(args: Namespace) -> nn.Module:
    if args.shape_model == 'Wu':
        from .fold.shape_layers import Wu
        return Wu(xi=0.774, mu=0.078, sigma=0.083, alpha=1.006, beta=1.404)
        
        # error をおこすparameter
        # return Wu(xi=2.8856, mu=0.0135, sigma=0.0100, alpha=0.7617, beta=1.3015)

    elif args.shape_model == 'Foo':
        from .fold.shape_layers import Foo
        return Foo(p_alpha=0.540, p_beta=1.390, u_alpha=1.006, u_beta=1.404)
    elif args.shape_model == 'MLP':
        from .loss.predict_shape import ShapeMLP
        return ShapeMLP()
    elif args.shape_model == 'External':
        from .loss.external_shape_predictor import ExternalShapePredictor
        # 例: args から外部モデルロード関数やパスを受け取る実装に置き換えてください
        return ExternalShapePredictor(predictor=None)
    else:
        raise ValueError(f'not implemented: {args.shape_model}')
