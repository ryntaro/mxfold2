from __future__ import annotations

import logging
import os
import random
import time
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, cast

import pytorch_optimizer as po
#import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
from torch.amp.grad_scaler import GradScaler
from torch.amp import autocast
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from . import interface
from .dataset import BPseqDataset, FastaDataset, ShapeDataset, MultiTaskDataset
from .fold.fold import AbstractFold
from .common import Common

# shape回帰のために導入
from .fold.layers import NeuralNet, NeuralNet1D 
from .shape_model import build_shape_model

try:
    from torch.utils.tensorboard.writer import SummaryWriter
except ImportError:
    pass


class Train(Common):
    step: int = 0
    disable_progress_bar: bool
    writer: Optional[SummaryWriter]

    def __init__(self):
        super(Train, self).__init__()
        self.log_file = None   # ここでは設定しない


    def train(self, epoch: int, model: AbstractFold, optimizer: optim.Optimizer, 
                loss_fn: nn.Module | dict[str, nn.Module], 
                data_loader: DataLoader[tuple[str, str, dict[str, torch.Tensor]]],
                loss_weight = defaultdict(lambda: 1.),
                clip_grad_value: float = 0.0, clip_grad_norm: float = 0.0,
                scaler: Optional[GradScaler] = None, 
                use_amp: bool = False) -> None:
        model.train()
        if not isinstance(loss_fn, dict):
            loss_fn = {'BPSEQ': loss_fn}
        n_dataset = len(cast(FastaDataset, data_loader.dataset))
        loss_total, num = 0., 0
        running_loss, n_running_loss = 0, 0
        start = time.time()
        with tqdm(total=n_dataset, disable=self.disable_progress_bar) as pbar:
            for fnames, seqs, vals in data_loader:
                logging.info(f"Step: {self.step}, {fnames}")
                self.step += 1
                n_batch = len(seqs)
                for i in range(n_batch):
                    optimizer.zero_grad()
                                        
                    # Use autocast for mixed precision if enabled
                    with autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                        if vals['type'][i]=='BPSEQ':
                            loss = torch.sum(loss_fn['BPSEQ'](seqs[i:i+1], vals['target'][i:i+1], fname=fnames[i:i+1]))
                        elif vals['type'][i]=='SHAPE': 
                            loss = torch.sum(loss_fn['SHAPE'](seqs[i:i+1], vals['target'][i:i+1], fname=fnames[i:i+1], dataset_id=vals['dataset_id'][i:i+1]))
                        elif vals['type'][i]=='MULTI':
                            seq = seqs[i:i+1]
                            # 構造ロス
                            struct_loss = loss_fn['BPSEQ'](seq, vals['target'][i:i+1], fname=fnames[i:i+1])
                            # shape回帰ロス 関数化したかったがloss_fnにくわえるのが難しそうなので直で書くことにした
                            tgt = vals['shape_target'][i:i+1]
                            msk = vals['shape_mask'][i:i+1]
                            shape_loss = loss_fn['SHAPE_regress'](seq, tgt, msk)

                            alpha, beta = loss_weight['MULTI_intra']
                            loss = torch.sum(alpha * struct_loss + beta * shape_loss)                        
                        else:
                            raise(RuntimeError('not implemented'))

                        # if torch.isnan(loss) or torch.isinf(loss):
                        #     logging.warning(f"Skip NaN sample: {fnames[i]}")
                        #     continue

                        # 交互学習のためにタスク形式で重みが異なる
                        loss = loss * loss_weight[vals['type'][i]]
                    
                    loss_total += loss.item()
                    running_loss += loss.item()
                    
                    # Scale loss and backward pass
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    # Gradient clipping with unscaling if using mixed precision
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        
                    if clip_grad_norm > 0.0:
                        nn.utils.clip_grad_norm_(model.parameters(),  max_norm=clip_grad_norm, norm_type=2)
                    elif clip_grad_value > 0.0:
                        nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_grad_value)
                    
                    # Step the optimizer
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    # --- パラメータ制約（Wu など shape_model 用） ---
                    with torch.no_grad():
                        if hasattr(loss_fn, "shape_model") and loss_fn.shape_model is not None:
                            for sm in loss_fn.shape_model:
                                if hasattr(sm, "xi"):
                                    sm.xi.clamp_(min=1e-2, max=2.0)
                                if hasattr(sm, "mu"):
                                    sm.mu.clamp_(min=0.0, max=2.0)
                                if hasattr(sm, "sigma"):
                                    sm.sigma.clamp_(min=1e-2, max=2.0)
                                if hasattr(sm, "alpha"):
                                    sm.alpha.clamp_(min=1e-2, max=5.0)
                                if hasattr(sm, "beta"):
                                    sm.beta.clamp_(min=1e-2, max=5.0)
                    
                    # バッチ単位の情報は標準出力へ出す（ジョブの .o に流れる）。
                    # loss.log にはエポック単位で一行だけ書くため、ここではファイル追記しない。
                    if torch.cuda.is_available():
                        alloc = torch.cuda.memory_allocated() / 1e6
                        resv = torch.cuda.memory_reserved() / 1e6
                        peak = torch.cuda.max_memory_allocated() / 1e6
                        print(f"[Epoch {epoch} Step {self.step}] iter={i} loss={loss.item():.6e} alloc={alloc:.1f}MB resv={resv:.1f}MB peak={peak:.1f}MB")
                    else:
                        print(f"[Epoch {epoch} Step {self.step}] iter={i} loss={loss.item():.6e}")

                num += n_batch
                pbar.set_postfix(train_loss='{:.3e}'.format(loss_total / num))
                pbar.update(n_batch)

                n_running_loss += n_batch
                if n_running_loss >= 100 or num >= n_dataset:
                    running_loss /= n_running_loss
                    if self.writer is not None:
                        self.writer.add_scalar("train/loss", running_loss, (epoch-1) * n_dataset + num)
                    running_loss, n_running_loss = 0, 0
        # elapsed_time = time.time() - start
        # print('Train Epoch: {}\tLoss: {:.6f}\tTime: {:.3f}s'.format(epoch, loss_total / num, elapsed_time))

        elapsed_time = time.time() - start
        avg_loss = loss_total / num
        log_line = f"Train Epoch: {epoch}\tLoss: {avg_loss:.6f}\tTime: {elapsed_time:.3f}s"
        print(log_line)
        # エポック単位のログは loss.log に一行だけ書く（alloc のみ記録、resv/peak は 0.0 で埋める）
        try:
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1e6
            else:
                alloc = 0.0
            with open(self.log_file, "a") as f:
                f.write(f"{epoch},train,{avg_loss:.6f},{elapsed_time:.3f},{self.step},{alloc:.1f}\n")
        except Exception:
            pass


    def test(self, epoch: int, model: AbstractFold | AveragedModel, 
                loss_fn: nn.Module | dict[str, nn.Module],
                data_loader: DataLoader[tuple[str, str, dict[str, torch.Tensor]]],
                use_amp: bool = False) -> None:
        model.eval()
        if not isinstance(loss_fn, dict):
            loss_fn = {'BPSEQ': loss_fn}
        n_dataset = len(cast(FastaDataset, data_loader.dataset))
        loss_total, num = 0, 0
        start = time.time()
        with torch.no_grad(), tqdm(total=n_dataset, disable=self.disable_progress_bar) as pbar:
            for fnames, seqs, vals in data_loader:
                n_batch = len(seqs)
                for i in range(n_batch):
                    with autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                        if vals['type'][i]=='BPSEQ':
                            loss = torch.sum(loss_fn['BPSEQ'](seqs[i:i+1], vals['target'][i:i+1], fname=fnames[i:i+1]))
                        elif vals['type'][i]=='SHAPE': 
                            loss = torch.sum(loss_fn['SHAPE'](seqs[i:i+1], vals['target'][i:i+1], fname=fnames[i:i+1], dataset_id=vals['dataset_id'][i:i+1]))
                        elif vals['type'][i]=='MULTI':
                            seq = seqs[i:i+1]
                            # 構造ロス
                            struct_loss = loss_fn['BPSEQ'](seq, vals['target'][i:i+1], fname=fnames[i:i+1])
                            # shape回帰ロス
                            tgt = vals['shape_target'][i:i+1]
                            msk = vals['shape_mask'][i:i+1]
                            shape_loss = loss_fn['SHAPE_regress'](seq, tgt, msk)

                            # loss = self.mt_alpha * struct_loss + self.mt_beta * shape_loss                        
                            
                            alpha, beta = loss_weight['MULTI_intra']
                            loss = torch.sum(alpha * struct_loss + beta * shape_loss)                        
                       
                        else:
                            raise(RuntimeError('not implemented'))
                    loss_total += loss.item()
                num += n_batch
                pbar.set_postfix(test_loss='{:.3e}'.format(loss_total / num))
                pbar.update(n_batch)

        # elapsed_time = time.time() - start
        # if self.writer is not None:
        #     self.writer.add_scalar("test/loss", loss_total / num, epoch * n_dataset)
        # print('Test Epoch: {}\tLoss: {:.6f}\tTime: {:.3f}s'.format(epoch, loss_total / num, elapsed_time))

        elapsed_time = time.time() - start
        avg_loss = loss_total / num
        log_line = f"Test Epoch: {epoch}\tLoss: {avg_loss:.6f}\tTime: {elapsed_time:.3f}s"
        print(log_line)

        with open(self.log_file, "a") as f:
            f.write(f"{epoch},test,{avg_loss:.6f},{elapsed_time:.3f}\n")


    def save_checkpoint(self, outdir: str, epoch: int, 
                        model: AbstractFold | AveragedModel, 
                        optimizer: optim.Optimizer, 
                        scheduler,
                        shape_model: Optional[list[nn.Module]] = None,
                        step: Optional[int] = None,
                        scaler: Optional[GradScaler] = None) -> None:
        filename = os.path.join(outdir, 'epoch-{}'.format(epoch))
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step if step is not None else self.step,
            'rng_state': {
                'torch': torch.get_rng_state(),
                'python': random.getstate(),
            }}
        if torch.cuda.is_available():
            checkpoint['rng_state']['cuda'] = torch.cuda.get_rng_state_all()
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        if shape_model is not None:
            checkpoint['shape_model_state_dict'] = [ sm.state_dict() for sm in shape_model ]
        if scaler is not None:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        torch.save(checkpoint, filename)


    def resume_checkpoint(self, filename: str, 
                        model: AbstractFold, 
                        optimizer: optim.Optimizer, 
                        scheduler,
                        shape_model: Optional[list[nn.Module]],
                        scaler: Optional[GradScaler] = None,
                        gpu: int = -1) -> tuple[int, Optional[GradScaler]]:
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore step counter
        if 'step' in checkpoint:
            self.step = checkpoint['step']
        
        # Restore random states
        if 'rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['rng_state']['torch'])
            random.setstate(checkpoint['rng_state']['python'])
            if 'cuda' in checkpoint['rng_state'] and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(checkpoint['rng_state']['cuda'])
        
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if shape_model is not None and 'shape_model_state_dict' in checkpoint:
            for i, sm in enumerate(shape_model):
                sm.load_state_dict(checkpoint['shape_model_state_dict'][i])
        
        # Load GradScaler state if available
        if 'scaler_state_dict' in checkpoint and scaler is None:
            device = f'cuda:{gpu}' if gpu >= 0 else 'cuda'
            scaler = GradScaler(device)
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        elif scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        return epoch, scaler


    def build_optimizer(self, optimizer: str, model: AbstractFold, lr: float, l2_weight: float,
                        shape_model: Optional[list[nn.Module]] = None) -> optim.Optimizer:
        # if hasattr(model, 'zuker') and hasattr(model, 'turner'):
        #     optim_params = [
        #         {'params': model.zuker.parameters(), 'lr': lr, 'weight_decay': l2_weight},
        #         {'params': model.turner.parameters(), 'lr': lr*10, 'weight_decay': l2_weight/10},
        #     ]
        # else:
        #     optim_params = [
        #         {'params': model.parameters(), 'lr': lr, 'weight_decay': l2_weight},
        #     ]
        optim_params = [
            {'params': model.parameters(), 'lr': lr, 'weight_decay': l2_weight},
        ]
        if shape_model is not None:
            for sm in shape_model:
                optim_params.append({'params': sm.parameters(), 'lr': lr, 'weight_decay': l2_weight})

        if optimizer == 'Adam':
            return optim.Adam(optim_params, amsgrad=False)
        elif optimizer =='AdamW':
            return optim.AdamW(optim_params, amsgrad=False)
        elif optimizer == 'RMSprop':
            return optim.RMSprop(optim_params)
        elif optimizer == 'SGD':
            return optim.SGD(optim_params, nesterov=True, momentum=0.9)
            #return optim.SGD(optim_params)
        elif optimizer == 'ASGD':
            return optim.ASGD(optim_params)
        elif optimizer == 'AdaBelief':
            return po.AdaBelief(optim_params)
        elif optimizer == 'Lion':
            return po.Lion(optim_params)
        else:
            raise(RuntimeError('not implemented'))


    def build_loss_function(self, loss_func: str, model: AbstractFold, args: Namespace) -> nn.Module:
        if loss_func == 'hinge' or loss_func == 'hinge_mix':
            from .loss.structured_loss import StructuredLoss
            return StructuredLoss(model,
                            loss_pos_paired=args.loss_pos_paired, loss_neg_paired=args.loss_neg_paired, 
                            loss_pos_unpaired=args.loss_pos_unpaired, loss_neg_unpaired=args.loss_neg_unpaired, 
                            l1_weight=args.l1_weight, l2_weight=args.l2_weight, sl_weight=args.score_loss_weight)

        if loss_func == 'fy' or loss_func == 'fy_mix':
            from .loss.fy_loss import FenchelYoungLoss
            return FenchelYoungLoss(model, perturb=args.perturb, l1_weight=args.l1_weight, l2_weight=args.l2_weight,
                                                sl_weight=args.score_loss_weight)

        if loss_func == 'f1':
            from .loss.f1_loss import F1Loss
            return F1Loss(model, perturb=args.perturb, nu=args.nu, l1_weight=args.l1_weight, l2_weight=args.l2_weight,
                            sl_weight=args.score_loss_weight)

        else:
            raise(ValueError(f'not implemented: {loss_func}'))

    # shape_model.pyに分離させた
    # def build_shape_model(self, args: Namespace) -> nn.Module:
    #     if args.shape_model == 'Wu':
    #         from .fold.shape_layers import Wu
    #         return Wu(xi=0.774, mu=0.078, sigma=0.083, alpha=1.006, beta=1.404)
    #     elif args.shape_model == 'Foo':
    #         from .fold.shape_layers import Foo
    #         return Foo(p_alpha=0.540, p_beta=1.390, u_alpha=1.006, u_beta=1.404)
    #     elif args.shape_model == 'MLP':
    #         from .loss.predict_shape import ShapeMLP
    #         return ShapeMLP()
    #     else:
    #         raise(ValueError(f'not implemented: {args.shape_model}'))


    def build_shape_loss_function(self, loss_func: str, model: AbstractFold, args: Namespace,
                                shape_model: Optional[nn.Module] = None) -> nn.Module:
        if loss_func == 'shape_nll':
            from .loss.shape_nll_loss import ShapeNLLLoss
            return ShapeNLLLoss(model=model,
                            shape_model=shape_model,
                            perturb=args.shape_perturb, nu=args.shape_nu, 
                            l1_weight=args.l1_weight, l2_weight=args.l2_weight,
                            sl_weight=args.score_loss_weight)

        elif loss_func == 'shape_fy':
            from .loss.shape_fy_loss import ShapeFenchelYoungLoss
            return ShapeFenchelYoungLoss(model,
                            perturb=args.shape_perturb, 
                            shape_intercept=args.shape_intercept, shape_slope=args.shape_slope,
                            l1_weight=args.l1_weight, l2_weight=args.l2_weight,
                            sl_weight=args.score_loss_weight)

        if loss_func == 'shape_mse':
            from .loss.shape_mse_loss import ShapeMSELoss
            return ShapeMSELoss(model=model,
                            shape_model=shape_model,
                            perturb=args.shape_perturb, nu=args.shape_nu, 
                            l1_weight=args.l1_weight, l2_weight=args.l2_weight,
                            sl_weight=args.score_loss_weight)

        else:
            raise(ValueError(f'not implemented: {loss_func}'))

    def build_shape_regress_loss_function(self, model: AbstractFold):
        """E0: SHAPE回帰(MSE)の損失関数を返すファクトリ"""
        def shape_e0_loss_fn(seq, target, mask):
            pred = model.zuker.net.predict_shape(seq)      
            # マスク付きMSE
            device = pred.device
            target = target.to(device)
            mask = mask.to(device)
            
            diff = (pred - target) * mask
            denom = mask.sum().clamp_min(1)
            return diff.pow(2).sum() / denom
        return shape_e0_loss_fn

    def build_scheduler(self, scheduler: str, optimizer: optim.Optimizer, args: Namespace):
        if scheduler == 'CyclicLR':
            return optim.lr_scheduler.CyclicLR(optimizer=optimizer, base_lr=0.001, max_lr=args.lr,
                                                step_size_up=args.scheduler_step_size, gamma=args.scheduler_gamma, mode="exp_range")
        if scheduler == 'CosineAnnealingLR':
            return optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.scheduler_step_size)
        
        return None


    def save_config(self, file: str, config: dict[str, Any]) -> None:
        with open(file, 'w') as f:
            for k, v in config.items():
                k = '--' + k.replace('_', '-')
                if isinstance(v, bool):
                    if v:
                        f.write('{}\n'.format(k))
                elif isinstance(v, list) or isinstance(v, tuple):
                    for vv in v:
                        f.write('{}\n{}\n'.format(k, vv))
                elif v is not None:
                    f.write('{}\n{}\n'.format(k, v))


    def run(self, args: Namespace, conf: Optional[str] = None) -> None:
        self.disable_progress_bar = args.disable_progress_bar
        loglevel = 'INFO' if args.verbose else args.loglevel
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=getattr(logging, loglevel, None))
        self.writer = None
        if args.log_dir is not None and 'SummaryWriter' in globals():
            self.writer = SummaryWriter(log_dir=args.log_dir)
        
        self.mt_alpha = args.mt_alpha
        self.mt_beta = args.mt_beta

        # loss.log の出力先を log_dir に変更
        if args.log_dir is not None:
            os.makedirs(args.log_dir, exist_ok=True)
            self.log_file = os.path.join(args.log_dir, "loss.log")
        else:
            os.makedirs("logs", exist_ok=True)
            self.log_file = os.path.join("logs", "loss.log")
        # ファイルをリセット
        with open(self.log_file, "w") as f:
            # 追加列: step, alloc_mb,
            f.write("epoch,phase,loss,time,step,alloc_mb\n")

        # train_dataset = BPseqDataset(args.input)
        # if args.shape is not None:
        #     shape_dataset = [ ShapeDataset(s, i) for i, s in enumerate(args.shape) ]
        #     train_dataset = ConcatDataset([train_dataset] + shape_dataset)

        # タスクによるデータセット生成の分岐（マルチタスクを生やす）
        task = args.task
        if task == 'Folding':
            # 構造のみ（BPSEQ系データセットだけを使う）
            train_dataset = BPseqDataset(args.input)
        elif task == 'Implicit_MLE':
            train_dataset = BPseqDataset(args.input)
            # SHAPE拘束つき構造予測（従来の ShapeDataset を追加）
            if not args.shape:
               raise ValueError("Implicit_MLE には --shape リストが必須です。")
            # ShapeDataset は従来の 'type':'SHAPE' を返すモードで作成
            shape_dataset = [ ShapeDataset(s, i) for i, s in enumerate(args.shape) ]
            train_dataset = ConcatDataset([train_dataset] + shape_dataset)
        elif task == 'Multitask':
            # 同一配列で BPSEQ + SHAPE を同時学習
            if not args.shape:
                raise ValueError("Multitask には --shape リストが必須です。")
            # 複数ファイルのときは dataset_id ごとに MultiTaskDataset を作って連結
            mt_datasets = []
            bpseq_lists = args.input if isinstance(args.input, (list,tuple)) else [args.input]
            shape_lists = args.shape if isinstance(args.shape, (list,tuple)) else [args.shape]
            if len(bpseq_lists) != len(shape_lists):
                raise ValueError("Multitask では BPSEQリストと SHAPEリストの本数を一致させてください。")
            for i, (bp, sh) in enumerate(zip(bpseq_lists, shape_lists)):
                mt_datasets.append(MultiTaskDataset(bpseq_list=bp, shape_list=sh, dataset_id=i))
            train_dataset = ConcatDataset(mt_datasets)
        else:
            raise ValueError(f"unknown task: {task}")

        # Create generator for reproducible shuffling
        generator = torch.Generator()
        if args.seed >= 0:
            generator.manual_seed(args.seed)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, generator=generator) # works well only for batch_size=1!!
        if args.test_input is not None:
            test_dataset = BPseqDataset(args.test_input)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # works well only for batch_size=1!!
        else:
            test_loader = None

        if args.seed >= 0:
            torch.manual_seed(args.seed)
            random.seed(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        model, config = self.build_model(args)
        config.update({ 'model': args.model, 'param': args.param, 'fold': args.fold })

        shape_model = None 
        if args.task == "Implicit_MLE":
            shape_model = [ build_shape_model(args) for _ in args.shape ]
        
        if args.init_param != '':
            init_param = Path(args.init_param)
            if not init_param.exists() and conf is not None:
                init_param = Path(conf) / init_param
            p = torch.load(init_param)
            if shape_model is not None and isinstance(p, dict) and 'shape_model_state_dict' in p:
                for i, sm in enumerate(shape_model):
                    sm.load_state_dict(p['shape_model_state_dict'][i])
            if isinstance(p, dict) and 'model_state_dict' in p:
                p = p['model_state_dict']
            model.load_state_dict(p)

        if args.gpu >= 0:
            model.to(torch.device("cuda", args.gpu))
            if shape_model is not None:
                for sm in shape_model:
                    sm.to(torch.device("cuda", args.gpu))

        torch.set_num_threads(args.threads)
        interface.set_num_threads(args.threads)

        optimizer = self.build_optimizer(args.optimizer, model, args.lr, args.l2_weight, shape_model=shape_model)

        loss_fn = {
            'BPSEQ': self.build_loss_function(args.loss_func, model, args), 
            'SHAPE': self.build_shape_loss_function(args.shape_loss_func, model, args, shape_model=shape_model), 
            'SHAPE_regress': self.build_shape_regress_loss_function(model) 
        }
       
        loss_weight = { 'BPSEQ': 1.0, 'SHAPE': args.shape_loss_weight, 'MULTI': 1.0, 'MULTI_intra': (args.mt_alpha, args.mt_beta) }
        scheduler = self.build_scheduler(args.scheduler, optimizer, args)

        # Initialize GradScaler for mixed precision training
        scaler = None
        use_amp = False
        if hasattr(args, 'use_amp') and args.use_amp and args.gpu >= 0:
            use_amp = True
            scaler = GradScaler(f'cuda:{args.gpu}')
            logging.info("Using Automatic Mixed Precision (AMP) training")

        checkpoint_epoch = 0
        if args.resume is not None:
            checkpoint_epoch, resumed_scaler = self.resume_checkpoint(args.resume, model, optimizer, scheduler, shape_model, scaler, args.gpu)
            if resumed_scaler is not None:
                scaler = resumed_scaler
                use_amp = True
                logging.info("Resumed with Automatic Mixed Precision (AMP) training")
        
        if args.swa:
            swa_model = AveragedModel(model)
            swa_start = args.swa_start if args.swa_start > 1.0 else args.epochs * args.swa_start
            swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr, 
                                    anneal_epochs=args.swa_anneal_epochs,
                                    anneal_strategy=args.swa_anneal_strategy)
        else:
            swa_start = args.epochs
            swa_model = None
            swa_scheduler = None

        for epoch in range(checkpoint_epoch+1, args.epochs+1):
            # Set random seed for this epoch to ensure reproducible shuffling
            if args.seed >= 0:
                epoch_seed = args.seed + epoch
                generator.manual_seed(epoch_seed)
            self.train(epoch, model=model, optimizer=optimizer, loss_fn=loss_fn, data_loader=train_loader,
                        loss_weight=loss_weight, clip_grad_value=args.clip_grad_value, clip_grad_norm=args.clip_grad_norm,
                        scaler=scaler, use_amp=use_amp)
            if swa_model is not None and swa_scheduler is not None and epoch > swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
                logging.info(f'LR = {swa_scheduler.get_last_lr()}')
            elif scheduler is not None:
                scheduler.step()
                logging.info(f'LR = {scheduler.get_last_lr()}')

            if test_loader is not None:
                self.test(epoch, model=swa_model or model, loss_fn=loss_fn, data_loader=test_loader, use_amp=use_amp)
            if args.log_dir is not None:
                self.save_checkpoint(args.log_dir, epoch, model, optimizer, scheduler, shape_model, step=self.step, scaler=scaler)

        if args.param is not None:
            torch.save((swa_model or model).state_dict(), args.param)
        if args.save_config is not None:
            self.save_config(args.save_config, config)
        
        #return self.model



    @classmethod
    def add_args(cls, parser):
        subparser = parser.add_parser('train', help='training')
        # input
        subparser.add_argument('input', type=str,
                            help='Training data of the list of BPSEQ-formatted files')
        subparser.add_argument('--test-input', type=str,
                            help='Test data of the list of BPSEQ-formatted files')
        subparser.add_argument('--gpu', type=int, default=-1, 
                            help='use GPU with the specified ID (default: -1 = CPU)')
        subparser.add_argument('--threads', type=int, default=1, metavar='N',
                            help='the number of threads (default: 1)')
        subparser.add_argument('--seed', type=int, default=0, metavar='S',
                            help='random seed (default: 0)')
        subparser.add_argument('--param', type=str, default='param.pth',
                            help='output file name of trained parameters')
        subparser.add_argument('--init-param', type=str, default='',
                            help='the file name of the initial parameters')
        subparser.add_argument('--shape', type=str, action='append', help='specify the file name that includes SHAPE reactivity')
        # subparser.add_argument('--shape-intercept', type=float, default=-0.8,
        #                     help='Specify an intercept used with SHAPE restraints. Default is -0.8 kcal/mol.')
        # subparser.add_argument('--shape-slope', type=float, default=2.6, 
        #                     help='Specify a slope used with SHAPE restraints. Default is 2.6 kcal/mol.')

        gparser = subparser.add_argument_group("Training environment")
        subparser.add_argument('--epochs', type=int, default=10, metavar='N',
                            help='number of epochs to train (default: 10)')
        subparser.add_argument('--log-dir', type=str, default=None,
                            help='Directory for storing logs')
        subparser.add_argument('--resume', type=str, default=None,
                            help='Checkpoint file for resume')
        subparser.add_argument('--save-config', type=str, default=None,
                            help='save model configurations')
        subparser.add_argument('--disable-progress-bar', action='store_true',
                            help='disable the progress bar in training')
        subparser.add_argument('--verbose', action='store_true',
                            help='enable verbose outputs for debugging')
        subparser.add_argument('--loglevel', choices=('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'),
                            default='WARNING', help="set the log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')")
        subparser.add_argument('--use-amp', action='store_true',
                            help='use automatic mixed precision (AMP) for faster training on GPUs')
        
        cls.add_task_args(subparser)
        
        cls.add_fold_args(subparser)

        gparser = subparser.add_argument_group("Setting for optimizer")
        gparser.add_argument('--optimizer', choices=('Adam', 'AdamW', 'RMSprop', 'SGD', 'ASGD', 'AdaBelief', 'Lion'), default='AdamW')
        gparser.add_argument('--lr', type=float, default=0.001,
                            help='the learning rate for optimizer (default: 0.001)')
        gparser.add_argument('--clip-grad-value', type=float, default=0.,
                            help='gradient clipping by values (default=0, no clipping)')
        gparser.add_argument('--clip-grad-norm', type=float, default=0.,
                            help='gradient clipping by norm (default=0, no clipping)')
        gparser.add_argument('--scheduler', choices=('None', 'CyclicLR', 'CosineAnnealingLR'), default='None',
                            help="learning rate scheduler ('None', 'CyclicLR', 'CosineAnnealingLR')")
        gparser.add_argument('--scheduler-step-size', type=int, default=5, help='scheduler step size (default=5)')
        gparser.add_argument('--scheduler-gamma', type=float, default=0.95, help='scheduler decoy rate (default: 0.95)')
        gparser.add_argument('--swa', default=False, action='store_true',
                            help='use stochastic weight averaging (SWA)')
        gparser.add_argument('--swa-start', type=float, default=0.75, 
                            help='start epoch of SWA: epochs * swa_start for swa_start<1.0, or swa_start otherwise')
        gparser.add_argument('--swa-anneal-epochs', type=int, default=10, help='SWA anneal-epochs (default: 10)')
        gparser.add_argument('--swa-anneal-strategy', choices=('linear', 'cos'), default='linear',
                            help="SWA anneal strategy ('linear', 'cos')")
        gparser.add_argument('--swa-lr', type=float, default=0.01, help='SWA learning rate (default: 0.01)')

        gparser = subparser.add_argument_group("Setting for loss function")
        gparser.add_argument('--loss-func', choices=('hinge', 'hinge_mix', 'fy', 'fy_mix', 'f1'), default='hinge',
                            help="loss fuction (default: hinge)")
        gparser.add_argument('--l1-weight', type=float, default=0.,
                            help='the weight for L1 regularization (default: 0)')
        gparser.add_argument('--l2-weight', type=float, default=0.,
                            help='the weight for L2 regularization (default: 0)')
        gparser.add_argument('--score-loss-weight', type=float, default=0.,
                            help='the weight for score loss for {hinge,fy} loss (default: 0)')
        gparser.add_argument('--perturb', type=float, default=0.1,
                            help='standard deviation of perturbation for fy loss (default: 0.1)')
        gparser.add_argument('--nu', type=float, default=0.1,
                            help='weight for distribution (default: 0.1)')
        gparser.add_argument('--loss-pos-paired', type=float, default=0.5,
                            help='the penalty for positive base-pairs for loss augmentation (default: 0.5)')
        gparser.add_argument('--loss-neg-paired', type=float, default=0.005,
                            help='the penalty for negative base-pairs for loss augmentation (default: 0.005)')
        gparser.add_argument('--loss-pos-unpaired', type=float, default=0.,
                            help='the penalty for positive unpaired bases for loss augmentation (default: 0)')
        gparser.add_argument('--loss-neg-unpaired', type=float, default=0.,
                            help='the penalty for negative unpaired bases for loss augmentation (default: 0)')
        # gparser.add_argument('--shape-model', choices=('Wu', 'Foo', 'MLP'), default='Wu',
        #                     help="shape model nll->Wu, Foo, MSE-> MLP (default: Wu)")
        gparser.add_argument('--shape-loss-func', choices=('shape_nll', 'shape_fy', 'shape_mse'), default='shape_nll',
                            help="loss fuction for SHAPE training data (default: shape)")
        gparser.add_argument('--shape-perturb', type=float, default=0.1,
                            help='standard deviation of perturbation for shape loss (default: 0.1)')
        gparser.add_argument('--shape-nu', type=float, default=0.1,
                            help='weight for distribution for shape loss (default: 0.1)')
        subparser.add_argument('--shape-intercept', type=float, default=-0.8,
                            help='Specify an intercept used with SHAPE restraints. Default is -0.8 kcal/mol.')
        subparser.add_argument('--shape-slope', type=float, default=2.6, 
                            help='Specify a slope used with SHAPE restraints. Default is 2.6.')
        gparser.add_argument('--shape-loss-weight', type=float, default=1.,
                            help='weight for SHAPE loss function (default=1) shape/shapeなしの交互学習における重み RNA間の重み')
        gparser.add_argument('--mt-alpha', type=float, default=1.0, help='multitask: weight for structure loss')
        gparser.add_argument('--mt-beta',  type=float, default=1.0, help='multitask: weight for SHAPE loss')

        cls.add_network_args(subparser)

        subparser.set_defaults(func = lambda args, conf: Train().run(args, conf))
