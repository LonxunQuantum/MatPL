"""
TNEP training orchestration.

Extends nep_network to handle Tensorial NEP training for dipole
and polarizability prediction. Reuses all data loading and neighbor
calculation infrastructure from regular NEP.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.user.input_param import InputParam
from src.PWMLFF.nep_network import nep_network
from src.model.tnep_net import TNEP


class tnep_network(nep_network):
    """
    Training orchestration for tNEP (Tensorial NEP).

    Inherits from nep_network to reuse:
      - Data loading (load_data)
      - Neighbor scaler computation (calculate_neighbor_scaler)
      - Checkpoint save/load
      - Multi-GPU DDP setup
      - LR schedulers

    Overrides:
      - load_model_optimizer: creates TNEP instead of NEP
      - train: uses tNEP-specific loss weighting and metrics
      - valid: reports dipole/polarizability metrics
    """

    def load_model_optimizer(self, energy_shift, avg_atom_num=1,
                             iterations=1, q_scaler=None,
                             max_NN_radial=-1, max_NN_angular=-1):
        """Create TNEP model and optimizer, overriding parent's NEP creation."""
        def _adjust_ckpt_keys(ckpt, new_ckpt):
            keys = list(ckpt['state_dict'].keys())
            new_dict = {}

            if 'q_scaler' in keys:
                if self.is_rank_0:
                    print("The checkpoint file from single gpu training!")
                for key in keys:
                    if self.input_param.world_size > 1:
                        new_dict[f'{module}{key}'] = ckpt['state_dict'][key]
                if self.input_param.world_size == 1:
                    new_dict = ckpt['state_dict']

                new_dict[f'{module}q_scaler'] = torch.tensor(
                    list(ckpt['state_dict']['q_scaler']),
                    dtype=new_ckpt.state_dict()[f'{module}c_param_2'].dtype,
                    device=new_ckpt.state_dict()[f'{module}c_param_2'].device)
                for key in ["C3B", "C4B", "C5B", "atom_type_device",
                            "max_NN_radial", "max_NN_angular"]:
                    new_dict[f'{module}{key}'] = new_ckpt.state_dict()[f'{module}{key}']
            else:
                if ("module." in keys[0] and self.input_param.world_size > 1) or \
                   ("module." not in keys[0] and self.input_param.world_size == 1):
                    new_dict = ckpt['state_dict']
                else:
                    for key in keys:
                        if "module." in keys[0] and self.input_param.world_size == 1:
                            new_dict[key.replace("module.", "")] = ckpt['state_dict'][key]
                        else:
                            new_dict[f'module.{key}'] = ckpt['state_dict'][key]
            ckpt['state_dict'] = new_dict
            return ckpt

        # Create TNEP model instead of NEP
        model = TNEP(self.input_param,
                     energy_shift,
                     q_scaler=q_scaler,
                     max_NN_radial=max_NN_radial,
                     max_NN_angular=max_NN_angular,
                     dtype=self.training_type,
                     device=self.device
                     ).to(self.training_type).to(self.device)

        # Wrap model with DDP for multi-GPU
        if torch.cuda.is_available() and self.input_param.world_size > 1:
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.input_param.local_rank],
                output_device=self.input_param.local_rank,
                find_unused_parameters=True)

        checkpoint = None
        model_path = None

        if self.input_param.inference:
            model_path = self.input_param.file_paths.model_load_path
        elif self.input_param.recover_train and \
             self.input_param.file_paths.model_load_path and \
             os.path.exists(self.input_param.file_paths.model_load_path):
            model_path = self.input_param.file_paths.model_load_path
        else:
            if self.input_param.nep_param.model_wb is None:
                if self.input_param.file_paths.model_load_path and \
                   os.path.exists(self.input_param.file_paths.model_load_path):
                    model_path = self.input_param.file_paths.model_load_path
                else:
                    model_path = self.input_param.file_paths.model_save_path
            else:
                model_path = None

        module = 'module.' if self.input_param.world_size > 1 else ''
        if model_path and os.path.isfile(model_path):
            checkpoint = torch.load(model_path, map_location=self.device,
                                    weights_only=False)
            checkpoint = _adjust_ckpt_keys(checkpoint, model)
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            if "epoch" in checkpoint:
                if self.input_param.optimizer_param.reset_epoch and \
                   self.input_param.inference is False:
                    if checkpoint["epoch"] != 1:
                        print(f"Rank {self.input_param.rank}: "
                              f"Resetting epoch to 1 from {checkpoint['epoch']}")
                    self.input_param.optimizer_param.start_epoch = 1
                else:
                    self.input_param.optimizer_param.start_epoch = \
                        checkpoint["epoch"] + 1
            if self.input_param.world_size > 1:
                print(f"Reload ckpt: Rank {self.input_param.rank}, "
                      f"LocalRank {self.input_param.local_rank}, "
                      f"start_epoch: {self.input_param.optimizer_param.start_epoch}")
                dist.barrier()

        # Optimizer setup (reuse Adam/AdamW/SGD/Muon/LKF/GKF from parent)
        scheduler = None
        opt_param = self.input_param.optimizer_param

        if opt_param.opt_name in ["ADAM", "ADAMW", "SGD", "MUON"]:
            if opt_param.warmup is not None:
                init_lr = opt_param.stop_lr
            else:
                init_lr = opt_param.learning_rate

            if opt_param.opt_name == "ADAM":
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=init_lr,
                    weight_decay=opt_param.lambda_2 or 0,
                    fused=torch.cuda.is_available(),
                )
            elif opt_param.opt_name == "ADAMW":
                optimizer = optim.AdamW(
                    model.parameters(),
                    lr=init_lr,
                    weight_decay=opt_param.lambda_2 or 0,
                    fused=torch.cuda.is_available(),
                )
            elif opt_param.opt_name == "SGD":
                optimizer = optim.SGD(
                    model.parameters(),
                    lr=init_lr,
                    momentum=opt_param.momentum or 0.9,
                    weight_decay=opt_param.lambda_2 or 0,
                )
            elif opt_param.opt_name == "MUON":
                from src.optimizer.hybrid_muon import HybridMuonOptimizer
                optimizer = HybridMuonOptimizer(
                    model.parameters(),
                    lr=init_lr,
                    weight_decay=opt_param.lambda_2 or 0,
                )
        elif opt_param.opt_name in ["LKF", "GKF"]:
            from src.optimizer.KFWrapper import KFOptimizerWrapper
            optimizer = KFOptimizerWrapper(
                model.parameters(),
                opt_name=opt_param.opt_name,
                start_lr=opt_param.learning_rate,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_param.opt_name}")

        # EMA
        ema = None
        if opt_param.use_ema:
            from src.optimizer.model_ema import ModelEMA
            ema = ModelEMA(model, decay=opt_param.ema_decay)

        return model, optimizer, scheduler, ema

    def train(self):
        """
        tNEP training loop.

        Reuses parent's data loading and neighbor scaler computation,
        then runs the training loop with tNEP-specific loss and metrics.
        """
        import torch.distributed as dist
        from src.PWMLFF.nep_network import calculate_neighbor_scaler
        from src.PWMLFF.tnep_mods.tnep_trainer import (
            tnep_train_step, tnep_valid_step)

        # Set up compute precision
        if self.input_param.precision == "float32":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        # Load data and compute scalers (same as NEP)
        energy_shift, train_loader, val_loader, forscaler_loader = self.load_data()
        if len(train_loader) < 1:
            print(f"ERROR! Training set size {len(train_loader)} is too small.")

        max_NN_radial, q_scaler = None, None
        local_global_max, local_global_min, local_max_NN_radial, local_min_NN_radial, \
            local_max_NN_angular, local_min_NN_angular = calculate_neighbor_scaler(
                forscaler_loader,
                self.input_param.nep_param.n_max[0],
                self.input_param.nep_param.basis_size[0],
                self.input_param.nep_param.n_max[1],
                self.input_param.nep_param.basis_size[1],
                self.input_param.nep_param.l_max[0],
                self.input_param.nep_param.l_max[1],
                self.input_param.nep_param.l_max[2],
                self.device,
                num_workers=self.input_param.workers)

        if self.input_param.world_size > 1:
            local_global_max_tensor = local_global_max.clone().detach().to(self.device)
            dist.all_reduce(local_global_max_tensor, op=dist.ReduceOp.MAX)
            global_max = local_global_max_tensor

            local_global_min_tensor = local_global_min.clone().detach().to(self.device)
            dist.all_reduce(local_global_min_tensor, op=dist.ReduceOp.MIN)
            global_min = local_global_min_tensor

            max_radial_tensor = torch.tensor([local_max_NN_radial],
                                             dtype=torch.int64, device=self.device)
            dist.all_reduce(max_radial_tensor, op=dist.ReduceOp.MAX)
            max_NN_radial = max_radial_tensor.item()

            max_angular_tensor = torch.tensor([local_max_NN_angular],
                                              dtype=torch.int64, device=self.device)
            dist.all_reduce(max_angular_tensor, op=dist.ReduceOp.MAX)
            max_NN_angular = max_angular_tensor.item()
        else:
            global_max = local_global_max
            global_min = local_global_min
            max_NN_radial = local_max_NN_radial
            max_NN_angular = local_max_NN_angular

        if self.input_param.nep_param.q_scaler is None:
            q_scaler = 1.0 / (global_max - global_min)
            q_scaler = q_scaler.tolist()
        else:
            q_scaler = self.input_param.nep_param.q_scaler
            if self.input_param.nep_param.max_nn_from_txt:
                max_NN_radial = max(self.input_param.nep_param.max_NN_radial,
                                    max_NN_radial)
                max_NN_angular = max(self.input_param.nep_param.max_NN_angular,
                                     max_NN_angular)

        if self.input_param.world_size > 1:
            dist.barrier()

        # Create model and optimizer
        model, optimizer, scheduler, ema = self.load_model_optimizer(
            energy_shift,
            avg_atom_num=1,
            iterations=len(train_loader),
            q_scaler=q_scaler,
            max_NN_radial=max_NN_radial,
            max_NN_angular=max_NN_angular)

        if self.is_rank_0 and not os.path.exists(
                self.input_param.file_paths.model_store_dir):
            os.makedirs(self.input_param.file_paths.model_store_dir)
        if self.input_param.world_size > 1:
            dist.barrier()

        train_mode = self.input_param.nep_param.train_mode
        mode_name = {1: "dipole", 2: "polarizability"}.get(train_mode, "potential")
        start_epoch = self.input_param.optimizer_param.start_epoch
        epochs = self.input_param.optimizer_param.epochs
        start_lr = self.input_param.optimizer_param.learning_rate
        stop_lr = self.input_param.optimizer_param.stop_lr

        print(f"tNEP training: mode={mode_name}, epochs={epochs}, "
              f"start_lr={start_lr}, stop_lr={stop_lr}")

        for epoch in range(start_epoch, epochs + 1):
            # Learning rate schedule
            real_lr = self._get_lr(epoch, start_lr, stop_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = real_lr

            # Training
            train_loss, train_rmse = tnep_train_step(
                train_loader, model, optimizer, epoch, real_lr, start_lr,
                self.device, self.input_param)

            # Validation
            if val_loader is not None and len(val_loader) > 0:
                valid_loss, valid_rmse = tnep_valid_step(
                    val_loader, model, epoch, self.device, self.input_param)

                if self.is_rank_0:
                    comp_names = {1: ["mu_x", "mu_y", "mu_z"],
                                  2: ["a_xx", "a_yy", "a_zz",
                                      "a_xy", "a_yz", "a_zx"]}.get(train_mode, [])
                    print(f"Epoch {epoch:4d} | "
                          f"Train {mode_name}_RMSE: {train_rmse:.6f} | "
                          f"Valid {mode_name}_RMSE: {valid_rmse:.6f} | "
                          f"LR: {real_lr:.2e}")
                    if comp_names and len(comp_names) <= 6:
                        # Per-component RMSE from valid step
                        pass  # detailed reporting handled in valid step

            # Save checkpoint
            if self.is_rank_0:
                self._save_checkpoint(model, optimizer, epoch, q_scaler)

        if self.is_rank_0:
            print(f"tNEP ({mode_name}) training completed.")

    def _get_lr(self, epoch, start_lr, stop_lr):
        """Compute learning rate for current epoch (exponential decay)."""
        from src.loss.dploss import adjust_lr
        stop_step = self.input_param.optimizer_param.stop_step
        decay_step = self.input_param.optimizer_param.decay_step
        if stop_step is None:
            stop_step = self.input_param.optimizer_param.epochs * 1000
        if decay_step is None:
            decay_step = max(1, stop_step // 200)
        return adjust_lr((epoch - 1) * 1000, start_lr, stop_step, decay_step, stop_lr)

    def _save_checkpoint(self, model, optimizer, epoch, q_scaler):
        """Save training checkpoint."""
        from src.PWMLFF.nep_network import save_checkpoint
        save_path = os.path.join(
            self.input_param.file_paths.model_store_dir,
            f"epoch_{epoch:03d}.ckpt")
        module = model.module if self.input_param.world_size > 1 else model
        state_dict = module.state_dict()
        if hasattr(module, 'get_checkpoint_state'):
            state_dict = module.get_checkpoint_state()
        save_checkpoint(
            save_path, epoch, state_dict,
            optimizer, self.input_param.file_paths.json_file,
            self.input_param.atom_type, q_scaler,
            self.input_param.nep_param.train_mode)

    def load_checkpoint(self):
        """Load checkpoint for inference/testing."""
        return super().train()  # Use parent's logic but with TNEP model
