
import os
import time 
import random
import numpy as np
from tqdm import tqdm
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

# Conditional imports for distributed
try:
    import torch.distributed as dist
    from torch.utils.data import DistributedSampler
    from torch.nn.parallel import DistributedDataParallel
    DIST_AVAILABLE = True
except:
    DIST_AVAILABLE = False

from _cfg import cfg
from _dataset import CustomDataset
from _model import ModelEMA, Net
from _utils import format_time

from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    import wandb
except ImportError:
    wandb = None

try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    wandb_key = user_secrets.get_secret("wandb_key")
    if wandb is not None and wandb_key:
        wandb.login(key=wandb_key)
except:
    pass

def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def setup(rank, world_size):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    return

def cleanup():
    dist.barrier()
    dist.destroy_process_group()
    return


def setup_gpu_mode(cfg):
    """Auto-detect GPU configuration and set distributed mode"""
    import os
    
    # Check if running with torchrun (distributed)
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        cfg.distributed = True
        cfg.local_rank = int(os.environ["RANK"])
        cfg.world_size = int(os.environ["WORLD_SIZE"])
        print(f"[Distributed] Rank: {cfg.local_rank}, World size: {cfg.world_size}")
    else:
        cfg.distributed = False
        cfg.local_rank = 0
        cfg.world_size = 1
        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if available_gpus == 0:
            print("[Single GPU] No GPUs available, using CPU")
            cfg.device = torch.device("cpu")
        else:
            print(f"[Single GPU] Using 1 GPU out of {available_gpus} available")
            cfg.device = torch.device("cuda:0")

def main_single_gpu(cfg):
    """Single GPU training"""
    print("="*25)
    print("Running on single GPU")
    print("="*25)
    
    # Print mode info
    mode_str = "72x72" if cfg.use_72x72_mode else "Full Resolution"
    print(f"Training Mode: {mode_str}")
    
    if cfg.use_wandb and wandb is not None:
        run_name = f"{cfg.run_name}_{cfg.seed}_single_gpu"
        if cfg.use_72x72_mode:
            run_name += "_72x72"
        wandb.init(
            project=cfg.wandb_project,
            config=vars(cfg),
            name=run_name
        )
    
    # Datasets
    print("Loading data..")
    train_ds = CustomDataset(cfg=cfg, mode="train")
    train_dl = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=cfg.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    
    valid_ds = CustomDataset(cfg=cfg, mode="valid")
    valid_dl = torch.utils.data.DataLoader(
        valid_ds, 
        batch_size=cfg.batch_size_val, 
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    
    # Model
    model = Net(backbone=cfg.backbone, use_72x72_mode=cfg.use_72x72_mode, use_smart_256x72_mode=cfg.use_smart_256x72_mode)

    model = model.to(cfg.device)
    
    compile_enabled = cfg.compile_model
    if compile_enabled:
        model = torch.compile(model, mode='default')
    
    if cfg.ema:
        print("Initializing EMA model..")
        ema_model = ModelEMA(model, decay=cfg.ema_decay, device=cfg.device)
    else:
        ema_model = None

    
    
    # Training setup
    criterion = nn.L1Loss()
    
    compile_enabled = cfg.compile_model
    if compile_enabled:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, fused=True)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    # Define autocast context
    if cfg.mixed_precision:
        autocast_context = autocast(device_type='cuda', dtype=cfg.precision_dtype)
    else:
        autocast_context = nullcontext()
    
    if cfg.use_grad_scaler and cfg.mixed_precision:
        scaler = GradScaler()
    else:
        scaler = None
    
    print("="*25)
    print("Starting training...")
    print("="*25)
    
    best_loss = 1_000_000
    val_loss = 1_000_000
    
    for epoch in range(1, cfg.epochs+1):
        tstart = time.time()
        
        # Train
        model.train()
        total_loss = []
        
        for i, (x, y) in enumerate(train_dl):
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            
            with autocast_context:
                logits = model(x)
            
            def smoothness_loss(pred_vel, alpha=0.1):
                grad_x = torch.abs(pred_vel[:, :, :, 1:] - pred_vel[:, :, :, :-1])
                grad_y = torch.abs(pred_vel[:, :, 1:, :] - pred_vel[:, :, :-1, :])
                return alpha * (grad_x.mean() + grad_y.mean())
            
            if cfg.smoothness_loss:
                mae_loss = criterion(logits, y)
                smooth_loss = smoothness_loss(logits, alpha=0.1)
                loss = mae_loss + smooth_loss
            else:
                loss = criterion(logits, y)
            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss.append(loss.item())
            
            if ema_model is not None:
                ema_model.update(model)
            
            if len(total_loss) >= cfg.logging_steps or i == 0:
                train_loss = np.mean(total_loss)
                total_loss = []
                print(f"Epoch {epoch}: Train MAE: {train_loss:.2f} Val MAE: {val_loss:.2f} "
                      f"Time: {format_time(time.time() - tstart)} Step: {i+1}/{len(train_dl)}")
            
            if cfg.use_wandb and wandb is not None and i % 50 == 0:
                wandb.log({
                    "train_loss": loss.item(),
                    "epoch": epoch,
                    "step": epoch * len(train_dl) + i
                })
        
        # Validation
        model.eval()
        val_logits = []
        val_targets = []
        
        with torch.no_grad():
            for x, y in tqdm(valid_dl):
                x = x.to(cfg.device)
                y = y.to(cfg.device)
                
                with autocast_context:
                    if ema_model is not None:
                        out = ema_model.module(x)
                    else:
                        out = model(x)
                
                val_logits.append(out.cpu())
                val_targets.append(y.cpu())
            
            val_logits = torch.cat(val_logits, dim=0)
            val_targets = torch.cat(val_targets, dim=0)
            val_loss = criterion(val_logits, val_targets).item()

        
        
        if cfg.use_wandb and wandb is not None and cfg.local_rank == 0:
            wandb.log({
                "val_loss": val_loss,
                "epoch": epoch,
                "best_loss": best_loss,
                "learning_rate": scheduler.get_last_lr()[0],
            })
            
        scheduler.step() 
        # Save best model
        if val_loss < best_loss:
            print(f"New best: {best_loss:.2f} -> {val_loss:.2f}")
            print("Saved weights..")
            best_loss = val_loss
            
            if ema_model is not None:
                torch.save(ema_model.module.state_dict(), f'best_model_{cfg.seed}.pt')
            else:
                torch.save(model.state_dict(), f'best_model_{cfg.seed}.pt')
            
            cfg.early_stopping["streak"] = 0
        else:
            cfg.early_stopping["streak"] += 1
            if cfg.early_stopping["streak"] > cfg.early_stopping["patience"]:
                print("Ending training (early_stopping).")
                break
    
    if cfg.use_wandb and wandb is not None:
        wandb.finish()

def main_multi_gpu(cfg):
    """Multi GPU training"""
    
    # Print mode info (only rank 0)
    if cfg.local_rank == 0:
        mode_str = "72x72" if cfg.use_72x72_mode else "Full Resolution"
        print(f"Training Mode: {mode_str}")
    
    if cfg.use_wandb and wandb is not None and cfg.local_rank == 0:
        run_name = f"{cfg.run_name}_{cfg.seed}"
        if cfg.use_72x72_mode:
            run_name += "_72x72"
        wandb.init(
            project=cfg.wandb_project,
            config=vars(cfg),
            name=run_name
        )

    # ========== Datasets / Dataloaders ==========
    if cfg.local_rank == 0:
        print("="*25)
        print("Loading data..")
    train_ds = CustomDataset(cfg=cfg, mode="train")
    sampler= DistributedSampler(
        train_ds, 
        num_replicas=cfg.world_size, 
        rank=cfg.local_rank,
    )
    train_dl = torch.utils.data.DataLoader(
        train_ds, 
        sampler= sampler,
        batch_size= cfg.batch_size, 
        num_workers= 4,
    )
    
    valid_ds = CustomDataset(cfg=cfg, mode="valid")
    sampler= DistributedSampler(
        valid_ds, 
        num_replicas=cfg.world_size, 
        rank=cfg.local_rank,
    )
    valid_dl = torch.utils.data.DataLoader(
        valid_ds, 
        sampler= sampler,
        batch_size= cfg.batch_size_val, 
        num_workers= 4,
    )

    # ========== Model / Optim ==========
    model = Net(backbone=cfg.backbone, use_72x72_mode=cfg.use_72x72_mode, use_smart_256x72_mode=cfg.use_smart_256x72_mode)

    model= model.to(cfg.local_rank)
    
    compile_enabled = cfg.compile_model
    if compile_enabled:
        model = torch.compile(model, mode='default')
    if cfg.ema:
        if cfg.local_rank == 0:
            print("Initializing EMA model..")
        ema_model = ModelEMA(
            model, 
            decay=cfg.ema_decay, 
            device=cfg.local_rank,
        )
    else:
        ema_model = None
    model= DistributedDataParallel(
        model, 
        device_ids=[cfg.local_rank],
        find_unused_parameters=True
        )
    
    criterion = nn.L1Loss()
    compile_enabled = cfg.compile_model
    if compile_enabled:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, fused=True)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    if cfg.mixed_precision and cfg.use_grad_scaler:
        scaler = GradScaler()
    else:
        scaler = None

    # ========== Training ==========
    if cfg.local_rank == 0:
        print("="*25)
        print("Give me warp {}, Mr. Sulu.".format(cfg.world_size))
        print("="*25)
    
    best_loss= 1_000_000
    val_loss= 1_000_000

    for epoch in range(0, cfg.epochs+1):
        if epoch != 0:
            tstart= time.time()
            train_dl.sampler.set_epoch(epoch)
    
            # Train loop
            model.train()
            total_loss = []
            for i, (x, y) in enumerate(train_dl):
                x = x.to(cfg.local_rank)
                y = y.to(cfg.local_rank)
        
                if cfg.mixed_precision:
                    autocast_context = autocast(device_type='cuda', dtype=cfg.precision_dtype)
                else:
                    autocast_context = nullcontext()
                
                with autocast_context:
                    logits = model(x)

                def smoothness_loss(pred_vel, alpha=0.1):
                        grad_x = torch.abs(pred_vel[:, :, :, 1:] - pred_vel[:, :, :, :-1])
                        grad_y = torch.abs(pred_vel[:, :, 1:, :] - pred_vel[:, :, :-1, :])
                        return alpha * (grad_x.mean() + grad_y.mean())

                if cfg.smoothness_loss:
                    # With:
                    mae_loss = criterion(logits, y)
                    smooth_loss = smoothness_loss(logits, alpha=0.1)
                    loss = mae_loss + smooth_loss
                else:
                    loss = criterion(logits, y)

        
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                    optimizer.step()
                    optimizer.zero_grad()
    
                total_loss.append(loss.item())
                
                if ema_model is not None:
                    ema_model.update(model)
                    
                if cfg.local_rank == 0 and (len(total_loss) >= cfg.logging_steps or i == 0):
                    train_loss = np.mean(total_loss)
                    total_loss = []
                    print("Epoch {}:     Train MAE: {:.2f}     Val MAE: {:.2f}     Time: {}     Step: {}/{}".format(
                        epoch, 
                        train_loss,
                        val_loss,
                        format_time(time.time() - tstart),
                        i+1, 
                        len(train_dl)+1, 
                    ))

                if cfg.use_wandb and wandb is not None and cfg.local_rank == 0 and i % 50 == 0:
                    wandb.log({
                        "train_loss": loss.item(),
                        "epoch": epoch,
                        "step": epoch * len(train_dl) + i
                    })
    
        # ========== Valid ==========
        model.eval()
        val_logits = []
        val_targets = []
        
        if cfg.mixed_precision:
            autocast_context = autocast(device_type='cuda', dtype=cfg.precision_dtype)
        else:
            autocast_context = nullcontext()
            
        with torch.no_grad():
            for x, y in tqdm(valid_dl, disable=cfg.local_rank != 0):
                x = x.to(cfg.local_rank)
                y = y.to(cfg.local_rank)
    
                with autocast_context:
                    if ema_model is not None:
                        out = ema_model.module(x)
                    else:
                        out = model(x)

                val_logits.append(out.cpu())
                val_targets.append(y.cpu())

            val_logits= torch.cat(val_logits, dim=0)
            val_targets= torch.cat(val_targets, dim=0)
                
            loss = criterion(val_logits, val_targets).item()

        # Gather loss
        v = torch.tensor([loss], device=cfg.local_rank)
        torch.distributed.all_reduce(v, op=dist.ReduceOp.SUM)
        val_loss = (v[0] / cfg.world_size).item()
        
        if cfg.use_wandb and wandb is not None and cfg.local_rank == 0:
            wandb.log({
                "val_loss": val_loss,
                "epoch": epoch,
                "best_loss": best_loss,
                "learning_rate": scheduler.get_last_lr()[0],
            })

        scheduler.step()
    
        # ========== Weights / Early stopping ==========
        stop_train = torch.tensor([0], device=cfg.local_rank)
        if cfg.local_rank == 0:
            es= cfg.early_stopping
            if val_loss < best_loss:
                print("New best: {:.2f} -> {:.2f}".format(best_loss, val_loss))
                print("Saved weights..")
                best_loss = val_loss
                if ema_model is not None:
                    torch.save(ema_model.module.state_dict(), f'best_model_{cfg.seed}.pt')
                else:
                    torch.save(model.state_dict(), f'best_model_{cfg.seed}.pt')
        
                es["streak"] = 0
            else:
                es= cfg.early_stopping
                es["streak"] += 1
                if es["streak"] > es["patience"]:
                    print("Ending training (early_stopping).")
                    stop_train = torch.tensor([1], device=cfg.local_rank)
        
        # Exits training on all ranks
        dist.broadcast(stop_train, src=0)
        if stop_train.item() == 1:
            return

    if cfg.use_wandb and wandb is not None and cfg.local_rank == 0:
        wandb.finish()

    return

if __name__ == "__main__":
    
    set_seed(cfg.seed)
    
    # Auto-detect GPU configuration
    setup_gpu_mode(cfg)
    
    if cfg.distributed:
        # Multi-GPU setup
        setup(cfg.local_rank, cfg.world_size)
        
        _, total = torch.cuda.mem_get_info(device=cfg.local_rank)
        time.sleep(cfg.local_rank)
        print(f"Rank: {cfg.local_rank}, World size: {cfg.world_size}, GPU memory: {total / 1024**3:.2f}GB", flush=True)
        time.sleep(cfg.world_size - cfg.local_rank)
        
        set_seed(cfg.seed + cfg.local_rank)  # Different seed per GPU
        
        main_multi_gpu(cfg)
        cleanup()
    else:
        # Single GPU setup
        main_single_gpu(cfg)