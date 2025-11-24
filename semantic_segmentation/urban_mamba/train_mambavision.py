#!/usr/bin/env python3
"""
MambaVision-NSST Training Script with VMamba-style Logging
Matches the logging format of VMamba-NSST-XLET training
"""

import os
import sys
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from datetime import datetime
from pathlib import Path
import logging
import argparse
import argparse

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from data.loveda_dataset import create_loveda_dataloaders
from models.model_v31 import UrbanMambaV31
from losses.loss import CompositeLoss
from losses.metrics import SegmentationMetrics


def setup_logging(log_dir, config):
    """Setup logging to both file and console"""
    # Create timestamped log filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f"mambavision_nsst_{config['model']['variant']}"
    log_file = os.path.join(log_dir, f"{model_name}_{timestamp}.log")
    
    # Create logger
    logger = logging.getLogger('mambavision_training')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file


def print_header(config, logger):
    """Print training configuration header"""
    gpu_name = torch.cuda.get_device_name(0)
    model_name = f"mambavision_nsst_{config['model']['variant']}_{gpu_name.lower().replace(' ', '_')}"
    
    logger.info("\n" + "="*80)
    logger.info(f"MambaVision-NSST Training Configuration: {model_name}")
    logger.info("="*80)
    
    logger.info("\n[Model Architecture]")
    logger.info(f"  Variant: {config['model']['variant']}")
    logger.info(f"  Num classes: {config['model']['num_classes']}")
    logger.info(f"  Dims: [96, 192, 384, 768]")
    logger.info(f"  Architecture: MambaVision Twin Tower + NSST")
    logger.info(f"  Fusion type: mamba")
    
    logger.info("\n[Training]")
    logger.info(f"  Dataset: loveda")
    logger.info(f"  Epochs: {config['training']['epochs']}")
    logger.info(f"  Batch size: {config['training']['batch_size']}")
    logger.info(f"  Accumulation steps: 1")
    logger.info(f"  Effective batch: {config['training']['batch_size']}")
    logger.info(f"  Use AMP: True")
    
    logger.info("\n[Optimizer]")
    logger.info(f"  Type: adamw")
    logger.info(f"  Base LR: {config['training']['lr']:.0e}")
    logger.info(f"  Weight decay: {config['training']['weight_decay']}")
    logger.info(f"  LR scheduler: cosine")
    logger.info(f"  Warmup epochs: {config['training'].get('warmup_epochs', 5)}")
    
    logger.info("\n[Loss]")
    logger.info(f"  Loss type: ce+lovasz")
    logger.info(f"  CE weight: 0.7")
    logger.info(f"  Lovasz weight: 0.3")
    
    logger.info("\n[Paths]")
    logger.info(f"  Data root: {config['data']['dataset_root']}")
    logger.info(f"  Checkpoint dir: {config['logging']['log_dir']}/checkpoints")
    
    logger.info("\n" + "="*80 + "\n")


def print_training_start(config, train_loader, val_loader, device, logger):
    """Print training start banner"""
    logger.info("="*80)
    logger.info("TRAINING STARTED")
    logger.info("="*80)
    logger.info(f"Model: MambaVision-NSST {config['model']['variant']}")
    logger.info(f"Dataset: LOVEDA (Train: {len(train_loader)} batches, Val: {len(val_loader)} batches)")
    logger.info(f"Epochs: {config['training']['epochs']} | Batch Size: {config['training']['batch_size']}")
    logger.info(f"Learning Rate: {config['training']['lr']:.2e} | AMP: {config['training'].get('use_amp', False)}")
    logger.info(f"Device: {device}")
    logger.info("="*80 + "\n")


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, num_epochs, num_classes, logger, accumulation_steps=2):
    """Train for one epoch with VMamba-style logging and metrics"""
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    
    ce_loss_sum = 0.0
    lovasz_loss_sum = 0.0
    
    # Add metrics tracker for training
    train_metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=255)
    
    start_time = time.time()
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        batch_start = time.time()
        
        images = images.to(device)
        labels = labels.to(device)
        
        # EXTREME FIX: Force entire model to FP32 by disabling AMP
        # NSST frequency data is too unstable for float16
        outputs = model(images)
        loss_dict = criterion(outputs, labels)
        loss = loss_dict['loss']
        
        # FIX 4: Enhanced NaN detection and recovery
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"\n⚠ NaN/Inf loss detected at batch {batch_idx}")
            logger.warning(f"  CE Loss: {loss_dict.get('loss_ce', 'N/A')}, Lovasz: {loss_dict.get('loss_lovasz', 'N/A')}")
            optimizer.zero_grad()  # Clear any corrupted gradients
            continue
        
        # No AMP scaling - direct backward pass in FP32
        loss.backward()
        
        # EXTREME FIX: Ultra-aggressive gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.05)  # EXTREME: 0.05
        
        # Skip update if gradient norm was inf/nan BEFORE clipping
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            logger.warning(f"⚠ NaN/Inf gradient norm at batch {batch_idx}, skipping update...")
            optimizer.zero_grad()
            continue
        
        # Log very large gradient norms (even after clipping)
        if grad_norm > 50.0:
            logger.info(f"⚠ Large gradient norm ({grad_norm:.2f}) at batch {batch_idx} (clipped to 0.05)")
        
        # Gradient accumulation support
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item()
        ce_loss_sum += loss_dict['loss_ce'].item()
        lovasz_loss_sum += loss_dict['loss_lovasz'].item()
        
        # Update training metrics
        with torch.no_grad():
            predictions = torch.argmax(outputs, dim=1)
            train_metrics.update(predictions, labels)
        
        batch_time = time.time() - batch_start
        avg_loss = total_loss / (batch_idx + 1)
        
        # Print progress every 50 batches
        if batch_idx % 50 == 0 or batch_idx == num_batches - 1:
            progress = (batch_idx / num_batches) * 100
            remaining_batches = num_batches - batch_idx - 1
            eta_seconds = remaining_batches * batch_time
            eta_minutes = eta_seconds / 60
            
            logger.info(f"Epoch [{epoch:3d}/{num_epochs}] [{progress:5.1f}%] "
                  f"Batch [{batch_idx:3d}/{num_batches}] | "
                  f"Loss: {loss.item():.4f} (avg: {avg_loss:.4f}) | "
                  f"Time: {batch_time:.2f}s | "
                  f"ETA: {eta_minutes:.1f}m")
    
    epoch_time = time.time() - start_time
    avg_loss = total_loss / num_batches
    avg_ce = ce_loss_sum / num_batches
    avg_lovasz = lovasz_loss_sum / num_batches
    
    # Get training metrics
    metric_results = train_metrics.get_metrics()
    
    return {
        'loss': avg_loss,
        'ce_loss': avg_ce,
        'lovasz_loss': avg_lovasz,
        'epoch_time': epoch_time,
        'mIoU': metric_results.get('mIoU', 0.0),
        'mean_acc': metric_results.get('mean_acc', 0.0)
    }


def validate(model, val_loader, criterion, device, num_classes, logger):
    """Validate with VMamba-style logging"""
    model.eval()
    total_loss = 0.0
    metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=255)
    
    logger.info("\n" + "="*80)
    logger.info("Validation Phase")
    logger.info("-"*80)
    
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            with autocast():
                outputs = model(images)
                loss_dict = criterion(outputs, labels)
                loss = loss_dict['loss']
            
            total_loss += loss.item()
            
            # Update metrics
            predictions = torch.argmax(outputs, dim=1)
            metrics.update(predictions, labels)
            
            # Print progress every 20 batches
            if batch_idx % 20 == 0 or batch_idx == num_batches - 1:
                progress = ((batch_idx + 1) / num_batches) * 100
                avg_loss = total_loss / (batch_idx + 1)
                logger.info(f"  Progress: [{progress:5.1f}%] Batch [{batch_idx:3d}/{num_batches}] Loss: {loss.item():.4f}")
    
    # Get final metrics
    metric_results = metrics.get_metrics()
    metric_results['val_loss'] = total_loss / num_batches
    
    return metric_results


def print_epoch_summary(epoch, num_epochs, train_metrics, val_metrics, lr, best_miou, 
                       is_best, checkpoint_path, total_time_hours, logger):
    """Print epoch summary in VMamba style"""
    logger.info("="*80)
    logger.info(f"EPOCH [{epoch:3d}/{num_epochs}] SUMMARY")
    logger.info("="*80)
    logger.info(f"Time: {train_metrics['epoch_time']:.1f}s (Total: {total_time_hours:.2f}h) | LR: {lr:.2e}")
    logger.info("-"*80)
    logger.info(f"{'Metric':<20} {'Training':<15} {'Validation':<15} {'Improvement':<15}")
    logger.info("-"*80)
    logger.info(f"{'Loss':<20} {train_metrics['loss']:<15.4f} {val_metrics['val_loss']:<15.4f}")
    
    improvement = ""
    if is_best:
        improvement = "[BEST]"
    logger.info(f"{'mIoU':<20} {train_metrics.get('mIoU', 0):<15.4f} {val_metrics.get('mIoU', 0):<15.4f} {improvement:<15}")
    logger.info(f"{'mAcc':<20} {train_metrics.get('mean_acc', 0):<15.4f} {val_metrics.get('mean_acc', 0):<15.4f}")
    logger.info(f"{'CE Loss':<20} {train_metrics['ce_loss']:<15.4f}")
    logger.info(f"{'Lovasz Loss':<20} {train_metrics['lovasz_loss']:<15.4f}")
    logger.info("-"*80)
    
    if is_best:
        miou_improvement = val_metrics.get('mIoU', 0) - best_miou
        logger.info(f"🎯 NEW BEST MODEL! mIoU: {val_metrics.get('mIoU', 0):.4f} (+{miou_improvement:.4f})")
        logger.info("-"*80)
        logger.info(f"💾 Saving best model (Epoch {epoch}, mIoU: {val_metrics.get('mIoU', 0):.4f})")
    
    logger.info("="*80 + "\n")


def save_checkpoint(epoch, model, optimizer, warmup_scheduler, cosine_scheduler, metrics, checkpoint_path, logger, scaler=None, best_miou=None):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'warmup_scheduler_state_dict': warmup_scheduler.state_dict(),
        'cosine_scheduler_state_dict': cosine_scheduler.state_dict(),
        'metrics': metrics
    }
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    if best_miou is not None:
        checkpoint['best_miou'] = best_miou
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"   Checkpoint saved: {checkpoint_path}")


def main():
    # Load configuration
    config_path = 'configs/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # GPU is set via CUDA_VISIBLE_DEVICES, so use device 0
    # Create checkpoint directory from config
    checkpoint_dir = config['checkpoint']['save_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup logging
    logger, log_file = setup_logging(config['logging']['log_dir'], config)
    
    # Setup device (use 0 because CUDA_VISIBLE_DEVICES is set in launch script)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    
    # Print configuration
    print_header(config, logger)
    
    gpu_name = torch.cuda.get_device_name(device)
    logger.info(f"[Rank 0] Using device: {device} ({gpu_name})")
    logger.info(f"[Rank 0] Creating MambaVision-NSST {config['model']['variant']}\n")
    
    # Log file location
    logger.info(f"Log file: {log_file}\n")
    
    # Create model with v3.1 (XLET Normalization Stem)
    logger.info("="*80)
    logger.info("MambaVision-NSST v3.1 Architecture (with XLET Stem)")
    logger.info("="*80 + "\n")
    model = UrbanMambaV31(
        num_classes=config['model']['num_classes'],
        variant=config['model']['variant'],
        use_xlet_stem=True,  # CRITICAL for stability
        pretrained_spatial=True,  # Use ImageNet weights
        freeze_spatial=False  # Allow fine-tuning
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("="*80)
    logger.info("MambaVision-NSST initialized successfully!")
    logger.info("="*80)
    logger.info(f"Total parameters: {num_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Model size: {num_params/1e6:.2f}M\n")
    
    # Parse arguments for checkpoint resuming
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load checkpoint if resuming
    start_epoch = 1
    best_miou = 0.0
    if args.resume and os.path.exists(args.resume):
        logger.info("="*80)
        logger.info(f"RESUMING from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_miou = checkpoint.get('best_miou', 0.0)
        logger.info(f"Resuming from epoch: {start_epoch}")
        logger.info(f"Previous best mIoU: {best_miou:.4f}")
        logger.info("="*80 + "\n")
    
    # Create dataloaders
    logger.info("Loading LOVEDA dataset...")
    train_loader, val_loader = create_loveda_dataloaders(
        dataset_root=config['data']['dataset_root'],
        train_list=config['data']['train_list'],
        val_list=config['data']['val_list'],
        batch_size=config['training']['batch_size'],
        crop_size=config['training']['crop_size'],
        num_workers=config['data']['num_workers']
    )
    
    logger.info(f"\n[DataLoaders Created]")
    logger.info(f"  Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    logger.info(f"  Val: {len(val_loader.dataset)} samples, {len(val_loader)} batches\n")
    
    # Loss, optimizer, scheduler
    criterion = CompositeLoss(
        num_classes=config['model']['num_classes'],
        ignore_index=255,
        ce_weight=0.7,
        lovasz_weight=0.3
    ).to(device)
    
    # Use differential learning rates (v3.1 feature)
    # Spatial: Low LR to preserve ImageNet knowledge
    # Frequency: High LR to learn NSST features
    base_lr = config['training']['lr']
    param_groups = model.get_param_groups(
        lr_spatial=base_lr * 0.1,  # 1e-5 if base is 1e-4
        lr_frequency=base_lr,       # 1e-4
        lr_fusion=base_lr,          # 1e-4
        lr_decoder=base_lr          # 1e-4
    )
    
    optimizer = optim.AdamW(
        param_groups,
        weight_decay=config['training']['weight_decay']
    )
    
    # Warmup + Cosine scheduler
    warmup_epochs = config['training'].get('warmup_epochs', 5)
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    
    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'] - warmup_epochs,
        eta_min=config['training'].get('min_lr', 1e-7)
    )
    
    # Only use scaler if AMP is enabled
    use_amp = config['training'].get('use_amp', False)
    scaler = GradScaler() if use_amp else None
    
    # Parse arguments for checkpoint resuming
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load checkpoint if resuming
    start_epoch = 1
    best_miou = 0.0
    if args.resume and os.path.exists(args.resume):
        logger.info("="*80)
        logger.info(f"RESUMING from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_miou = checkpoint.get('best_miou', 0.0)
        logger.info(f"Resuming from epoch: {start_epoch}")
        logger.info(f"Previous best mIoU: {best_miou:.4f}")
        logger.info("="*80 + "\n")
    
    # Print training start
    if start_epoch == 1:
        print_training_start(config, train_loader, val_loader, device, logger)
    else:
        logger.info("="*80)
        logger.info(f"TRAINING RESUMED from Epoch {start_epoch}")
        logger.info("="*80)
        logger.info(f"Model: MambaVision-NSST {config['model']['variant']}")
        logger.info(f"Dataset: LOVEDA (Train: {len(train_loader)} batches, Val: {len(val_loader)} batches)")
        logger.info(f"Resuming to Epoch: {config['training']['epochs']} | Batch Size: {config['training']['batch_size']}")
        logger.info(f"Learning Rate: {config['training']['lr']:.2e} | AMP: {config['training'].get('use_amp', False)}")
        logger.info(f"Device: {device}")
        logger.info(f"Previous best mIoU: {best_miou:.4f}")
        logger.info("="*80 + "\n")
    
    # Training loop
    start_time = time.time()
    total_time_hours = 0.0
    
    # Get accumulation steps from config
    accumulation_steps = config['training'].get('accumulation_steps', 2)
    
    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            epoch, config['training']['epochs'], config['model']['num_classes'], logger,
            accumulation_steps=accumulation_steps
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device,
            config['model']['num_classes'], logger
        )
        
        # Update learning rate (warmup for first epochs, then cosine)
        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Check if best model
        current_miou = val_metrics.get('mIoU', 0)
        is_best = current_miou > best_miou
        
        # Calculate total time
        total_time = time.time() - start_time
        total_time_hours = total_time / 3600
        
        # Save checkpoints
        if is_best:
            # Save best model with mIoU in filename
            best_checkpoint_path = os.path.join(checkpoint_dir, f'mambavision_best_miou_{current_miou:.4f}.pth')
            save_checkpoint(epoch, model, optimizer, warmup_scheduler, cosine_scheduler, val_metrics, best_checkpoint_path, logger, scaler=scaler, best_miou=current_miou)
            best_miou = current_miou
            
            # Also save as latest best (for easy loading)
            latest_best_path = os.path.join(checkpoint_dir, 'mambavision_best.pth')
            save_checkpoint(epoch, model, optimizer, warmup_scheduler, cosine_scheduler, val_metrics, latest_best_path, logger, scaler=scaler, best_miou=current_miou)
        
        # Save periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            periodic_checkpoint_path = os.path.join(checkpoint_dir, f'mambavision_epoch_{epoch}.pth')
            save_checkpoint(epoch, model, optimizer, warmup_scheduler, cosine_scheduler, val_metrics, periodic_checkpoint_path, logger, scaler=scaler, best_miou=best_miou)
        
        # Print summary
        print_epoch_summary(
            epoch, config['training']['epochs'],
            train_metrics, val_metrics, current_lr,
            best_miou if not is_best else 0.0,
            is_best, '', total_time_hours, logger
        )
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETED")
    logger.info("="*80)
    logger.info(f"Best mIoU: {best_miou:.4f}")
    logger.info(f"Total time: {total_time_hours:.2f} hours")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)


if __name__ == '__main__':
    main()
