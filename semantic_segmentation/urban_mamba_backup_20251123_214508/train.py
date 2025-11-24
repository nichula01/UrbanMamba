"""
Training Script for UrbanMamba
Implements complete training pipeline with on-the-fly NSST generation,
composite loss, gradient accumulation, and cosine annealing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime
from typing import Dict, Optional

try:
    from models.model import UrbanMamba, create_urban_mamba
    from models.transforms import NSSTDecomposition
    from losses.lovasz_loss import CompositeLoss
    from losses.metrics import SegmentationMetrics
except ImportError:
    from .models.model import UrbanMamba, create_urban_mamba
    from .models.transforms import NSSTDecomposition
    from .losses.lovasz_loss import CompositeLoss
    from .losses.metrics import SegmentationMetrics


class UrbanSegmentationDataset(Dataset):
    """
    Urban segmentation dataset loader.
    Modify this class according to your dataset structure.
    """
    
    def __init__(self, data_root: str, split: str = 'train', transform=None):
        """
        Initialize dataset.
        
        Args:
            data_root: Root directory of dataset
            split: 'train', 'val', or 'test'
            transform: Data augmentation transforms
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        
        # Load image and label paths
        self.image_paths = sorted((self.data_root / split / 'images').glob('*.png'))
        self.label_paths = sorted((self.data_root / split / 'labels').glob('*.png'))
        
        assert len(self.image_paths) == len(self.label_paths), \
            f"Mismatch: {len(self.image_paths)} images vs {len(self.label_paths)} labels"
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Load image and label."""
        # Load image (implement according to your data format)
        # For example using PIL:
        # from PIL import Image
        # image = Image.open(self.image_paths[idx]).convert('RGB')
        # label = Image.open(self.label_paths[idx])
        
        # Placeholder: Load as torch tensors
        # Replace with actual loading code
        image = torch.randn(3, 512, 512)  # Placeholder
        label = torch.randint(0, 6, (512, 512))  # Placeholder
        
        if self.transform:
            image, label = self.transform(image, label)
        
        return image, label


class Trainer:
    """
    UrbanMamba Trainer with NSST feature generation and composite loss.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device(config['device'])
        
        # Create UrbanMamba v3 model (Twin Tower Architecture)
        print(f"🚀 Creating UrbanMamba v3 (Twin Tower Architecture)")
        self.model = create_urban_mamba(
            num_classes=config['model']['num_classes'],
            variant=config['model']['size'],
            pretrained_spatial=config['model'].get('pretrained_spatial', None)
        ).to(self.device)
        
        # Create loss function
        self.criterion = create_loss_function(
            num_classes=config['model']['num_classes'],
            loss_type='composite',
            lambda_ce=config['loss']['lambda_ce'],
            lambda_lovasz=config['loss']['lambda_lovasz'],
            class_weights=config['loss'].get('class_weights', None),
            ignore_index=config['loss'].get('ignore_index', -100)
        )
        
        # Create optimizer with parameter groups
        param_groups = self.model.get_params_groups(config['optimizer']['base_lr'])
        
        self.optimizer = optim.AdamW(
            param_groups,
            lr=config['optimizer']['base_lr'],
            weight_decay=config['optimizer']['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler with warm-up and cosine annealing
        self.scheduler = self._create_scheduler()
        
        # Gradient scaler for mixed precision
        self.scaler = GradScaler() if config['training']['use_amp'] else None
        
        # Gradient accumulation
        self.accumulation_steps = config['training']['gradient_accumulation_steps']
        
        # Metrics
        self.train_metrics = SegmentationMetrics(
            num_classes=config['model']['num_classes'],
            class_names=config['data'].get('class_names', None)
        )
        self.val_metrics = SegmentationMetrics(
            num_classes=config['model']['num_classes'],
            class_names=config['data'].get('class_names', None)
        )
        
        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_miou = 0.0
        
        # Create output directory
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warm-up and cosine annealing."""
        total_steps = self.config['training']['epochs'] * \
                     self.config['training'].get('steps_per_epoch', 1000)
        warmup_steps = self.config['scheduler']['warmup_epochs'] * \
                      self.config['training'].get('steps_per_epoch', 1000)
        
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warm-up
                return float(step) / float(max(1, warmup_steps))
            else:
                # Cosine annealing
                progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return 0.5 * (1.0 + np.cos(np.pi * progress))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.train_metrics.reset()
        
        epoch_loss = 0.0
        epoch_loss_ce = 0.0
        epoch_loss_lovasz = 0.0
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # On-the-fly NSST feature extraction
            with torch.no_grad():
                xlet_features = nsct_decomposition_to_tensor(
                    images,
                    scales=3,
                    directions_profile=[2, 3, 4]
                )
            
            # Forward pass with mixed precision
            if self.scaler:
                with autocast():
                    logits = self.model(images, xlet_features=xlet_features)
                    loss_dict = self.criterion(logits, labels)
                    loss = loss_dict['loss'] / self.accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
            else:
                logits = self.model(images, xlet_features=xlet_features)
                loss_dict = self.criterion(logits, labels)
                loss = loss_dict['loss'] / self.accumulation_steps
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1
            
            # Update metrics
            self.train_metrics.update(logits.detach(), labels)
            
            # Track losses
            epoch_loss += loss_dict['loss'].item()
            epoch_loss_ce += loss_dict['loss_ce'].item()
            epoch_loss_lovasz += loss_dict['loss_lovasz'].item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss_dict['loss'].item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })
        
        # Compute epoch metrics
        num_batches = len(train_loader)
        metrics = self.train_metrics.compute_all()
        metrics['loss'] = epoch_loss / num_batches
        metrics['loss_ce'] = epoch_loss_ce / num_batches
        metrics['loss_lovasz'] = epoch_loss_lovasz / num_batches
        
        return metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        self.val_metrics.reset()
        
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Extract NSST features
                xlet_features = nsct_decomposition_to_tensor(
                    images,
                    scales=3,
                    directions_profile=[2, 3, 4]
                )
                
                # Forward pass
                logits = self.model(images, xlet_features=xlet_features)
                loss_dict = self.criterion(logits, labels)
                
                # Update metrics
                self.val_metrics.update(logits, labels)
                self.val_metrics.update_boundary(logits, labels, threshold=2)
                
                val_loss += loss_dict['loss'].item()
        
        # Compute metrics
        metrics = self.val_metrics.compute_all()
        metrics['loss'] = val_loss / len(val_loader)
        
        return metrics
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = self.output_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best checkpoint (mIoU: {metrics['mIoU']:.4f})")
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
        """
        print(f"\n{'='*60}")
        print(f"Training UrbanMamba")
        print(f"{'='*60}")
        print(f"Model size: {self.config['model']['size']}")
        print(f"Number of classes: {self.config['model']['num_classes']}")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            print(f"\nEpoch {epoch} Training Results:")
            print(f"  Loss: {train_metrics['loss']:.4f}")
            print(f"  mIoU: {train_metrics['mIoU']:.4f}")
            print(f"  Pixel Accuracy: {train_metrics['pixel_accuracy']:.4f}")
            
            # Validate
            if val_loader and (epoch + 1) % self.config['training']['val_interval'] == 0:
                val_metrics = self.validate(val_loader)
                
                print(f"\nEpoch {epoch} Validation Results:")
                print(f"  Loss: {val_metrics['loss']:.4f}")
                print(f"  mIoU: {val_metrics['mIoU']:.4f}")
                print(f"  Pixel Accuracy: {val_metrics['pixel_accuracy']:.4f}")
                print(f"  Boundary F1: {val_metrics.get('boundary_f1', 0.0):.4f}")
                
                # Save checkpoint
                is_best = val_metrics['mIoU'] > self.best_miou
                if is_best:
                    self.best_miou = val_metrics['mIoU']
                
                self.save_checkpoint(val_metrics, is_best=is_best)
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best mIoU: {self.best_miou:.4f}")
        print(f"{'='*60}\n")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train UrbanMamba')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed for reproducibility
    torch.manual_seed(config.get('seed', 42))
    np.random.seed(config.get('seed', 42))
    
    # Create datasets (modify according to your dataset)
    # train_dataset = UrbanSegmentationDataset(
    #     data_root=config['data']['root'],
    #     split='train'
    # )
    # val_dataset = UrbanSegmentationDataset(
    #     data_root=config['data']['root'],
    #     split='val'
    # )
    
    # Create data loaders (placeholder)
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=config['training']['batch_size'],
    #     shuffle=True,
    #     num_workers=config['training']['num_workers'],
    #     pin_memory=True
    # )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=config['training']['batch_size'],
    #     shuffle=False,
    #     num_workers=config['training']['num_workers'],
    #     pin_memory=True
    # )
    
    # Create trainer
    trainer = Trainer(config)
    
    # Start training
    # trainer.train(train_loader, val_loader)
    
    print("Training script loaded successfully!")
    print("Modify the dataset loading code according to your data structure.")


if __name__ == "__main__":
    main()
