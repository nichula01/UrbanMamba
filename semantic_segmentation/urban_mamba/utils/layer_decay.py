"""
Layer-wise Learning Rate Decay utility for VMamba/MambaVision models.

This implements the layer-wise LR decay strategy used in VMamba and Swin Transformer
training, where earlier layers get smaller learning rates than deeper layers.
"""

import torch
import json


def get_layer_id_for_mambavision(name, num_layers):
    """
    Assign a layer ID for layer-wise learning rate decay.
    
    Args:
        name: Parameter name
        num_layers: Total number of layers in the encoder
        
    Returns:
        layer_id: Integer representing the depth of this parameter
    """
    if name.startswith('patch_embed') or name.startswith('xlet_stem'):
        return 0
    elif name.startswith('levels'):
        # Extract level number from name like 'levels.0.blocks.2.norm1.weight'
        parts = name.split('.')
        if len(parts) > 1 and parts[1].isdigit():
            return int(parts[1]) + 1
        return num_layers
    elif name.startswith('fusions') or name.startswith('decoder'):
        return num_layers
    else:
        return num_layers


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=0.75):
    """
    Create parameter groups with layer-wise learning rate decay.
    
    This is critical for VMamba training! Without this, early layers (especially
    the pretrained spatial encoder) change too fast and destabilize training.
    
    Args:
        model: The model (UrbanMambaV31)
        weight_decay: Base weight decay (0.05 is standard for transformers)
        no_weight_decay_list: List of parameter names that shouldn't have weight decay
        layer_decay: Decay rate for each layer (0.75 means layer N gets 0.75^(max_layer-N) * base_lr)
        
    Returns:
        List of parameter groups for optimizer
        
    Example:
        param_groups = param_groups_lrd(model, weight_decay=0.05, layer_decay=0.75)
        optimizer = optim.AdamW(param_groups, lr=1e-4)
    """
    param_group_names = {}
    param_groups = {}
    
    # Determine number of layers
    num_layers = 4  # MambaVision has 4 levels
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # No weight decay for bias, norm layers, and positional embeddings
        if len(param.shape) == 1 or name.endswith(".bias") or name in no_weight_decay_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        
        # Get layer ID for this parameter
        layer_id = get_layer_id_for_mambavision(name, num_layers)
        group_name = f"layer_{layer_id}_{group_name}"
        
        if group_name not in param_group_names:
            # Calculate the learning rate scale for this layer
            # Deeper layers get higher LR
            scale = layer_decay ** (num_layers - layer_id)
            
            param_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "param_names": [],
                "lr_scale": scale,
                "layer_id": layer_id,
            }
            param_groups[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
                "layer_id": layer_id,
            }
        
        param_group_names[group_name]["params"].append(param)
        param_group_names[group_name]["param_names"].append(name)
        param_groups[group_name]["params"].append(param)
    
    # Print group information for debugging
    print("Parameter groups for layer-wise LR decay:")
    for group_name in sorted(param_group_names.keys()):
        print(f"  {group_name}: "
              f"scale={param_group_names[group_name]['lr_scale']:.3f}, "
              f"wd={param_group_names[group_name]['weight_decay']}, "
              f"params={len(param_group_names[group_name]['params'])}")
    
    return list(param_groups.values())


def freeze_spatial_encoder_early_layers(model):
    """
    Freeze the patch_embed and first level of the spatial encoder.
    
    This prevents the pretrained weights from being destroyed by gradients
    from the randomly initialized frequency branch in the first few epochs.
    
    Args:
        model: UrbanMambaV31 instance
        
    Returns:
        List of frozen parameter names
    """
    frozen_params = []
    
    for name, param in model.named_parameters():
        # Freeze spatial encoder's patch_embed and level 0
        if 'spatial_encoder' in name:
            if 'patch_embed' in name or 'levels.0' in name:
                param.requires_grad = False
                frozen_params.append(name)
    
    print(f"\nFrozen {len(frozen_params)} parameters in spatial encoder early layers:")
    for name in frozen_params[:5]:  # Show first 5
        print(f"  ✓ {name}")
    if len(frozen_params) > 5:
        print(f"  ... and {len(frozen_params) - 5} more")
    
    return frozen_params


def unfreeze_all_parameters(model):
    """
    Unfreeze all parameters in the model.
    
    Call this after the first 5-10 epochs to allow full fine-tuning.
    
    Args:
        model: UrbanMambaV31 instance
        
    Returns:
        Number of unfrozen parameters
    """
    count = 0
    for param in model.parameters():
        if not param.requires_grad:
            param.requires_grad = True
            count += 1
    
    print(f"\n✓ Unfroze {count} parameters - full model training enabled")
    return count


if __name__ == "__main__":
    # Test the layer-wise decay function
    print("Testing layer-wise LR decay...")
    
    # Create a dummy model structure
    import torch.nn as nn
    
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = nn.Linear(3, 96)
            self.levels = nn.ModuleList([
                nn.Linear(96, 96) for _ in range(4)
            ])
            self.decoder = nn.Linear(96, 7)
        
        def named_parameters(self):
            for name, param in super().named_parameters():
                yield name, param
    
    model = DummyModel()
    param_groups = param_groups_lrd(model, weight_decay=0.05, layer_decay=0.75)
    
    print(f"\n✓ Created {len(param_groups)} parameter groups")
    for i, group in enumerate(param_groups):
        print(f"  Group {i}: lr_scale={group['lr_scale']:.3f}, "
              f"params={len(group['params'])}, wd={group['weight_decay']}")
