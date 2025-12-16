#!/usr/bin/env python3
"""
Create Optimal Model Variant

Creates and saves the optimal model variant (PPC 5×5) for deployment.
"""

import sys
from pathlib import Path
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO

def replace_attention_modules(model, variant_type='ppc', variant_value=7):
    """
    Replace all Attention modules in a model with specified variant.
    
    Args:
        model: YOLO model instance
        variant_type: 'ppc' for Position Perceiver Convolution kernel size, or 'pe' for PE variant
        variant_value: For 'ppc': kernel size (3, 5, 7, 9). For 'pe': 'none', 'rpe', 'rope'
    
    Returns:
        List of replaced modules for potential restoration
    """
    from ultralytics.nn.modules.block import (
        Attention, AttentionNoPE, AttentionRPE, AttentionRoPE,
        AttentionPPC3x3, AttentionPPC5x5, AttentionPPC7x7, AttentionPPC9x9,
        PSABlock, PSA, C2PSA, C2fPSA
    )
    import torch.nn as nn
    
    replaced_modules = []
    
    # Map variants to classes
    if variant_type == 'ppc':
        variant_map = {
            3: AttentionPPC3x3,
            5: AttentionPPC5x5,
            7: AttentionPPC7x7,
            9: AttentionPPC9x9
        }
        if variant_value not in variant_map:
            raise ValueError(f"Unknown PPC kernel size: {variant_value}. Must be one of {list(variant_map.keys())}")
        new_class = variant_map[variant_value]
        variant_name = f"PPC {variant_value}x{variant_value}"
    elif variant_type == 'pe':
        variant_map = {
            'none': AttentionNoPE,
            'rpe': AttentionRPE,
            'rope': AttentionRoPE
        }
        if variant_value not in variant_map:
            raise ValueError(f"Unknown PE variant: {variant_value}. Must be one of {list(variant_map.keys())}")
        new_class = variant_map[variant_value]
        variant_name = variant_value.upper()
    else:
        raise ValueError(f"Unknown variant_type: {variant_type}. Must be 'ppc' or 'pe'")
    
    # Recursively find and replace Attention modules
    def _replace_module(module, name=''):
        for child_name, child_module in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name
            
            # Check if this is an Attention module
            if isinstance(child_module, Attention):
                # Get the module's parameters
                dim = child_module.head_dim * child_module.num_heads
                num_heads = child_module.num_heads
                attn_ratio = child_module.key_dim / child_module.head_dim
                
                # Create new module
                if variant_type == 'ppc':
                    new_module = new_class(dim, num_heads, attn_ratio)
                elif variant_type == 'pe':
                    if variant_value == 'rope':
                        new_module = new_class(dim, num_heads, attn_ratio, rope_theta=10000.0)
                    else:
                        new_module = new_class(dim, num_heads, attn_ratio)
                
                # Copy weights if compatible
                try:
                    if hasattr(child_module, 'qkv') and hasattr(new_module, 'qkv'):
                        new_module.qkv.weight.data.copy_(child_module.qkv.weight.data)
                        if hasattr(child_module.qkv, 'bias') and child_module.qkv.bias is not None:
                            if new_module.qkv.bias is not None:
                                new_module.qkv.bias.data.copy_(child_module.qkv.bias.data)
                    
                    if hasattr(child_module, 'proj') and hasattr(new_module, 'proj'):
                        new_module.proj.weight.data.copy_(child_module.proj.weight.data)
                        if hasattr(child_module.proj, 'bias') and child_module.proj.bias is not None:
                            if new_module.proj.bias is not None:
                                new_module.proj.bias.data.copy_(child_module.proj.bias.data)
                    
                    # For PPC variants, try to copy PPC weights if kernel size matches
                    if variant_type == 'ppc':
                        if hasattr(child_module, 'ppc') and hasattr(new_module, 'ppc'):
                            if child_module.ppc_kernel_size == new_module.ppc_kernel_size:
                                new_module.ppc.weight.data.copy_(child_module.ppc.weight.data)
                                if hasattr(child_module.ppc, 'bias') and child_module.ppc.bias is not None:
                                    if new_module.ppc.bias is not None:
                                        new_module.ppc.bias.data.copy_(child_module.ppc.bias.data)
                except Exception as e:
                    print(f"  Warning: Could not copy all weights for {full_name}: {e}")
                
                # Replace the module
                setattr(module, child_name, new_module)
                replaced_modules.append((full_name, child_module, new_module))
                print(f"  Replaced Attention at {full_name} with {variant_name}")
            
            # Recursively check nested modules
            elif isinstance(child_module, (PSABlock, PSA, C2PSA, C2fPSA, nn.Sequential, nn.ModuleList)):
                _replace_module(child_module, full_name)
            else:
                # Check deeper
                if len(list(child_module.children())) > 0:
                    _replace_module(child_module, full_name)
    
    # Start replacement from model's model attribute (the actual nn.Module)
    if hasattr(model, 'model'):
        _replace_module(model.model)
    else:
        _replace_module(model)
    
    return replaced_modules

def create_optimal_variant():
    """Create and save the optimal PPC 5×5 variant."""
    print("=" * 80)
    print("CREATING OPTIMAL MODEL VARIANT (PPC 5×5)")
    print("=" * 80)
    
    # Paths
    baseline_path = Path(__file__).parent.parent / 'model' / 'yolov12n-seg.pt'
    output_path = Path(__file__).parent / 'yolov12n-seg-ppc5x5.pt'
    
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline model not found: {baseline_path}")
    
    print(f"\nLoading baseline model: {baseline_path}")
    model = YOLO(str(baseline_path))
    
    print(f"\nApplying PPC 5×5 modification...")
    replaced = replace_attention_modules(model, variant_type='ppc', variant_value=5)
    
    if len(replaced) == 0:
        print("Warning: No Attention modules found to replace!")
        print("The model may already be using PPC 5×5 or may not contain Attention modules.")
    else:
        print(f"Successfully replaced {len(replaced)} Attention module(s)")
    
    print(f"\nSaving optimal variant to: {output_path}")
    model.save(str(output_path))
    
    # Verify file was created
    if output_path.exists():
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✓ Model saved successfully ({file_size_mb:.2f} MB)")
    else:
        raise RuntimeError(f"Failed to save model to {output_path}")
    
    print("\n" + "=" * 80)
    print("OPTIMAL VARIANT CREATED SUCCESSFULLY")
    print("=" * 80)
    
    return str(output_path)

if __name__ == '__main__':
    try:
        output_path = create_optimal_variant()
        print(f"\nOptimal model variant saved to: {output_path}")
    except Exception as e:
        print(f"\nError creating optimal variant: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

