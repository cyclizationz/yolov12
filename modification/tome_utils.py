
import torch
import torch.nn as nn
from ultralytics.nn.modules.block import AAttn, ABlock

class TokenMergingWrapperAAttn(nn.Module):
    """
    ToMe wrapper specifically for AAttn modules in YOLOv12.
    """
    def __init__(self, attn_module, merge_ratio=0.25):
        super().__init__()
        self.attn = attn_module
        self.merge_ratio = merge_ratio
        
    def forward(self, x):
        if self.merge_ratio > 0:
            # Simplified spatial reduction for ablation
            # Real ToMe uses bipartite matching, here we use interpolation 
            # to simulate token reduction impact on throughput
            B, C, H, W = x.shape
            scale = 1.0 - self.merge_ratio
            h_new = max(1, int(H * scale))
            w_new = max(1, int(W * scale))
            
            # Downsample
            x_small = nn.functional.interpolate(x, size=(h_new, w_new), mode='bilinear', align_corners=False)
            
            # Attention
            out_small = self.attn(x_small)
            
            # Upsample
            return nn.functional.interpolate(out_small, size=(H, W), mode='bilinear', align_corners=False)
        else:
            return self.attn(x)

def apply_tome_to_aattn(model, merge_ratio=0.25, placement='all'):
    """
    Wrap AAttn modules with ToMe.
    Placement: 'all', 'backbone', 'neck', 'early'
    """
    print(f"Applying ToMe (r={merge_ratio}) to {placement}...")
    count = 0
    
    # Heuristic for backbone vs neck:
    # In YOLOv12n-seg, layers 0-9 are typically backbone, 10+ are neck/head.
    # We'll use a simple index-based split if possible, or just all for now.
    # Layer indices in inspect output: 6, 8 (Backbone?), 11, 14, 17 (Neck?)
    
    backbone_indices = [6, 8]
    neck_indices = [11, 14, 17]
    
    # Note: We need to find the 'i' (index) of the parent module to decide placement
    
    def _replace(module, parent_idx=None):
        nonlocal count
        
        # Check if module has index
        current_idx = getattr(module, 'i', parent_idx)
        
        for name, child in module.named_children():
            if isinstance(child, ABlock):
                # Determine if we should wrap based on placement
                should_wrap = False
                if placement == 'all':
                    should_wrap = True
                elif placement == 'backbone':
                    # If we can track index, use it. Otherwise wrap "early" ones?
                    # Hard to track exact index recursively without context.
                    # We'll assume the user passes a "model" which is the YOLO object or Sequential
                    pass
                elif placement == 'after_patch_embed':
                    # Only the very first one we find
                    if count == 0: should_wrap = True
                
                # Logic fix: Use global list traversal in the outer loop instead of recursion for placement
                
            # Recurse
            _replace(child, current_idx)

    # Better approach: Linear scan of main layers
    seq = model.model.model if hasattr(model.model, 'model') else model.model
    
    for i, m in enumerate(seq):
        # Check placement condition
        is_target = False
        if placement == 'all':
            is_target = True
        elif placement == 'backbone' and i < 10: # Approx cutoff
            is_target = True
        elif placement == 'neck_pre_fpn' and i >= 10:
            is_target = True
        elif placement == 'after_patch_embed' and count == 0 and i > 2: # First block
            is_target = True
            
        if not is_target and placement != 'all' and placement != 'after_patch_embed':
            continue
            
        # Search for ABlock/AAttn inside this layer
        # Note: m could be A2C2f which contains ABlock
        
        def _wrap_recursive(submod):
            nonlocal count
            for n, c in submod.named_children():
                if isinstance(c, ABlock) and hasattr(c, 'attn') and isinstance(c.attn, AAttn):
                    # Don't double wrap
                    if not isinstance(c.attn, TokenMergingWrapperAAttn):
                        c.attn = TokenMergingWrapperAAttn(c.attn, merge_ratio)
                        count += 1
                        if placement == 'after_patch_embed': return # Only one
                _wrap_recursive(c)
                
        if is_target:
            _wrap_recursive(m)
            if placement == 'after_patch_embed' and count > 0: break

    print(f"Wrapped {count} AAttn modules.")
    return count




