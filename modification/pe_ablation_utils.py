
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import AAttn, ABlock

# Custom AAttn that supports variable kernel size for PPC
class AAttnCustom(AAttn):
    def __init__(self, dim, num_heads, area=1, kernel_size=5):
        super().__init__(dim, num_heads, area)
        # Override self.pe with custom kernel size
        # Padding = kernel_size // 2 for same spatial resolution
        all_head_dim = self.head_dim * self.num_heads
        if kernel_size == 0: # NoPE
             self.pe = nn.Identity()
        else:
             self.pe = Conv(all_head_dim, dim, kernel_size, 1, kernel_size // 2, g=dim, act=False)

def replace_aattn_modules(model, kernel_size=5):
    """
    Replace AAttn modules in the model with AAttnCustom having specified PPC kernel size.
    kernel_size=0 implies NoPE.
    """
    print(f"Replacing AAttn modules with PPC kernel_size={kernel_size}...")
    count = 0
    
    # Traverse model to find ABlock and replace its attn attribute
    # We look for ABlock because AAttn is usually inside it
    
    def _replace(module):
        nonlocal count
        for name, child in module.named_children():
            if isinstance(child, ABlock):
                if hasattr(child, 'attn') and isinstance(child.attn, AAttn):
                    old_attn = child.attn
                    # Create new attn with same params but new kernel
                    new_attn = AAttnCustom(
                        dim=old_attn.qk.conv.in_channels, # Input dim from qk conv
                        num_heads=old_attn.num_heads,
                        area=old_attn.area,
                        kernel_size=kernel_size
                    )
                    
                    # Copy weights (except PE)
                    try:
                        new_attn.qk.load_state_dict(old_attn.qk.state_dict())
                        new_attn.v.load_state_dict(old_attn.v.state_dict())
                        new_attn.proj.load_state_dict(old_attn.proj.state_dict())
                        
                        # If kernel sizes match (e.g. 5 to 5), copy PE too
                        if kernel_size == 5 and hasattr(old_attn.pe, 'conv'): 
                             new_attn.pe.load_state_dict(old_attn.pe.state_dict())
                    except Exception as e:
                        print(f"Warning: weight copy failed for {name}: {e}")
                        
                    child.attn = new_attn
                    count += 1
            
            # Recurse
            _replace(child)

    if hasattr(model, 'model'):
        _replace(model.model)
    else:
        _replace(model)
        
    print(f"Replaced {count} AAttn modules.")
    return count




