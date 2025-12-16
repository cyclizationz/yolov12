
import sys
import os
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules.block import A2C2f, C3k2, Attention, ABlock, AAttn

def inspect_deep(model):
    print("Deep inspection of model structure...")
    seq = model.model.model if hasattr(model.model, 'model') else model.model
    
    has_attention = False
    for i, m in enumerate(seq):
        if isinstance(m, A2C2f):
            print(f"Layer {i} is A2C2f.")
            if hasattr(m, 'm'):
                for j, sub in enumerate(m.m):
                    print(f"  Block {j}: {type(sub)}")
                    
                    # Recurse children
                    for n1, c1 in sub.named_children():
                        print(f"    {n1}: {type(c1)}")
                        if isinstance(c1, (ABlock, AAttn, Attention)):
                             print(f"      -> FOUND ATTENTION TYPE: {type(c1)}")
                             has_attention = True
                        # One more level
                        for n2, c2 in c1.named_children():
                             if isinstance(c2, (ABlock, AAttn, Attention)):
                                 print(f"        -> FOUND ATTENTION TYPE: {type(c2)}")
                                 has_attention = True

    if not has_attention:
        print("\nCONCLUSION: No Attention blocks found.")
    else:
        print("\nCONCLUSION: Attention blocks FOUND.")

if __name__ == "__main__":
    path_n = '/home/tiehangz/proj/yolov12/modification/model/yolov12n-seg.pt'
    if os.path.exists(path_n):
        inspect_deep(YOLO(path_n))
