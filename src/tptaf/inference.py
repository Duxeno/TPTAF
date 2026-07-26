"""Inference utilities for tiled full-resolution fusion."""
from __future__ import annotations
import torch
from torch import Tensor

def tiled_fusion(infrared:Tensor,visible:Tensor,runner,tile_size=256,overlap=32)->Tensor:
    if infrared.shape!=visible.shape: raise ValueError('inputs must match')
    b,c,h,w=infrared.shape; out=torch.zeros_like(infrared); weight=torch.zeros_like(infrared); step=max(tile_size-overlap,1)
    ys=list(range(0,max(h-tile_size,0)+1,step)); xs=list(range(0,max(w-tile_size,0)+1,step))
    if not ys or ys[-1]!=max(h-tile_size,0): ys.append(max(h-tile_size,0))
    if not xs or xs[-1]!=max(w-tile_size,0): xs.append(max(w-tile_size,0))
    for y in ys:
        for x in xs:
            patch=runner(infrared[:,:,y:y+tile_size,x:x+tile_size],visible[:,:,y:y+tile_size,x:x+tile_size]); out[:,:,y:y+patch.shape[-2],x:x+patch.shape[-1]]+=patch; weight[:,:,y:y+patch.shape[-2],x:x+patch.shape[-1]]+=1
    return out/weight.clamp_min(1)
