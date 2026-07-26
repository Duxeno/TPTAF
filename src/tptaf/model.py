"""TPTAF fusion network."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
from torch import Tensor, nn
from .modules import DepthwiseSeparableConv,DiscriminativeDetailEnhancementBlock,GlobalLocalAttention,HaarDWT,PhaseCoherentStructureBlock,PositionalEmbedding,TripartiteAttention

class DecoupledEncoder(nn.Module):
    def __init__(self,in_channels=1,channels=64):
        super().__init__(); self.embedding=DepthwiseSeparableConv(in_channels,channels,3); self.position=PositionalEmbedding(channels); self.dwt=HaarDWT(); self.pcsb=PhaseCoherentStructureBlock(channels); self.ddeb=DiscriminativeDetailEnhancementBlock(channels)
    def forward(self,image):
        x=self.position(self.embedding(image)); ll,lh,hl,hh=self.dwt(x); size=x.shape[-2:]; return self.pcsb(ll,size),self.ddeb(lh,hl,hh,size)

class ReconstructionDecoder(nn.Module):
    def __init__(self,channels=64,out_channels=1,window_size=8):
        super().__init__(); self.net=nn.Sequential(DepthwiseSeparableConv(channels,channels,3),GlobalLocalAttention(channels,4,window_size),DepthwiseSeparableConv(channels,channels//2,3),nn.Conv2d(channels//2,out_channels,3,padding=1))
    def forward(self,feature,infrared,visible): return torch.sigmoid(self.net(feature)+.5*(infrared+visible))

@dataclass
class TPTAFOutput:
    fused: Tensor; structural_ir: Tensor; structural_vi: Tensor; discriminative_ir: Tensor; discriminative_vi: Tensor; attention: dict[str,Tensor]

class TPTAF(nn.Module):
    def __init__(self,channels=64,detection_channels=64,num_heads=4,num_points=8,window_size=8,prior_strength=1.0):
        super().__init__(); self.encoder_ir=DecoupledEncoder(1,channels); self.encoder_vi=DecoupledEncoder(1,channels); self.tam=TripartiteAttention(channels,detection_channels,num_heads,num_points,window_size,prior_strength); self.decoder=ReconstructionDecoder(channels,1,window_size)
    def forward(self,infrared,visible,detection_semantics):
        if infrared.shape!=visible.shape or infrared.ndim!=4 or infrared.shape[1]!=1: raise ValueError('infrared and visible must be matching [B,1,H,W] tensors')
        si,di=self.encoder_ir(infrared); sv,dv=self.encoder_vi(visible); task,attn=self.tam(si,di,sv,dv,detection_semantics); fused=self.decoder(task,infrared,visible); return TPTAFOutput(fused,si,sv,di,dv,attn)
    def load_checkpoint(self,path:str,strict=True)->dict[str,Any]:
        ckpt=torch.load(path,map_location='cpu',weights_only=False); state=ckpt.get('fusion_model',ckpt.get('model',ckpt)); inc=self.load_state_dict(state,strict=strict); return {'missing_keys':list(inc.missing_keys),'unexpected_keys':list(inc.unexpected_keys)}
