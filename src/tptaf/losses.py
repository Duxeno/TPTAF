"""Fusion losses and six-term uncertainty-weighted learning."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .model import TPTAFOutput

def _ssim(x: Tensor, y: Tensor, c1: float, c2: float, window_size: int = 11) -> Tensor:
    p=window_size//2; mx=F.avg_pool2d(x,window_size,1,p); my=F.avg_pool2d(y,window_size,1,p)
    sx=F.avg_pool2d(x*x,window_size,1,p)-mx.square(); sy=F.avg_pool2d(y*y,window_size,1,p)-my.square(); sxy=F.avg_pool2d(x*y,window_size,1,p)-mx*my
    return (((2*mx*my+c1)*(2*sxy+c2))/((mx.square()+my.square()+c1)*(sx+sy+c2)).clamp_min(1e-12)).mean()

class SimilarityLoss(nn.Module):
    def __init__(self,c1=1e-4,c2=9e-4): super().__init__(); self.c1=c1; self.c2=c2
    def forward(self,fused,infrared,visible): return sum((1-_ssim(s,fused,self.c1,self.c2))+F.mse_loss(fused,s) for s in (infrared,visible))

def _corr(x,y,eps):
    x=x.flatten(2); y=y.flatten(2); x=x-x.mean(2,keepdim=True); y=y-y.mean(2,keepdim=True)
    return ((x*y).mean(2)/(x.square().mean(2).sqrt()*y.square().mean(2).sqrt()+eps)).mean()

class DecouplingLoss(nn.Module):
    def __init__(self,c3=.01): super().__init__(); self.c3=c3
    def forward(self,o): return _corr(o.discriminative_ir,o.discriminative_vi,self.c3).square()/(1+_corr(o.structural_ir,o.structural_vi,self.c3)).clamp_min(1e-6)

class SobelGradient(nn.Module):
    def __init__(self):
        super().__init__(); k=torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]); self.register_buffer('kx',k[None,None],persistent=False); self.register_buffer('ky',k.T[None,None],persistent=False)
    def forward(self,x): return torch.cat((F.conv2d(x,self.kx.to(x.dtype),padding=1),F.conv2d(x,self.ky.to(x.dtype),padding=1)),1)

class GradientLoss(nn.Module):
    def __init__(self,infrared_weight=.5): super().__init__(); self.w=infrared_weight; self.g=SobelGradient()
    def forward(self,fused,infrared,visible): return F.l1_loss(self.g(fused),self.w*self.g(infrared)+(1-self.w)*self.g(visible))

class UncertaintyWeightedLoss(nn.Module):
    names=("sime","deco","grad","box","cls","dfl")
    def __init__(self,clamp_min=-4.6,clamp_max=4.6): super().__init__(); self.log_variances=nn.Parameter(torch.zeros(6)); self.clamp_min=clamp_min; self.clamp_max=clamp_max
    def forward(self,losses: Sequence[Tensor]): return sum(torch.exp(-self.log_variances[i])*loss+.5*self.log_variances[i] for i,loss in enumerate(losses))
    @torch.no_grad()
    def clamp_(self): self.log_variances.clamp_(self.clamp_min,self.clamp_max)
    @torch.no_grad()
    def effective_weights(self): return dict(zip(self.names,torch.exp(-self.log_variances).cpu().tolist()))

class TPTAFLoss(nn.Module):
    def __init__(self,c1=1e-4,c2=9e-4,c3=.01,infrared_gradient_weight=.5,log_var_min=-4.6,log_var_max=4.6):
        super().__init__(); self.similarity=SimilarityLoss(c1,c2); self.decoupling=DecouplingLoss(c3); self.gradient=GradientLoss(infrared_gradient_weight); self.uwl=UncertaintyWeightedLoss(log_var_min,log_var_max)
    def forward(self,output:TPTAFOutput,infrared:Tensor,visible:Tensor,detection_losses:Mapping[str,Tensor]|Sequence[Tensor]):
        if isinstance(detection_losses,Mapping): box,cls,dfl=(detection_losses[k] for k in ('box','cls','dfl'))
        else: box,cls,dfl=detection_losses
        c={'sime':self.similarity(output.fused,infrared,visible),'deco':self.decoupling(output),'grad':self.gradient(output.fused,infrared,visible),'box':box,'cls':cls,'dfl':dfl}
        return self.uwl([c[n] for n in self.uwl.names]),c
