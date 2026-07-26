"""Core TPTAF modules: embedding, PCSB, DDEB, DPG and TAM."""
from __future__ import annotations
import math
from typing import Tuple
import torch
from torch import Tensor, nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,kernel_size:int=3,stride:int=1,activation:bool=True):
        super().__init__(); p=kernel_size//2
        layers=[nn.Conv2d(in_channels,in_channels,kernel_size,stride,p,groups=in_channels,bias=False),nn.BatchNorm2d(in_channels),nn.Conv2d(in_channels,out_channels,1,bias=False),nn.BatchNorm2d(out_channels)]
        if activation: layers.append(nn.SiLU(inplace=True))
        self.block=nn.Sequential(*layers)
    def forward(self,x): return self.block(x)

def _pe(c,h,w,device,dtype):
    if c%4: raise ValueError('channels must be divisible by 4')
    q=c//4; o=1/(10000**(torch.arange(q,device=device,dtype=torch.float32)/max(q-1,1)))
    y=torch.arange(h,device=device)[:,None]*o[None]; x=torch.arange(w,device=device)[:,None]*o[None]
    py=torch.cat((y.sin(),y.cos()),1).T[:,:,None].expand(-1,-1,w); px=torch.cat((x.sin(),x.cos()),1).T[:,None,:].expand(-1,h,-1)
    return torch.cat((py,px),0)[None].to(dtype)

class PositionalEmbedding(nn.Module):
    def __init__(self,channels): super().__init__(); self.channels=channels; self.alpha=nn.Parameter(torch.zeros(1))
    def forward(self,x): return x+self.alpha*_pe(self.channels,x.shape[-2],x.shape[-1],x.device,x.dtype)

class HaarDWT(nn.Module):
    def __init__(self):
        super().__init__(); f=torch.tensor([[[1.,1.],[1.,1.]],[[-1.,-1.],[1.,1.]],[[-1.,1.],[-1.,1.]],[[1.,-1.],[-1.,1.]]])/2; self.register_buffer('filters',f[:,None],persistent=False)
    def forward(self,x):
        b,c,h,w=x.shape
        if h%2 or w%2: x=F.pad(x,(0,w%2,0,h%2),mode='reflect')
        y=F.conv2d(x,self.filters.repeat(c,1,1,1).to(x.dtype),stride=2,groups=c).reshape(b,c,4,x.shape[-2]//2,x.shape[-1]//2)
        return y[:,:,0],y[:,:,1],y[:,:,2],y[:,:,3]

class SqueezeExcitation(nn.Module):
    def __init__(self,c,reduction=4):
        super().__init__(); h=max(c//reduction,4); self.g=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(c,h,1),nn.SiLU(),nn.Conv2d(h,c,1),nn.Sigmoid())
    def forward(self,x): return x*self.g(x)

class ChannelSpatialGate(nn.Module):
    def __init__(self,c,reduction=4,spatial_kernel=7):
        super().__init__(); h=max(c//reduction,4); self.cm=nn.Sequential(nn.Conv2d(c,h,1),nn.SiLU(),nn.Conv2d(h,c,1)); self.sp=nn.Conv2d(2,1,spatial_kernel,padding=spatial_kernel//2,bias=False)
    def forward(self,x,residual=False):
        ch=torch.sigmoid(self.cm(F.adaptive_avg_pool2d(x,1))).expand_as(x); sp=torch.sigmoid(self.sp(torch.cat((x.mean(1,keepdim=True),x.amax(1,keepdim=True)),1))); g=ch*sp
        return (1+g)*x if residual else g*x

class PhaseCoherentStructureBlock(nn.Module):
    def __init__(self,c,kernel_size=3,reduction=4):
        super().__init__(); self.mask=nn.Conv2d(c,c,kernel_size,padding=kernel_size//2,groups=c); self.gamma=nn.Parameter(torch.zeros(1)); self.proj=nn.Conv2d(c,c,3,padding=1,bias=False); self.se=SqueezeExcitation(c,reduction)
    def forward(self,low,output_size):
        s=torch.fft.fft2(low,norm='ortho'); amp=s.abs(); phase=torch.angle(s); amp=amp*(1+self.gamma*torch.sigmoid(self.mask(amp))); x=torch.fft.ifft2(torch.polar(amp,phase),norm='ortho').real
        return self.se(F.silu(self.proj(F.interpolate(x,size=output_size,mode='bilinear',align_corners=False))))

class DiverseBranchBlock(nn.Module):
    def __init__(self,cin,cout,k=3):
        super().__init__(); p=k//2
        self.a=nn.Conv2d(cin,cout,k,padding=p,bias=False); self.b=nn.Conv2d(cin,cout,1,bias=False); self.c=nn.Sequential(nn.AvgPool2d(k,1,p),nn.Conv2d(cin,cout,1,bias=False)); self.d=nn.Sequential(nn.Conv2d(cin,cin,k,padding=p,groups=cin,bias=False),nn.Conv2d(cin,cout,1,bias=False)); self.n=nn.BatchNorm2d(cout)
    def forward(self,x): return F.silu(self.n(self.a(x)+self.b(x)+self.c(x)+self.d(x)))

class DiscriminativeDetailEnhancementBlock(nn.Module):
    def __init__(self,c,reduction=4): super().__init__(); self.db=DiverseBranchBlock(3*c,c); self.se=SqueezeExcitation(c,reduction); self.g=ChannelSpatialGate(c,reduction)
    def forward(self,lh,hl,hh,output_size):
        x=torch.cat([F.interpolate(b,size=output_size,mode='bilinear',align_corners=False) for b in (lh,hl,hh)],1); return self.g(self.se(self.db(x)),residual=True)

class GlobalLocalAttention(nn.Module):
    def __init__(self,c,num_heads=4,window_size=8):
        super().__init__(); self.norm=nn.BatchNorm2d(c); self.attn=nn.MultiheadAttention(c,num_heads,batch_first=True); self.local=DepthwiseSeparableConv(c,c,3); self.mix=nn.Conv2d(2*c,c,1)
    def forward(self,x):
        b,c,h,w=x.shape; t=self.norm(x).flatten(2).transpose(1,2); a,_=self.attn(t,t,t,need_weights=False); a=a.transpose(1,2).reshape(b,c,h,w); return x+F.silu(self.mix(torch.cat((a,self.local(x)),1)))

class DetectionPriorGenerator(nn.Module):
    def __init__(self,c,num_heads=4,num_points=8,window_size=8):
        super().__init__(); self.num_heads=num_heads; self.num_points=num_points; self.temp=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(c,num_heads,1)); self.support=nn.Conv2d(c,1,1); self.ref=nn.Conv2d(c,2,1); self.off=nn.Conv2d(c,2*num_points,3,padding=1)
    def forward(self,x):
        b,_,h,w=x.shape; t=F.softplus(self.temp(x))+.05; p=torch.softmax(self.support(x).flatten(2),-1).reshape(b,1,h,w); r=torch.sigmoid(self.ref(x)); o=.25*torch.tanh(self.off(x)).view(b,self.num_points,2,h,w); return t,p,(r[:,None]+o).clamp(0,1)

def _sample(f,pos):
    b,c,h,w=f.shape; m=pos.shape[1]; g=pos.permute(0,3,4,1,2).reshape(b,h,w*m,2)*2-1; s=F.grid_sample(f,g,mode='bilinear',padding_mode='border',align_corners=True); return s.view(b,c,h,w,m).permute(0,1,4,2,3)

class TripartiteAttention(nn.Module):
    def __init__(self,channels,detection_channels,num_heads=4,num_points=8,window_size=8,prior_strength=1.0):
        super().__init__(); self.c=channels; self.h=num_heads; self.d=channels//num_heads; self.m=num_points; self.beta=prior_strength
        self.ir=nn.Sequential(nn.Conv2d(2*channels,channels,1),GlobalLocalAttention(channels,num_heads,window_size)); self.vi=nn.Sequential(nn.Conv2d(2*channels,channels,1),GlobalLocalAttention(channels,num_heads,window_size)); self.q=nn.Conv2d(2*channels,channels,1); self.v=nn.Conv2d(2*channels,channels,1); self.kd=nn.Conv2d(2*channels,channels,1); self.kt=nn.Conv2d(detection_channels,channels,1); self.dpg=DetectionPriorGenerator(channels,num_heads,num_points,window_size); self.out=nn.Conv2d(channels,channels,1)
    def forward(self,si,di,sv,dv,det):
        size=si.shape[-2:]; det=F.interpolate(det,size=size,mode='bilinear',align_corners=False); fi=self.ir(torch.cat((si,di),1)); fv=self.vi(torch.cat((sv,dv),1)); q=self.q(torch.cat((fi,fv),1)); v=self.v(torch.cat((fi,fv),1)); kd=self.kd(torch.cat((di,dv),1)); kt=self.kt(det); temp,prior,pos=self.dpg(kt); sk=_sample(kd,pos); vv=_sample(v,pos); sp=_sample(prior,pos).squeeze(1); b,_,h,w=q.shape; q=q.view(b,self.h,self.d,h,w); sk=sk.view(b,self.h,self.d,self.m,h,w); vv=vv.view(b,self.h,self.d,self.m,h,w); score=(q.unsqueeze(3)*sk).sum(2)/torch.sqrt(self.d*temp).clamp_min(1e-8).unsqueeze(2)+self.beta*torch.log(sp[:,None]+1e-8); weight=torch.softmax(score,2); y=(weight.unsqueeze(2)*vv).sum(3).reshape(b,self.c,h,w); return self.out(y)+.5*(fi+fv),{'temperature':temp,'spatial_prior':prior,'sampling_positions':pos,'attention_weights':weight}
