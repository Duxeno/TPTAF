"""TPTAF-owned detector preprocessing and external detector interface."""
from __future__ import annotations
from abc import ABC,abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
import torch
from torch import Tensor,nn
import torch.nn.functional as F
from .modules import ChannelSpatialGate,DepthwiseSeparableConv,GlobalLocalAttention

@dataclass
class DetectorOutput:
    semantics: Tensor; predictions: Any|None=None; losses: dict[str,Tensor]|None=None; detector_input: Tensor|None=None

class SemanticPreprocess(nn.Module):
    def __init__(self,c): super().__init__(); self.conv=DepthwiseSeparableConv(1,c,3); self.gate=ChannelSpatialGate(c)
    def forward(self,x):
        f=self.conv(x); return f+self.gate(f)

class DetectionInputPreprocessor(nn.Module):
    def __init__(self,channels,output_channels=3,num_heads=4,window_size=8):
        super().__init__(); self.ir=SemanticPreprocess(channels); self.vi=SemanticPreprocess(channels); self.merge=DepthwiseSeparableConv(2*channels,2*channels,3); self.gla=GlobalLocalAttention(2*channels,num_heads,window_size); self.to_detector=nn.Conv2d(2*channels,output_channels,1)
    def forward(self,ir,vi): return torch.sigmoid(self.to_detector(self.gla(self.merge(torch.cat((self.ir(ir),self.vi(vi)),1)))))

class DetectionSemanticsProvider(nn.Module,ABC):
    @abstractmethod
    def forward(self,infrared:Tensor,visible:Tensor,labels:Mapping[str,Tensor]|None=None)->DetectorOutput: raise NotImplementedError
    def postprocess_predictions(self,predictions:Any,confidence_threshold:float,iou_threshold:float,max_detections:int)->list[Tensor]: raise NotImplementedError

class ExternalDetectorAdapter(DetectionSemanticsProvider):
    def __init__(self,semantic_channels:int): super().__init__(); self.semantic_channels=semantic_channels
    def forward(self,infrared,visible,labels=None): raise NotImplementedError('Connect a separately obtained detector and return DetectorOutput.')

def validate_detection_losses(losses):
    if losses is None: return None
    missing=[k for k in ('box','cls','dfl') if k not in losses]
    if missing: raise KeyError(f'Missing detection losses: {missing}')
    return {k:losses[k] for k in ('box','cls','dfl')}

def crop_semantic_feature(semantics:Tensor,crop_boxes:Tensor,source_size:tuple[int,int])->Tensor:
    sh,sw=source_size; fh,fw=semantics.shape[-2:]; crops=[]
    for i,(top,left,height,width) in enumerate(crop_boxes.detach().cpu().tolist()):
        y0=max(0,min(fh-1,round(top*fh/sh))); x0=max(0,min(fw-1,round(left*fw/sw))); y1=max(y0+1,min(fh,round((top+height)*fh/sh))); x1=max(x0+1,min(fw,round((left+width)*fw/sw))); crops.append(semantics[i:i+1,:,y0:y1,x0:x1])
    h=max(x.shape[-2] for x in crops); w=max(x.shape[-1] for x in crops); return torch.cat([F.interpolate(x,size=(h,w),mode='bilinear',align_corners=False) for x in crops],0)
