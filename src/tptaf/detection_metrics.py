"""Lightweight class-wise AP and mAP50-95 accumulation."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch import Tensor

def box_iou(a:Tensor,b:Tensor)->Tensor:
    area1=(a[:,2]-a[:,0]).clamp(0)*(a[:,3]-a[:,1]).clamp(0); area2=(b[:,2]-b[:,0]).clamp(0)*(b[:,3]-b[:,1]).clamp(0); lt=torch.maximum(a[:,None,:2],b[None,:,:2]); rb=torch.minimum(a[:,None,2:],b[None,:,2:]); inter=(rb-lt).clamp(0).prod(2); return inter/(area1[:,None]+area2[None]-inter+1e-12)
def xywhn_to_xyxy(labels:Tensor,h:int,w:int)->Tensor:
    if labels.numel()==0:return torch.empty((0,5))
    c,x,y,bw,bh=labels.T; return torch.stack((c,(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h),1)

def _ap(rec,prec):
    mrec=np.r_[0,rec,1]; mpre=np.r_[0,prec,0]; mpre=np.maximum.accumulate(mpre[::-1])[::-1]; return float(np.trapz(np.interp(np.linspace(0,1,101),mrec,mpre),np.linspace(0,1,101)))
@dataclass
class DetectionSummary:
    map50:float; map50_95:float; per_class:list[dict]
    def to_dict(self): return {'map50':self.map50,'map50_95':self.map50_95,'per_class':self.per_class}
class DetectionMetricAccumulator:
    def __init__(self,class_names,thresholds=None): self.names=list(class_names); self.th=torch.tensor(thresholds or np.arange(.5,1,.05)); self.records=[]
    def update(self,pred:Tensor,target:Tensor): self.records.append((pred.cpu(),target.cpu()))
    def compute(self):
        rows=[]; allaps=[]
        for ci,name in enumerate(self.names):
            aps=[]
            for t in self.th:
                scores=[]; hits=[]; total=0
                for pred,target in self.records:
                    p=pred[pred[:,5].long()==ci] if pred.numel() else pred.reshape(0,6); g=target[target[:,0].long()==ci]; total+=len(g); used=set()
                    for row in p[p[:,4].argsort(descending=True)]:
                        scores.append(float(row[4]));
                        if len(g):
                            ious=box_iou(row[:4][None],g[:,1:5]).squeeze(0); j=int(torch.argmax(ious)); ok=float(ious[j])>=float(t) and j not in used
                        else: ok=False
                        hits.append(1 if ok else 0)
                        if ok: used.add(j)
                if scores:
                    order=np.argsort(scores)[::-1]; tp=np.cumsum(np.array(hits)[order]); fp=np.cumsum(1-np.array(hits)[order]); rec=tp/max(total,1); prec=tp/np.maximum(tp+fp,1); aps.append(_ap(rec,prec))
                else: aps.append(0.)
            rows.append({'class':name,'ap50':aps[0],'ap50_95':float(np.mean(aps))}); allaps.append(aps)
        arr=np.array(allaps) if allaps else np.zeros((1,len(self.th))); return DetectionSummary(float(arr[:,0].mean()),float(arr.mean()),rows)
