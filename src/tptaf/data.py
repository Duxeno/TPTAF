"""Paired infrared-visible dataset helpers."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
EXT={'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}
@dataclass(frozen=True)
class PairedSample:
    name:str; infrared:Path; visible:Path; label:Path|None=None

def _index(folder:Path): return {p.stem:p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXT}
def discover_pairs(root:str|Path,split='val',require_labels=False):
    base=Path(root)/split; ir=_index(base/'ir'); vi=_index(base/'vi'); names=sorted(ir.keys()&vi.keys()); out=[]
    for n in names:
        label=base/'labels'/f'{n}.txt'; label=label if label.is_file() else None
        if require_labels and label is None: continue
        out.append(PairedSample(n,ir[n],vi[n],label))
    return out

def _gray(path): return torch.from_numpy(np.asarray(Image.open(path).convert('L'),dtype=np.float32)/255.).unsqueeze(0)
def _labels(path):
    if path is None:return torch.empty((0,5),dtype=torch.float32)
    rows=[]
    for line in path.read_text().splitlines():
        if line.strip(): rows.append([float(v) for v in line.split()[:5]])
    return torch.tensor(rows,dtype=torch.float32) if rows else torch.empty((0,5),dtype=torch.float32)

class PairedFusionDetectionDataset(Dataset):
    def __init__(self,root,split='train',require_labels=False): self.samples=discover_pairs(root,split,require_labels)
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        s=self.samples[i]; return {'name':s.name,'infrared':_gray(s.infrared),'visible':_gray(s.visible),'labels':_labels(s.label)}

def collate_paired_batch(batch):
    return {'name':[x['name'] for x in batch],'infrared':torch.stack([x['infrared'] for x in batch]),'visible':torch.stack([x['visible'] for x in batch]),'labels':[x['labels'] for x in batch]}
