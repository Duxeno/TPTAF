"""Protocol-oriented TPTAF validation pipeline."""
from __future__ import annotations
import csv,json
from dataclasses import dataclass,asdict
from pathlib import Path
from typing import Any,Protocol
import numpy as np
from PIL import Image
import torch
from torch import Tensor
from .data import discover_pairs
from .metrics import compute_fusion_metrics
class FusionRunner(Protocol):
    def __call__(self,infrared:Tensor,visible:Tensor)->Tensor: ...
class DetectionRunner(Protocol):
    def __call__(self,fused_rgb:np.ndarray)->Tensor: ...
class LossRunner(Protocol):
    def __call__(self,infrared:Tensor,visible:Tensor,labels:Tensor)->tuple[dict[str,Tensor],dict[str,float]]: ...
@dataclass
class ValidationOptions:
    data_root:Path; split:str='val'; output_dir:Path=Path('outputs/tptaf_validation'); fused_dir:Path|None=None; metrics:tuple[str,...]=('ssim','vif','qabf','scd','cc','fmi_dct'); save_grayscale:bool=True; save_color:bool=False; max_samples:int|None=None; device:str|None=None; evaluate_detection:bool=False; evaluate_losses:bool=False
@dataclass
class ValidationResult:
    samples:int; fusion_metrics:dict[str,float]; detection:dict[str,Any]|None; losses:dict[str,float]|None; uwl_weights:dict[str,float]|None; mean_inference_ms:float|None; output_dir:str; warnings:list[str]
    def to_dict(self): return asdict(self)
def _read(path): return np.asarray(Image.open(path).convert('L'),dtype=np.float32)/255.
def _find(folder:Path,name:str):
    matches=[p for p in folder.iterdir() if p.is_file() and p.stem==name]
    if len(matches)!=1: raise FileNotFoundError(f'Expected one fused image for {name}, found {len(matches)}')
    return matches[0]
def _write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: path.write_text(''); return
    with path.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def validate(config:dict[str,Any],options:ValidationOptions,*,fusion_runner:FusionRunner|None=None,detection_runner:DetectionRunner|None=None,loss_runner:LossRunner|None=None)->ValidationResult:
    if options.fused_dir is None and fusion_runner is None: raise ValueError('Provide fused_dir or fusion_runner')
    samples=discover_pairs(options.data_root,options.split)
    if options.max_samples is not None:samples=samples[:options.max_samples]
    options.output_dir.mkdir(parents=True,exist_ok=True); rows=[]
    for s in samples:
        ir=_read(s.infrared); vi=_read(s.visible)
        if options.fused_dir is not None: fused=_read(_find(options.fused_dir,s.name))
        else:
            ti=torch.from_numpy(ir)[None,None]; tv=torch.from_numpy(vi)[None,None]; fused=fusion_runner(ti,tv).detach().squeeze().cpu().numpy()
        if fused.shape!=ir.shape: raise ValueError(f'{s.name}: fused shape {fused.shape} != {ir.shape}')
        if options.save_grayscale: Image.fromarray((np.clip(fused,0,1)*255).astype(np.uint8)).save(options.output_dir/f'{s.name}.png')
        rows.append({'name':s.name,**compute_fusion_metrics(ir,vi,fused,options.metrics)})
    _write_csv(options.output_dir/'fusion_per_image.csv',rows); summary={k:float(np.mean([r[k] for r in rows])) for k in options.metrics} if rows else {k:0. for k in options.metrics}; result=ValidationResult(len(rows),summary,None,None,None,None,str(options.output_dir),[]); (options.output_dir/'summary.json').write_text(json.dumps(result.to_dict(),indent=2),encoding='utf-8'); return result
