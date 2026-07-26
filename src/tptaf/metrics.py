"""Fusion metrics used by the released validation pipeline."""
from __future__ import annotations
import numpy as np
from scipy import ndimage

def _as(x):
    a=np.asarray(x,dtype=np.float64)
    if a.ndim!=2: raise ValueError('metric inputs must be grayscale HxW arrays')
    return np.clip(a,0,1)

def ssim(x,y):
    x=_as(x); y=_as(y); ux=ndimage.uniform_filter(x,11); uy=ndimage.uniform_filter(y,11); vx=ndimage.uniform_filter(x*x,11)-ux*ux; vy=ndimage.uniform_filter(y*y,11)-uy*uy; cxy=ndimage.uniform_filter(x*y,11)-ux*uy
    return float(np.mean(((2*ux*uy+1e-4)*(2*cxy+9e-4))/((ux*ux+uy*uy+1e-4)*(vx+vy+9e-4)+1e-12)))
def cc(x,y):
    x=_as(x).ravel(); y=_as(y).ravel(); x=x-x.mean(); y=y-y.mean(); return float((x@y)/(np.linalg.norm(x)*np.linalg.norm(y)+1e-12))
def scd(ir,vi,f): return cc(f-ir,vi)+cc(f-vi,ir)
def vif(source,fused):
    source=_as(source); fused=_as(fused); num=den=0.
    for sigma in (1.2,2.4,4.8,9.6):
        mu1=ndimage.gaussian_filter(source,sigma); mu2=ndimage.gaussian_filter(fused,sigma); v1=np.maximum(ndimage.gaussian_filter(source*source,sigma)-mu1*mu1,0); v2=np.maximum(ndimage.gaussian_filter(fused*fused,sigma)-mu2*mu2,0); cov=ndimage.gaussian_filter(source*fused,sigma)-mu1*mu2; g=cov/(v1+1e-10); sv=np.maximum(v2-g*cov,1e-10); num+=np.log1p(g*g*v1/sv).sum(); den+=np.log1p(v1/1e-4).sum()
    return float(num/(den+1e-12))
def _grad(x): return np.hypot(ndimage.sobel(x,0),ndimage.sobel(x,1))
def qabf(ir,vi,f):
    gi,gv,gf=_grad(_as(ir)),_grad(_as(vi)),_grad(_as(f)); qi=np.minimum(gi,gf)/(np.maximum(gi,gf)+1e-12); qv=np.minimum(gv,gf)/(np.maximum(gv,gf)+1e-12); wi=gi**2; wv=gv**2; return float(((qi*wi+qv*wv).sum())/(wi.sum()+wv.sum()+1e-12))
def fmi_dct(ir,vi,f):
    from scipy.fft import dctn
    def corr(a,b): return cc(np.abs(dctn(_as(a),norm='ortho')),np.abs(dctn(_as(b),norm='ortho')))
    return .5*(corr(ir,f)+corr(vi,f))
METRIC_REGISTRY={'ssim':lambda i,v,f:.5*(ssim(i,f)+ssim(v,f)),'vif':lambda i,v,f:.5*(vif(i,f)+vif(v,f)),'qabf':qabf,'scd':scd,'cc':lambda i,v,f:.5*(cc(i,f)+cc(v,f)),'fmi_dct':fmi_dct}
def compute_fusion_metrics(ir,vi,fused,metrics=None):
    names=tuple(metrics or METRIC_REGISTRY); return {n:float(METRIC_REGISTRY[n](ir,vi,fused)) for n in names}
