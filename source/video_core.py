"""Predictive memory model."""
import hashlib
import numpy as np
import cv2
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

FEATURES=['log_detail_energy','log_innovation','relative_innovation','log_age',
 'capped_age','flow_inconsistency','flow_speed','log_gradient','log_curvature',
 'log_preview_noise','old_gradient','detail_gradient_ratio','short_change',
 'log_texture','log_anchor_error','relative_anchor_error','cumulative_innovation']

def seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:4],'little')

def down(x,k):
    h,w,c=x.shape
    return x.reshape(h//k,k,w//k,k,c).mean((1,3))


def _cubic(x):
    ax=np.abs(x); ax2=ax*ax; ax3=ax2*ax
    return ((1.5*ax3-2.5*ax2+1)*(ax<=1)+(-.5*ax3+2.5*ax2-4*ax+2)*((ax>1)&(ax<=2))).astype(np.float64)

def _imresize_weights(in_len,out_len,scale):
    """MATLAB imresize bicubic weights with antialiasing for downscaling (scale<1)."""
    width=4./scale
    x=np.arange(1,out_len+1,dtype=np.float64)
    u=x/scale+.5*(1-1/scale)
    left=np.floor(u-width/2)
    taps=int(np.ceil(width))+2
    idx=left[:,None]+np.arange(taps)[None,:]-1
    w=scale*_cubic(scale*(u[:,None]-idx-1))
    w=w/w.sum(1,keepdims=True)
    # Symmetric boundary handling, as in MATLAB.
    aux=np.concatenate([np.arange(in_len),np.arange(in_len-1,-1,-1)])
    idx=aux[np.mod(idx.astype(int),2*in_len)]
    keep=np.any(w!=0,axis=0)
    return w[:,keep],idx[:,keep]

def matlab_bicubic_down(x,k):
    """Antialiased MATLAB-compatible bicubic downsampling by an integer factor.

    This is the degradation used to train standard bicubic (BI) super-resolution
    checkpoints, so pretrained baselines are evaluated in their native regime.
    """
    x=np.asarray(x,np.float64); h,w=x.shape[:2]
    wy,iy=_imresize_weights(h,h//k,1./k); wx,ix=_imresize_weights(w,w//k,1./k)
    y=np.einsum('ot,otwc->owc',wy,x[iy]) if x.ndim==3 else np.einsum('ot,otw->ow',wy,x[iy])
    y=np.einsum('ot,hotc->hoc',wx,y[:,ix]) if x.ndim==3 else np.einsum('ot,hot->ho',wx,y[:,ix])
    return y.astype(np.float32)

def up(x,k):return x.repeat(k,0).repeat(k,1)

def base_image(p,k):
    b=cv2.resize(p,(p.shape[1]*k,p.shape[0]*k),interpolation=cv2.INTER_CUBIC)
    return (b+up(p-down(b,k),k)).astype(np.float32)

def detail(x,k):return x-base_image(down(x,k),k)

def blocks(x,k):
    if x.ndim==3:x=x.mean(2)
    h,w=x.shape
    return x.reshape(h//k,k,w//k,k).mean((1,3))

def expand(x,k):return x.repeat(k,0).repeat(k,1)

def remap(x,flow):
    yy,xx=np.indices(flow.shape[:2],dtype=np.float32)
    return cv2.remap(x,xx+flow[...,0],yy+flow[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)

def gray(x):return cv2.cvtColor(x.astype(np.float32),cv2.COLOR_RGB2GRAY)

def gradient_energy(x):
    y=gray(x) if x.ndim==3 else x
    gx=cv2.Sobel(y,cv2.CV_32F,1,0,ksize=3)/8
    gy=cv2.Sobel(y,cv2.CV_32F,0,1,ksize=3)/8
    return gx*gx+gy*gy

class RiskMemory:
    """Weighted noisy-observation risk minimisation; no clean pixel target."""
    def __init__(self,family='extra',use_history=True,leaf=40):
        self.family=family;self.use_history=use_history;self.leaf=leaf
    def _x(self,X):return X if self.use_history else X[:,:14]
    def fit(self,X,C,H,R):
        eps=1e-6;weights=(H+eps)/(H+eps).mean();X=self._x(X)
        if self.family=='extra':
            self.gain_model=ExtraTreesRegressor(n_estimators=80,min_samples_leaf=self.leaf,max_depth=16,n_jobs=1,random_state=42)
            self.risk_model=ExtraTreesRegressor(n_estimators=80,min_samples_leaf=self.leaf,max_depth=16,n_jobs=1,random_state=43)
        else:
            args=dict(max_iter=120,max_leaf_nodes=15,min_samples_leaf=self.leaf,l2_regularization=1.,learning_rate=.06,early_stopping=False)
            self.gain_model=HistGradientBoostingRegressor(random_state=42,**args)
            self.risk_model=HistGradientBoostingRegressor(loss='poisson',random_state=43,**args)
        self.gain_model.fit(X,C/(H+eps),sample_weight=weights)
        self.risk_model.fit(X,np.maximum(R,1e-8));return self
    def predict(self,X,H):
        raw=self.gain_model.predict(self._x(X));g=np.clip(raw,0,1)
        R=np.maximum(self.risk_model.predict(self._x(X)),0)
        return g,np.maximum(R-2*g*raw*(H+1e-6)+g*g*H,0)

def feature_maps(bank,age,anchor,history,obs,cfg):
    k=cfg.block;b=obs['b'];h=detail(bank,cfg.factor)
    H=blocks(h*h,k);e=blocks(obs['innovation'],k)
    a=blocks(age,k);grad=blocks(gradient_energy(b),k)
    old=blocks(gradient_energy(b+h),k)
    curv=blocks(cv2.Laplacian(gray(b),cv2.CV_32F)**2,k)
    ae=blocks((b-anchor)**2,k)
    hist=blocks(history,k)
    v=blocks(obs['fb'],k);speed=blocks(np.sum(obs['flow']**2,2),k)
    noise=obs['noise']**2; eps=1e-6
    log=lambda x:np.log1p(np.maximum(x,0)/eps)
    arrays=[log(H),log(e),np.minimum(e/(grad+noise+eps),100),np.log1p(a),np.minimum(a,64)/64,
        np.log1p(v),np.log1p(speed),log(grad),log(curv),np.full_like(H,log(noise)),log(old),
        np.minimum(H/(grad+noise+eps),100),log(e),log(blocks((b-cv2.GaussianBlur(b,(0,0),2.))**2,k)),
        log(ae),np.minimum(ae/(grad+noise+eps),100),log(hist)]
    X=np.stack(arrays,-1).reshape(-1,len(arrays)).astype(np.float32)
    return X,h,H,a,e,grad,ae,v

