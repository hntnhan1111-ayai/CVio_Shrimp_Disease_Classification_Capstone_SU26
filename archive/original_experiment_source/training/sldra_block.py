
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .block import C3k2

def _divisor(c, g):
    for x in range(min(c,g),0,-1):
        if c%x==0: return x
    return 1

class CircularChannel(nn.Module):
    def __init__(self,k,d):
        super().__init__(); self.p=((k-1)*d)//2
        self.conv=nn.Conv1d(1,1,k,dilation=d,bias=False)
    def forward(self,x):
        return self.conv(F.pad(x.unsqueeze(1),(self.p,self.p),mode="circular")).squeeze(1)

class SpatialEvidence(nn.Module):
    def __init__(self,c,g,k):
        super().__init__(); self.g=_divisor(c,g); q=c//self.g; p=k//2
        self.h=nn.Conv2d(q,q,(1,k),padding=(0,p),groups=q,bias=False)
        self.v=nn.Conv2d(q,q,(k,1),padding=(p,0),groups=q,bias=False)
        self.n=nn.GroupNorm(1,q); self.o=nn.Conv2d(q,1,1)
    def forward(self,x):
        b,c,h,w=x.shape; z=x.reshape(b*self.g,c//self.g,h,w)
        z=self.o(F.silu(self.n(self.h(z)+self.v(z))))
        return z.reshape(b,self.g,h,w).mean(1,keepdim=True)

class BoundaryEvidence(nn.Module):
    def forward(self,x):
        z=x.abs().mean(1,keepdim=True)
        q=(z-F.avg_pool2d(z,3,1,1)).abs()
        q=F.avg_pool2d(q,3,1,1)
        return q/q.mean((2,3),keepdim=True).clamp_min(1e-6)

class SLDRA(nn.Module):
    def __init__(self,c,variant,family,ck,cd,sk,groups,robust,fusion,alpha,temp,bw):
        super().__init__()
        self.variant=int(variant); self.family=int(family); self.robust=int(robust)
        self.fusion=int(fusion); self.alpha=float(alpha); self.temp=float(temp); self.bw=float(bw)
        self.cm=CircularChannel(int(ck),int(cd))
        self.sm=SpatialEvidence(c,int(groups),int(sk))
        self.be=BoundaryEvidence()
        self.cb=nn.Parameter(torch.zeros(1,c))
        self.balance=nn.Parameter(torch.tensor(0.0))
        self.confscale=nn.Parameter(torch.tensor(1.0))
    def desc(self,x):
        if self.robust==0:
            m=x.mean((2,3),keepdim=True); return (x-m).abs().mean((2,3))
        if self.robust==1:
            return x.square().mean((2,3)).add(1e-6).sqrt()
        z=x.abs().flatten(2); k=max(1,z.shape[-1]//8)
        return z.topk(k,dim=-1).values.mean(-1)
    def forward(self,x):
        d=self.desc(x); c=self.cm(d)+self.cb
        s=self.sm(x); b=self.be(x)*self.bw
        cc=c[...,None,None]
        if self.fusion==0: e=cc+s+b
        elif self.fusion==1: e=cc*torch.sigmoid(s)+s+b
        else:
            q=torch.sigmoid(self.balance); e=q*cc+(1-q)*s+b
        if self.family==0: e=e+0.10*cc
        elif self.family==1: e=e+0.15*torch.tanh(cc)
        elif self.family==2: e=e+0.10*s
        elif self.family==3: e=e+self.bw*self.be(x)
        elif self.family==4:
            q=torch.sigmoid(self.confscale*d.std(1,keepdim=True))[...,None,None]; e=e*q
        else: e=e*(0.5+torch.sigmoid(cc*s))
        return x*(1.0+self.alpha*torch.tanh(e/max(self.temp,1e-3)))

class C3k2SLDRA(nn.Module):
    def __init__(self,c1,c2,n=1,shortcut=False,e=0.25,variant=0,family=0,ck=3,cd=1,sk=3,
                 groups=8,robust=0,fusion=0,alpha=0.15,temp=0.85,bw=0.10):
        super().__init__()
        self.block=C3k2(c1,c2,n,shortcut,e)
        self.attn=SLDRA(c2,variant,family,ck,cd,sk,groups,robust,fusion,alpha,temp,bw)
    def forward(self,x): return self.attn(self.block(x))
