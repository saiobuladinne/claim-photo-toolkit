"""v9 localization net: ConvNeXt-T (IN-22k) RGB encoder + SRM/learnable high-pass noise
stream, FPN/U-Net decoder fused at strides 2/4/8/16, per-pixel logit map at stride 2
(upsampled to full res). Image score = mean of top-K map logits (K fixed in pixels)."""
import torch, torch.nn as nn, torch.nn.functional as F, timm
TOPK=256  # stride-2 cells (~1024 full-res px)
def srm_kernels():
    k1=torch.tensor([[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]],dtype=torch.float32)/4
    k2=torch.tensor([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]],dtype=torch.float32)/12
    k3=torch.tensor([[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]],dtype=torch.float32)/2
    w=torch.zeros(9,3,5,5)
    for i,k in enumerate([k1,k2,k3]):
        for c in range(3): w[i*3+c,c]=k
    return w
def cbr(i,o,s=1): return nn.Sequential(nn.Conv2d(i,o,3,s,1,bias=False),nn.BatchNorm2d(o),nn.ReLU(inplace=True))
class Noise(nn.Module):
    def __init__(s):
        super().__init__()
        s.register_buffer("srm",srm_kernels())
        s.learn=nn.Conv2d(3,6,5,1,2,bias=False)
        s.s1=nn.Sequential(cbr(15,32),cbr(32,32))
        s.s2=nn.Sequential(cbr(32,48,2),cbr(48,48))
        s.s4=nn.Sequential(cbr(48,64,2),cbr(64,64))
        s.s8=nn.Sequential(cbr(64,96,2),cbr(96,96))
        s.s16=nn.Sequential(cbr(96,128,2),cbr(128,128))
    def forward(s,x):  # x in [0,1]
        n=torch.cat([F.conv2d(x*255,s.srm,padding=2).clamp(-3,3)/3, s.learn(x*4)],1)
        a=s.s1(n); b=s.s2(a); c=s.s4(b); d=s.s8(c); e=s.s16(d)
        return b,c,d,e
class LocNet(nn.Module):
    def __init__(s,pretrained=True):
        super().__init__()
        s.enc=timm.create_model("convnext_tiny.fb_in22k",pretrained=pretrained,features_only=True)
        ch=s.enc.feature_info.channels()  # 96,192,384,768 @ 4,8,16,32
        s.noise=Noise(); D=128
        s.lat=nn.ModuleList([nn.Conv2d(c,D,1) for c in ch])
        s.f16=cbr(D+128,D); s.f8=cbr(D+96,D); s.f4=cbr(D+64,D)
        s.f2=nn.Sequential(cbr(D+48,64),cbr(64,64))
        s.head=nn.Conv2d(64,1,1)
        s.register_buffer("mean",torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        s.register_buffer("std",torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))
        nn.init.constant_(s.head.bias,-3.0)
    def forward(s,x):
        H,W=x.shape[-2:]
        f4,f8,f16,f32=s.enc((x-s.mean)/s.std)
        n2,n4,n8,n16=s.noise(x)
        up=lambda t,r: F.interpolate(t,size=r.shape[-2:],mode="bilinear",align_corners=False)
        p=s.lat[3](f32)
        p=s.f16(torch.cat([s.lat[2](f16)+up(p,f16),n16],1))
        p=s.f8(torch.cat([s.lat[1](f8)+up(p,f8),n8],1))
        p=s.f4(torch.cat([s.lat[0](f4)+up(p,f4),n4],1))
        p=s.f2(torch.cat([up(p,n2),n2],1))
        return s.head(p)  # logits at stride 2
def img_score(logits,k=TOPK):
    f=logits.flatten(1); k=min(k,f.shape[1])
    return f.topk(k,dim=1).values.mean(1)
