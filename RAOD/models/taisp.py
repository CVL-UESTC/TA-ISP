from collections import OrderedDict
from torch import nn
import torch
import torch.nn.functional as F
import cv2
from typing import Optional, Tuple, List
from torch import Tensor
from torch.nn.functional import conv2d, pad as torch_pad
import numpy as np

class MultiScaleSpatialAttention(nn.Module):
    """
    Multi-scale spatial attention that produces a single-channel attention map (B,1,H,W).
    Branch maps (from different kernels) are fused by per-branch weights predicted
    from global pooled descriptor (GAP over channels).
    """
    def __init__(self, in_channels, kernel_sizes=(3,5)):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_sizes = tuple(kernel_sizes)
        self.num_branches = len(self.kernel_sizes)

        # Convs operating on pooled (avg+max) input -> produce one map per branch
        self.branch_convs = nn.ModuleList()
        for k in self.kernel_sizes:
            pad = (k - 1) // 2
            # input channel = 2 (avg+max pooled concatenation)
            self.branch_convs.append(nn.Conv2d(2, 1, kernel_size=k, padding=pad, bias=True))

        # Gate: from global descriptor (B, C, 1, 1) -> (B, num_branches, 1, 1)
        self.gate_proj = nn.Conv2d(in_channels, self.num_branches, kernel_size=1, bias=True)

    def forward(self, x):
        # x: (B,C,H,W)
        B,C,H,W = x.shape

        # pooled spatial maps
        avg_pool = torch.mean(x, dim=1, keepdim=True)          # (B,1,H,W)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]        # (B,1,H,W)
        pooled = torch.cat([avg_pool, max_pool], dim=1)        # (B,2,H,W)

        # branch maps (B,1,H,W) each
        branch_maps = []
        for conv in self.branch_convs:
            m = conv(pooled)           # (B,1,H,W)
            m = torch.sigmoid(m)       # normalize to (0,1)
            branch_maps.append(m)

        # global gate weights per-branch
        desc = F.adaptive_avg_pool2d(x, 1)     # (B,C,1,1)
        gate_logits = self.gate_proj(desc)     # (B,num_branches,1,1)
        gate = F.softmax(gate_logits, dim=1)   # (B,num_branches,1,1)

        # fuse branches into single-channel map
        attn = torch.zeros((B,1,H,W), device=x.device, dtype=x.dtype)
        for i, m in enumerate(branch_maps):
            wi = gate[:, i:i+1, :, :]          # (B,1,1,1)
            attn = attn + wi * m              # broadcast to (B,1,H,W)

        attn = torch.clamp(attn, 0.0, 1.0)
        out = x * attn                       # broadcasting channel-wise
        return out

class MultipleMaskISP_Conv(nn.Module):
    """
    Conv-based mask prediction with brightness gain and FiLM-conditioned masks,
    with dataset-adaptable channel weight range.

    1. Compute global stats (mean, var).
    2. MLP derives per-channel dgain (>1) applied to raw.
    3. MLP derives FiLM γ/β to condition mask logits.
    4. Predict K spatial masks via 1×1 conv on adjusted raw.
    5. Fuse masks with original raw.
    6. Channel weights are learned and mapped to a user-specified [w_min,w_max].
    """
    def __init__(self,
                 in_ch=3,
                 num_masks=16,
                 mlp_hidden=64,
                 weight_range=(1.0, 3.0),
                 tau=0.1):
        super().__init__()
        self.in_ch = in_ch
        self.num_masks = num_masks
        self.tau = tau
        self.w_min, self.w_max = weight_range
    

        # MLP for dgain (brightness adjustment)
        self.gain_mlp = nn.Sequential(
            nn.Linear(in_ch * 2, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, in_ch),
            nn.Softplus()  # gain >= 0, then +1 in forward
        )
        # self.mask_head = MultiScalePyramid(in_ch, num_masks)
        self.mask_head = nn.Conv2d(in_ch, num_masks, kernel_size=3,padding=1)
        
        self.weight_mlp = nn.Sequential(
            nn.Linear(num_masks, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, num_masks)
        )
        self.attn = MultiScaleSpatialAttention(in_channels=3)

    def forward(self, raw):
        B, C, H, W = raw.shape
        mean = raw.view(B, C, -1).mean(dim=-1)
        var  = raw.view(B, C, -1).var(dim=-1, unbiased=False)
        stats = torch.cat([mean, var], dim=1)  # (B, 2C)
        
        dgain = self.gain_mlp(stats) + 1.0
        raw_adj = (raw * dgain.view(B, C, 1, 1)).clamp(1e-6,1)
        raw_attn = self.attn(raw_adj).clamp(1e-6,1)

        logits = self.mask_head(raw_attn)
        masks = F.gumbel_softmax(logits, dim=1, tau=self.tau)

        logits_pool = logits.view(B, self.num_masks, -1).mean(dim=-1)
        w_logit = self.weight_mlp(logits_pool)           
        w_norm = torch.sigmoid(w_logit)            
        w = self.w_min + (self.w_max - self.w_min) * w_norm
        w = w.view(B, self.num_masks, 1, 1, 1)

        raw_attn_d = raw_attn.unsqueeze(1)
        raw_attn_d = raw_attn_d.pow(1/w).clamp(1e-6,1)
        out = (masks.unsqueeze(2) * raw_attn_d).sum(dim=1)
        return out


    
if __name__ == "__main__":
    from thop import profile
    stride = 640
    net = MultipleMaskISP_Conv().to('cuda')
    img = torch.zeros((1, 3, stride, stride)).to('cuda')
    flops, params = profile(net, inputs=(img,), verbose=True)
    params /= 1e6
    flops /= 1e9
    info = "Params: {:.2f}M, Gflops: {:.2f}".format(params, flops)
    print(info)

