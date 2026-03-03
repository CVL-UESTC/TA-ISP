from collections import OrderedDict
from torch import nn
import torch
import torch.nn.functional as F
import cv2
from typing import Optional, Tuple, List
from torch import Tensor
from torch.nn.functional import conv2d, pad as torch_pad
import numpy as np

class fc(nn.Module):
    def __init__(self, inc, outc, activation=None, is_BN=False):
        super(fc, self).__init__()
        if is_BN:
            self.fc = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(inc, outc)),
                ("bn", nn.BatchNorm1d(outc)),
                ("act", activation),
            ]))
        elif activation is not None:
            self.fc = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(inc, outc)),
                ("act", activation),
            ]))
        else:
            self.fc = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(inc, outc)),
            ]))

    def forward(self, input):
        return self.fc(input)

class Guide(nn.Module):
    '''
    pointwise neural net
    '''
    def __init__(self, mode="PointwiseNN"):
        super(Guide, self).__init__()
        if mode == "PointwiseNN":
            self.mode = "PointwiseNN"
            self.conv1 = conv_block(3, 16, kernel_size=1, padding=0, is_BN=True)
            self.conv2 = conv_block(16, 1, kernel_size=1, padding=0, activation=nn.Tanh())

        elif mode == "PointwiseCurve":
            # ccm: color correction matrix
            self.ccm = nn.Conv2d(3, 3, kernel_size=1)

            pixelwise_weight = torch.FloatTensor([1, 0, 0, 0, 1, 0, 0, 0, 1]) + torch.randn(1) * 1e-4
            pixelwise_bias = torch.FloatTensor([0, 0, 0])

            self.conv1x1.weight.data.copy_(pixelwise_weight.view(3, 3, 1, 1))
            self.conv1x1.bias.data.copy_(pixelwise_bias)

            # per channel curve
            pass

            # conv2d: num_output = 1
            self.conv1x1 = nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, x):
        if self.mode == "PointwiseNN":
            guidemap = self.conv2(self.conv1(x))

        return guidemap

class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, bilateral_grid, guidemap):
        N, _, H, W = guidemap.shape
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
        hg = hg.type(torch.cuda.FloatTensor).repeat(N, 1, 1).unsqueeze(3) / (H-1) * 2 - 1
        wg = wg.type(torch.cuda.FloatTensor).repeat(N, 1, 1).unsqueeze(3) / (W-1) * 2 - 1
        guidemap = guidemap.permute(0,2,3,1).contiguous()
        guidemap_guide = torch.cat([guidemap, hg, wg], dim=3).unsqueeze(1)

        coeff = F.grid_sample(bilateral_grid, guidemap_guide)

        return coeff.squeeze(2)

class Transform(nn.Module):
    def __init__(self):
        super(Transform, self).__init__()

    def forward(self, coeff, full_res_input):
        R = torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True) + coeff[:, 3:4, :, :]
        G = torch.sum(full_res_input * coeff[:, 4:7, :, :], dim=1, keepdim=True) + coeff[:, 7:8, :, :]
        B = torch.sum(full_res_input * coeff[:, 8:11, :, :], dim=1, keepdim=True) + coeff[:, 11:12, :, :]

        return torch.cat([R, G, B], dim=1)
    
class conv_block(nn.Module):
    def __init__(self, inc , outc, kernel_size=3, padding=1, stride=1, use_bias=True, activation=nn.ReLU(inplace=True), is_BN=False):
        super(conv_block, self).__init__()
        if is_BN:
            self.conv = nn.Sequential(OrderedDict([
                ("conv", nn.Conv2d(inc, outc, kernel_size, padding=padding, stride=stride, bias=use_bias)),
                ("bn", nn.BatchNorm2d(outc)),
                ("act", activation)
            ]))
        elif activation is not None:
            self.conv = nn.Sequential(OrderedDict([
                ("conv", nn.Conv2d(inc, outc, kernel_size, padding=padding, stride=stride, bias=use_bias)),
                ("act", activation)
            ]))
        else:
            self.conv = nn.Sequential(OrderedDict([
                ("conv", nn.Conv2d(inc, outc, kernel_size, padding=padding, stride=stride, bias=use_bias)),
            ]))

    def forward(self, input):
        return self.conv(input)
    
class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(self, in_channels, out_channels, ksize=3, stride=1, bias=True):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, bias=bias)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

def sigmoid_range(l=0.5, r=2.0):
    def get_activation(left, right):
        def activation(x):
            return (torch.sigmoid(x)) * (right - left) + left
        return activation
    return get_activation(l, r)



import functools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F



    
class GFE(nn.Module):
    def __init__(self, in_ch=3, nf=32):
        super(GFE, self).__init__()
        self.conv1 = conv_block(in_ch, nf, stride=2, is_BN=False)
        self.conv2 = conv_block(nf, nf, stride=2, is_BN=False)
        self.conv3 = conv_block(nf, nf, stride=2, is_BN=False)
        self.pool = nn.AdaptiveAvgPool2d(1)  

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        return x
    

class param_generator(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.r1_base = nn.Parameter(torch.FloatTensor([0.05]), requires_grad=False)
        self.r2_base = nn.Parameter(torch.FloatTensor([1]), requires_grad=False)
        self.long_fc = nn.Linear(nf, 1)
        self.short_fc = nn.Linear(nf, 1)
    def forward(self, global_feature):
        r1 = self.long_fc(global_feature)
        r2 = self.short_fc(global_feature)
        r1 = 0.1 * r1 +  self.r1_base
        r2 = 0.1 * r2 +  self.r2_base
        # print(r1,r2)
        
        return r1, r2

def getGaussianKernel(ksize, device,sigma=0):
    if sigma <= 0:
        # 根据 kernelsize 计算默认的 sigma，和 OpenCV 保持一致
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8 
    center = ksize // 2
    # 使用 torch 而不是 numpy 计算元素与矩阵中心的横向距离
    xs = (torch.arange(ksize, dtype=torch.float32) - center).to(device)
    # 计算一维高斯卷积核
    kernel1d = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
    # 利用矩阵乘法计算二维高斯卷积核
    kernel = kernel1d[..., None] @ kernel1d[None, ...]
    # 归一化
    kernel = kernel / kernel.sum()
    return kernel

def bilateralFilter(batch_img, ksize, sigmaColor=None, sigmaSpace=None):
    device = batch_img.device
    if sigmaSpace is None:
        sigmaSpace = 0.15 * ksize + 0.35  # 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    if sigmaColor is None:
        sigmaColor = sigmaSpace
    
    pad = (ksize - 1) // 2
    batch_img_pad = F.pad(batch_img, pad=[pad, pad, pad, pad], mode='reflect')
    
    # batch_img 的维度为 BxcxHxW, 因此要沿着第 二、三维度 unfold
    # patches.shape:  B x C x H x W x ksize x ksize
    patches = batch_img_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
    patch_dim = patches.dim() # 6 
    # 求出像素亮度差
    diff_color = patches - batch_img.unsqueeze(-1).unsqueeze(-1)
    # 根据像素亮度差，计算权重矩阵
    weights_color = torch.exp(-(diff_color ** 2) / (2 * sigmaColor ** 2))
    # 归一化权重矩阵
    weights_color = weights_color / weights_color.sum(dim=(-1, -2), keepdim=True)
    
    # 获取 gaussian kernel 并将其复制成和 weight_color 形状相同的 tensor
    weights_space = getGaussianKernel(ksize, device, sigmaSpace)
    weights_space_dim = (patch_dim - 2) * (1,) + (ksize, ksize)
    weights_space = weights_space.view(*weights_space_dim).expand_as(weights_color)
    
    # 两个权重矩阵相乘得到总的权重矩阵
    weights = weights_space * weights_color
    # 总权重矩阵的归一化参数
    weights_sum = weights.sum(dim=(-1, -2))
    # 加权平均
    weighted_pix = (weights * patches).sum(dim=(-1, -2)) / weights_sum
    return weighted_pix

class DetailModule(nn.Module):
    def __init__(self, strength=1.0):
        super().__init__()
        # 可学习的细节放大强度
        self.strength = nn.Parameter(torch.tensor(strength))
        # 定义 Laplacian kernel
        kernel = torch.tensor([[0, -1, 0],
                               [-1, 4, -1],
                               [0, -1, 0]], dtype=torch.float32)
        self.register_buffer('kernel', kernel.unsqueeze(0).unsqueeze(0))  # 1×1×3×3
    
    def forward(self, x):
        # x: B×C×H×W
        # 对每个通道做高通滤波
        detail = F.conv2d(x, self.kernel.expand(x.size(1), -1, -1, -1),
                          padding=1, groups=x.size(1))
        # 放大后加回
        return x + self.strength * detail


class TAISP(nn.Module):
    def __init__(self, in_ch=3, nf=32, gamma_range=[2.2, 3.0], d_range=[1.0, 1.1]):
        super().__init__()
        self.d_range = d_range
        self.gamma_range = gamma_range
        self.activation = nn.ReLU(inplace=True)

        self.gfe = GFE(in_ch=in_ch, nf=nf)
        # self.param_generator = param_generator(nf)

        # 输出为5x5x3的gamma图
        self.gamma_fc = fc(nf, 25 * 3, activation=None, is_BN=False)
        self.dgain_fc = fc(nf, 1, activation=None, is_BN=False)
        self.ccm_fc = fc(nf, 27, activation=None, is_BN=False)

        # 卷积层用于融合
        # self.adapt = nn.Conv2d(in_ch,in_ch,1,1,0)

    def apply_gamma(self, img, gamma_map):
        gamma_map = sigmoid_range(self.gamma_range[0], self.gamma_range[1])(gamma_map)
        gamma_map = F.interpolate(gamma_map, size=img.shape[2:], mode='bilinear', align_corners=True)
        
        out_images = []
        for b in range(img.size(0)):
            gamma_corrected = [img[b] ** (1.0 / gamma_map[b, i]) for i in range(gamma_map.shape[1])]
            gamma_corrected_avg = torch.stack(gamma_corrected, dim=0).mean(dim=0)
            out_images.append(gamma_corrected_avg)
        
        out_image = torch.stack(out_images, dim=0)
        return out_image

    def apply_dgain(self, img, dgain):
        dgain = sigmoid_range(self.d_range[0], self.d_range[1])(dgain)[..., None, None]
        out_img = img * dgain
        return out_img.clamp(1e-6, 1)

    def apply_ccm(self, img, ccm):
        img_flat = img.view(img.size(0), img.size(1), -1)
        ccm = ccm.view(ccm.size(0), 3, 9)

        R, G, B = img_flat[:, 0, :], img_flat[:, 1, :], img_flat[:, 2, :]
        
        R2, G2, B2 = R ** 2, G ** 2, B ** 2
        RG, RB, GB = R * G, R * B, G * B
        X = torch.stack([R, G, B, R2, G2, B2, RG, RB, GB], dim=1)

        img_corrected_flat = torch.bmm(ccm, X)
        img_corrected = img_corrected_flat.view(img.size())
        return img_corrected.clamp(1e-6, 1)

    def forward(self, img):
        img_down = F.interpolate(img, (256, 256), mode='bilinear', align_corners=True)
        bs, _, _, _ = img_down.size()

        global_info = self.gfe(img_down)
        # r1, r2 = self.param_generator(global_info.view(bs, -1))

        img_de = (bilateralFilter(img, 3, 0.05, 1) + 
                  0.5 * (img - bilateralFilter(img, 3, 0.05, 1))).clamp(1e-6, 1.0)

        dgain_params = self.dgain_fc(global_info.view(bs, -1))
        img_inter = self.apply_dgain(img_de, dgain_params)

        out_wb = torch.zeros_like(img_inter)

        mean_r = img_inter[:, 0, :, :].mean()
        mean_g = img_inter[:, 1, :, :].mean()
        mean_b = img_inter[:, 2, :, :].mean()

        out_wb[:, 0, :, :] = img_inter[:, 0, :, :] * (mean_g / mean_r)
        out_wb[:, 1, :, :] = img_inter[:, 1, :, :]  # 保持绿色通道不变
        out_wb[:, 2, :, :] = img_inter[:, 2, :, :] * (mean_g / mean_b)

        out_wb = out_wb.clamp(1e-6, 1)

        ccm_params = self.ccm_fc(global_info.view(bs, -1)).view(bs, 3, 9)
        base_ccm = torch.ones(bs, 3, 9, device=img.device)
        final_ccm = 2 * ccm_params + base_ccm
        out_ccm = self.apply_ccm(out_wb, final_ccm)

        gamma_params = self.gamma_fc(global_info.view(bs, -1)).view(bs, 3, 5, 5)
        out_gamma = self.apply_gamma(out_ccm, gamma_params)
        # out = self.adapt(out_gamma)

        
        return out_gamma
   
# class MultipleMaskISP_Conv(nn.Module):
#     """
#     Conv-based mask prediction with brightness gain and FiLM-conditioned masks,
#     with dataset-adaptable channel weight range.

#     1. Compute global stats (mean, var).
#     2. MLP derives per-channel dgain (>1) applied to raw.
#     3. MLP derives FiLM γ/β to condition mask logits.
#     4. Predict K spatial masks via 1×1 conv on adjusted raw.
#     5. Fuse masks with original raw.
#     6. Channel weights are learned and mapped to a user-specified [w_min,w_max].
#     """
#     def __init__(self,
#                  in_ch=3,
#                  num_masks=16,
#                  mlp_hidden=64,
#                  weight_range=(1.0, 3.0),
#                  tau=1.0):
#         super().__init__()
#         self.in_ch = in_ch

#         # MLP for dgain (brightness adjustment)
#         self.gain_mlp = nn.Sequential(
#             nn.Linear(in_ch * 2, mlp_hidden),
#             nn.ReLU(inplace=True),
#             nn.Linear(mlp_hidden, in_ch),
#             nn.Softplus()  # gain >= 0, then +1 in forward
#         )


#     def forward(self, raw):
#         B, C, H, W = raw.shape
#         # 1. global stats
#         mean = raw.view(B, C, -1).mean(dim=-1)
#         var  = raw.view(B, C, -1).var(dim=-1, unbiased=False)
#         stats = torch.cat([mean, var], dim=1)  # (B, 2C)

#         # 2. brightness gain
#         dgain = self.gain_mlp(stats) + 1.0
#         # print(dgain)
#         raw_adj = (raw * dgain.view(B, C, 1, 1)).clamp(1e-6,1)

#         return raw_adj



class SpatialAttention(nn.Module):
    """
    Simple spatial attention via max & avg pooling
    """
    def __init__(self, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding)
        # self.conv2 = nn.Conv2d(2, 1, kernel_size, padding=padding*3, dilation=3)
    def forward(self, x):
        # x: [B,C,H,W]
        avg = torch.mean(x, dim=1, keepdim=True)
        mx = torch.max(x, dim=1, keepdim=True)[0]
        cat = torch.cat([avg, mx], dim=1)
        attn = torch.sigmoid(self.conv(cat))  # [B,1,H,W]
        # attn = F.softplus(self.conv(cat))  # [B,1,H,W]
        return x * attn
    
class PerChannelSpatialAttention(nn.Module):
    """
    For each input channel c: apply a dedicated 2D conv on that single-channel map,
    then sigmoid, then concatenate to produce attn (B, C, H, W).
    Implemented efficiently via depthwise conv (groups=in_channels).
    Returns out = x * attn (elementwise) and optionally attn.
    """
    def __init__(self, in_channels, kernel_size=3, bias=True, return_map=False):
        """
        in_channels: number of input channels (e.g., 3 for RGB)
        kernel_size: odd recommended (3,5,7)
        bias: whether each per-channel conv has bias
        return_map: if True, forward returns (out, attn)
        """
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd"
        self.in_ch = in_channels
        self.ks = kernel_size
        self.pad = (kernel_size - 1) // 2
        # Depthwise conv: in_channels -> in_channels, groups=in_channels
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                padding=self.pad, groups=in_channels, bias=bias)
        # init
        nn.init.kaiming_normal_(self.dwconv.weight, mode='fan_out', nonlinearity='relu')
        if bias:
            nn.init.constant_(self.dwconv.bias, 0.0)

        self.return_map = return_map
        self.act = nn.Sigmoid()

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: out (B, C, H, W)  (and attn if return_map True)
        """
        # depthwise conv yields one feature map per channel (B, C, H, W)
        attn = self.dwconv(x)        # per-channel conv responses
        attn = self.act(attn)        # per-channel sigmoid -> (0,1)
        out = x * attn               # element-wise modulation
        if self.return_map:
            return out, attn
        return out
    
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
        # self.mask_head = nn.Sequential(
        #     nn.Conv2d(in_ch, num_masks, kernel_size=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(num_masks, num_masks, kernel_size=3,padding=1,groups=num_masks)
        # )
        # MLP for raw channel weight logits
        
        self.weight_mlp = nn.Sequential(
            nn.Linear(num_masks, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, num_masks)
        )
        # self.attn = SpatialAttention(kernel_size=3)
        # self.attn = PerChannelSpatialAttention(in_channels=3)
        self.attn = MultiScaleSpatialAttention(in_channels=3)
        # self.attn = SelectiveKernelBlock(in_ch=3)

    def forward(self, raw):
        # print(raw.max())
        B, C, H, W = raw.shape
        # 1. global stats
        mean = raw.view(B, C, -1).mean(dim=-1)
        var  = raw.view(B, C, -1).var(dim=-1, unbiased=False)
        stats = torch.cat([mean, var], dim=1)  # (B, 2C)
        
        # 2. brightness gain
        dgain = self.gain_mlp(stats) + 1.0
        raw_adj = (raw * dgain.view(B, C, 1, 1)).clamp(1e-6,1)
        raw_attn = self.attn(raw_adj).clamp(1e-6,1)

        # 4. mask logits and soft masks
        logits = self.mask_head(raw_attn)
        # masks = F.softmax(logits, dim=1)
        masks = F.gumbel_softmax(logits, dim=1, tau=self.tau)

        # 5. compute channel weights in [w_min, w_max]
        logits_pool = logits.view(B, self.num_masks, -1).mean(dim=-1)
        w_logit = self.weight_mlp(logits_pool)           
        w_norm = torch.sigmoid(w_logit)            
        w = self.w_min + (self.w_max - self.w_min) * w_norm
        w = w.view(B, self.num_masks, 1, 1, 1)
        # print(w)

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
    params /= 1e3
    flops /= 1e9
    info = "Params: {:.2f}K, Gflops: {:.2f}".format(params, flops)
    print(info)
