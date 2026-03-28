import random
import torch
import torch.nn as nn
from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
# 2023-04-19 Modified by Huawei, import adaptive adjustment module
from .taisp import MultipleMaskISP_Conv
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2


class YOLOX(nn.Module):
    def __init__(self, backbone=None, head=None, nf=16, 
                 gamma_range=[1.0,4.0]):
        super().__init__()
        # self.TAISP = MultipleMaskISP_Conv(weight_range=[3.0,6.0])
        self.TAISP = MultipleMaskISP_Conv(weight_range=[5.0,9.0])
        
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)
        self.backbone = backbone
        self.head = head
        
    def forward(self, x, targets=None, return_xtm=False):
        x_tm_01 = self.TAISP(x)
        x_tm = torch.clamp(x_tm_01, 1e-6, 1) * 255.0
        if return_xtm:
            return torch.round(x_tm)

        fpn_outs = self.backbone(x_tm)

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(fpn_outs, targets, x)
            outputs = {
                "total_loss": loss, "iou_loss": iou_loss, "l1_loss": l1_loss, "conf_loss": conf_loss, "cls_loss": cls_loss,
                "num_fg": num_fg
            }
        else:
            outputs = self.head(fpn_outs)
        
        return outputs
