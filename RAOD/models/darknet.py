#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import torch
from torch import nn
import time
from .network_blocks import BaseConv, CSPLayer, DWConv, Focus, ResLayer, SPPBottleneck
from .network_blocks import BaseConvStem
# from network_blocks import BaseConv, CSPLayer, DWConv, Focus, ResLayer, SPPBottleneck
# from network_blocks import BaseConvStem


class Darknet(nn.Module):
    # number of blocks from dark2 to dark5.
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_out_channels=32,
        out_features=("dark3", "dark4", "dark5"),
    ):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output chanels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        self.stem = nn.Sequential(
            BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu"),
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2),
        )
        in_channels = stem_out_channels * 2  # 64

        num_blocks = Darknet.depth2blocks[depth]
        # create darknet with `stem_out_channels` and `num_blocks` layers.
        # to make model structure more clear, we don't use `for` statement in python.
        self.dark2 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[0], stride=2)
        )
        in_channels *= 2  # 128
        self.dark3 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[1], stride=2)
        )
        in_channels *= 2  # 256
        self.dark4 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[2], stride=2)
        )
        in_channels *= 2  # 512

        self.dark5 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),
            *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
        )

    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu"),
            *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)],
        ]

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation="lrelu",
                ),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
            ]
        )
        return m

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


class CSPDarknet(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        # self.stem = Focus(3, base_channels, ksize=3, act=act)
        self.stem = BaseConvStem(3, base_channels, ksize=4, stride=4, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        # print(x.shape)
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}
    
class CSPDarknet_adapter(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
        light_mode='low',
        w_lut=True,   # with or without 3DLUT
        lut_dim=32,
        k_size=3,  # LOD dataset, 9
        fea_c_s = [24, 48, 96],
        ada_c_s = [24, 48, 96],
        mid_c_s = [32, 32, 64],
        merge_ratio=1.0,
    ):
        super().__init__()
        # fea_c_s = [24, 48, 96], yolox-s
        # ada_c_s = [24, 48, 96],
        # mid_c_s = [32, 32, 64],
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        # print(base_channels)
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # RAW-Adapter
        self.w_lut = w_lut
        self.pre_encoder = Input_level_Adapeter(mode=light_mode, lut_dim=lut_dim, k_size=k_size, w_lut=self.w_lut)
        self.model_adapter = Model_level_Adapeter(in_c=3, in_dim=ada_c_s[0], w_lut=self.w_lut)
        # print(wid_mul*fea_c_s[0])
        # print(wid_mul)
        # print(fea_c_s[0])
        self.merge_1 = Merge_block(fea_c=fea_c_s[0], ada_c=ada_c_s[0], mid_c=mid_c_s[0], return_ada=True)
        self.merge_2 = Merge_block(fea_c=fea_c_s[1], ada_c=ada_c_s[1], mid_c=mid_c_s[1], return_ada=True)
        self.merge_3 = Merge_block(fea_c=fea_c_s[2], ada_c=ada_c_s[2], mid_c=mid_c_s[2], return_ada=False)
        self.merge_blocks = [self.merge_1, self.merge_2, self.merge_3]
        self.merge_ratio = merge_ratio
        # stem
        # self.stem = Focus(3, base_channels, ksize=3, act=act)
        self.stem = BaseConvStem(3, base_channels, ksize=4, stride=4, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark3 256
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark4 512
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark5 1024
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )

    def forward(self, x):
        x = self.pre_encoder(x) # Input-level Adapter
        # x = [(torch.clamp(item,1e-6,1) * 255.0) for item in x]
        if self.w_lut:  # I1, I2, I3, I4
            ada = self.model_adapter([x[0], x[1], x[2], x[3]])
        else:   # I1, I2, I3
            ada = self.model_adapter([x[0], x[1], x[2]])
        
        x = x[-1]  
        outputs = {}
        x = self.stem(x) # 24
        # print(x.shape)
        # print(ada.shape)
        x, ada = self.merge_blocks[0](x, ada, ratio=self.merge_ratio)
        outputs["stem"] = x
        x = self.dark2(x)
        # print(x.shape) #48
        # print(ada.shape)
        x, ada = self.merge_blocks[1](x, ada, ratio=self.merge_ratio)
        outputs["dark2"] = x
        x = self.dark3(x)
        # print(x.shape) #96
        # print(ada.shape)
        x, ada = self.merge_blocks[2](x, ada, ratio=self.merge_ratio)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}

if __name__ == '__main__':
    import torch
    from thop import profile
    # 创建输入张量
    stride = 640
    input_tensor = torch.zeros((1, 3, stride, stride))

    # 创建模型
    model = CSPDarknet(0.33, 0.25)
    # model = CSPDarknet_adapter(0.33, 0.25)
    model.eval()

    # 移动到 CUDA（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    params /= 1e6
    flops /= 1e9
    info = "Params: {:.2f}M, Gflops: {:.2f}".format(params, flops)
    print(info)

    # 计算输入数据量 (3 channels, 640x640, 每个像素1个字节)
    input_size = input_tensor.nelement() * input_tensor.element_size()  # 总字节数

    # 开始计时
    start_time = time.time()

    # 进行推理
    with torch.no_grad():
        output = model(input_tensor)

    # 结束计时
    total_time = time.time() - start_time

    output_size = 0
    # print(output)
    for k in output:
        output_size += output[k].nelement() * output[k].element_size()
    
    # 计算总带宽
    total_data = input_size + output_size
    bandwidth = total_data / total_time

    print(f"输入大小: {input_size / 1e6:.2f} MB")
    print(f"输出大小: {output_size / 1e6:.2f} MB")
    print(f"推理时间: {total_time:.6f} 秒")
    print(f"总带宽: {bandwidth / 1e6:.2f} MB/s")
    # for k, v in out.items():
    #     print(k, v.shape)
