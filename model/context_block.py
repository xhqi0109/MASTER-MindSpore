# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 10/4/2020 14:17

import torch
from torch import nn
import mindspore as ms
from mindspore import nn as mnn
class MultiAspectGCAttention(mnn.Cell):

    def __init__(self,
                 inplanes,
                 ratio,
                 headers,
                 pooling_type='att',
                 att_scale=False,
                 fusion_type='channel_add'):
        super(MultiAspectGCAttention, self).__init__()
        assert pooling_type in ['avg', 'att']

        assert fusion_type in ['channel_add', 'channel_mul', 'channel_concat']
        assert inplanes % headers == 0 and inplanes >= 8  # inplanes must be divided by headers evenly

        self.headers = headers
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_type = fusion_type
        self.att_scale = False

        self.single_header_inplanes = int(inplanes / headers)
        # # 卷积层，输入的通道数为6，输出的通道数为16，卷积核大小为5*5
        # self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        if pooling_type == 'att':
            # self.conv_mask = nn.Conv2d(self.single_header_inplanes, 1, kernel_size=1)
            # self.softmax = nn.Softmax(dim=2)

            self.conv_mask = mnn.Conv2d(self.single_header_inplanes, 1, kernel_size=1)
            self.softmax = mnn.Softmax(axis=2)
        else:
            # self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.avg_pool = ms.ops.AdaptiveAvgPool2d(1)

        if fusion_type == 'channel_add':
            # self.channel_add_conv = nn.Sequential(
            #     nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
            #     nn.LayerNorm([self.planes, 1, 1]),
            #     nn.ReLU(inplace=True),
            #     nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
            # self.channel_add_conv = mnn.SequentialCell([
            #     mnn.Conv2d(self.inplanes, self.planes, kernel_size=1),
            #     mnn.LayerNorm([1,self.planes, 1, 1],begin_norm_axis=1, begin_params_axis=1),
            #     mnn.ReLU(),
            #     mnn.Conv2d(self.planes, self.inplanes, kernel_size=1)])
            self.convtt = mnn.Conv2d(self.inplanes, self.planes, kernel_size=1),
            self.lntt = mnn.LayerNorm([1,self.planes, 1, 1],begin_norm_axis=1, begin_params_axis=1),
     
            self.channel_add_conv = mnn.SequentialCell([
                # mnn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                # mnn.LayerNorm([1,self.planes, 1, 1],begin_norm_axis=1, begin_params_axis=1),
                mnn.ReLU(),
                mnn.Conv2d(self.planes, self.inplanes, kernel_size=1)])

        elif fusion_type == 'channel_concat':
            # self.channel_concat_conv = nn.Sequential(
            #     nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
            #     nn.LayerNorm([self.planes, 1, 1]),
            #     nn.ReLU(inplace=True),
            #     nn.Conv2d(self.planes, self.inplanes, kernel_size=1))

            # self.channel_concat_conv = mnn.SequentialCell(
            #     mnn.Conv2d(self.inplanes, self.planes, kernel_size=1),
            #     mnn.LayerNorm([self.planes, 1,1]),
            #     mnn.ReLU(),      
            #     mnn.Conv2d(self.planes, self.inplanes, kernel_size=1))
            # self.convtt = mnn.Conv2d(self.inplanes, self.planes, kernel_size=1)
            # self.lntt = mnn.LayerNorm([self.planes, 1, 1],begin_norm_axis=1,begin_params_axis=1)
     
            self.channel_concat_conv = mnn.SequentialCell([
                mnn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                mnn.LayerNorm([self.planes, 1, 1],begin_norm_axis=1, begin_params_axis=1),
                mnn.ReLU(),
                mnn.Conv2d(self.planes, self.inplanes, kernel_size=1)])

            # for concat
            # self.cat_conv = nn.Conv2d(2 * self.inplanes, self.inplanes, kernel_size=1)
            self.cat_conv = mnn.Conv2d(2 * self.inplanes, self.inplanes, kernel_size=1)
        elif fusion_type == 'channel_mul':
            # self.channel_mul_conv = nn.Sequential(
            #     nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
            #     nn.LayerNorm([self.planes, 1, 1]),
            #     nn.ReLU(inplace=True),
            #     nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
            self.channel_mul_conv = mnn.SequentialCell(
                mnn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                mnn.LayerNorm([self.planes, 1, 1]),
                mnn.ReLU(),
                mnn.Conv2d(self.planes, self.inplanes, kernel_size=1))



    def spatial_pool(self, x):
        # print("data/qxh22/code/MASTER-pytorch-main/model/context_block1.py:",x.size())
        batch, channel, height, width = x.shape
        if self.pooling_type == 'att':
            # [N*headers, C', H , W] C = headers * C'
            x = x.view(batch * self.headers, self.single_header_inplanes, height, width)
            input_x = x

            # [N*headers, C', H * W] C = headers * C'
            # input_x = input_x.view(batch, channel, height * width)
            input_x = input_x.view(batch * self.headers, self.single_header_inplanes, height * width)

            # [N*headers, 1, C', H * W]
            # input_x = input_x.unsqueeze(1)
            expand = ms.ops.ExpandDims()
            input_x = expand(input_x,1)
            # [N*headers, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N*headers, 1, H * W]
            context_mask = context_mask.view(batch * self.headers, 1, height * width)

            # scale variance
            if self.att_scale and self.headers > 1:
                context_mask = context_mask /ms.ops.Sqrt(self.single_header_inplanes)

            # [N*headers, 1, H * W]
            context_mask = self.softmax(context_mask)

            # [N*headers, 1, H * W, 1]
            # context_mask = context_mask.unsqueeze(-1)
            context_mask = expand(context_mask,-1)


            # [N*headers, 1, C', 1] = [N*headers, 1, C', H * W] * [N*headers, 1, H * W, 1]
            # context = torch.matmul(input_x, context_mask)
            context = ms.ops.matmul(input_x, context_mask)
            # [N, headers * C', 1, 1]
            context = context.view(batch, self.headers * self.single_header_inplanes, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def construct(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x

        if self.fusion_type == 'channel_mul':
            # [N, C, 1, 1]
            # channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            channel_mul_term = mindspore.ops.Sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        elif self.fusion_type == 'channel_add':
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        else:
            # [N, C, 1, 1]
            # print("1111111111")
            # context = self.convtt(context)
            # print("22222222:",context.shape)
            # context = self.lntt(context)
            # print("333333333")
            channel_concat_term = self.channel_concat_conv(context)
            # use concat
            _, C1, _, _ = channel_concat_term.shape
            N, C2, H, W = out.shape
            # print("out.shape:",out.shape,"channel_concat_term:",channel_concat_term.shape)
            channel_concat_term = channel_concat_term.broadcast_to((-1, -1, H, W))
            concat_op = ms.ops.Concat(axis=1)
            # cast_op = ms.ops.Cast()
            # output = concat_op((cast_op(out),cast_op( channel_concat_term)))
            out = concat_op([out,channel_concat_term])

            out = self.cat_conv(out)
            # out = nn.functional.layer_norm(out, [self.inplanes, H, W])
            
            # out = ms.ops.LayerNorm(out, , begin_params_axis=1)
            self.lntt = mnn.LayerNorm([self.inplanes, H, W],begin_norm_axis=1, begin_params_axis=1)
            out = self.lntt(out)
            print("==========:",out.shape)
            relu = ms.ops.ReLU()
            out = relu(out)
        return out


