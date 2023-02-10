# -*- coding: utf-8 -*-

import mindspore.nn as mnn
import mindspore as ms
from model.context_block import MultiAspectGCAttention


# CNN for Feature Extraction + Multi-Aspect GCAttention

def conv3x3(in_planes, out_planes, stride=1):
    # "3x3 convolution with padding"
    return mnn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return mnn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride)


class BasicBlock(mnn.Cell):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_gcb=False, gcb_config=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = nn.BatchNorm2d(planes, momentum=0.9)
        #与pytorch中的momentum是1-momentum的关系
        self.bn1 = mnn.BatchNorm2d(planes, momentum=0.1)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = mnn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes, momentum=0.9)
        self.bn2 = mnn.BatchNorm2d(planes, momentum=0.1)
        self.downsample = downsample
        self.stride = stride
        self.use_gcb = use_gcb

        if self.use_gcb:
            gcb_ratio = gcb_config['ratio']
            gcb_headers = gcb_config['headers']
            att_scale = gcb_config['att_scale']
            fusion_type = gcb_config['fusion_type']
            self.context_block = MultiAspectGCAttention(inplanes=planes,
                                                        ratio=gcb_ratio,
                                                        headers=gcb_headers,
                                                        att_scale=att_scale,
                                                        fusion_type=fusion_type)

    def construct(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_gcb:
            out = self.context_block(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(mnn.Cell):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = mnn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = mnn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = mnn.BatchNorm2d(planes * self.expansion)
        self.relu = mnn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(mnn.Cell):

    def __init__(self, block, layers, zero_init_residual=False, gcb=None, in_channels=1):
        super(ResNet, self).__init__()
        gcb_config = gcb

        self.inplanes = 128
        self.conv1 = mnn.Conv2d(in_channels, 64, kernel_size=3, stride=1, has_bias=False)
        self.bn1 = mnn.BatchNorm2d(64)
        self.relu1 = mnn.ReLU()

        self.conv2 = mnn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.bn2 = mnn.BatchNorm2d(128)
        self.relu2 = mnn.ReLU()
        self.maxpool1 = mnn.MaxPool2d(kernel_size=2, stride=2)

        self.layer1 = self._make_layer(block, 256, layers[0], stride=1, gcb_config=gcb_config,
                                       use_gcb=gcb_config['layers'][0])

        self.conv3 = mnn.Conv2d(256, 256, kernel_size=3, stride=1, has_bias=False)
        self.bn3 = mnn.BatchNorm2d(256)
        self.relu3 = mnn.ReLU()

        self.maxpool2 = mnn.MaxPool2d(kernel_size=2, stride=2)

        self.layer2 = self._make_layer(block, 256, layers[1], stride=1, gcb_config=gcb_config,
                                       use_gcb=gcb_config['layers'][1])

        self.conv4 = mnn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn4 = mnn.BatchNorm2d(256)
        self.relu4 = mnn.ReLU()


        self.maxpool3 = mnn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.layer3 = self._make_layer(block, 512, layers[2], stride=1, gcb_config=gcb_config,
                                       use_gcb=gcb_config['layers'][2])

        self.conv5 = mnn.Conv2d(512, 512, kernel_size=3, stride=1)
        self.bn5 = mnn.BatchNorm2d(512)
        self.relu5 = mnn.ReLU()

        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, gcb_config=gcb_config,
                                       use_gcb=gcb_config['layers'][3])

        self.conv6 = mnn.Conv2d(512, 512, kernel_size=3, stride=1)
        self.bn6 = mnn.BatchNorm2d(512)
        self.relu6 = mnn.ReLU()

        for m in self.cells():
            if isinstance(m, mnn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                ms.common.initializer.HeNormal(m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, mnn.BatchNorm2d):
            #     # nn.init.constant_(m.weight, 1)
            #     ########################################################
            #     #在mindspore中输入为numpy，但是m.weight可能为tensor。
            #     constant_init = ms.common.initializer.Constant(value=1)
            #     constant_init(m.weight)

                # nn.init.constant_(m.bias, 0)
                ########################################################
             

        if zero_init_residual:
            for m in self.cells():
                if isinstance(m, Bottleneck):
                    # nn.init.constant_(m.bn3.weight, 0)
                    constant_init = ms.common.initializer.Constant(value=0)
                    constant_init(m.bias)

                elif isinstance(m, BasicBlock):
                    # nn.init.constant_(m.bn2.weight, 0)
                    constant_init = ms.common.initializer.Constant(value=0)
                    constant_init(m.bias)


    def _make_layer(self, block, planes, blocks, stride=1, use_gcb=False, gcb_config=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = mnn.SequentialCell(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                mnn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_gcb=use_gcb, gcb_config=gcb_config))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return mnn.SequentialCell(layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool1(x)

        x = self.layer1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.maxpool2(x)
        x = self.layer2(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.maxpool3(x)
        x = self.layer3(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.layer4(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        return x


def resnet50(gcb_kwargs, in_channels=1):
    model = ResNet(BasicBlock, [1, 2, 5, 3], gcb=gcb_kwargs, in_channels=in_channels)
    return model


class ConvEmbeddingGC(mnn.Cell):

    def __init__(self, gcb_kwargs, in_channels=1):
        super().__init__()
        ###28个
        self.backbone = resnet50(gcb_kwargs, in_channels=in_channels)

    def construct(self, x):
        feature = self.backbone(x)
        b, c, h, w = feature.shape  # （B， C， H/8, W/4）
        feature = feature.view(b, c, h * w)
        # feature = feature.permute((0, 2, 1))
        feature = feature.transpose((0, 2, 1))
        return feature
