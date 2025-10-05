"""
DDRNet (Deep Dual-Resolution Network) implementation for semantic segmentation.
Student model in the knowledge distillation framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic residual block for DDRNet"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)


class Bottleneck(nn.Module):
    """Bottleneck block for deeper DDRNet variants"""
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)


class DAPPM(nn.Module):
    """Deep Aggregation Pyramid Pooling Module"""
    def __init__(self, inplanes, branch_planes, outplanes):
        super(DAPPM, self).__init__()
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale0 = nn.Sequential(
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.process1 = nn.Sequential(
            nn.BatchNorm2d(branch_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process2 = nn.Sequential(
            nn.BatchNorm2d(branch_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process3 = nn.Sequential(
            nn.BatchNorm2d(branch_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process4 = nn.Sequential(
            nn.BatchNorm2d(branch_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )        
        self.compression = nn.Sequential(
            nn.BatchNorm2d(branch_planes * 5),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]        
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                        size=[height, width],
                        mode='bilinear', align_corners=True) + x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                        size=[height, width],
                        mode='bilinear', align_corners=True) + x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                        size=[height, width],
                        mode='bilinear', align_corners=True) + x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                        size=[height, width],
                        mode='bilinear', align_corners=True) + x_list[3])))
       
        out = self.compression(torch.cat(x_list, dim=1)) + self.shortcut(x)
        return out 


class DDRNet(nn.Module):
    """
    Deep Dual-Resolution Network for semantic segmentation.
    
    Args:
        block: BasicBlock or Bottleneck
        layers: list of layer numbers for each stage
        num_classes: number of segmentation classes
        planes: base channel number (default: 64)
        spp_planes: channels for SPP module (default: 128)
        head_planes: channels for segmentation head (default: 128)
        augment: whether to use auxiliary head for training
    """
    
    def __init__(self, block, layers, num_classes=19, planes=64, spp_planes=128, 
                 head_planes=128, augment=False):
        super(DDRNet, self).__init__()
        
        highres_planes = planes * 2
        self.augment = augment
        
        # Stem
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )

        # Low-resolution branch
        self.layer1 = self._make_layer(block, planes, planes, layers[0])
        self.layer2 = self._make_layer(block, planes, planes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, planes * 2, planes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, planes * 4, planes * 8, layers[3], stride=2)

        # High-resolution branch
        self.compression3 = nn.Sequential(
            nn.Conv2d(planes * 4, highres_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(highres_planes),
        )
        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 8, highres_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(highres_planes),
        )
        
        self.layer3_ = self._make_layer(block, highres_planes, highres_planes, 2)
        self.layer4_ = self._make_layer(block, highres_planes, highres_planes, 2)
        self.layer5_ = self._make_layer(Bottleneck, highres_planes, highres_planes, 1)
        
        # Bilateral fusion
        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 8, 1, stride=2)
        self.spp = DAPPM(planes * 16, spp_planes, planes * 4)
        
        # Final prediction heads
        self.final_layer = nn.Sequential(
            nn.Conv2d(planes * 4, head_planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(head_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_planes, num_classes, kernel_size=1, bias=True)
        )
        
        if self.augment:
            # Auxiliary head for deep supervision
            self.seghead_extra = nn.Sequential(
                nn.Conv2d(highres_planes, head_planes, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(head_planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_planes, num_classes, kernel_size=1, bias=True)
            )

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks-1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8
        
        x = self.conv1(x)
        x = self.layer1(x)
        x = F.relu(self.layer2(F.relu(x)))
        
        x_ = self.layer3_(x)
        x_d = self.layer3(x)
        
        # Compression with spatial matching
        x_d_compressed = self.compression3(x_d)
        if x_.shape[-2:] != x_d_compressed.shape[-2:]:
            x_d_compressed = F.interpolate(x_d_compressed, size=x_.shape[-2:],
                                          mode='bilinear', align_corners=False)
        x = self.layer4_(F.relu(x_ + x_d_compressed))
        x_d = self.layer4(F.relu(x_d))
        
        x_d_compressed = self.compression4(x_d)
        if x.shape[-2:] != x_d_compressed.shape[-2:]:
            x_d_compressed = F.interpolate(x_d_compressed, size=x.shape[-2:],
                                          mode='bilinear', align_corners=False)
        x_ = self.layer5_(F.relu(x + x_d_compressed))
        x_d = F.relu(self.layer5(F.relu(x_d)))
        
        x = self.spp(x_d)
        x = self.final_layer(x)
        
        x = F.interpolate(x, size=[height_output, width_output], 
                         mode='bilinear', align_corners=True)
        
        if self.augment:
            x_ = self.seghead_extra(x_)
            x_ = F.interpolate(x_, size=[height_output, width_output],
                             mode='bilinear', align_corners=True)
            return [x, x_]
        else:
            return x


def DDRNet23(num_classes=19, pretrained=False, **kwargs):
    """
    DDRNet-23 model
    """
    model = DDRNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)
    return model


def DDRNet23Slim(num_classes=19, pretrained=False, **kwargs):
    """
    DDRNet-23-Slim model with reduced channels
    """
    model = DDRNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes,
                   planes=32, spp_planes=64, head_planes=64, **kwargs)
    return model


def DDRNet39(num_classes=19, pretrained=False, **kwargs):
    """
    DDRNet-39 model
    """
    model = DDRNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)
    return model
