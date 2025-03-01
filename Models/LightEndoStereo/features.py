from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from Models.LightEndoStereo.submodule import *
import math

class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Mobilenetv4Feature_depth(SubModule):
    def __init__(self):
        super(Mobilenetv4Feature_depth, self).__init__()
        # model = timm.create_model('mobilenetv2_100', pretrained=pretrained, features_only=True)
        model = timm.create_model("mobilenetv4_conv_medium", pretrained=False)
        model.load_state_dict(torch.load("checkpoints/mobilenetv4_conv_medium.bin"))
        layers = [1,2,3,4,5]
        self.in_norm = nn.InstanceNorm2d(1)
        self.conv_stem = model.conv_stem
        self.block0 = torch.nn.Sequential(*model.blocks[0:layers[0]])
        self.block1 = torch.nn.Sequential(*model.blocks[layers[0]:layers[1]])
        self.block2 = torch.nn.Sequential(*model.blocks[layers[1]:layers[2]])
        self.block3 = torch.nn.Sequential(*model.blocks[layers[2]:layers[3]]) 
        self.weight_init()
    
    def forward(self, x):
        x2 = self.conv_stem(self.in_norm(x)) # B,32,H,W
        x4 = self.block0(x2) # B,48,H/2,W/2
        x8 = self.block1(x2) # B,80,H/4,W/4
        x16 = self.block2(x4) # B,160,H/8,W/8
        x32 = self.block3(x8)
        return [x4,x8,x16,x32]

class UnetUpConv2x(nn.Module):
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.doubleConv = nn.Sequential(
            nn.Conv2d(2*out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,up_feature, skip_feature):
        up_feature = self.upconv(up_feature)
        up_feature = torch.cat([up_feature, skip_feature], dim=1)
        up_feature = self.doubleConv(up_feature)
        return up_feature

class Mobilenetv4Feature(SubModule):
    def __init__(self, out_channels=0):
        super(Mobilenetv4Feature, self).__init__()
        self.out_channels = out_channels
        # model = timm.create_model('mobilenetv2_100', pretrained=pretrained, features_only=True)
        model = timm.create_model("mobilenetv4_conv_medium", pretrained=False)
        model.load_state_dict(torch.load("checkpoints/mobilenetv4_conv_medium.bin"))
        layers = [1,2,3,4,5]
        chans = [32,48,80,160,256]
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.block0 = torch.nn.Sequential(*model.blocks[0:layers[0]])
        self.block1 = torch.nn.Sequential(*model.blocks[layers[0]:layers[1]])
        self.block2 = torch.nn.Sequential(*model.blocks[layers[1]:layers[2]])
        self.block3 = torch.nn.Sequential(*model.blocks[layers[2]:layers[3]])
        # self.block4 = torch.nn.Sequential(*model.blocks[layers[3]:layers[4]])
        self.deconv32_16 = Conv2x_IN(chans[4], chans[3], deconv=True, concat=True)
        self.deconv16_8 = Conv2x_IN(chans[3]*2, chans[2], deconv=True, concat=True)
        self.deconv8_4 = Conv2x_IN(chans[2]*2, chans[1], deconv=True, concat=True)

        # self.deconv32_16 = UnetUpConv2x(chans[4], chans[3])
        # self.deconv16_8 = UnetUpConv2x(chans[3], chans[2])
        # self.deconv8_4 = UnetUpConv2x(chans[2], chans[1])
        if out_channels > 0:
            self.out_proj = nn.Conv2d(2*chans[1], self.out_channels, kernel_size=3, stride=1, padding=1)
        # self.weight_init()
        # self.deconv4_2 = Conv2x_IN(chans[1]*2, chans[0], deconv=True, concat=True)
        # self.deconv2_0 = nn.ConvTranspose2d(chans[0]*2, chans[0], kernel_size=4, stride=2, padding=1)
        # self.conv4 = BasicConv_IN(chans[1]*2, 320, kernel_size=3, stride=1, padding=1)
        # self.conv2 = BasicConv_IN(chans[1]*2, chans[0], kernel_size=3, stride=1, padding=1)
    
    
    def forward(self, x, refine_feature=False):
        x2 = self.bn1(self.conv_stem(x)) # B,32,H/2,W/2
        x4 = self.block0(x2) # B,48,H/4,W/4
        x8 = self.block1(x4) # B,80,H/4,W/4
        
        x16 = self.block2(x8) # B,160,H/16,W/16
        x32 = self.block3(x16) # B,256,H/32,W/32

        x16_up = self.deconv32_16(x32, x16) # B,320,H/16,W/16
        x8_up = self.deconv16_8(x16_up, x8) # B,160,H/8,W/8
        x4_up = self.deconv8_4(x8_up, x4) # B,96,H/4,W/4
        if self.out_channels > 0:
            gwc_feature = self.out_proj(x4_up) # B,out_channels,H/4,W/4
        else:
            gwc_feature = x4_up # B,out_channels,H/4,W/4
        # x2 = self.deconv4_2(x4, x2) # B,48,H/2,W/2
        # x0 = self.deconv2_0(x2) # B,24,H,W
        if refine_feature:
            return {"gwc_feature": gwc_feature, "refine_feature": x4_up}
        return {'gwc_feature': gwc_feature, }
 

class gwc_feature(nn.Module):
    def __init__(self, concat_feature=False, concat_feature_channel=12):
        super(gwc_feature, self).__init__()
        self.concat_feature = concat_feature
        self.inplanes = 32
        self.out_channels = 320
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)
        if self.concat_feature:
            self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                    bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        gwc_feature = torch.cat((l2, l3, l4), dim=1) # shape: B, 320, H/4, W/4

        if not self.concat_feature:
            return {"gwc_feature": gwc_feature}
        else:
            concat_feature = self.lastconv(gwc_feature)
            return {"gwc_feature": gwc_feature, "concat_feature": concat_feature}
