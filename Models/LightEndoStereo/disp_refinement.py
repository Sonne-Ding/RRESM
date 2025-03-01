from torch import nn
from .features import Mobilenetv4Feature_depth
import torch
from torch.nn import functional as F
from .submodule import CBAM

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

def context_upsample(disp_low, up_weights, scale_factor=4):
    # disp_low [b,1,h,w]
    # up_weights [b,9,4*h,4*w]
    b, c, h, w = disp_low.shape
    disp_unfold = F.unfold(disp_low, kernel_size=3, dilation=1, padding=1)  # [bz, 3x3, hxw]
    disp_unfold = disp_unfold.reshape(b, -1, h, w)  # [bz, 3x3, h, w]
    disp_unfold = F.interpolate(disp_unfold, (h * scale_factor, w * scale_factor), mode='nearest')  # [bz, 3x3, 4h, 4w]
    disp = (disp_unfold - up_weights).sum(1)  # # [bz, 4h, 4w]

    return disp

class LDE(nn.Module):
    def __init__(self, in_filter, num_filter):
        super(LDE, self).__init__()
        self.conv1 = ConvBlock(in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.conv2 = ConvBlock(num_filter, num_filter , 1, 1, 0, activation='prelu', norm=None)
        # self.conv11 = ConvBlock(in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        # self.conv22 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.conv_ch1 = ConvBlock(num_filter, num_filter // 2, 1, 1, 0, activation=None, bias=False, norm=None)
        self.conv_ch2 = ConvBlock(num_filter, num_filter // 2, 1, 1, 0, activation=None, bias=False, norm=None)
        self.conv_ch3 = ConvBlock(num_filter, num_filter // 2, 1, 1, 0, activation=None, bias=False, norm=None)
        self.conv_ch4 = ConvBlock(num_filter // 2, num_filter, 1, 1, 0, activation=None, bias=False, norm=None)
        self.softmax = torch.nn.Softmax(-1)

    def forward(self, rgb, depth):
        b, c, h, w = depth.shape
        c0 = self.conv2(self.conv1(rgb))
        c1 = self.conv_ch1(c0).view(b, c//2, -1)  # B * C/2 * (H * W)
        c1 = c1.permute(0, 2, 1)  # B * (H * W) * C/2

        d0 = self.conv2(self.conv1(depth))
        d1 = self.conv_ch2(d0).view(b, c//2, -1)  # B * C/2 * (H * W)
        d1 = d1.permute(0, 2, 1)  # B * (H * W) * C/2
        d2 = self.conv_ch3(d0).view(b, c//2, -1)  # B * C/2 * (H * W)

        self_map = self.softmax(torch.matmul(d1, d2))  # B * (H * W) * (H * W)
        guided = torch.matmul(self_map, c1)  # B * (H * W) * C/2
        guided = guided.permute(0, 2, 1).contiguous().view(b, c//2, h, w)
        guided = self.conv_ch4(guided)

        out = guided + d0
        return out

class simple_refine(nn.Module):
    def __init__(self,rgb_in_features) -> None:
        super().__init__()
        self.cbam = CBAM(rgb_in_features)
    
    def forward(self, rgb_f, disp_low):
        """
            :param rgb: B,C1,H,W
            :param disp_low: B,1,H,W
            :return: refined disparity
        """
        # fused_feature = self.rgb_inproj(rgb_f)+self.depth_inproj(depth_f)
        att = self.cbam(rgb_f) # B,1,H,W

        