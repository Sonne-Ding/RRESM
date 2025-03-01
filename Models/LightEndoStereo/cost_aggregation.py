from torch import nn
from torch.nn import functional as F
from .submodule import convbn_3d, coordinate_attention_linear, coordinate_attention_mamba
import torch
class MCAHG(nn.Module):
    """
        Mamba Coordinate Attention Hourglass Aggregation
    """
    def __init__(self, in_channels,linear=False):
        super(MCAHG, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))
        
        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))
        if linear:
            self.coordAtten5 = coordinate_attention_linear(in_channels*2)
            self.coordAtten6 = coordinate_attention_linear(in_channels)
        else:
            self.coordAtten5 = coordinate_attention_mamba(in_channels*2)
            self.coordAtten6 = coordinate_attention_mamba(in_channels)
        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x) # 1/2
        conv2 = self.conv2(conv1) # 1/2

        conv3 = self.conv3(conv2) # 1/4
        conv4 = self.conv4(conv3) # 1/4
        conv5_4 = self.conv5(conv4)
        redir2 = self.redir2(conv2)
        conv5 = F.relu(self.coordAtten5(conv5_4 + redir2), inplace=True) # 1/2
        conv6_5 = self.conv6(conv5)
        redir1 = self.redir1(x)
        conv6 = F.relu(self.coordAtten6(conv6_5 + redir1), inplace=True) # 1/1
        return conv6

class MCAHG2(nn.Module):
    def __init__(self, in_channels, mca=True):
        super().__init__()
        self.mca = mca
        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))
        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))
        if self.mca:
            self.coordAtten5 = coordinate_attention_mamba(in_channels*2)
            self.coordAtten6 = coordinate_attention_mamba(in_channels)
        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x) # 1/2
        conv2 = self.conv2(conv1) # 1/2
        conv3 = self.conv3(conv2) # 1/4
        conv4 = self.conv4(conv3) # 1/4
        if self.mca:
            upconv2 = self.coordAtten5(self.redir2(conv2))
            upx = self.coordAtten6(self.redir1(x))
            conv5 = F.relu(self.conv5(conv4) + upconv2, inplace=True) # 1/2
            conv6 = F.relu(self.conv6(conv5) + upx, inplace=True) # 1/1
            # conv5 = F.relu(self.conv5(conv4) + self.coordAtten5(self.redir2(conv2)), inplace=True) # 1/2
            # conv6 = F.relu(self.conv6(conv5) + self.coordAtten6(self.redir1(x)), inplace=True) # 1/1
        else:
            conv5 = F.relu(self.conv5(conv4) +self.redir2(conv2), inplace=True) # 1/2
            conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True) # 1/1
        return conv6



class HG(nn.Module):
    def __init__(self, in_channels):
        super(HG, self).__init__()
        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))
        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))
        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x) # 1/2
        conv2 = self.conv2(conv1) # 1/2
        conv3 = self.conv3(conv2) # 1/4
        conv4 = self.conv4(conv3) # 1/4
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True) # 1/2
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True) # 1/1
        return conv6


if __name__=="__main__":
    pass