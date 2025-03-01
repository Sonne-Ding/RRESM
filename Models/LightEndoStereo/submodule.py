"""
参考：
1. coordinate attention：https://github.com/houqb/CoordAttention/blob/main/coordatt.py
2. mamba
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from torch.autograd.function import Function
from mamba_ssm.ops.triton import ssd_combined
import torch.nn.functional as F
import numpy as np

from .nd_mamba import NdMamba2_1d
from mamba_ssm import Mamba2
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn

def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_channels))


def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False),
                         nn.BatchNorm3d(out_channels))


def disparity_compute_interp(x, maxdisp, shape):
    if x.dim()==4:
        x = x.unsqueeze(1)
    x = F.interpolate(x, [shape[0], shape[1], shape[2]], mode='trilinear') # x should be [b,1,d,h,w]
    x = x.squeeze(1)
    x = F.softmax(x, 1) # x.shape = [b, d, h, w]
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    disp = torch.sum(x * disp_values, 1, keepdim=False)
    return disp
    

def disparity_compute(x, maxdisp, scale=1):
    if x.dim()==5:
        x = x.squeeze(1)
    x = F.softmax(x, 1) # x.shape = [B, D, H, W]
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    disp = torch.sum(x * disp_values, 1, keepdim=False)
    if scale > 1:
        disp = scale*disp
        return disp
    else:
        return disp

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out

class BasicConv_IN(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, IN=True, relu=True, **kwargs):
        super(BasicConv_IN, self).__init__()
        self.relu = relu
        self.use_in = IN
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_in:
            x = self.IN(x)
        if self.relu:
            x = nn.LeakyReLU()(x)#, inplace=True)
        return x


class Conv2x_IN(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, IN=True, relu=True, keep_dispc=False):
        super(Conv2x_IN, self).__init__()
        self.concat = concat
        self.is_3d = is_3d 
        if deconv and is_3d: 
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3
        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv_IN(in_channels, out_channels, deconv, is_3d, IN=True, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv_IN(in_channels, out_channels, deconv, is_3d, IN=True, relu=True, kernel_size=kernel, stride=2, padding=1)
        if self.concat: 
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv_IN(out_channels*2, out_channels*mul, False, is_3d, IN, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv_IN(out_channels, out_channels, False, is_3d, IN, relu, kernel_size=3, stride=1, padding=1)
            
    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(
                x,
                size=(rem.shape[-2], rem.shape[-1]),
                mode='nearest')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else: 
            x = x + rem
        x = self.conv2(x)
        return x

class Mamba2Block(nn.Module):
    def __init__(
        self, in_channels,out_channels=None, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=True
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.norm = norm_cls(in_channels)
        self.mixer = Mamba2(in_channels, expand=8, headdim=int(in_channels*8/8))
        if out_channels is not None:
            self.norm2 = norm_cls(in_channels)
            self.mlp = nn.Linear(in_channels, out_channels)
        else:
            self.mlp = None
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states, residual = None, inference_params=None, **mixer_kwargs
    ):
        # 防止混合精度使用后出现的数值问题
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.to(torch.float32)
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm)
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)

        if self.mlp is not None:
            if not self.fused_add_norm:
                residual = hidden_states + residual
                hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.norm2.weight,
                    self.norm2.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm2.eps,
                    is_rms_norm=isinstance(self.norm2, RMSNorm)
                )
            hidden_states = self.mlp(hidden_states)
        # return hidden_states, residual
        return hidden_states

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)       

class AixsPosition(nn.Module):
    def __init__(self, feature_dim, proj_type="mamba") -> None:
        """
            :param scale: mamba_state_dim = 64*scale 
            :param proj_type: choice ["mamba", "linear"]
        """
        super().__init__()
        self.pooling = nn.AdaptiveMaxPool2d((1,1))
        # self.pooling = nn.AdaptiveAvgPool2d((1,1))
        if proj_type == "mamba":
            # self.proj = NdMamba2_1d(feature_dim, feature_dim, cmid=64)
            self.proj = Mamba2Block(feature_dim, feature_dim)
        elif proj_type == "linear":
            self.proj = nn.Linear(feature_dim, feature_dim)
        else:
            raise ValueError("Invalid proj_type")
        
        # self.mambaRegression = Mamba2Block(feature_dim, feature_dim)
        
    def forward(self, data):
        """
            data.shape = B, C, dim1, dim2, dim3
        """
        B, C, dim1, dim2, dim3 = data.shape
        data = data.contiguous()
        data = data.view(B, C*dim1, dim2, dim3) 
        data = self.pooling(data)# B, C*dim1, 1, 1
        data = data.view(B, C, dim1).transpose(1,2) # B, dim1, C
        dataReg = self.proj(data).transpose(1,2)
        return dataReg

class CoordAttention(nn.Module):
    def __init__(self, feature_in, scale=1, proj_type="linear") -> None:
        """
        
        :param proj_type: choice ['linear', 'mamba']
        """
        super().__init__()
        self.in_norm = nn.InstanceNorm3d(num_features=feature_in)
        axis_proj = "linear"
        self.PosAttenEncoder1 = AixsPosition(feature_in, axis_proj)
        self.PosAttenEncoder2 = AixsPosition(feature_in, axis_proj)
        self.PosAttenEncoder3 = AixsPosition(feature_in, axis_proj)
        self.mamba2_forward = Mamba2Block(feature_in, feature_in)
        self.mamba2_backward = Mamba2Block(feature_in, feature_in)
        # self.axis_corr = nn.Conv1d(feature_in,feature_in,3,1,1)
        self.nonlinear1 = nn.Sigmoid()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, data):
        """
            :param data: shape = B, C, dim1, dim2, dim3
        """
        usq = False
        if data.dim() == 4:
            data = data.unsqueeze(1)
            usq = True
        B, C, dim1, dim2, dim3 = data.shape
        data = self.in_norm(data)
        dataDim1 = data
        dataDim2 = data.permute(0,1,3,2,4) # B, C, dim2, dim1, dim3
        dataDim3 = data.permute(0,1,4,2,3) # B, C, dim3, dim1, dim2
        dim1CoordAtten = self.PosAttenEncoder1(dataDim1) # B, C, dim1
        dim2CoordAtten = self.PosAttenEncoder2(dataDim2) # B, C, dim2
        dim3CoordAtten = self.PosAttenEncoder3(dataDim3) # B, C, dim3
        ConcatAttention = torch.cat([dim1CoordAtten, dim2CoordAtten, dim3CoordAtten], dim=-1) # B, C, dim1+dim2+dim3
        CoordAttention = ConcatAttention.transpose(1,2) # B, dim1+dim2, C
        ca1 = self.mamba2_forward(CoordAttention)
        ca2 = self.mamba2_backward(CoordAttention.flip(1))
        ca = ca1 + ca2.flip(1)
        ca = ca.transpose(1,2)
        ca = self.nonlinear1(ca)
        # ConcatAttention = self.nonlinear1(self.mambaRegression(ConcatAttention))
        # ConcatAttention = self.nonlinear1(self.axis_corr(ConcatAttention))
 
        dim1CoordAtten, dim2CoordAtten, dim3CoordAtten = torch.split(ca, split_size_or_sections=[dim1, dim2, dim3], dim=-1)
        dim1CoordAtten = self.sigmoid(dim1CoordAtten).view(B,C,dim1,1,1)
        dim2CoordAtten = self.sigmoid(dim2CoordAtten).view(B,C,1,dim2,1)
        dim3CoordAtten = self.sigmoid(dim3CoordAtten).view(B,C,1,1,dim3)
        torch.save(dim1CoordAtten, "results/ablation11/dim1.pth")
        torch.save(dim2CoordAtten, "results/ablation11/dim2.pth")
        torch.save(dim3CoordAtten, "results/ablation11/dim3.pth")
        data = data*dim1CoordAtten*dim2CoordAtten*dim3CoordAtten
        # data = (data*dim1CoordAtten + data*dim2CoordAtten + data*dim3CoordAtten)/3
        if usq:
            data = data.squeeze(1)
        return data


def retrieve_corr_feature(corr_volume, disp):
    """
        disp: B, 1, H, W
        corr_volume: B, C, D, H, W
    """
    if disp.dim()==3:
        disp = disp.unsqueeze(1)
    B, C, D, H, W = corr_volume.shape
    disp_index = torch.round(disp).long().expand(B,C,1,H,W)
    retrieved_feature = torch.gather(corr_volume, 2, disp_index).squeeze(2) # B, C, H, W
    return retrieved_feature

class SpatialAttentionExtractor(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionExtractor, self).__init__()

        self.samconv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.samconv(x)
        return self.sigmoid(x)

class ChannelAttentionEnhancement(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttentionEnhancement, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class ChannelAttentionEnhancement3D(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttentionEnhancement3D, self).__init__()
        # 使用3D自适应池化
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # 输出形状为 [N, C, 1, 1, 1]
        self.max_pool = nn.AdaptiveMaxPool3d(1)  # 输出形状为 [N, C, 1, 1, 1]
        
        # 使用3D卷积层
        self.fc = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_planes // ratio, in_planes, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入x的形状为 [N, C, D, H, W]
        avg_out = self.fc(self.avg_pool(x))  # 平均池化后的特征
        max_out = self.fc(self.max_pool(x))  # 最大池化后的特征
        out = avg_out + max_out  # 特征融合
        return self.sigmoid(out)  # 输出形状为 [N, C, 1, 1, 1]

class CMCAM(nn.Module):
    """
       channel mamba coordinate attention module 
    """
    def __init__(self, in_channels,ca_type) -> None:
        """
            :param ca_type: linear, mamba
        """
        super().__init__()
        self.cae = ChannelAttentionEnhancement3D(in_channels)
        self.sae = CoordAttention(in_channels, proj_type=ca_type)
    
    def forward(self, x):
        x = self.cae(x)*x
        x = self.sae(x)
        return x
def coordinate_attention_linear(feature_in):
    return CoordAttention(feature_in, proj_type='linear')

def coordinate_attention_mamba(feature_in):
    return CoordAttention(feature_in, proj_type='mamba')

def cmcam_linear(feature_in):
    return CMCAM(feature_in, ca_type='linear')

def cmcam_mamba(feature_in):
     return CMCAM(feature_in, ca_type='mamba')