import torch
from torch import nn
from torch.nn import functional as F
from .submodule import CMCAM

def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, i:] = refimg_fea[:, :, :, i:]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume


def groupwise_correlation(fea1, fea2, num_groups, use_cosine):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    if use_cosine:
        fea1 = fea1.view([B, num_groups, channels_per_group, H, W])
        fea2 = fea2.view([B, num_groups, channels_per_group, H, W])
        cost = 0.5-0.5*F.cosine_similarity(fea1, fea2, dim=2)
    else:
        cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups, use_cosine=False):
    B, C, H, W = refimg_fea.shape
    # 先对特征进行标准化
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:   
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups, use_cosine)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups, use_cosine)
    volume = volume.contiguous()
    return volume


class LowHighFreqVolume(nn.Module):
    def __init__(self,in_channels, maxdisp, num_groups):
        super().__init__()
        self.freq_filter = CMCAM(in_channels)
        self.maxdisp = maxdisp
        self.num_groups = num_groups
        
    def forward(self, refimg_fea, targetimg_fea,):
        """
            :param refimg_fea: B,C,H,W
        """
        ref_hf_mask = self.freq_filter(refimg_fea) # B,1,H,W
        tg_hf_mask = self.freq_filter(targetimg_fea) 
        ref_hf_feature = refimg_fea*ref_hf_mask
        ref_ctr_feature = refimg_fea*(1-ref_hf_mask)
        tg_hf_feature = targetimg_fea*tg_hf_mask
        tg_ctr_feature = targetimg_fea*(1-tg_hf_mask)
        corr_hf = build_gwc_volume(ref_hf_feature, tg_hf_feature, self.maxdisp, self.num_groups)
        corr_ctr = build_gwc_volume(ref_ctr_feature, tg_ctr_feature, self.maxdisp, self.num_groups)
        corr = torch.cat([corr_hf, corr_ctr],dim=1)
        return corr 