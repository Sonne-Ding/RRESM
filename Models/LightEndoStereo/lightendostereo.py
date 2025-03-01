from __future__ import print_function
import math
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
# from scipy.stats import multivariate_normal
from .WTrefinement import WTRefine
from .features import gwc_feature, Mobilenetv4Feature, Mobilenetv4Feature_depth
from .submodule import convbn_3d, disparity_compute, retrieve_corr_feature, disparity_compute_interp, coordinate_attention_mamba, cmcam_mamba
from .cost_volume import build_concat_volume, build_gwc_volume, LowHighFreqVolume
from .cost_aggregation import MCAHG, HG, MCAHG2
from .hourglass import HourglassRefine
from timm.models import register_model
# from .disp_refinement import MultiScaleRefine

class LightEndoStero_ablation_1(nn.Module):
    """使用mamba attention替代复杂的hourglass网络

    """
    def __init__(self, maxdisp=192, dispRefine=False,**kwargs):
        """
            :param featureNet: choice ['gwcnet', 'mobilenetv4']
            :param agg_net: choice ['HG', 'MCAHG', 'CAHG']
            :param dispRefine: choice ["LDE", "simple"]
            :param costVolume: choice ["gwc", "LHfv"] 
        """
        super().__init__()
        self.dispRefine = dispRefine
        self.maxdisp = maxdisp
        self.costVolume = "gwc"
        self.use_concat_volume = False
        # 12x8=96, 40x8 = 320
        self.num_groups = 32
        self.concat_channels = 12 if self.use_concat_volume else 0
        self.feature_extraction = Mobilenetv4Feature(out_channels=320)
        self.dres0 = nn.Sequential(
            convbn_3d(self.num_groups + self.concat_channels * 2, 32, 3, 1, 1),
            # coordinate_attention_mamba(32),
            # convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True)
            )
        self.classif_head0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                nn.ReLU(inplace=True),
                                nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
        self.classif_head1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                nn.ReLU(inplace=True),
                                nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
        self.classif_head2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                nn.ReLU(inplace=True),
                                nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
        self.classif_head3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                nn.ReLU(inplace=True),
                                nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
        self.dres2 = HG(32)
        self.dres3 = HG(32)
        self.dres4 = HG(32)
        self.classif_head1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
        if self.dispRefine:
            self.disp_refine = WTRefine(96, 1)
        # self.classif_head = nn.Conv3d(32, 1, kernel_size=5, padding=2, stride=1, )
        # self.mamba_regreesion1 = coordinate_attention_mamba(32)
        # self.mamba_regreesion2 = coordinate_attention_mamba(32)
        # self.depth_features = Mobilenetv4Feature_depth()
        
    def forward(self, left, right):
        features_left = self.feature_extraction(left, refine_feature=self.dispRefine)
        features_right = self.feature_extraction(right) # [B, 320, H/4, W/4]
        if self.use_concat_volume:
            gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], self.maxdisp // 4, self.num_groups)
            concat_volume = build_concat_volume(features_left["concat_feature"], features_right["concat_feature"],self.maxdisp // 4)
            volume = torch.cat((gwc_volume, concat_volume), 1)
        elif self.costVolume=="gwc":
            gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], self.maxdisp // 4, self.num_groups)
            # B, 40, disp//4, h//4, w//4
            volume = gwc_volume
        else:
            raise NotImplementedError("Not implemented cost volume.")
        
        cost0 = self.dres0(volume)
        out1 = self.dres2(cost0)
        out2 = self.dres3(out1)
        out3 = self.dres4(out2)
        full_shape = [self.maxdisp, left.size()[2], left.size()[3]]
        # true true
        if self.training:
            pred0 = self.classif_head0(cost0)
            pred0 = disparity_compute_interp(pred0, self.maxdisp, full_shape)
            
            pred1 = self.classif_head1(out1)
            pred1 = disparity_compute_interp(pred1, self.maxdisp, full_shape)

            pred2 = self.classif_head2(out2)
            pred2 = disparity_compute_interp(pred2, self.maxdisp, full_shape)
        
            pred3 = self.classif_head3(out3)
            pred3 = disparity_compute_interp(pred3, self.maxdisp, full_shape)
            if self.dispRefine:
                pred3 = self.disp_refine(features_left['refine_feature'], pred3)
            return [pred0, pred1, pred2, pred3]
        else:
            pred1 = self.classif_head3(out3)  
            pred1 = disparity_compute_interp(pred1, self.maxdisp, full_shape)
            if self.dispRefine:
                pred1 = self.disp_refine(features_left['refine_feature'], pred1)
            return [pred1]

        
            
class GwcDyNetv4_ablation(nn.Module):
    """使用mamba attention替代复杂的hourglass网络

    """
    def __init__(self, maxdisp=192, agg_net=False, dispRefine=False,**kwargs):
        """
            :param featureNet: choice ['gwcnet', 'mobilenetv4']
            :param agg_net: choice ['HG', 'MCAHG', 'CAHG']
            :param dispRefine: choice ["LDE", "simple"]
            :param costVolume: choice ["gwc", "LHfv"] 
        """
        super().__init__()
        self.aggNet = agg_net
        self.dispRefine = dispRefine
        self.maxdisp = maxdisp
        self.costVolume = "gwc"
        self.use_concat_volume = False
        # 12x8=96, 40x8 = 320
        self.num_groups = 32
        self.concat_channels = 12 if self.use_concat_volume else 0
        self.feature_extraction = Mobilenetv4Feature(out_channels=320)
        self.dres0 = nn.Sequential(
            convbn_3d(self.num_groups + self.concat_channels * 2, 32, 3, 1, 1),
            # coordinate_attention_mamba(32),
            # convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True)
            )
        self.classif_head0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                nn.ReLU(inplace=True),
                                nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
        if self.aggNet:
            self.cost_agg = MCAHG2(32)
            self.classif_head1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
        if self.dispRefine:
            self.disp_refine = WTRefine(96, 1)
        # self.classif_head = nn.Conv3d(32, 1, kernel_size=5, padding=2, stride=1, )
        # self.mamba_regreesion1 = coordinate_attention_mamba(32)
        # self.mamba_regreesion2 = coordinate_attention_mamba(32)
        # self.depth_features = Mobilenetv4Feature_depth()
        
    def forward(self, left, right):
        features_left = self.feature_extraction(left, refine_feature=self.dispRefine)
        features_right = self.feature_extraction(right) # [B, 320, H/4, W/4]
        if self.use_concat_volume:
            gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], self.maxdisp // 4, self.num_groups)
            concat_volume = build_concat_volume(features_left["concat_feature"], features_right["concat_feature"],self.maxdisp // 4)
            volume = torch.cat((gwc_volume, concat_volume), 1)
        elif self.costVolume=="gwc":
            gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], self.maxdisp // 4, self.num_groups)
            # B, 40, disp//4, h//4, w//4
            volume = gwc_volume
        else:
            raise NotImplementedError("Not implemented cost volume.")
        
        cost0 = self.dres0(volume)
        full_shape = [self.maxdisp, left.size()[2], left.size()[3]]
        if self.aggNet:
            if self.dispRefine:
                # true true
                cost1 = self.cost_agg(cost0)
                # if self.training:
                pred0 = self.classif_head0(cost0)
                pred0 = disparity_compute_interp(pred0, self.maxdisp, full_shape)
                pred1 = self.classif_head1(cost1)
                pred1 = disparity_compute_interp(pred1, self.maxdisp, full_shape) # b, height, width
                pred2 = self.disp_refine(features_left['refine_feature'], pred1)
                return [pred0, pred1, pred2]
                # pred1 = self.classif_head1(cost1)  
                # pred1 = disparity_compute_interp(pred1, self.maxdisp, full_shape)
                # pred1 = self.disp_refine(features_left['refine_feature'], pred1)
                # return [pred1]
            else:
                cost1 = self.cost_agg(cost0)
                if self.training:
                    pred0 = self.classif_head0(cost0)
                    pred0 = disparity_compute_interp(pred0, self.maxdisp, full_shape)
                    
                    pred1 = self.classif_head1(cost1)
                    pred1 = disparity_compute_interp(pred1, self.maxdisp, full_shape) # b, height, width
                    return [pred0, pred1]
                pred1 = self.classif_head1(cost1)  
                pred1 = disparity_compute_interp(pred1, self.maxdisp, full_shape)
                return [pred1]
        else:
            if self.dispRefine:
                if self.training:
                    pred0 = self.classif_head0(cost0)
                    pred0 = disparity_compute_interp(pred0, self.maxdisp, full_shape)
                    pred1 = self.disp_refine(features_left['refine_feature'], pred0)
                    return [pred0, pred1]
                pred0 = self.classif_head0(cost0)
                pred0 = disparity_compute_interp(pred0, self.maxdisp, full_shape)
                pred1 = self.disp_refine(features_left['refine_feature'], pred0)
                return [pred1]
            else:
                pred0 = self.classif_head0(cost0)
                pred0 = disparity_compute_interp(pred0, self.maxdisp, full_shape)
                return [pred0]
            


class LightEndoStereo(nn.Module):
    """使用mamba attention替代复杂的hourglass网络

    """
    def __init__(self, maxdisp=192, use_concat_volume=False, featureNet="mobilenetv4", 
                 contentLoss=False,**kwargs):
        """
            :param featureNet: choice ['gwcnet', 'mobilenetv4']
            :param agg_net: choice ['HG', 'MCAHG', 'CAHG']
            :param dispRefine: choice ["LDE", "simple"]
            :param costVolume: choice ["gwc", "LHfv"] 
        """
        super().__init__()
        self.maxdisp = maxdisp
        self.costVolume = "gwc"
        self.use_concat_volume = use_concat_volume
        self.contentLoss = contentLoss
        # 12x8=96, 40x8 = 320
        self.num_groups = 32
        self.concat_channels = 12 if self.use_concat_volume else 0
        

        if featureNet=="gwcnet":
            self.feature_extraction = gwc_feature(concat_feature=self.use_concat_volume,
                                                         concat_feature_channel=self.concat_channels)
        elif featureNet=="mobilenetv4":
            self.feature_extraction = Mobilenetv4Feature(out_channels=320)
        else:
            raise NotImplementedError("featureNet type not implemented")
        self.dres0 = nn.Sequential(
            convbn_3d(self.num_groups + self.concat_channels * 2, 32, 3, 1, 1),
            # coordinate_attention_mamba(32),
            # convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True)
            )
        # self.dres1 = nn.Sequential(
        #     convbn_3d(32, 32, 3, 1, 1),
        #     # coordinate_attention_mamba(32),
        #     cmcam_mamba(32),
        #     convbn_3d(32, 32, 3, 1, 1),
        #     nn.ReLU(inplace=True))
        self.cost_agg = MCAHG2(32)
        self.disp_refine = WTRefine(96, 1)
        
        self.classif_head00 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels * 2, 32, 3, 1, 1),
                                nn.ReLU(inplace=True),
                                nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
        self.classif_head0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                nn.ReLU(inplace=True),
                                nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
        self.classif_head1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
        # self.classif_head = nn.Conv3d(32, 1, kernel_size=5, padding=2, stride=1, )
        # self.mamba_regreesion1 = coordinate_attention_mamba(32)
        # self.mamba_regreesion2 = coordinate_attention_mamba(32)
        # self.depth_features = Mobilenetv4Feature_depth()
        
    def forward(self, left, right):
        features_left = self.feature_extraction(left, refine_feature=True)
        features_right = self.feature_extraction(right) # [B, 320, H/4, W/4]
        if self.use_concat_volume:
            gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], self.maxdisp // 4, self.num_groups)
            concat_volume = build_concat_volume(features_left["concat_feature"], features_right["concat_feature"],self.maxdisp // 4)
            volume = torch.cat((gwc_volume, concat_volume), 1)
        elif self.costVolume=="gwc":
            gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], self.maxdisp // 4, self.num_groups)
            # B, 40, disp//4, h//4, w//4
            volume = gwc_volume
        else:
            raise NotImplementedError("Not implemented cost volume.")
        
        cost0 = self.dres0(volume)
        cost1 = self.cost_agg(cost0)
        full_shape = [self.maxdisp, left.size()[2], left.size()[3]]
        if self.training:
            pred00 = self.classif_head00(volume)
            pred00 = disparity_compute_interp(pred00, self.maxdisp, full_shape)
            pred0 = self.classif_head0(cost0)
            pred0 = disparity_compute_interp(pred0, self.maxdisp, full_shape)
            pred1 = self.classif_head1(cost1)
            pred1 = disparity_compute_interp(pred1, self.maxdisp, full_shape) # b, height, width
            pred1 = self.disp_refine(features_left['refine_feature'], pred1)
            return [pred00, pred0, pred1]
        pred1 = self.classif_head1(cost1)  
        pred1 = disparity_compute_interp(pred1, self.maxdisp, full_shape)
        
        pred1 = self.disp_refine(features_left['refine_feature'], pred1)
        return [pred1]
        # cost0 = self.mamba_regreesion1(cost0)
        # cost0 = self.dres1(cost0) + cost0 # B,32,disp//4,h//4,w//4
        # cost0 = self.mamba_regreesion2(cost0)
        # cost1 = self.cost_agg(cost0)
        # full_shape = [self.maxdisp, left.size()[2], left.size()[3]]
        # out1 = self.classif_head1(cost1) # B, 1, disp//4, h//4, w//4
        # pred1 = disparity_compute_interp(out1, self.maxdisp, full_shape) # B, 1，H，W
        # return [pred1] 


@register_model
def gwcdynetv4_base(**kwargs):
    return GwcDyNetv4(**kwargs)

@register_model
def gwcdynet_abl(**kwargs):
    return GwcDyNet_ablation(**kwargs)

@register_model
def gwcdynet_abl_1(**kwargs):
    return GwcDyNet_ablation_1(**kwargs)