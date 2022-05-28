from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as tf
from torch.nn import Transformer
import logging

from .correlation_package.correlation import Correlation

from .modules_sceneflow import get_grid, WarpingLayer_SF
from .modules_sceneflow import initialize_msra, upsample_outputs_as
from .modules_sceneflow import upconv
from .modules_sceneflow import FeatureExtractor, MonoSceneFlowDecoder, ContextNetwork, ContextNetworkPred

from utils.interpolation import interpolate2d_as
from utils.sceneflow_util import flow_horizontal_flip, intrinsic_scale, get_pixelgrid, post_processing

from swin_model import build_model
from config_swin import get_config
import time

class MonoSceneFlow(nn.Module):
    def __init__(self, args):
        super(MonoSceneFlow, self).__init__()

        self._args = args
        self.num_chs = [3, 32, 64, 96, 128, 192]
        self.search_range = 4
        self.output_level = 4  # 4
        self.num_levels = 7

        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        if self._args.version == "correlation" or self._args.version == "diff" or self._args.version == "stacked" \
                or self._args.version == "predict":
            self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        if self._args.version == "concatenated_inputs":
            self.feature_pyramid_extractor = FeatureExtractor([2*3, 2*32, 2*64, 2*96, 2*128, 2*192])

        self.warping_layer_sf = WarpingLayer_SF()

        self.flow_estimators = nn.ModuleList()
        if self._args.version == "predict":
            self.flow_estimators_pred = nn.ModuleList()
        self.upconv_layers = nn.ModuleList()

        self.dim_corr = (self.search_range * 2 + 1) ** 2
        if self._args.version == "predict":
            self.attention_chs = [120] * 5
        if self._args.version == "correlation" or self._args.version == "diff" or self._args.version == "stacked" or self._args.version == "concatenated_inputs":
            self.attention_chs = [81]*5

        if self._args.version=="correlation":
            for l, ch in enumerate(self.num_chs[::-1]):
                if l > self.output_level:
                    break
                if l == 0:
                    num_ch_in = ch + self.attention_chs[l]
                else:
                    num_ch_in = ch + 32 + 3 + 1 + self.attention_chs[l]
                    self.upconv_layers.append(upconv(32, 32, 3, 2))
                layer_sf = MonoSceneFlowDecoder(num_ch_in)
                self.flow_estimators.append(layer_sf)

        if self._args.version=="concatenated_inputs":
            for l, ch in enumerate(self.num_chs[::-1]):
                if l > self.output_level:
                    break
                if l == 0:
                    num_ch_in = 2*ch
                else:
                    num_ch_in = 2*ch + 32 + 3 + 1
                    self.upconv_layers.append(upconv(32, 32, 3, 2))
                layer_sf = MonoSceneFlowDecoder(num_ch_in)
                self.flow_estimators.append(layer_sf)

        if self._args.version=="stacked" or self._args.version=="diff":
            for l, ch in enumerate(self.num_chs[::-1]):
                if l > self.output_level:
                    break
                if l == 0:
                    num_ch_in = 2*ch + self.attention_chs[l]
                else:
                    num_ch_in = 2*ch + 32 + 3 + 1 + self.attention_chs[l]
                    self.upconv_layers.append(upconv(32, 32, 3, 2))
                layer_sf = MonoSceneFlowDecoder(num_ch_in)
                self.flow_estimators.append(layer_sf)

        if self._args.version == "predict":
            for l, ch in enumerate(self.num_chs[::-1]):
                if l > self.output_level:
                    break
                if l == 0:
                    num_ch_in = 2 * ch + self.attention_chs[l]
                    num_ch_in_pred = 2 * ch + self.attention_chs[l]
                else:
                    num_ch_in = 2 * ch + 32 + 3 + 1 + self.attention_chs[l]
                    num_ch_in_pred = 2 * ch + 2 * 32 + 2 * 3 + 2 * 1 + self.attention_chs[l]
                    self.upconv_layers.append(upconv(32, 32, 3, 2))
                layer_sf = MonoSceneFlowDecoder(num_ch_in)
                self.flow_estimators.append(layer_sf)
                layer_sf_pred = MonoSceneFlowDecoder(num_ch_in_pred)
                self.flow_estimators_pred.append(layer_sf_pred)

        self.corr_params = {"pad_size": self.search_range, "kernel_size": 1, "max_disp": self.search_range,
                            "stride1": 1, "stride2": 1, "corr_multiply": 1}
        self.context_networks = ContextNetwork(32 + 3 + 1, self._args)
        if self._args.version=="predict":
            self.context_networks_pred = ContextNetworkPred(32 + 3 + 1)

        self.sigmoid = torch.nn.Sigmoid()

        initialize_msra(self.modules())

        if self._args.version=="stacked" or self._args.version=="diff" or self._args.version=="predict":
            if args.version=="stacked":
                factor = 2
            elif args.version=="diff":
                factor = 1
            elif args.version=="predict":
                factor = 3

            self.corr_swin = nn.ModuleList()
            # config, args, im_size, channel_size, embed_dim, num_heads, window_size, ape
            config0 = get_config(args, (4, 13), factor*192, self.attention_chs[0], [1], [4, 13])
            self.corr_swin.append(build_model(config0, args))
            config1 = get_config(args, (8, 26), factor*128, self.attention_chs[1], [1], [4, 13])
            self.corr_swin.append(build_model(config1, args))
            config2 = get_config(args, (16, 52), factor*96, self.attention_chs[2], [1], [4, 13])
            self.corr_swin.append(build_model(config2, args))
            config3 = get_config(args, (32, 104), factor*64, self.attention_chs[3], [1], [4, 13])
            self.corr_swin.append(build_model(config3, args))
            config4 = get_config(args, (64, 208), factor*32, self.attention_chs[4], [1], [4, 13])
            self.corr_swin.append(build_model(config4, args))


    def run_pwc(self, input_dict, x1_raw=None, x2_raw=None, k1=None, k2=None,
                mode=''):  # input_dict, input_dict['input_l1_aug'], input_dict['input_l2_aug'], input_dict['input_k_l1_aug'], input_dict['input_k_l2_aug']

        if not self._args.test:
            if mode == "l_aug":
                string_im = "input_"
                string_k = "input_k_"
                string_view = "l"
                string_mode = "_aug"

            elif mode == "r_flip":
                string_im = "input_"
                string_k = "k_"
                string_view = "r"
                string_mode = "_flip"

            # second last frame
            x1_raw = input_dict[string_im + string_view + str(1) + string_mode]
            k1 = input_dict[string_k + string_view + str(1) + string_mode]
            # last frame
            x2_raw = input_dict[string_im + string_view + str(2) + string_mode]
            k2 = input_dict[string_k + string_view + str(2) + string_mode]

        if self._args.version == "predict":
            x3_raw = input_dict[string_im + string_view + str(3) + string_mode]
            k3 = input_dict[string_k + string_view + str(3) + string_mode]

        start = time.time()
        if self._args.version == "predict":
            x0_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
            x1_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]
            x2_pyramid = self.feature_pyramid_extractor(x3_raw) + [x3_raw]
        if self._args.version == "stacked" or self._args.version == "diff" or self._args.version == "correlation":
            x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
            x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]
        if self._args.version == "concatenated_inputs":
            x12_raw = torch.cat((x1_raw, x2_raw), dim=1)
            x12_pyramid = self.feature_pyramid_extractor(x12_raw) + [x12_raw]
            x1_pyramid, x2_pyramid = [], []
            for l in range(0,len(x12_pyramid)):
                x1_pyramid.append(x12_pyramid[l][:, :int(x12_pyramid[l].size(1) / 2), :, :])
                x2_pyramid.append(x12_pyramid[l][:, int(x12_pyramid[l].size(1) / 2):, :, :])

        output_dict = {}
        sceneflows_f, sceneflows_b, disps_1, disps_2, pts_1, pts_2, pts_1tf, pts_2tf, output_depth_1, output_depth_2, attention_f_list,attention_b_list, corr_f_list, corr_b_list = [], [], [], [], [], [], [], [], [], [], [], [], [], []
        if self._args.version == "predict":
            sceneflows_f_pred, sceneflows_b_pred, disps_2_pred, disps_3_pred = [], [], [], []
        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x1_warp = x1
                x2_warp = x2

            else:
                flow_f = interpolate2d_as(flow_f, x1, mode="bilinear")
                flow_b = interpolate2d_as(flow_b, x1, mode="bilinear")
                disp_l1 = interpolate2d_as(disp_l1, x1, mode="bilinear")
                disp_l2 = interpolate2d_as(disp_l2, x1, mode="bilinear")
                x1_out = self.upconv_layers[l - 1](x1_out)
                x2_out = self.upconv_layers[l - 1](x2_out)
                x2_warp, pts1, pts1_tf, output_depth1 = self.warping_layer_sf(x2, flow_f, disp_l1, k1, input_dict[
                    'aug_size'])  # becuase K can be changing when doing augmentation
                x1_warp, pts2, pts2_tf, output_depth2 = self.warping_layer_sf(x1, flow_b, disp_l2, k2,
                                                                              input_dict['aug_size'])
                if self._args.version == "predict":
                    flow_f_pred = interpolate2d_as(flow_f_pred, x1, mode="bilinear")
                    flow_b_pred = interpolate2d_as(flow_b_pred, x1, mode="bilinear")
                    disp_l2_pred = interpolate2d_as(disp_l2_pred, x1, mode="bilinear")
                    disp_l3_pred = interpolate2d_as(disp_l3_pred, x1, mode="bilinear")
                    x2_out_pred = self.upconv_layers[l - 1](x2_out_pred)
                    x1_out_pred = self.upconv_layers[l - 1](x1_out_pred)

            # corr or swin
            if self._args.version=="stacked":
                attention_f = self.corr_swin[l](torch.cat((x1, x2_warp),dim=1), l, self._args)
                attention_b = self.corr_swin[l](torch.cat((x2, x1_warp),dim=1), l, self._args)
            elif self._args.version=="diff":
                attention_f = self.corr_swin[l](x1 - x2_warp, l, self._args)
                attention_b = self.corr_swin[l](x2 - x1_warp, l, self._args)
            elif self._args.version=="correlation":
                out_corr_f = Correlation.apply(x1, x2_warp, self.corr_params)
                out_corr_b = Correlation.apply(x2, x1_warp, self.corr_params)
                out_corr_relu_f = self.leakyRELU(out_corr_f)
                out_corr_relu_b = self.leakyRELU(out_corr_b)
            elif self._args.version == "predict":
                attention_f = self.corr_swin[l](torch.cat((x0_pyramid[l], x1, x2), dim=1), l, self._args)
                attention_b = self.corr_swin[l](torch.cat((x2, x1, x0_pyramid[l]), dim=1), l, self._args)

            # monosf estimator
            if l == 0:
                if self._args.version=="correlation":
                    x1_out, flow_f, disp_l1 = self.flow_estimators[l](torch.cat([x1, out_corr_relu_f], dim=1))
                    x2_out, flow_b, disp_l2 = self.flow_estimators[l](torch.cat([x2, out_corr_relu_b], dim=1))
                elif self._args.version=="stacked" or self._args.version=="diff" or self._args.version == "predict":
                    x1_out, flow_f, disp_l1 = self.flow_estimators[l](torch.cat([x1, attention_f, x2_warp], dim=1))
                    x2_out, flow_b, disp_l2 = self.flow_estimators[l](torch.cat([x2, attention_b, x1_warp], dim=1))
                elif self._args.version=="concatenated_inputs":
                    x1_out, flow_f, disp_l1 = self.flow_estimators[l](torch.cat([x1, x2_warp], dim=1))
                    x2_out, flow_b, disp_l2 = self.flow_estimators[l](torch.cat([x2, x1_warp], dim=1))
                if self._args.version == "predict":
                    x1_out_pred, flow_f_pred, disp_l2_pred = self.flow_estimators_pred[l](torch.cat([x1, attention_f, x2_warp], dim=1))
                    x2_out_pred, flow_b_pred, disp_l3_pred = self.flow_estimators_pred[l](torch.cat([x2, attention_b, x1_warp], dim=1))

            else:
                if self._args.version == "correlation":
                    x1_out, flow_f_res, disp_l1 = self.flow_estimators[l](torch.cat([out_corr_relu_f, x1, x1_out, flow_f, disp_l1], dim=1))
                    x2_out, flow_b_res, disp_l2 = self.flow_estimators[l](torch.cat([out_corr_relu_b, x2, x2_out, flow_b, disp_l2], dim=1))
                elif self._args.version == "stacked" or self._args.version == "diff" or self._args.version == "predict":
                    x1_out, flow_f_res, disp_l1 = self.flow_estimators[l](torch.cat([attention_f, x1, x1_out, flow_f, disp_l1, x2_warp], dim=1))
                    x2_out, flow_b_res, disp_l2 = self.flow_estimators[l](torch.cat([attention_b, x2, x2_out, flow_b, disp_l2, x1_warp], dim=1))
                elif self._args.version=="concatenated_inputs":
                    x1_out, flow_f_res, disp_l1 = self.flow_estimators[l](torch.cat([x2_warp, x1, x1_out, flow_f, disp_l1], dim=1))
                    x2_out, flow_b_res, disp_l2 = self.flow_estimators[l](torch.cat([x1_warp, x2, x2_out, flow_b, disp_l2], dim=1))
                if self._args.version == "predict":
                    x1_out_pred, flow_f_res_pred, disp_l2_pred = self.flow_estimators_pred[l](torch.cat(
                        [attention_f, x1, x1_out, flow_f, disp_l1, x2_warp, x1_out_pred, flow_f_pred, disp_l2_pred],
                        dim=1))
                    x2_out_pred, flow_b_res_pred, disp_l3_pred = self.flow_estimators_pred[l](torch.cat(
                        [attention_b, x2, x2_out, flow_b, disp_l2, x1_warp, x2_out_pred, flow_b_pred, disp_l3_pred],
                        dim=1))
                    flow_f_pred = flow_f_pred + flow_f_res_pred
                    flow_b_pred = flow_b_pred + flow_b_res_pred
                flow_f = flow_f + flow_f_res
                flow_b = flow_b + flow_b_res

            # upsampling or post-processing

            # attention_f_list.append(attention_f)
            # attention_b_list.append(attention_b)
            # # corr_f_list.append(out_corr_relu_f)
            # # corr_b_list.append(out_corr_relu_b)
            if l != self.output_level:
                disp_l1 = self.sigmoid(disp_l1) * 0.3
                disp_l2 = self.sigmoid(disp_l2) * 0.3
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)
                if self._args.version == "predict":
                    disp_l2_pred = self.sigmoid(disp_l2_pred) * 0.3
                    disp_l3_pred = self.sigmoid(disp_l3_pred) * 0.3
                    sceneflows_f_pred.append(flow_f_pred)
                    sceneflows_b_pred.append(flow_b_pred)
                    disps_2_pred.append(disp_l2_pred)
                    disps_3_pred.append(disp_l3_pred)
            else:

                flow_res_f, disp_l1 = self.context_networks(torch.cat([x1_out, flow_f, disp_l1], dim=1))
                flow_res_b, disp_l2 = self.context_networks(torch.cat([x2_out, flow_b, disp_l2], dim=1))

                if self._args.version == "predict":
                    flow_res_f_pred, disp_l2_pred = self.context_networks_pred(
                        torch.cat([x1_out_pred, flow_f_pred, disp_l2_pred], dim=1))
                    flow_res_b_pred, disp_l3_pred = self.context_networks_pred(
                        torch.cat([x2_out_pred, flow_b_pred, disp_l3_pred], dim=1))
                    flow_f_pred = flow_f_pred + flow_res_f_pred
                    flow_b_pred = flow_b_pred + flow_res_b_pred
                    sceneflows_f_pred.append(flow_f_pred)
                    sceneflows_b_pred.append(flow_b_pred)
                    sceneflows_f_pred.append(flow_f_pred)
                    sceneflows_b_pred.append(flow_b_pred)
                    disps_2_pred.append(disp_l2_pred)
                    disps_3_pred.append(disp_l3_pred)

                flow_f = flow_f + flow_res_f
                flow_b = flow_b + flow_res_b
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)
                break
            if l > 0:
                pts_1.append(pts1)
                pts_2.append(pts2)
                pts_1tf.append(pts1_tf)
                pts_2tf.append(pts2_tf)
                output_depth_1.append(output_depth1)
                output_depth_2.append(output_depth2)
            # ---------------------------------------------------------------------------------------------------------
            end = time.time()
            elapsed = end-start

        x1_rev = x1_pyramid[::-1]

        # output_depth_1.append(output_depth1)
        # output_depth_2.append(output_depth2)
        # pts_1.append(pts1)
        # pts_2.append(pts2)
        # pts_1tf.append(pts1_tf)
        # pts_2tf.append(pts2_tf)
        _, pts1, pts1_tf, output_depth1 = self.warping_layer_sf(x2_raw, interpolate2d_as(flow_f, x2_raw),
                                                                interpolate2d_as(disp_l1, x2_raw), k1,
                                                                input_dict['aug_size'])
        _, pts2, pts2_tf, output_depth2 = self.warping_layer_sf(x1_raw, interpolate2d_as(flow_b, x1_raw),
                                                                interpolate2d_as(disp_l2, x1_raw), k2,
                                                                input_dict['aug_size'])
        output_depth_1.append(output_depth1)
        output_depth_2.append(output_depth2)
        pts_1.append(pts1)
        pts_2.append(pts2)
        pts_1tf.append(pts1_tf)
        pts_2tf.append(pts2_tf)

        output_dict['flow_f'] = upsample_outputs_as(sceneflows_f[::-1], x1_rev)
        output_dict['flow_b'] = upsample_outputs_as(sceneflows_b[::-1], x1_rev)
        output_dict['flowf'] = sceneflows_f[::-1]
        output_dict['flowb'] = sceneflows_b[::-1]
        output_dict['disp_l1'] = upsample_outputs_as(disps_1[::-1], x1_rev)
        output_dict['disp_l2'] = upsample_outputs_as(disps_2[::-1], x1_rev)
        output_dict['disp_r1'] = upsample_outputs_as(disps_1[::-1], x1_rev)
        output_dict['displ1'] = disps_1[::-1]
        output_dict['displ2'] = disps_2[::-1]
        # output_dict['pts_1'] = upsample_outputs_as(pts_1[::-1], x1_rev)
        # output_dict['pts_2'] = upsample_outputs_as(pts_2[::-1], x1_rev)
        # output_dict['pts_1tf'] = upsample_outputs_as(pts_1tf[::-1], x1_rev)
        # output_dict['pts_2tf'] = upsample_outputs_as(pts_2tf[::-1], x1_rev)
        # output_dict['depth_1'] = upsample_outputs_as(output_depth_1[::-1], x1_rev)
        # output_dict['depth_2'] = upsample_outputs_as(output_depth_2[::-1], x1_rev)
        output_dict['input_k_l1_aug'] = input_dict['input_k_l1_aug']
        output_dict['k1'] = k1
        output_dict['k2'] = k2
        output_dict['x1_raw'] = x1_raw
        output_dict['x2_raw'] = x2_raw


        if self._args.version == "predict":
            output_dict['k3'] = k3
            output_dict['x3_raw'] = x3_raw
        output_dict['flow_f'] = upsample_outputs_as(sceneflows_f[::-1], x1_rev)
        output_dict['flow_b'] = upsample_outputs_as(sceneflows_b[::-1], x1_rev)
        output_dict['disp_l1'] = upsample_outputs_as(disps_1[::-1], x1_rev)
        output_dict['disp_l2'] = upsample_outputs_as(disps_2[::-1], x1_rev)
        output_dict['disp_r1'] = upsample_outputs_as(disps_1[::-1], x1_rev)
        # output_dict['pts_1'] = upsample_outputs_as(pts_1[::-1], x1_rev)
        # output_dict['pts_2'] = upsample_outputs_as(pts_2[::-1], x1_rev)
        # output_dict['pts_1tf'] = upsample_outputs_as(pts_1tf[::-1], x1_rev)
        # output_dict['pts_2tf'] = upsample_outputs_as(pts_2tf[::-1], x1_rev)
        # output_dict['depth_1'] = upsample_outputs_as(output_depth_1[::-1], x1_rev)
        # output_dict['depth_2'] = upsample_outputs_as(output_depth_2[::-1], x1_rev)
        output_dict['x2_warp'] = x2_warp
        output_dict['input_k_l1_aug'] = input_dict['input_k_l1_aug']
        if self._args.version == "predict":
            output_dict['flow_f_pred'] = upsample_outputs_as(sceneflows_f_pred[::-1], x1_rev)
            output_dict['flow_b_pred'] = upsample_outputs_as(sceneflows_b_pred[::-1], x1_rev)
            output_dict['disp_l2_pred'] = upsample_outputs_as(disps_2_pred[::-1], x1_rev)
            output_dict['disp_l3_pred'] = upsample_outputs_as(disps_3_pred[::-1], x1_rev)
            output_dict['flowf_pred'] = sceneflows_f_pred[::-1]
            output_dict['flowb_pred'] = sceneflows_b_pred[::-1]
            output_dict['displ2_pred'] = disps_2_pred[::-1]
            output_dict['displ3_pred'] = disps_3_pred[::-1]

        return output_dict

    def forward(self, input_dict):

        output_dict = {}

        ## Left
        # output_dict = self.run_pwc(input_dict, input_dict['input_l1_aug'], input_dict['input_l2_aug'], input_dict['input_k_l1_aug'], input_dict['input_k_l2_aug'])
        if not self._args.test:
            output_dict = self.run_pwc(input_dict, mode="l_aug")
        else:
            output_dict = self.run_pwc(input_dict, x1_raw=input_dict['input_l1_aug'], x2_raw=input_dict['input_l2_aug'],
                                       k1=input_dict['input_k_l1_aug'], k2=input_dict['input_k_l2_aug'])
        ## Right
        ## ss: train val
        ## ft: train
        if self.training or (not self._args.finetuning and not self._args.evaluation):  # training, test, validation
            if not self._args.test:
                for i in range(0, self._args.num_of_past_frames):
                    input_dict["input_r" + str(i + 1) + "_flip"] = torch.flip(
                        input_dict['input_r' + str(i + 1) + '_aug'], [3])
                    input_dict["k_r" + str(i + 1) + "_flip"] = input_dict["input_k_r" + str(i + 1) + "_flip_aug"]

                output_dict_r = self.run_pwc(input_dict, mode="r_flip")

                for ii in range(0, len(output_dict_r['flow_f'])):
                    output_dict_r['flow_f'][ii] = flow_horizontal_flip(output_dict_r['flow_f'][ii])
                    output_dict_r['flow_b'][ii] = flow_horizontal_flip(output_dict_r['flow_b'][ii])
                    output_dict_r['disp_l1'][ii] = torch.flip(output_dict_r['disp_l1'][ii], [3])
                    output_dict_r['disp_l2'][ii] = torch.flip(output_dict_r['disp_l2'][ii], [3])
                    if self._args.version=="predict":
                        output_dict_r['flow_f_pred'][ii] = flow_horizontal_flip(output_dict_r['flow_f_pred'][ii])
                        output_dict_r['flow_b_pred'][ii] = flow_horizontal_flip(output_dict_r['flow_b_pred'][ii])
                        output_dict_r['disp_l2_pred'][ii] = torch.flip(output_dict_r['disp_l2_pred'][ii], [3])
                        output_dict_r['disp_l3_pred'][ii] = torch.flip(output_dict_r['disp_l3_pred'][ii], [3])

                output_dict['output_dict_r'] = output_dict_r
            else:
                input_r1_flip = torch.flip(input_dict['input_r1_aug'], [3])
                input_r2_flip = torch.flip(input_dict['input_r2_aug'], [3])
                k_r1_flip = input_dict["input_k_r1_flip_aug"]
                k_r2_flip = input_dict["input_k_r2_flip_aug"]
                output_dict_r = self.run_pwc(input_dict, input_r1_flip, input_r2_flip, k_r1_flip, k_r2_flip)
                for ii in range(0, len(output_dict_r['flow_f'])):
                    output_dict_r['flow_f'][ii] = flow_horizontal_flip(output_dict_r['flow_f'][ii])
                    output_dict_r['flow_b'][ii] = flow_horizontal_flip(output_dict_r['flow_b'][ii])
                    output_dict_r['disp_l1'][ii] = torch.flip(output_dict_r['disp_l1'][ii], [3])
                    output_dict_r['disp_l2'][ii] = torch.flip(output_dict_r['disp_l2'][ii], [3])

                output_dict['output_dict_r'] = output_dict_r

        ## Post Processing
        ## ss:           eval
        ## ft: train val eval
        if self._args.test or self._args.evaluation or self._args.finetuning:  # test

            input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
            input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
            k_l1_flip = input_dict["input_k_l1_flip_aug"]
            k_l2_flip = input_dict["input_k_l2_flip_aug"]

            output_dict_flip = self.run_pwc(input_dict, x1_raw=input_l1_flip, x2_raw=input_l2_flip, k1=k_l1_flip,
                                            k2=k_l2_flip)

            flow_f_pp = []
            flow_b_pp = []
            disp_l1_pp = []
            disp_l2_pp = []

            for ii in range(0, len(output_dict_flip['flow_f'])):
                flow_f_pp.append(
                    post_processing(output_dict['flow_f'][ii], flow_horizontal_flip(output_dict_flip['flow_f'][ii])))
                flow_b_pp.append(
                    post_processing(output_dict['flow_b'][ii], flow_horizontal_flip(output_dict_flip['flow_b'][ii])))
                disp_l1_pp.append(
                    post_processing(output_dict['disp_l1'][ii], torch.flip(output_dict_flip['disp_l1'][ii], [3])))
                disp_l2_pp.append(
                    post_processing(output_dict['disp_l2'][ii], torch.flip(output_dict_flip['disp_l2'][ii], [3])))

            output_dict['flow_f_pp'] = flow_f_pp
            output_dict['flow_b_pp'] = flow_b_pp
            output_dict['disp_l1_pp'] = disp_l1_pp
            output_dict['disp_l2_pp'] = disp_l2_pp

        return output_dict