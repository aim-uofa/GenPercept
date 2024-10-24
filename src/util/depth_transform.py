# --------------------------------------------------------
# What Matters When Repurposing Diffusion Models for General Dense Perception Tasks? (https://arxiv.org/abs/2403.06090)
# Github source: https://github.com/aim-uofa/GenPercept
# Copyright (c) 2024, Advanced Intelligent Machines (AIM)
# Licensed under The BSD 2-Clause License [see LICENSE for details]
# Author: Guangkai Xu (https://github.com/guangkaixu/)
# --------------------------------------------------------------------------
# This code is based on Marigold and diffusers codebases
# https://github.com/prs-eth/marigold
# https://github.com/huggingface/diffusers
# --------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/aim-uofa/GenPercept#%EF%B8%8F-citation
# More information about the method can be found at https://github.com/aim-uofa/GenPercept
# --------------------------------------------------------------------------

import torch
import logging


def get_depth_normalizer(cfg_normalizer):
    if cfg_normalizer is None:

        def identical(x):
            return x

        depth_transform = identical

    elif "scale_shift_depth" == cfg_normalizer.type:
        depth_transform = ScaleShiftDepthNormalizer(
            norm_min=cfg_normalizer.norm_min,
            norm_max=cfg_normalizer.norm_max,
            min_max_quantile=cfg_normalizer.min_max_quantile,
            clip=cfg_normalizer.clip,
        )
    elif "scale_shift_disparity" == cfg_normalizer.type:
        depth_transform = ScaleShiftDisparityNormalizer(
            norm_min=cfg_normalizer.norm_min,
            norm_max=cfg_normalizer.norm_max,
            min_max_quantile=cfg_normalizer.min_max_quantile,
            clip=cfg_normalizer.clip,
        )
    else:
        raise NotImplementedError
    return depth_transform


class DepthNormalizerBase:
    is_absolute = None
    far_plane_at_max = None

    def __init__(
        self,
        norm_min=-1.0,
        norm_max=1.0,
    ) -> None:
        self.norm_min = norm_min
        self.norm_max = norm_max
        raise NotImplementedError

    def __call__(self, depth, valid_mask=None, clip=None):
        raise NotImplementedError

    def denormalize(self, depth_norm, **kwargs):
        # For metric depth: convert prediction back to metric depth
        # For relative depth: convert prediction to [0, 1]
        raise NotImplementedError


class ScaleShiftDepthNormalizer(DepthNormalizerBase):
    """
    Use near and far plane to linearly normalize depth,
        i.e. d' = d * s + t,
        where near plane is mapped to `norm_min`, and far plane is mapped to `norm_max`
    Near and far planes are determined by taking quantile values.
    """

    is_absolute = False
    far_plane_at_max = True

    def __init__(
        self, norm_min=-1.0, norm_max=1.0, min_max_quantile=0.02, clip=True
    ) -> None:
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.norm_range = self.norm_max - self.norm_min
        self.min_quantile = min_max_quantile
        self.max_quantile = 1.0 - self.min_quantile
        self.clip = clip

    def __call__(self, depth_linear, valid_mask=None, clip=None):
        clip = clip if clip is not None else self.clip

        if valid_mask is None:
            valid_mask = torch.ones_like(depth_linear).bool()
        valid_mask = valid_mask & (depth_linear > 0)

        try:
            # Take quantiles as min and max
            _min, _max = torch.quantile(
                depth_linear[valid_mask],
                torch.tensor([self.min_quantile, self.max_quantile]),
            )
        except:
            _min = depth_linear.min()
            _max = depth_linear.max()

        # scale and shift
        depth_norm_linear = (depth_linear - _min) / (
            _max - _min
        ) * self.norm_range + self.norm_min

        if clip:
            depth_norm_linear = torch.clip(
                depth_norm_linear, self.norm_min, self.norm_max
            )

        return depth_norm_linear

    def scale_back(self, depth_norm):
        # scale to [0, 1]
        depth_linear = (depth_norm - self.norm_min) / self.norm_range
        return depth_linear

    def denormalize(self, depth_norm, **kwargs):
        logging.warning(f"{self.__class__} is not revertible without GT")
        return self.scale_back(depth_norm=depth_norm)


class ScaleShiftDisparityNormalizer(DepthNormalizerBase):
    """
    Use near and far plane to linearly normalize depth,
        i.e. d' = d * s + t,
        where near plane is mapped to `norm_min`, and far plane is mapped to `norm_max`
    Near and far planes are determined by taking quantile values.
    """

    is_absolute = False
    far_plane_at_max = True

    def __init__(
        self, norm_min=-1.0, norm_max=1.0, min_max_quantile=0.02, clip=True
    ) -> None:
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.norm_range = self.norm_max - self.norm_min
        self.min_quantile = min_max_quantile
        self.max_quantile = 1.0 - self.min_quantile
        self.clip = clip

    def __call__(self, depth_linear, valid_mask=None, clip=None):
        clip = clip if clip is not None else self.clip

        if valid_mask is None:
            valid_mask = torch.ones_like(depth_linear).bool()
        valid_mask = valid_mask & (depth_linear > 0)

        try:
            # Take quantiles as min and max
            _min, _max = torch.quantile(
                depth_linear[valid_mask],
                torch.tensor([self.min_quantile, self.max_quantile]),
            )
        except:
            _min = depth_linear.min()
            _max = depth_linear.max()
        
        disp_linear = 1 / depth_linear
        del depth_linear
        disp_min = 1 / _max
        disp_max = 1 / _min
        disp_norm_linear = (disp_linear - disp_min) / (disp_max - disp_min)
        del disp_linear
        disp_norm_linear[~valid_mask] = disp_norm_linear.min()
        disp_norm_linear = (disp_norm_linear - disp_norm_linear.min())/(disp_norm_linear.max() - disp_norm_linear.min())  * self.norm_range + self.norm_min
        disp_norm_linear[~valid_mask] = self.norm_min
        del valid_mask

        if clip:
            disp_norm_linear = torch.clip(
                disp_norm_linear, self.norm_min, self.norm_max
            )

        return disp_norm_linear

    def scale_back(self, disp_norm):
        # scale to [0, 1]
        disp_linear = (disp_norm - self.norm_min) / self.norm_range
        return disp_linear

    def denormalize(self, disp_norm, **kwargs):
        logging.warning(f"{self.__class__} is not revertible without GT")
        return self.scale_back(disp_norm=disp_norm)
